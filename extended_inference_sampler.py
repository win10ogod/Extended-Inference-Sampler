#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended Inference Sampler (EIS) – 推理时间扩展采样器
====================================================

本模块实现了一种推理时扩展采样器，核心思想为：
    对于当前生成状态 x，
    1. 首先计算下一个 token 的概率分布，从中取 top_k 候选 token（初始对数概率 log p(c|x)）。
    2. 对于每个候选 token c，通过 rollout 生成后续 L 个 token（前瞻步数 lookahead_steps），
       计算累计折扣对数概率：
           S_rollout(c) = Σ_{t=1}^L [γ^t * log p(c_t | x, c, c_{1:t-1})]
    3. 扩展得分 S_ext(c) = log p(c|x) + S_rollout(c)
    4. 根据扩展得分选择最佳候选 token 更新生成序列。

【优化内容】
1. **缓存优化**  
   - 若 model_kwargs 包含 past_key_values（模型缓存），在主生成循环中将其更新，避免重复计算。
2. **采样策略扩展**  
   - 添加 temperature 参数（默认1.0）以调节温度。
   - 添加 top_p（nucleus采样）参数（默认1.0表示不开启）。
3. **早停机制**  
   - 当所有样本均生成 EOS 时提前退出生成循环。
4. **生成质量指标**  
   - 在候选 rollout 时记录扩展得分，并在生成结束后（或调试模式下）统计平均扩展得分。

使用示例：
    sampler = ExtendedInferenceSampler(model, tokenizer,
                                         lookahead_steps=3,
                                         discount_factor=0.9,
                                         top_k=5,
                                         temperature=0.8,
                                         top_p=0.95,
                                         use_cache=True,
                                         debug=True)
    output_ids = sampler.generate(input_ids, generation_config=gen_config, **model_kwargs)

请根据需要进一步扩展缓存更新部分（本代码中 candidate rollout 依然采用 deepcopy 以确保独立计算）。

Author: [Your Name]
Date: [发布日期]
"""

import copy
import torch
import torch.nn.functional as F

def top_p_filtering(logits, top_p=1.0, min_tokens_to_keep=1, filter_value=-float("Inf")):
    """
    对 logits 应用 nucleus (top-p) 过滤。保持累计概率大于 top_p 的最小集合，
    其余位置设为 filter_value。
    
    Args:
        logits (torch.Tensor): [batch, vocab_size]
        top_p (float): 累计概率阈值
        min_tokens_to_keep (int): 最少保留 token 数
        filter_value (float): 被过滤 logits 的值
    Returns:
        filtered logits (torch.Tensor)
    """
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 找到累计概率超过 top_p 的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        # 确保至少保留 min_tokens_to_keep 个
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # 将对应位置的 logits 设置为 filter_value
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

class ExtendedInferenceSampler:
    def __init__(self, model, tokenizer, lookahead_steps=3, discount_factor=0.9, top_k=5,
                 temperature=1.0, top_p=1.0, rollout_mode="greedy", use_cache=True, debug=False):
        """
        初始化 ExtendedInferenceSampler
        
        Args:
            model: 支持自回归生成的 transformers 模型（要求实现 forward(inputs, **model_kwargs)）
            tokenizer: 与模型对应的 tokenizer
            lookahead_steps (int): 前瞻 rollout 长度 L
            discount_factor (float): 折扣因子 γ (0 < γ <= 1)
            top_k (int): 每步从候选 token 中选取数量
            temperature (float): 采样温度（调节 softmax 分布平滑程度，1.0为原始）
            top_p (float): nucleus 采样参数（<1.0 时启用）
            rollout_mode (str): rollout token 选择策略，“greedy” 或 “sample”
            use_cache (bool): 是否在主生成循环中更新并复用 past_key_values 缓存
            debug (bool): 是否输出扩展得分等调试指标
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lookahead_steps = lookahead_steps
        self.discount_factor = discount_factor
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        if rollout_mode not in ["greedy", "sample"]:
            raise ValueError("rollout_mode 必须为 'greedy' 或 'sample'")
        self.rollout_mode = rollout_mode
        self.use_cache = use_cache
        self.debug = debug

    def _rollout_batch(self, candidate_ids, model_kwargs):
        """
        对候选序列（形状 [B, seq_len]）进行 rollout，支持批量计算。
        
        Args:
            candidate_ids (torch.Tensor): [B, seq_len] 每行为一个候选扩展序列（含候选 token）
            model_kwargs: 模型 forward 所需的参数（复制使用以确保候选间独立）
        Returns:
            rollout_logprobs (torch.Tensor): [B] 累计折扣对数概率
        """
        B = candidate_ids.size(0)
        rollout_logprobs = torch.zeros(B, device=candidate_ids.device)
        rollout_ids = candidate_ids

        for t in range(1, self.lookahead_steps + 1):
            # 复制 kwargs 以避免缓存干扰
            outputs = self.model(rollout_ids, **copy.deepcopy(model_kwargs))
            next_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            # 应用温度调整
            if self.temperature != 1.0:
                next_logits = next_logits / self.temperature
            # 应用 nucleus (top-p) 过滤（若启用）
            next_logits = top_p_filtering(next_logits, top_p=self.top_p, min_tokens_to_keep=1)
            next_logprobs = F.log_softmax(next_logits, dim=-1)
            if self.rollout_mode == "greedy":
                next_tokens = next_logprobs.argmax(dim=-1, keepdim=True)
                selected_logprobs = next_logprobs.gather(dim=-1, index=next_tokens).squeeze(1)
            else:  # "sample"
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                selected_logprobs = next_logprobs.gather(dim=-1, index=next_tokens).squeeze(1)
            rollout_logprobs += (self.discount_factor ** t) * selected_logprobs
            rollout_ids = torch.cat([rollout_ids, next_tokens], dim=-1)
        return rollout_logprobs

    def generate(self, input_ids, generation_config=None, **model_kwargs):
        """
        使用扩展采样器生成序列，每步先计算 top_k 候选，再对候选进行 rollout 计算扩展得分，
        选择扩展得分最高的 token 更新生成序列；同时支持缓存更新、早停及生成指标记录。
        
        Args:
            input_ids (torch.Tensor): 初始 token ids，形状 [batch, seq_len]
            generation_config: 生成配置对象（需包含 max_new_tokens、eos_token_id 等，若未指定则默认）
            model_kwargs: 模型 forward 的其他参数（如 attention_mask, past_key_values 等）
        Returns:
            output_ids (torch.Tensor): 完整生成序列
        """
        max_new_tokens = (generation_config.max_new_tokens if generation_config is not None and hasattr(generation_config, "max_new_tokens")
                          else 20)
        eos_token_id = (generation_config.eos_token_id if generation_config is not None and hasattr(generation_config, "eos_token_id")
                        else None)
        batch_size = input_ids.size(0)
        output_ids = input_ids.clone()
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        extended_score_metrics = []  # 用于记录每步候选扩展得分的平均值（调试/评估）

        # 如果启用缓存，在主生成循环中复用 past_key_values
        if self.use_cache and "past_key_values" not in model_kwargs:
            model_kwargs["past_key_values"] = None

        for step in range(max_new_tokens):
            outputs = self.model(output_ids, **model_kwargs)
            logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            logprobs = F.log_softmax(logits, dim=-1)
            # 应用温度和 nucleus 过滤
            if self.temperature != 1.0:
                logits = logits / self.temperature
            logits = top_p_filtering(logits, top_p=self.top_p, min_tokens_to_keep=1)
            logprobs = F.log_softmax(logits, dim=-1)
            topk_logprobs, topk_indices = torch.topk(logprobs, self.top_k, dim=-1)  # [batch, top_k]

            # 针对未结束样本批量构造候选扩展序列
            unfinished_idx = torch.nonzero(unfinished, as_tuple=False).squeeze(1)
            if unfinished_idx.numel() > 0:
                current_ids = output_ids[unfinished_idx]  # [N, seq_len]
                N, seq_len = current_ids.size()
                # 重复当前序列：扩展为 [N, top_k, seq_len]
                repeated_ids = current_ids.unsqueeze(1).expand(N, self.top_k, seq_len)
                candidate_tokens = topk_indices[unfinished_idx].unsqueeze(-1)  # [N, top_k, 1]
                candidate_ids = torch.cat([repeated_ids, candidate_tokens], dim=-1)  # [N, top_k, seq_len+1]
                candidate_ids = candidate_ids.view(-1, seq_len + 1)  # [N * top_k, seq_len+1]
                candidate_logprobs = topk_logprobs[unfinished_idx].view(-1)  # [N * top_k]
                rollout_logprobs = self._rollout_batch(candidate_ids, copy.deepcopy(model_kwargs))  # [N * top_k]
                total_scores = candidate_logprobs + rollout_logprobs
                total_scores = total_scores.view(N, self.top_k)
                best_candidate_idx = torch.argmax(total_scores, dim=-1)  # [N]
                best_tokens = topk_indices[unfinished_idx, :][torch.arange(N, device=input_ids.device), best_candidate_idx]
                if self.debug:
                    avg_ext_score = total_scores.mean().item()
                    extended_score_metrics.append(avg_ext_score)
            else:
                best_tokens = torch.empty((0,), dtype=torch.long, device=input_ids.device)

            # 构造整批新 token（对已结束样本填 eos）
            new_tokens = []
            ptr = 0
            for i in range(batch_size):
                if not unfinished[i]:
                    new_tokens.append(torch.tensor(eos_token_id, device=input_ids.device))
                else:
                    new_tokens.append(best_tokens[ptr])
                    ptr += 1
            new_tokens = torch.stack(new_tokens).unsqueeze(1)  # [batch, 1]
            output_ids = torch.cat([output_ids, new_tokens], dim=-1)
            if eos_token_id is not None:
                unfinished = unfinished & (new_tokens.squeeze(1) != eos_token_id)
            # 若启用缓存，更新 model_kwargs 缓存（简化示例，直接使用模型返回的 past_key_values）
            if self.use_cache and hasattr(outputs, "past_key_values"):
                model_kwargs["past_key_values"] = outputs.past_key_values
            if not unfinished.any():
                break

        if self.debug and extended_score_metrics:
            avg_metric = sum(extended_score_metrics) / len(extended_score_metrics)
            print(f"[DEBUG] Average extended rollout score per step: {avg_metric:.4f}")
        return output_ids
