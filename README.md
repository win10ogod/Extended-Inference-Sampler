```
# Extended Inference Sampler (EIS)

Extended Inference Sampler (EIS) 是一個基於推理時擴展計算（Extended Inference）的採樣器，專門用於自回歸文本生成任務。它在生成每個 token 之前，會從模型輸出中選取 top‑K 候選 token，並對每個候選 token 執行一定步數的前瞻 rollout（lookahead rollout），計算出延伸得分，從而使模型在“花更多推理時間”後再做出選擇。此設計靈感來自於 OpenAI o1/o3‑mini 系列模型以及相關推理時計算（inference‑time compute）的研究成果。

## 特性

- **前瞻 Rollout**  
  - 為每個候選 token 執行可配置步數（lookahead_steps）的前瞻 rollout，計算延伸得分  
  - 得分計算公式：  
    \[
    S_{\text{ext}}(c) = \log p(c \mid x) + \sum_{t=1}^{L} \gamma^t \, \log p(c_t \mid x, c, c_{1:t-1})
    \]
    其中：  
    - \( x \) 為當前生成上下文  
    - \( c \) 為候選 token  
    - \( L \) 為 rollout 步數  
    - \( \gamma \) 為折扣因子

- **多策略採樣**  
  - 支持基於 top‑K 候選的採樣  
  - 可選擇 nucleus (top‑p) 採樣（通過 `top_p` 參數啟用）  
  - 支持溫度（temperature）調節，以控制 softmax 分布的平滑程度  
  - Rollout 選擇策略可設置為 “greedy”（貪婪）或 “sample”（隨機采樣）

- **緩存與早停**  
  - 可啟用模型自身的 past_key_values 緩存，避免重複計算，從而提高推理效率  
  - 當所有生成序列均遇到 EOS 時提前停止生成

- **生成質量評估**  
  - Debug 模式下會記錄每一步候選的平均延伸得分，便於生成質量的評估與調試

## 目錄

- [Extended Inference Sampler (EIS)](#extended-inference-sampler-eis)
  - [特性](#特性)
  - [項目結構](#項目結構)
  - [安裝要求](#安裝要求)
  - [安裝](#安裝)
  - [使用方法](#使用方法)
    - [基本用法](#基本用法)
    - [參數說明](#參數說明)
  - [API 參考](#api-參考)
  - [潛在優化空間](#潛在優化空間)
  - [貢獻](#貢獻)
  - [授權](#授權)

## 項目結構

該項目主要包含以下模塊：
- **ExtendedInferenceSampler**  
  主類，封裝了擴展推理採樣器的所有功能。主要方法包括：
  - `__init__`：初始化模型、tokenizer 以及各項採樣策略參數。
  - `_rollout_batch`：對一批候選序列並行執行前瞻 rollout，計算累積折扣對數概率。
  - `generate`：主生成循環，每步根據 top‑K 候選 token 的延伸得分選取最佳 token，並更新輸出序列；同時支持緩存更新與早停。
- **top_p_filtering**  
  用於實現 nucleus 採樣的輔助函數，將 logits 中不屬於 top‑p 候選的部分過濾掉。

## 安裝要求

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- 其他依賴見項目 requirements.txt（若有）

## 安裝

可以通過 pip 安裝依賴：
```bash
pip install torch transformers
```

然後將本項目克隆到本地或直接將 `extended_inference_sampler.py` 模塊添加到您的項目中。

## 使用方法

### 基本用法

下面是一個簡單示例，展示如何使用 EIS 採樣器生成文本：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from extended_inference_sampler import ExtendedInferenceSampler

# 載入模型與分詞器（以 GPT-2 為例）
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 準備提示文本
prompt = "It was a dark and stormy"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 可選生成配置
gen_config = GenerationConfig(max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)

# 初始化 ExtendedInferenceSampler
sampler = ExtendedInferenceSampler(
    model, tokenizer,
    lookahead_steps=3,
    discount_factor=0.9,
    top_k=5,
    temperature=0.8,
    top_p=0.95,
    rollout_mode="greedy",   # "greedy" 或 "sample"
    use_cache=True,
    debug=True
)

# 生成文本
output_ids = sampler.generate(input_ids, generation_config=gen_config)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:", generated_text)
```

### 參數說明

- **lookahead_steps**: 前瞻 rollout 長度 L（例如 3），即在候選 token 之後模擬生成 3 個 token。
- **discount_factor**: 折扣因子 γ（例如 0.9），控制後續 rollout 得分的重要性，γ 越小後續影響越低。
- **top_k**: 每步從概率分布中選取的候選 token 數量。
- **temperature**: 採樣溫度，調節 softmax 分布平滑程度；值越低越接近貪婪選擇。
- **top_p**: nucleus 採樣參數；當小於 1.0 時會過濾掉累積概率低於該值的 token。
- **rollout_mode**: Rollout 時 token 選擇方式，可選 "greedy"（貪婪）或 "sample"（隨機采樣）。
- **use_cache**: 是否在生成循環中使用並更新模型的 past_key_values 緩存。
- **debug**: 是否啟用調試模式，輸出每步平均延伸得分等信息。

## API 參考

### ExtendedInferenceSampler

- `__init__(model, tokenizer, lookahead_steps, discount_factor, top_k, temperature, top_p, rollout_mode, use_cache, debug)`  
  初始化採樣器，設置所有參數。

- `_rollout_batch(candidate_ids, model_kwargs)`  
  對形狀 [B, seq_len] 的候選序列進行 rollout，返回形狀 [B] 的累積折扣對數概率。

- `generate(input_ids, generation_config, **model_kwargs)`  
  主生成方法。對每個生成步驟：
  1. 計算當前 logits 並選取 top_k 候選 token。
  2. 批量構造候選擴展序列並通過 `_rollout_batch` 計算 rollout 得分。
  3. 選擇擴展得分最高的 token，更新生成序列。
  4. 更新模型緩存（如果啟用）與未完成標誌，支持早停。
  
- `top_p_filtering(logits, top_p, min_tokens_to_keep, filter_value)`  
  輔助函數，用於對 logits 進行 nucleus 採樣過濾。

## 潛在優化空間

1. **緩存優化**  
   - 當模型支持 past_key_values 緩存時，可進一步優化 rollout 與主生成循環中對緩存的更新和復用，減少重複計算。

2. **內存優化**  
   - 處理超長序列時，可能存在內存瓶頸，可考慮結合 gradient checkpointing 或分塊計算策略來降低內存占用（對於推理可考慮內存映射）。

3. **採樣策略擴展**  
   - 除了目前的貪婪和隨機采樣 rollout 策略，還可添加更多策略（例如 nucleus 採樣、混合多策略採樣）以進一步平衡探索與利用。

4. **早停機制**  
   - 除了遇到 EOS 時停止外，可根據生成質量指標（例如延伸得分低於某閾值）提前終止生成，提高計算效率。

5. **生成質量指標**  
   - 增加更多指標（如每步候選的平均延伸得分、最終序列得分統計等），便於在調試和自動化評估中衡量生成質量。

## 貢獻

歡迎任何形式的貢獻，包括但不限於：
- 提交 issue 反饋使用中遇到的問題或建議
- 提交 pull request 改進緩存機制、內存優化或新增採樣策略
- 提出並討論早停策略和生成質量指標的改進方案

請參閱 [CONTRIBUTING.md](CONTRIBUTING.md) 以了解詳細指南。

## 授權

本項目基於 MIT 許可證，詳情請參閱 [LICENSE](LICENSE) 文件。

---
希望本項目能夠幫助研究者和開發者在推理時計算擴展中獲得更好的生成效果，並期待社區的貢獻與交流！
```
