# Contributing to Extended Inference Sampler

感謝您有興趣為Extended Inference Sampler (EIS)專案做出貢獻！本文件將幫助您了解如何參與項目開發。

## 目錄
- [行為準則](#行為準則)
- [開始貢獻](#開始貢獻)
- [開發流程](#開發流程)
- [提交指南](#提交指南)
- [代碼風格](#代碼風格)
- [測試指南](#測試指南)
- [文檔貢獻](#文檔貢獻)

## 行為準則

我們期望所有貢獻者都能夠：
- 保持友善和尊重
- 接受建設性的批評
- 關注什麼對社群最有利
- 展現同理心

## 開始貢獻

1. Fork 此專案到您的 GitHub 帳戶
2. Clone 您的 fork:
```bash
git clone https://github.com/YOUR-USERNAME/extended-inference-sampler.git
```
3. 創建新的分支:
```bash
git checkout -b feature/your-feature-name
```

## 開發流程

1. **環境設置**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **運行測試**
```bash
pytest tests/
```

3. **本地開發**
- 確保您的更改不會破壞現有功能
- 添加適當的測試用例
- 更新相關文檔

## 提交指南

1. **Commit 信息格式**
```
<type>(<scope>): <subject>

<body>

<footer>
```

Type 可以是:
- feat: 新功能
- fix: Bug修復
- docs: 文檔更新
- style: 代碼格式調整
- refactor: 代碼重構
- test: 添加測試
- chore: 構建過程或輔助工具的變動

2. **Pull Request 流程**
- 確保PR描述清晰地說明了更改的內容和原因
- 關聯相關的issue
- 確保所有測試都通過
- 請求至少一位維護者審查

## 代碼風格

1. **Python 代碼風格**
- 遵循 PEP 8 規範
- 使用 Black 進行代碼格式化
- 使用 isort 排序導入
- 最大行長度為88字符

2. **文檔字符串**
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    簡短的功能描述。

    詳細的功能描述，可以包含多行。

    Args:
        param1: 參數1的描述
        param2: 參數2的描述

    Returns:
        返回值的描述

    Raises:
        可能拋出的異常
    """
```

## 測試指南

1. **單元測試要求**
- 每個新功能都必須有對應的測試
- 測試覆蓋率應保持在85%以上
- 使用 pytest 框架
- 測試文件放在 `tests/` 目錄下

2. **運行測試**
```bash
# 運行所有測試
pytest

# 運行特定測試
pytest tests/test_specific.py

# 查看覆蓋率報告
pytest --cov=eis tests/
```

## 文檔貢獻

1. **文檔更新**
- 確保示例代碼是最新的
- 修正任何錯別字或語法錯誤
- 保持文檔結構清晰

2. **生成文檔**
```bash
cd docs/
make html
```

## 問題與討論

- 使用 GitHub Issues 報告bug或提出新功能建議
- 在提交新issue之前，請搜索是否已存在相似的issue
- 使用提供的issue模板來提交問題

## 發布流程

維護者發布新版本時需要：
1. 更新版本號 (`__version__.py`)
2. 更新 CHANGELOG.md
3. 創建新的 release tag
4. 發布到 PyPI

感謝您的貢獻！如有任何問題，請隨時在 GitHub Issues 上與我們聯繫。
