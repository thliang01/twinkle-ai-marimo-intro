# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "watchdog==6.0.0",
#     "openai>=1.40.0",
#     "python-dotenv",
#     "huggingface_hub",
#     "datasets",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="background-color: #f0f8ff; border: 2px solid #4682b4; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 14px; color: #2c5282;">
            ℹ️ <strong>Fork Notice:</strong> This is a fork version adapted for molab/marimo from the original repository:
            <a href="https://github.com/ai-twinkle/llm-lab" target="_blank" style="color: #2c5282; text-decoration: underline;">https://github.com/ai-twinkle/llm-lab</a>
        </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🚀 LLM 對話資料集生成課程索引

    <div align="left" style="line-height: 1;">
      <a href="https://discord.gg/Cx737yw4ed" target="_blank" style="margin: 2px;">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-Twinkle%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://huggingface.co/twinkle-ai" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Twinkle%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://github.com/ai-twinkle" target="_blank" style="margin: 2px;">
        <img alt="GitHub" src="https://img.shields.io/badge/GitHub-ai--twinkle-181717?logo=github&logoColor=white&color=181717" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://molab.marimo.io" target="_blank" style="margin: 2px;">
            <img src="https://molab.marimo.io/molab-shield.png" alt="Open in molab" style="display: inline-block; vertical-align: middle;"/>
      </a>
    </div>

    ## 📚 課程概述

    本課程將帶您完整體驗從 **LLM API 呼叫** 到 **對話資料集生成與發佈** 的完整流程。透過實作學習如何：

    - 🔧 **環境設定與 API 串接**：學會安全管理 API Key 並呼叫 LLM 服務
    - 📊 **資料生成策略**：掌握 Reference-free 與 Reference-based 兩種生成方式
    - 🔍 **品質控制流程**：實作規則式檢查與 LLM-as-Judge 評估
    - 🚀 **資料集發佈**：上傳到 Hugging Face Hub 並撰寫完整的 Dataset Card

    ### 🎯 學習目標
    - 建立可重現的 LLM 資料生成 pipeline
    - 掌握對話資料的品質控制方法
    - 學會專業的資料集發佈流程
    - 理解不同對話格式的應用場景
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📋 課程大綱與導航

    ### 📁 課程結構
    本課程採用 **Colab-first** 設計，所有 notebook 都可在 Google Colab 中直接執行，同時提供 **marimo** 版本供本地互動式開發，以及 molab 雲端版本供線上執行。

    ---

    ### 📖 Notebook 清單

    #### 🔧 **00 - 環境設定與首次呼叫 LLM API**
    > *設定開發環境，學習安全的 API Key 管理，並完成第一次 LLM API 呼叫*

    - **Marimo 本地版本**: `00_setup_and_llm_api_call_marimo.py`
    - **Colab 版本**: [00_setup_and_api_call.ipynb](https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-0820-llm-dialogue-dataset/00_setup_and_api_call.ipynb)
    - **學習重點**:
      - 🔐 使用 python-dotenv 安全管理 API Key
      - 🌐 串接 Nebius AI 的 Gemma-3-27B-it-fast API
      - 📝 理解 OpenAI SDK 的基本用法
      - ✨ 完成第一次成功的 API 呼叫

    ---

    #### 📊 **01 - 對話資料生成與格式介紹**
    > *學習兩種主要的資料生成策略，並了解常見的對話資料格式*

    - **Marimo 本地版本**: `01_dialogue_generation_and_formats_marimo.py`
    - **Colab 版本**: [01_generate_dialogs.ipynb](https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/01_generate_dialogs.ipynb)
    - **學習重點**:
      - 🎯 **Reference-free**: 基於 seed 任務的自主生成
      - 📚 **Reference-based**: 基於參考文本的問答生成
      - 📋 格式比較：Alpaca vs ShareGPT vs OpenAI Messages
      - 💾 JSONL vs Parquet 格式選擇與應用

    ---

    #### 🔍 **02 - 資料品質檢查與過濾**
    > *實作完整的品質控制流程，確保資料集的品質與安全性*

    - **Marimo 本地版本**: `02_quality_checks_marimo.py`
    - **Colab 版本**: [02_quality_checks.ipynb](https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/02_quality_checks.ipynb)
    - **學習重點**:
      - 📏 **結構檢查**: 驗證對話格式完整性
      - 🔒 **敏感詞過濾**: 移除個資與不當內容
      - 📊 **統計分析**: 生成品質報表與分佈統計
      - 🤖 **LLM-as-Judge**: 語義層面的品質評估

    ---

    #### 🚀 **03 - 上傳到 Hugging Face Hub**
    > *將清洗後的資料集發佈到 Hugging Face Hub，建立完整的 Dataset Card*

    - **Marimo 本地版本**: `03_upload_to_hfhub_marimo.py`
    - **Colab 版本**: [03_upload_to_hfhub.ipynb](https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/03_upload_to_hfhub.ipynb)
    - **學習重點**:
      - 🔑 HF Token 安全管理與權限設定
      - 📦 多格式上傳：JSONL + Parquet
      - 📄 撰寫專業的 Dataset Card
      - 🏷️ 正確的 metadata 與 licensing 設定
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🛠️ 開發環境設定

    ### 📋 前置需求

    #### 1. **Python 環境**
    ```bash
    # 建議使用 Python 3.8+
    python --version
    ```

    #### 2. **虛擬環境設定**
    ```bash
    # 使用 uv 安裝 Python 環境 (推薦)
    # 參考: https://docs.astral.sh/uv/guides/install-python/
    uv python install 3.12
    uv venv twinkle-labs
    source twinkle-labs/bin/activate

    # 或傳統方式啟動現有虛擬環境
    source twinkle-labs/bin/activate
    ```

    #### 3. **必要套件**
    主要依賴已包含在虛擬環境中，額外需要：

    - `openai>=1.40.0` - OpenAI SDK
    - `python-dotenv` - 環境變數管理
    - `huggingface_hub` - HF Hub 上傳
    - `datasets` - 資料處理
    - `pyarrow` - Parquet 格式支援

    ### 🔐 API Key 設定

    建立 `.env` 檔案：
    ```env
    NEBIUS_API_KEY=your_nebius_api_key_here
    HF_TOKEN=hf_your_hugging_face_token_here
    HF_USERNAME=your_hf_username
    ```

    **重要提醒：**

    - 前往 [Nebius AI Studio](https://studio.nebius.ai) 註冊並取得 `NEBIUS_API_KEY`
    - 從 [HF Settings > Access Tokens](https://huggingface.co/settings/tokens) 取得 `HF_TOKEN` (需要 **Write** 權限)
    - 確保 `.env` 已加入 `.gitignore`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 學習路徑建議

    ### 🔰 **初學者路徑** (完整體驗)
    1. **00 - 環境設定** → 建立基礎開發環境
    2. **01 - 資料生成** → 理解兩種生成策略
    3. **02 - 品質控制** → 學會資料清洗流程
    4. **03 - 資料發佈** → 完成專業資料集發佈

    ### ⚡ **進階用戶路徑** (重點實作)
    - 熟悉 LLM API → 直接從 **01** 開始
    - 已有資料集 → 直接學習 **02** 品質控制
    - 專注發佈流程 → 重點學習 **03** 上傳流程

    ### 📊 **研究導向路徑** (深度理解)
    - 詳細研讀每個 notebook 的理論說明
    - 嘗試修改生成策略與品質標準
    - 比較不同格式與方法的效果

    ---

    ## 🔗 相關資源

    ### 📚 **延伸學習**

    - [Hugging Face Datasets 文件](https://huggingface.co/docs/datasets/)
    - [OpenAI API 參考](https://platform.openai.com/docs/api-reference)
    - [Self-Instruct 論文](https://arxiv.org/abs/2212.10560)

    ### 🤝 **社群支持**

    - 💬 [Discord - Twinkle AI](https://discord.gg/Cx737yw4ed)
    - 🐙 [GitHub - ai-twinkle](https://github.com/ai-twinkle)
    - 🤗 [Hugging Face - twinkle-ai](https://huggingface.co/twinkle-ai)

    ### 📋 **課程反饋**
    如有任何問題或建議，歡迎透過以下方式聯繫：

    - GitHub Issues
    - Discord 社群討論
    - Hugging Face 社群空間

    ---

    ## 🚀 開始學習

    準備好開始您的 LLM 對話資料集生成之旅了嗎？

    **建議從 `00_setup_and_llm_api_call_marimo.py` 開始！** 🎯
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
