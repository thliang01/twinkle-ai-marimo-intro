# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "watchdog",
#     "openai>=1.40.0",
#     "python-dotenv==1.1.1",
#     "huggingface-hub==0.34.4",
#     "datasets==4.0.0",
#     "pyarrow",
#     "pandas==2.3.2",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


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
    # 03 — 將資料集上傳到 Hugging Face Hub（Dataset Repo）
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
      <a href="https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/03_upload_to_hfhub.ipynb" target="_blank" style="margin: 2px;">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 03_upload_to_hfhub In Colab" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://molab.marimo.io/notebooks/nb_zNsH5TyPTT6sjhDdVvw2Nx" target="_blank" style="margin: 2px;">
            <img src="https://molab.marimo.io/molab-shield.png" alt="Open in molab" style="display: inline-block; vertical-align: middle;"/>
      </a>
    </div>

    本章目標：

    1. 準備要上傳的檔案（預期：`outputs/datasets.jsonl`）
    2. 使用 `huggingface_hub` 建立或覆用 **Dataset repo**
    3. 上傳 `data/train.jsonl`（選配：同時上傳 `train.parquet`）
    4. 建立 / 更新 Dataset Card（`README.md`）
    """)
    return


@app.cell
def _():
    # 確保 outputs 目錄存在
    # 在 marimo 中我們假設 datasets.jsonl 已經從前面的 notebook 產生
    import os
    os.makedirs("outputs", exist_ok=True)
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 環境變數設定

    **步驟一：建立 .env 檔案**

    在你的專案根目錄建立一個 `.env` 檔案，內容如下：
    ```
    TWINKLE_API_KEY=your_twinkle_api_key_here
    HF_TOKEN=hf_your_hugging_face_token_here
    HF_USERNAME=your_hugging_face_username
    ```

    **步驟二：取得 Hugging Face Token 和用戶名**

    - 前往 [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
    - 建立新的 token，選擇 **Write** 權限（用於上傳 dataset）
    - 複製 token 並加入到 `.env` 檔案中
    - 同時加入你的 HF 用戶名（確保你有該帳號的寫入權限）

    **步驟三：確保 .env 已加入 .gitignore**

    ```
    .env
    *.env
    ```

    **重要提醒：**

    - 使用你自己的 HF 用戶名，不要使用組織名稱（除非你有該組織的寫入權限）
    - Token 必須有 **Write** 權限才能上傳檔案
    - 如果遇到 403 錯誤，檢查用戶名和權限設定

    ✅ 使用**環境變數**來安全管理 API Key 和 Token！
    """)
    return


@app.cell
def _(os):
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
    from huggingface_hub import login as hf_login
    from pathlib import Path
    from dotenv import load_dotenv
    import json, time

    # 載入 .env 檔案中的環境變數
    load_dotenv()

    # === 基本設定（請依實際調整） ===
    HF_TOKEN = os.getenv('HF_TOKEN')  # 從環境變數取得 HF Token

    # 重要：使用你的個人帳號，而不是組織帳號（除非你有該組織的寫入權限）
    # ORG_OR_USER = "tw-llama"  # 你的組織/個人帳號
    # 建議改為你的個人 HF 用戶名，例如：
    ORG_OR_USER = os.getenv('HF_USERNAME')  # 從環境變數取得 HF 用戶名

    # 如果環境變數未設定，則在執行時動態取得用戶名
    if not ORG_OR_USER:
        print("⚠️ 未設定 HF_USERNAME 環境變數，將在登入後自動取得用戶名")
        ORG_OR_USER = "temp-placeholder"  # 暫時占位符，稍後會被替換

    DATASET_NAME = "twinkle-dialogue-gemma3-2025-08"  # 建議有日期與主題
    REPO_ID = f"{ORG_OR_USER}/{DATASET_NAME}"         # 例：your-username/twinkle-dialogue-gemma3-2025-08

    LOCAL_JSONL = Path("outputs/datasets.jsonl")      # 01/02 章節累積的主檔，註：這裡仍先以 datasets.jsonl，使用者可以再考慮要不要上傳 clean.jsonl

    # 驗證 HF Token 是否存在
    if not HF_TOKEN:
        print("❌ 錯誤：找不到 HF_TOKEN 環境變數")
        print("請確保：")
        print("1. 已建立 .env 檔案")
        print("2. 已從 https://huggingface.co/settings/tokens 取得 token")
        print("3. 在 .env 檔案中設定 HF_TOKEN=你的HuggingFace金鑰")
        raise ValueError("HF Token 未設定")
    else:
        print(f"✅ 成功載入 HF Token (前 8 字元: {HF_TOKEN[:8]}...)")

    if not LOCAL_JSONL.exists():
        print(f"❌ 找不到 {LOCAL_JSONL}，請先完成前面章節生成資料")
        raise FileNotFoundError(f"找不到 {LOCAL_JSONL}")

    # 可選：是否也上傳 Parquet（HF Hub 也會在後台自動生成 parquet 分支，但這裡示範手動輸出一次）
    ALSO_UPLOAD_PARQUET = True

    print("Repo:", REPO_ID)
    print("Local file:", LOCAL_JSONL.resolve())
    return (
        ALSO_UPLOAD_PARQUET,
        DATASET_NAME,
        HF_TOKEN,
        HfApi,
        LOCAL_JSONL,
        ORG_OR_USER,
        Path,
        hf_login,
        json,
        upload_file,
    )


@app.cell
def _(Path):
    CARD_PATH = Path("outputs/README.md")
    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 注意：HF 會讀取 README.md 頂端的 YAML 區塊作為中繼資料
    card_md = f"""---
    pretty_name: Twinkle Dialogue (Gemma-3-27B-it-fast, 2025-08)
    tags:
    - dialog
    - instruction-tuning
    - sft
    - openai-messages
    license: cc-by-4.0
    task_categories:
    - text-generation
    dataset_info:
      features:
      - name: messages
        sequence:
          - name: role
            dtype: string
          - name: content
            dtype: string
      splits:
      - name: train
        num_bytes: 123456
        num_examples: 1000
      download_size: 123456
      dataset_size: 123456
    language:
    - zh
    ---

    # Twinkle Dialogue (Gemma-3-27B-it-fast, 2025-08)
    <div align="left" style="line-height: 1;">
      <a href="https://discord.gg/Cx737yw4ed" target="_blank" style="margin: 2px;">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-Twinkle%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://huggingface.co/twinkle-ai" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Twinkle%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
      </a>
    </div>

    本資料集由 **Gemma-3-27B-it-fast（Nebius AI）** 生成之對話資料，採用 **OpenAI Chat Messages** 格式（`.jsonl`），並整合：
    - Reference-free（由 seed 派生單輪問答）
    - Reference-based（依據參考文本生成單輪問答）

    > 檔案路徑：`data/train.jsonl`（選配：`data/train.parquet`）

    ## 結構說明
    - 每列為一筆樣本：`{{"id": "...", "type": "...", "messages": [{{"role":"system","content":"..."}}, ...]}}`
    - 訓練時可擷取第一個 `user` 與對應 `assistant` 形成 (instruction, response) pair，或直接使用 chat 格式的 trainer。

    ## 來源與限制
    - Model: google/gemma-3-27b-it-fast（Nebius AI）
    - 語言：繁體中文
    - 使用情境：教學示範用；不代表專業意見

    ## 授權
    - 建議使用 **CC BY 4.0**；若另有需求請調整 `license` 欄位。
    """

    CARD_PATH.write_text(card_md, encoding="utf-8")
    print("✅ 產生 Dataset Card：", CARD_PATH.resolve())
    return (CARD_PATH,)


@app.cell
def _(ALSO_UPLOAD_PARQUET, LOCAL_JSONL, Path, json):
    if ALSO_UPLOAD_PARQUET:
        from datasets import Dataset
        import pandas as pd

        # 讀 jsonl → Dataset → parquet
        rows = []
        with LOCAL_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        ds = Dataset.from_pandas(pd.DataFrame(rows))
        PARQUET_PATH = Path("outputs/train.parquet")
        ds.to_parquet(PARQUET_PATH)
        print("✅ 產生 parquet：", PARQUET_PATH.resolve())
    else:
        PARQUET_PATH = None
    return (PARQUET_PATH,)


@app.cell
def _(Path):
    # HF 對部分副檔名會自動 LFS，但 .jsonl 有時未必；這裡顯式指定
    GITATTR_PATH = Path("outputs/.gitattributes")
    gitattributes = """*.jsonl filter=lfs diff=lfs merge=lfs -text
    *.parquet filter=lfs diff=lfs merge=lfs -text
    """
    GITATTR_PATH.write_text(gitattributes, encoding="utf-8")
    print("✅ 產生 .gitattributes")
    return (GITATTR_PATH,)


@app.cell
def _(DATASET_NAME, HF_TOKEN, HfApi, ORG_OR_USER, hf_login):
    from huggingface_hub import whoami

    hf_login(token=HF_TOKEN)  # 一次性登入本機快取
    user_info = whoami()
    actual_username = user_info["name"]
    print("✅ Logged in as:", actual_username)
    print("✅ User type:", user_info.get("type", "user"))

    # 如果環境變數未設定或設錯了，使用實際的登入用戶名
    if ORG_OR_USER == "temp-placeholder" or ORG_OR_USER != actual_username:
        print(f"🔄 更新用戶名：{ORG_OR_USER} → {actual_username}")
        final_username = actual_username
    else:
        final_username = ORG_OR_USER

    final_repo_id = f"{final_username}/{DATASET_NAME}"
    print("✅ Target repo:", final_repo_id)

    # ==== 先建立（或覆用）Dataset repo ====
    api = HfApi()
    try:
        repo_info = api.create_repo(
            repo_id=final_repo_id,
            repo_type="dataset",
            exist_ok=True,   # 已存在則不報錯
            private=False    # 需要私有可改 True
        )
        print("✅ Repository ready:", repo_info)
    except Exception as e:
        print(f"❌ Repository creation failed: {e}")
        print("💡 提示：確保你的 token 有 Write 權限")
        raise e
    return (final_repo_id,)


@app.cell
def _(
    CARD_PATH,
    GITATTR_PATH,
    LOCAL_JSONL,
    PARQUET_PATH,
    final_repo_id,
    upload_file,
):
    # 建議的 Hub 目錄結構
    REMOTE_JSONL = "data/train.jsonl"
    REMOTE_PARQUET = "data/train.parquet" if PARQUET_PATH else None
    REMOTE_CARD = "README.md"
    REMOTE_GITATTR = ".gitattributes"

    # 逐檔上傳（huggingface_hub 會自動處理 commit）
    upload_file(
        path_or_fileobj=str(LOCAL_JSONL),
        path_in_repo=REMOTE_JSONL,
        repo_id=final_repo_id,
        repo_type="dataset",
    )

    upload_file(
        path_or_fileobj=str(CARD_PATH),
        path_in_repo=REMOTE_CARD,
        repo_id=final_repo_id,
        repo_type="dataset",
    )

    upload_file(
        path_or_fileobj=str(GITATTR_PATH),
        path_in_repo=REMOTE_GITATTR,
        repo_id=final_repo_id,
        repo_type="dataset",
    )

    if PARQUET_PATH and PARQUET_PATH.exists():
        upload_file(
            path_or_fileobj=str(PARQUET_PATH),
            path_in_repo=REMOTE_PARQUET,
            repo_id=final_repo_id,
            repo_type="dataset",
        )

    print("✅ 上傳完成")
    print(f"👉 瀏覽： https://huggingface.co/datasets/{final_repo_id}")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
