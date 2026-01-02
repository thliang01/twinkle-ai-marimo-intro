# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "openai==1.105.0",
#     "python-dotenv==1.1.1",
#     "watchdog",
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
    # 00 - 環境設定與首次呼叫 LLM API 🚀

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
      <a href="https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/00_setup_and_api_call.ipynb" target="_blank" style="margin: 2px;">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 00_setup_and_api_call In Colab" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://molab.marimo.io/notebooks/nb_zNsH5TyPTT6sjhDdVvw2Nx" target="_blank" style="margin: 2px;">
            <img src="https://molab.marimo.io/molab-shield.png" alt="Open in molab" style="display: inline-block; vertical-align: middle;"/>
      </a>
    </div>

    在這個 Notebook 中，你將學會：

    - 如何設定環境與 API Key（使用環境變數）
    - 如何呼叫 Nebius AI 提供的 **Gemma-3-27B-it-fast** 模型 API
    - 如何撰寫最小化的 API client
    - 實際體驗一次最簡單的 Prompt → Response 流程
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 安裝必要套件

    我們將使用 [OpenAI Python SDK](https://pypi.org/project/openai/) 和 python-dotenv
    這個 SDK 與多數 **OpenAI 相容 API** 完全相容，適合拿來呼叫 Nebius AI 提供的端點。
    python-dotenv 讓我們能安全地管理 API Key。
    """)
    return


@app.cell
def _():
    # 🛠️ 安裝最新版本 OpenAI SDK 和 python-dotenv
    # (use marimo's built-in package management features instead) 
    # !pip -q install --upgrade openai>=1.40.0 python-dotenv
    import os
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 設定環境變數與 .env 檔案

    **第一步：建立 .env 檔案**

    在你的專案根目錄建立一個 `.env` 檔案，內容如下：
    ```
    NEBIUS_API_KEY=your_actual_api_key_here
    ```

    **第二步：將 .env 加入 .gitignore**

    確保 `.env` 檔案不會被提交到版本控制系統：
    ```
    .env
    *.env
    ```

    **第三步：取得 Nebius AI 的 API Key**

    - **API Key**：這是存取 LLM API 服務的金鑰，請前往 [Nebius AI Studio](https://studio.nebius.ai) 註冊並取得。
    - **Base URL**：我們使用的服務端點是 `https://api.studio.nebius.ai/v1`

    ✅ 現在我們使用**環境變數**來安全管理 API Key，而不是直接寫在程式碼中！
    """)
    return


@app.cell
def _(os):
    from openai import OpenAI
    from dotenv import load_dotenv

    # ✅ Correct format
    # TWINKLE_API_KEY=sk-eT_04...

    # # Method 1: Set environment variable directly in code (for testing)
    # os.environ['NEBIUS_API_KEY'] = ''

    # Method 2: Try to get from environment variable
    # 載入 .env 檔案中的環境變數
    load_dotenv()

    # 從環境變數取得 API Key
    API_KEY = os.getenv('NEBIUS_API_KEY')
    BASE_URL = "https://api.studio.nebius.ai/v1"

    # 驗證 API Key 是否存在
    if not API_KEY:
        print("❌ 錯誤：找不到 NEBIUS_API_KEY 環境變數")
        print("請確保：")
        print("1. 已建立 .env 檔案")
        print("2. 前往 https://studio.nebius.ai 註冊並取得 API Key")
        print("3. 在 .env 檔案中設定 NEBIUS_API_KEY=你的API金鑰")

        raise ValueError("API Key 未設定")
    else:
        print(f"✅ 成功載入 API Key (前 8 字元: {API_KEY[:8]}...)")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL  # BASE_URL 已包含 /v1
    )

    MODEL = "google/gemma-3-27b-it-fast"  # Nebius AI 提供的可用模型
    return MODEL, client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 替代方案：直接設定環境變數

    如果你不想使用 .env 檔案，也可以直接在系統中設定環境變數：

    **Windows (Command Prompt):**
    ```cmd
    set NEBIUS_API_KEY=your_api_key_here
    marimo run 00_setup_and_llm_api_call_marimo.py
    ```

    **macOS/Linux (Terminal):**
    ```bash
    export NEBIUS_API_KEY=your_api_key_here
    marimo run 00_setup_and_llm_api_call_marimo.py
    ```

    **或者在執行時設定：**
    ```bash
    NEBIUS_API_KEY=your_api_key_here marimo run 00_setup_and_llm_api_call_marimo.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 發送第一次 Chat Completion

    我們來嘗試一個最小化的對話呼叫：

    - `system`：這是對模型的「角色指令」（system prompt），用來告訴模型要以什麼身份、什麼語氣來回應。例如，你可以指定它是「專業助理」、「法律顧問」或「客服人員」。這個設定會影響模型回應的風格與用詞。
    - `user`：代表使用者的輸入內容，也就是我們真正想問的問題或任務描述。模型會依照前面 system 的角色設定來解讀並生成回答。
    - `temperature`：控制生成的多樣性（0.7 代表中等創意）
    - `max_tokens`：限制模型回傳的字數

    如果呼叫成功，會收到一個包含回應的 JSON 結構。
    """)
    return


@app.cell
def _(MODEL, client):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是專業的助理，使用繁體中文回答。"},
                {"role": "user", "content": "請用一句話介紹什麼是大型語言模型（LLM）。"}
            ],
            temperature=0.7,
            max_tokens=256,
        )
        print("✅ 呼叫成功")
    except Exception as e:
        print("❌ 呼叫失敗，請檢查 API Key / base_url / 模型名稱是否正確。")
        print(f"錯誤詳情：{str(e)}")
        raise e
    return (resp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 解析並顯示模型回應

    成功呼叫後，回傳物件中會包含多個 `choices`，每個 choice 都有一段 `message.content`。
    """)
    return


@app.cell
def _(resp):
    if resp.choices:
        print("=== Model Output ===")
        print(resp.choices[0].message.content)
        print("\n=== 其他資訊 ===")
        print(f"模型：{resp.model}")
        print(f"使用 tokens：{resp.usage.total_tokens if resp.usage else 'N/A'}")
    else:
        import json
        print("⚠️ 非預期回傳格式：")
        print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 環境變數最佳實踐總結

    ✅ **做到了：**

    - 使用 `python-dotenv` 載入環境變數
    - API Key 不再硬編碼在程式碼中
    - 加入了 API Key 存在性驗證
    - 提供了多種設定環境變數的方法

    🔒 **安全性提升：**

    - `.env` 檔案不會被提交到版本控制
    - API Key 與程式碼分離
    - 可以在不同環境使用不同的 API Key

    📚 **後續學習：**

    - 在生產環境中，建議使用雲端服務的密鑰管理系統（如 AWS Secrets Manager、Azure Key Vault）
    - 可以設定多個環境變數來管理不同的配置（開發、測試、生產）
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
