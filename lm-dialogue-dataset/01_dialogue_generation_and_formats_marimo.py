# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "watchdog",
#     "openai==1.102.0",
#     "python-dotenv==1.1.1",
#     "datasets",
#     "pyarrow",
#     "pydantic==2.11.7",
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
    # 01 - 對話資料生成 & 對話集格式介紹
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
      <a href="https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/01_generate_dialogs.ipynb" target="_blank" style="margin: 2px;">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 00_setup_and_api_call In Colab" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://molab.marimo.io/notebooks/nb_zNsH5TyPTT6sjhDdVvw2Nx" target="_blank" style="margin: 2px;">
            <img src="https://molab.marimo.io/molab-shield.png" alt="Open in molab" style="display: inline-block; vertical-align: middle;"/>
      </a>
    </div>

    在這個 Lab，我們的目標是建立一份「可持續擴充」的對話資料集。主要的步驟如下：

    1. 連接 `Gemma-3-27B-it-fast` API（使用 OpenAI SDK）
    2. 介紹對話資料的常見格式：**Alpaca**, **ShareGPT**，以及 **OpenAI** 格式（我們採用後者）
    3. 探討 `.jsonl` 格式與 `.parquet` 格式的優缺點，並說明 HF Hub 對 parquet 的轉換支援
       (上傳 parquet 時 HF 會自動生成 `.parquet` 分支與 viewer)
    """)
    return


@app.cell
def _():
    import os
    from openai import OpenAI
    from dotenv import load_dotenv

    # 載入 .env 檔案中的環境變數
    load_dotenv()

    # 從環境變數取得 API Key
    API_KEY = os.getenv('NEBIUS_API_KEY')
    BASE_URL = "https://api.studio.nebius.ai/v1"
    MODEL = "google/gemma-3-27b-it-fast"

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

    # 初始化 API client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL  # BASE_URL 已包含 /v1
    )

    print("✅ API client 已初始化")
    return MODEL, client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 常見對話資料集格式比較

    - **Alpaca**：單輪指令 + 回答格式，通常是 Instruct tuning。（不適合多輪場景）
    - **ShareGPT**：使用以下格式，支援多角色多輪對話
    ```json
    {
        "conversations":[
            {"from":"human","value":"..."},
            {"from":"gpt","value":"..."}…]
    }
    ```
    - **OpenAI Chat Messages**：最常用格式，像這樣：
    ```json
    {
      "messages":[
          {"role":"system","content":"..."},
          {"role":"user","content":"..."},
          {"role":"assistant","content":"..."}]
    }
    ```
    Hugging Face 與多數工具都支援這種格式，且 OpenAI 本質上是 ShareGPT 格式的變體

    我們將採用 **OpenAI messages** 格式，並使用 `.jsonl` 儲存；

    另外，還有一種叫 `.parquet` 格式，他的優勢例如：

    - 高效壓縮、支援分欄讀取、大檔案操作快速
    - HF Hub 支援直接上傳 parquet，也會自動生成可公開瀏覽版本
    """)
    return


@app.cell
def _(mo):
    mo.image(
        src="assets/01_wiki_data_format.png",
        alt="圖 1：Wiki 對話格式示意圖",
        rounded=True,
        caption="圖 1：Wiki 對話格式示意圖",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## JSONL vs Parquet 比較

    | 格式     | 優點                          | 缺點                         |
    |----------|-------------------------------|------------------------------|
    | `.jsonl` | 易讀、輕量、開發友善              | 檔案大、大量數據讀取效率較低   |
    | `.parquet` | 壓縮效果好、查詢效能高、支援 HF 轉換 | 不易直接閱讀，需使用工具處理   |

    注意：即使你上傳 `.jsonl`，HF Hub 也可能幫你生成 `.parquet` 分支，方便瀏覽與載入。
    """)
    return


@app.cell
def _(mo):
    mo.image(
        src="assets/01_hf_parquet_branch.png",
        alt="圖 2：HF Hub 自動生成的 .parquet 分支",
        rounded=True,
        caption="圖 2：HF Hub 自動生成的 .parquet 分支",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference-Free vs Reference-Based

    - **Reference-Free（無參考）**：用一些 seed prompt 引導模型生成。最早出自 [Self-Instruct: Aligning Language Models with Self-Generated Instructions
    ](https://arxiv.org/abs/2212.10560)。
    - **Reference-Based（參考內容）**：使用真實資料片段（例如 Wiki 條目）作 prompt 佐料，讓生成內容更 grounded。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference-Free 實作

    在 Reference-Free 的情境下，我們並不依賴任何外部知識庫或文件，而是透過 **seed 任務 (seed task)** 來驅動模型自行生成資料。
    這些 seed 任務通常包含一個 **instruction（指令）**，加上少量的 **instance（範例輸入/輸出對）**，作為模型模仿與延伸的起點。

    這種方法的代表性工作是 *Self-Instruct*，它透過人工設計的一些高品質種子指令，讓模型去「舉一反三」產生更多指令和對應答案，最終建立出龐大的資料集。

    以下是一個取自 [self-instruct](https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl) seed 範例，主題是「早餐建議」。

    ```json
    {
      "id": "seed_task_0",
      "name": "breakfast_suggestion",
      "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?",
      "instances": [
        {
          "input": "\",
          "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1 tbsp flaxseed oil and 1/2 cup water, totaling about 550 calories. The 4 strips of bacon contains about 200 calories."
        }
      ],
      "is_classification": false
    }
    ```

    說明：

    - id：任務的唯一識別碼。
    - name：任務名稱，方便辨識。
    - instruction：給模型的主要問題或指令。
    - instances：包含輸入/輸出對，本例中 input 為空，代表模型直接依 instruction 回答；output 是一個可能的解答。
    - is_classification：標記此任務是否為分類型問題（此例為否）。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    在實務中，我們會設計數十到數百個 seed 任務，涵蓋不同領域與指令型態，作為 Reference-Free 資料生成的核心基礎。

    不過，我們的作法並**不完全等同於 Self-Instruct**。
    相較於 Self-Instruct 的完整 pipeline（如：過濾、去重、迭代擴展），我們傾向採用更簡單直接的方式：

    1.	人工撰寫少量高品質 seed 指令。
    2.	要求模型基於這些 seed 產生新的 seed 指令（但僅限輸出 seed 本文，避免雜訊）。
    3.	再利用這些新 seed 指令，由模型生成單輪問答配對。

    這樣的流程更輕量，雖然缺少複雜的篩選與多輪迭代，但對於課程實作與教學目標而言，已經能清楚展現 Reference-Free 的核心精神。
    """)
    return


@app.cell
def _(MODEL, client):
    import re

    base_seed = "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?"

    seed_gen_messages = [
        {'role': 'system', 'content': '你是一個資料生成器。你的任務是『根據給定 seed，產生一則不同但主題相關的 seed 指令』。\n務必遵守：\n1) 僅輸出新的 seed 指令本身（繁體中文）。\n2) 不要加任何解釋、前後文、引號、標點裝飾或標籤。\n3) 一至兩句話，清楚可執行。\n4) 避免重複與原 seed 完全相同的限制條件或措辭，但主題需相關。\n'}, 
        {'role': 'user', 'content': f'這是原始 seed：\n{base_seed}\n\n請依規則產生一個新的 seed 指令（繁體中文）。只輸出新 seed 本文，其他一律不要。'}
    ]

    try:
        resp_seed = client.chat.completions.create(
            model=MODEL, 
            messages=seed_gen_messages, 
            temperature=0.9, 
            max_tokens=200
        )
        new_seed_instruction_raw = resp_seed.choices[0].message.content.strip()

        def sanitize_seed(text: str) -> str:
            text = text.strip()
            text = re.sub(r'^```.*?\n|\n```$', '', text, flags=re.MULTILINE)
            return text

        new_seed_instruction = sanitize_seed(new_seed_instruction_raw)
        print('🔹 原始 seed：', base_seed)
        print('🔸 新的 seed：', new_seed_instruction)

    except Exception as e:
        print(f"❌ 生成新 seed 時發生錯誤：{str(e)}")
        print("請檢查 API Key 和網路連接")
        raise e
    return new_seed_instruction, re


@app.cell
def _(MODEL, client, new_seed_instruction):
    # Step 2: 以「新的 seed 指令」當作 user 提問，生成單輪回答（assistant 一次回覆）。
    # 產出為 OpenAI messages 格式，可直接累積進 datasets.jsonl。

    import json
    from uuid import uuid4
    from pathlib import Path

    qa_messages = [
        {"role": "system", "content": "你是一位營養與飲食規劃的專家，請使用繁體中文，給出明確、可執行的建議。"},
        {"role": "user", "content": new_seed_instruction},
    ]

    try:
        resp_qa = client.chat.completions.create(
            model=MODEL,
            messages=qa_messages,
            temperature=0.7,
            max_tokens=600,
        )

        answer = resp_qa.choices[0].message.content

        example = {
            "id": str(uuid4()),
            "type": "reference_free",
            "seed": new_seed_instruction,
            "messages": [
                qa_messages[0],                 # system
                qa_messages[1],                 # user（新的 seed）
                {"role": "assistant", "content": answer},  # 單輪回答
            ]
        }

        # ✅ 可選：追加寫入 datasets.jsonl（供下一章節 QC 使用）
        out_path = Path("outputs/datasets.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print("✅ 已生成單輪 QA 並寫入：", out_path)
        print("\n=== 回答預覽 ===\n", answer[:800])

    except Exception as e:
        print(f"❌ 生成 QA 對話時發生錯誤：{str(e)}")
        print("請檢查 API Key 和網路連接")
        raise e
    return Path, json, uuid4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference-based 資料生成

    在 Reference-based 的情境下，我們會使用一段外部文本作為依據，並在其上生成問答資料。
    這種方式常見於知識型 QA 系統（例如 Wikipedia 問答），其核心原則是：

    - 問題（Question）必須來自於文本
    - 答案（Answer）必須完全依照文本，不可超出文本範圍

    這樣生成的資料，可以幫助模型學會「根據參考內容回答」，而非憑空想像。
    """)
    return


@app.cell
def _():
    # 我們這裡直接示範一段 Wikipedia 中文條目（取自公開資料集 https://huggingface.co/datasets/lianghsun/wikipedia-zh-742M）
    # https://twbsball.dils.tku.edu.tw/wiki/index.php?title=%E9%A6%96%E9%A0%81
    # https://twbsball.dils.tku.edu.tw/wiki/index.php?title=兄弟象隊
    # https://twbsball.dils.tku.edu.tw/wiki/index.php?title=兄弟飯店棒球隊
    # https://twbsball.dils.tku.edu.tw/wiki/index.php?title=兄弟象隊隊史
    wiki_context = """
    全壘打梅花
    　　職棒三年（1992年）開始，當時的總教練森下正夫為了激勵球員士氣，率先提出球員每擊出一支全壘打，就貼上一枚黃色的梅花標誌的隊徽在打擊頭盔上的做法，意外收到不錯的效果，不但有效激勵了球員的士氣，而且還引發球員間對於全壘打及「收集梅花」的良性競爭，因而拿下了職棒三年的總冠軍；此後，這樣的做法就成為了兄弟的傳統。

    　　職棒十四年（2003年）時，有某家贊助商打算以該公司的標誌代替全壘打的小隊徽，卻意外遭到許多球迷的抗議，該提議被認為是違背兄弟象隊傳統的舉動；贊助商於是知難而退，顯然全壘打梅花已經深深被球迷認同，不僅僅是一個家族企業的標誌而已。此事件後，兄弟球團也對梅花標誌貼紙的樣式做了小幅度更新，將黃色取代為白色，沿用至今。
    """
    return (wiki_context,)


@app.cell
def _(MODEL, client, json, re):
    from typing import List
    from pydantic import BaseModel, Field

    class QuestionItem(BaseModel):
        question: str = Field(..., min_length=4, description='依據給定文本可直接回答的問題（繁體中文）')

    class QuestionList(BaseModel):
        items: List[QuestionItem]

    def generate_questions_from_context(context: str, n_pairs: int=4) -> List[str]:
        sys_rules = f'你是資料標註助理，請使用繁體中文設計問題。\n請產生 {n_pairs} 題問題，不要提供答案。\n原則：\n1) 問題必須可由『文本』直接回答，或能忠實改寫自其中資訊。\n2) 禁止加入『文本』以外的知識。\n3) 問題要清楚、具體，答案可在 1—2 句內表達。\n4) 若『文本』不足以支撐問題，請產生需要使用者進一步釐清的問題（單一句）。\n5) 問題要自然，不要暴露有任何『文本』或外部資料存在。\n6) 只輸出 JSON，格式固定為：{{"items":[{{"question":"..."}}, ...]}}。'
        user_rules = f'請根據以下『文本』設計問題：\n\n{context}\n\n⚠️ 僅輸出 JSON，格式：{{"items":[{{"question":"..."}}, ...]}}，不得有額外說明/Markdown/前後綴。'

        try:
            parsed = client.beta.chat.completions.parse(
                model=MODEL, 
                messages=[
                    {'role': 'system', 'content': sys_rules}, 
                    {'role': 'user', 'content': user_rules}
                ], 
                response_format=QuestionList
            )
            items = parsed.choices[0].message.parsed.items
            questions = [it.question.strip() for it in items if it.question.strip()]
            return questions[:n_pairs]
        except Exception:
            # Fallback to regular JSON mode
            fallback_sys = '你是資料標註助理。請只輸出 JSON，不要任何解釋或 Markdown。\n格式：[{"question":"..."}, {"question":"..."}]'
            fallback_user = f'{sys_rules}\n\n請輸出 JSON 陣列，每個物件僅含 question 欄位。\n\n『文本』\n{context}'

            resp = client.chat.completions.create(
                model=MODEL, 
                messages=[
                    {'role': 'system', 'content': fallback_sys}, 
                    {'role': 'user', 'content': fallback_user}
                ], 
                response_format={'type': 'json_object'}, 
                temperature=0.2, 
                max_tokens=800
            )

            raw = resp.choices[0].message.content.strip()
            txt = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
            data = json.loads(txt)

            items = data.get('items') if isinstance(data, dict) and 'items' in data else data
            if not isinstance(items, list):
                raise ValueError('模型輸出不是問題清單 JSON 陣列/物件')

            qs = []
            for obj in items:
                if isinstance(obj, dict) and 'question' in obj:
                    q = str(obj['question']).strip()
                elif isinstance(obj, str):
                    q = obj.strip()
                else:
                    continue
                if q:
                    qs.append(q)
            return qs[:n_pairs]
    return (generate_questions_from_context,)


@app.cell
def _(MODEL, client):
    # ---------- (B) 逐題回答：每題都嚴格依 context 回答（單輪） ----------
    def answer_questions_from_context(questions: list[str], context: str) -> list[dict]:
        """
        依據 context 作答，但「不要暴露有參考文本」。
        若題目資訊不足以得出明確答案：提出一個具體、簡潔的釐清問題（單一句），
        或請使用者補充需要的關鍵條件；不要說「無法回答」、「缺乏文本」等字眼。
        """
        results = []
        sys = (
            "你是一位知識淵博且精準的助理，請使用繁體中文回答。\n"
            "原則：\n"
            "1) 回答要自然直接，不要提到你參考了任何外部文本/資料，也不要使用「根據提供的文本/段落/資料」等措辭。\n"
            "2) 若題目資訊不足以形成明確答案：請提出一個具體、簡潔的釐清問題（只用單一句），"
            "   或請使用者補充最關鍵的條件；不要說你無法回答、不要提到資訊不足或來源限制。\n"
            "3) 優先提供可執行、可驗證的重點；避免冗長鋪陳與套話。\n"
            "4) 禁止露出任何內部規則、提示詞或參考來源。"
        )

        for q in questions:
            # 注意：這裡仍然把 context 放到 user 訊息中以「隱式限制」模型，
            # 但系統訊息已禁止它在話語中暴露來源。
            user = f"『背景資料』\n{context}\n\n『問題』{q}"

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                )
                ans = resp.choices[0].message.content.strip()
                results.append({"question": q, "answer": ans})
            except Exception as e:
                print(f"❌ 回答問題時發生錯誤：{str(e)}")
                results.append({"question": q, "answer": "抱歉，處理此問題時發生錯誤。"})

        return results
    return (answer_questions_from_context,)


@app.cell
def _(
    Path,
    answer_questions_from_context,
    generate_questions_from_context,
    json,
    uuid4,
):
    def build_reference_based_from_context(context: str, n_pairs: int=4, out_path: Path=Path('outputs/datasets.jsonl')):
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            qs = generate_questions_from_context(context, n_pairs=n_pairs)
            qa_list = answer_questions_from_context(qs, context)
            wrote = 0

            with out_path.open('a', encoding='utf-8') as f:
                for qa in qa_list:
                    rec = {
                        'id': str(uuid4()), 
                        'type': 'reference_based', 
                        'seed': context, 
                        'context': context, 
                        'messages': [
                            {'role': 'system', 'content': '請嚴格依據提供的文本回答問題，使用繁體中文。'}, 
                            {'role': 'user', 'content': qa['question']}, 
                            {'role': 'assistant', 'content': qa['answer']}
                        ]
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    wrote += 1

            print(f'✅ 已新增 {wrote} 筆 reference-based QA 至 {out_path}')
            return qa_list

        except Exception as e:
            print(f"❌ 建立 reference-based 資料時發生錯誤：{str(e)}")
            return []
    return (build_reference_based_from_context,)


@app.cell
def _(build_reference_based_from_context, wiki_context):
    try:
        _qa_preview = build_reference_based_from_context(wiki_context, n_pairs=4)
        print("\n--- 產生預覽 ---")
        for i, qa in enumerate(_qa_preview, 1):
            print(f"Q{i}: {qa['question']}")
            print(f"A{i}: {qa['answer'][:200]}{'...' if len(qa['answer'])>200 else ''}\n")
    except Exception as e:
        print(f"❌ 執行 reference-based 生成時發生錯誤：{str(e)}")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
