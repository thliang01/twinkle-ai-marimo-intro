# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "watchdog",
#     "openai>=1.40.0",
#     "python-dotenv",
#     "datasets",
#     "pyarrow",
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
    # 02 - 資料品質檢查與過濾（Quality Checks）
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
      <a href="https://colab.research.google.com/github/ai-twinkle/llm-lab/blob/main/courses/2025-08-llm-dialogue-dataset/02_quality_checks.ipynb" target="_blank" style="margin: 2px;">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open 02_quality_checks In Colab" style="display: inline-block; vertical-align: middle;"/>
      </a>
      <a href="https://molab.marimo.io/notebooks/nb_zNsH5TyPTT6sjhDdVvw2Nx" target="_blank" style="margin: 2px;">
            <img src="https://molab.marimo.io/molab-shield.png" alt="Open in molab" style="display: inline-block; vertical-align: middle;"/>
      </a>
    </div>

    目標：

    - 載入 `datasets.jsonl`
    - 規則式檢查：敏感詞 / 結構完整 / 長度門檻 / 不含 placeholder
    - 產出 `clean.jsonl`
    - 生成摘要報表（通過/剔除統計、剔除原因分佈）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 註：不論如何，禁用 [opencc-python](https://github.com/yichen0831/opencc-python) 做任何轉換

    雖然 OpenCC 的簡轉繁功能很方便，但它只是機械式轉換，繁體字有時會被誤判或錯轉，導致語意錯誤或不符合在地用法，因此並不適合需要精準繁體輸出的情境。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 準備路徑與依賴

    我們將載入前一步驟產生的 `datasets.jsonl` 檔案進行品質檢查。
    """)
    return


@app.cell
def _():
    # 確保 outputs 目錄存在
    # 在 marimo 中我們假設 datasets.jsonl 已經從前一個 notebook 產生
    import os
    os.makedirs("outputs", exist_ok=True)
    return


@app.cell
def _():
    from pathlib import Path
    import json
    import re
    import statistics
    from collections import Counter

    INPUT_PATH = Path("outputs/datasets.jsonl")
    OUTPUT_DIR = Path("outputs")
    OUTPUT_CLEAN = OUTPUT_DIR / "clean.jsonl"
    OUTPUT_REPORT = OUTPUT_DIR / "qc_report.json"

    assert INPUT_PATH.exists(), f"找不到 {INPUT_PATH}，請先完成 01_generate_dialogs.ipynb"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("✅ 讀取來源：", INPUT_PATH)
    print("✅ 乾淨輸出：", OUTPUT_CLEAN)
    return (
        Counter,
        INPUT_PATH,
        OUTPUT_CLEAN,
        OUTPUT_REPORT,
        json,
        re,
        statistics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 載入資料

    逐行讀取 JSONL，存到 list。這裡不做任何變形，只檢視基本鍵值。
    """)
    return


@app.cell
def _(INPUT_PATH, json):
    records = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception as e:
                # 若出現無法解析的行，記錄並跳過
                print("⚠️ 無法解析的行，已略過：", e)

    len(records)
    return (records,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 品質規則定義

    本課採「規則式（rule-based）」檢查以快速過濾：

    1. **結構**：`messages` 至少包含 `system`、`user`、`assistant` 三則；且對話文本不為空。
    2. **多輪性**：對話需包含至少 3 輪（可鬆綁為 1 輪以上，但本課先採至少 3 輪）。
    3. **長度**：合併文本長度至少 80 字（避免過短）。
    4. **敏感詞**：過濾個資或敏感詞（示例黑名單）。
    5. **Placeholder**：不得包含 `XXX`、`<填充>` 類佔位符。
    """)
    return


@app.cell
def _(re):
    # 1) 結構/角色檢查
    def has_min_roles(msgs):
        roles = [m.get("role") for m in msgs]
        return {"system", "user", "assistant"}.issubset(set(roles))

    # 2) 多輪性（這裡以訊息數 >= 3 視為最低門檻；若需要更嚴謹可解析回合）
    def has_min_turns(msgs, min_msgs=3):
        return len(msgs) >= min_msgs

    # 3) 長度門檻
    def meet_min_length(msgs, min_chars=30):
        total = sum(len((m.get("content") or "").strip()) for m in msgs)
        return total >= min_chars

    # 4) 敏感詞（示例）：身分證/電話/地址/Email/信用卡/生日
    SENSITIVE_PATTERNS = [
        r"\b[A-Z][12]\d{8}\b",                         # 台灣身分證格式
        r"\b09\d{8}\b|\b0\d{1,2}-\d{6,8}\b",          # 手機或市話
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
        r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",   # 信用卡 16 碼
        r"\b(19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}\b",    # 西元生日 yyyy/mm/dd 或 yyyy-mm-dd
    ]

    def has_sensitive(text):
        return any(re.search(p, text) for p in SENSITIVE_PATTERNS)

    # 5) Placeholder 過濾
    PLACEHOLDER_PATTERNS = [r"XXX", r"<填充>", r"\[PLACEHOLDER\]"]

    def has_placeholder(text):
        return any(re.search(p, text, flags=re.IGNORECASE) for p in PLACEHOLDER_PATTERNS)
    return (
        PLACEHOLDER_PATTERNS,
        SENSITIVE_PATTERNS,
        has_min_roles,
        has_min_turns,
        has_placeholder,
        has_sensitive,
        meet_min_length,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 單筆檢查與原因標註

    輸入一筆記錄，回傳 (是否通過, 剔除原因集合)。
    """)
    return


@app.cell
def _(
    has_min_roles,
    has_min_turns,
    has_placeholder,
    has_sensitive,
    meet_min_length,
):
    def join_text_by_roles(msgs, roles=("assistant",)):
        return "\n".join((m.get("content") or "").strip()
                         for m in msgs if m.get("role") in roles)

    def quality_check(record):
        reasons = []

        msgs = record.get("messages", [])
        if not isinstance(msgs, list) or not msgs:
            return False, {"bad_structure"}

        if not has_min_roles(msgs):
            reasons.append("missing_roles")

        if not has_min_turns(msgs, min_msgs=3):
            reasons.append("too_few_messages")

        # ⬇️ 只看 assistant 文字，避免掃到 user 提示內的「例如 身分證/電話…」
        text = join_text_by_roles(msgs, roles=("assistant",))

        if not meet_min_length(msgs, min_chars=30):
            reasons.append("too_short")

        if has_sensitive(text):
            reasons.append("sensitive_content")

        if has_placeholder(text):
            reasons.append("placeholder_found")

        return (len(reasons) == 0), set(reasons)
    return (quality_check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 執行過濾並輸出 `clean.jsonl`
    """)
    return


@app.cell
def _(OUTPUT_CLEAN, json, quality_check, records):
    kept, dropped = ([], [])
    for rec in records:
        ok, reasons = quality_check(rec)
        if ok:
            kept.append(rec)
        else:
            dropped.append((rec.get('id'), reasons))
    with OUTPUT_CLEAN.open('w', encoding='utf-8') as f_1:
        for r in kept:
            f_1.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'✅ 通過：{len(kept)}  筆')
    print(f'❌ 剔除：{len(dropped)} 筆')
    return dropped, kept


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 產出品質報表

    統計剔除原因分佈、長度分佈（通過者），並輸出 `qc_report.json` 方便保存與追蹤。
    """)
    return


@app.cell
def _(Counter, OUTPUT_REPORT, dropped, json, kept, records, statistics):
    reason_counter = Counter()
    for _, reasons_1 in dropped:
        reason_counter.update(reasons_1)
    lengths = []
    for r_1 in kept:
        lengths.append(sum((len((m.get('content') or '').strip()) for m in r_1['messages'])))
    report = {'input_total': len(records), 'kept': len(kept), 'dropped': len(dropped), 'drop_reasons': dict(reason_counter), 'length_stats_kept': {'min': min(lengths) if lengths else 0, 'max': max(lengths) if lengths else 0, 'mean': round(statistics.mean(lengths), 2) if lengths else 0, 'median': statistics.median(lengths) if lengths else 0}}
    with OUTPUT_REPORT.open('w', encoding='utf-8') as f_2:
        json.dump(report, f_2, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 抽樣檢視通過樣本（前 2 筆）

    確認清洗後的資料結構與內容是否符合預期。
    """)
    return


@app.cell
def _(OUTPUT_CLEAN, json):
    preview = []
    with OUTPUT_CLEAN.open('r', encoding='utf-8') as f_3:
        for i, line_1 in enumerate(f_3):
            if i >= 2:
                break
            preview.append(json.loads(line_1))
    for i, s in enumerate(preview, 1):
        print(f'\n--- Clean Sample {i} / topic={s.get('topic')} ---')
        text = s['messages'][-1]['content']
        print(text[:500] + ('...' if len(text) > 500 else ''))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.（可選）LLM 輔助檢查（實務建議）
    > 所謂的 LLM-as-Judge

    在規則式檢查後，可抽樣使用 LLM 來做語義層面的檢查（如：是否符合主題、語氣、是否含危險建議等）。
    以下為示意程式（預設註解，不影響主流程）。
    """)
    return


@app.cell
def _():
    # from openai import OpenAI
    # from dotenv import load_dotenv
    # import os
    #
    # load_dotenv()
    # API_KEY = os.getenv('NEBIUS_API_KEY')
    # BASE_URL = "https://api.studio.nebius.ai/v1"
    # client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    #
    # def llm_qc_judgement(text: str) -> bool:
    #     """回傳 True 視為通過；False 視為不通過"""
    #     prompt = f"請閱讀以下對話是否符合：主題連貫、語氣正式友善、無敏感資料、無危險建議。\n\n{text}\n\n請只回答：PASS 或 FAIL。"
    #     resp = client.chat.completions.create(
    #         model="google/gemma-3-27b-it-fast",
    #         messages=[{"role":"user","content": prompt}],
    #         temperature=0.0,
    #         max_tokens=10,
    #     )
    #     ans = resp.choices[0].message.content.strip().upper()
    #     return ans.startswith("PASS")
    #
    # # 示例（只檢查前 3 筆）
    # for s in preview:
    #     ok = llm_qc_judgement("\n".join(m["content"] for m in s["messages"]))
    #     print("LLM QC ->", "PASS" if ok else "FAIL")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.（可選）如果生成資料集一直沒通過
    """)
    return


@app.cell
def _(PLACEHOLDER_PATTERNS, SENSITIVE_PATTERNS, re):
    def _ctx(text: str, start: int, end: int, width: int=50) -> str:
        s = max(0, start - width)
        e = min(len(text), end + width)
        return text[s:start] + '【' + text[start:end] + '】' + text[end:e]

    def debug_scan_record(rec: dict, show_only_hits: bool=True):
        rid = rec.get('id', '<no-id>')
        topic = rec.get('topic', '')
        msgs = rec.get('messages', [])
        text = '\n'.join((m.get('content') or '' for m in msgs if m.get('role') == 'assistant'))
        sens_hits = []
        for p in SENSITIVE_PATTERNS:
            for m in re.finditer(p, text, flags=re.IGNORECASE):
                sens_hits.append((p, m.start(), m.end(), m.group(0)))
        ph_hits = []
        for p in PLACEHOLDER_PATTERNS:
            for m in re.finditer(p, text, flags=re.IGNORECASE):
                ph_hits.append((p, m.start(), m.end(), m.group(0)))
        if sens_hits or ph_hits or (not show_only_hits):
            print(f'\n=== Record id={rid} | topic={topic} ===')
            if sens_hits:
                print(f'Sensitive matches ({len(sens_hits)}):')
                for p, s, e, g in sens_hits:
                    print(f' - pattern: {p}  | match: {g!r}')
                    print('   ...', _ctx(text, s, e), '...')
            if ph_hits:
                print(f'Placeholder matches ({len(ph_hits)}):')
                for p, s, e, g in ph_hits:
                    print(f' - pattern: {p}  | match: {g!r}')
                    print('   ...', _ctx(text, s, e), '...')
        return (bool(sens_hits), bool(ph_hits))

    def debug_scan_all(recs: list[dict], limit: int | None=None):
        n = 0
        total_sens = total_ph = 0
        for rec in recs:
            sens, ph = debug_scan_record(rec)
            total_sens = total_sens + int(sens)
            total_ph = total_ph + int(ph)
            n = n + 1
            if limit and n >= limit:
                break
        print(f'\nSummary: scanned {n} records | with_sensitive={total_sens} | with_placeholder={total_ph}')
    return (debug_scan_all,)


@app.cell
def _(debug_scan_all, records):
    # 假設你已在前面載入 records = [...]（從 raw.jsonl）
    debug_scan_all(records)          # 掃全部
    # 或只看前 10 筆
    # debug_scan_all(records, limit=10)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
