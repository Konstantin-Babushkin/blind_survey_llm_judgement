import os
import pandas as pd
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# === CONSTANT SETTINGS ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL_NAME")
INPUT_FILE = os.getenv("INPUT_FILE")
OUTPUT_FILE = os.getenv("OUTPUT_FILE")
FEEDBACK_COLUMN = "Feedback"
START_TAG = 7
SYSTEM_PROMPT = (
    "You are a concise assistant. Summarize the following employee feedback "
    "keeping important details and sentiment. Be as concise as possible. Length: 150-300 characters"
)
# ==========================

def call_openrouter(content):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "temperature": 0.2,
        "max_tokens": 160
    }

    backoff = 1
    while True:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        elif resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
        else:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

def main():
    if not OPENROUTER_API_KEY:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    df = pd.read_csv(INPUT_FILE, dtype=str, encoding="utf-8", on_bad_lines="skip")
    summaries = []
    try:
        for text in df[FEEDBACK_COLUMN]:
            summaries.append(call_openrouter(text if isinstance(text, str) else ""))
    except Exception as e:
        print(f"An error occurred after processing {len(summaries)} items: {e}")

    if not summaries:
        print("No summaries were generated.")
        return

    summary_df = pd.DataFrame({
        "tag": [f"{START_TAG + i}" for i in range(len(summaries))],
        "summary": summaries,
        FEEDBACK_COLUMN: df[FEEDBACK_COLUMN][:len(summaries)],
    })
    summary_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Done. Wrote {len(summaries)} summaries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
