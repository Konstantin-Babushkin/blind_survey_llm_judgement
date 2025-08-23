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
START_TAG = 1
SYSTEM_PROMPT = (
    "You are an employee performance feedback interpreter. "
    "Given a feedback note, explain to the employee: "
    "(1) what it means, "
    "(2) what it implies for growth/standing, "
    "and (3) how to respond now with 2–4 concrete actions that address issues. "
    "Preserve the original sentiment; "
    "don’t add new facts. "
    "Tone: supportive, direct, non-judgmental. "
    "Use third person (use name instead of pronouns). "
    "Length: 200–300 characters. "
    "Output a single paragraph."
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
    model_output = []
    try:
        for text in df[FEEDBACK_COLUMN]:
            model_output.append(call_openrouter(text if isinstance(text, str) else ""))
    except Exception as e:
        print(f"An error occurred after processing {len(model_output)} items: {e}")

    if not model_output:
        print("No model_output were generated.")
        return

    result_df = pd.DataFrame({
        "tag": [f"{START_TAG + i}" for i in range(len(model_output))],
        "expanded_text": model_output,
        FEEDBACK_COLUMN: df[FEEDBACK_COLUMN][:len(model_output)],
    })
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Done. Wrote {len(model_output)} to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
