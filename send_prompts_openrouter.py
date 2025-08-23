#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Send prompts (one by one) from a single .txt file to OpenRouter and save results.

Requirements:
  pip install python-dotenv pandas requests

Env vars:
  OPENROUTER_API_KEY=xxxx
  MODEL_NAME=anthropic/claude-3.5-sonnet  (example; or pass --model)
  SYSTEM_PROMPT="You are a careful writing evaluator. Return JSON only."

Usage:
  python send_prompts_openrouter.py \
    --input prompts_A_V2.txt \
    --output results_A_V2.jsonl \
    --model "anthropic/claude-3.5-sonnet" \
    --start 1 \
    --max -1

Notes:
- Splits prompts using lines that match:  ^----- PROMPT .* -----$
- Each chunk (prompt) is sent as the *user* message. A short system prompt is used.
- Retries with exponential backoff for 429/5xx.
- Writes a JSONL with: idx, title, request_tokens, response_tokens, raw_text, parsed(form_id/criterion/rating/justifications), and error if any.
"""

import os
import re
import json
import time
import argparse
import requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# -------- Helpers --------

HEADER_RE = re.compile(r"^----- PROMPT\s+(\S+):\s+(.*?)\s+-----\s*$")

def split_prompts(text: str) -> List[Tuple[str, str, str]]:
    """
    Return a list of (idx, title, body) for each prompt block.
    Blocks are separated by header lines that match HEADER_RE.
    """
    lines = text.splitlines()
    blocks: List[Tuple[str, str, str]] = []

    current_idx = None
    current_title = None
    current_body_lines: List[str] = []

    def flush():
        nonlocal current_idx, current_title, current_body_lines
        if current_idx is not None:
            blocks.append((current_idx, current_title, "\n".join(current_body_lines).strip()))
        current_idx, current_title, current_body_lines = None, None, []

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            # new header => flush previous
            flush()
            current_idx = m.group(1)
            current_title = m.group(2)
        else:
            if current_idx is not None:
                current_body_lines.append(line)

    flush()
    return blocks


def call_openrouter(api_key: str, model: str, system_prompt: str, user_content: str,
                    temperature: float = 0.0, max_tokens: int = 800,
                    extra_headers: Dict[str, str] = None) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    backoff = 1
    while True:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            return data
        elif resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
        else:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")


def try_extract_json(text: str) -> Tuple[dict, str]:
    """
    Try to parse the assistant content as JSON.
    If it fails, try to find the first {...} block and parse that.
    Returns (obj or None, err_msg or "").
    """
    # direct parse
    try:
        return json.loads(text), ""
    except Exception as e1:
        pass
    # find first {...} block (naive but often enough)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate), ""
        except Exception as e2:
            return None, f"JSON parse error: {e2}"
    return None, "No JSON object found in response."


def tokens_from_usage(resp_json: dict) -> Tuple[int, int]:
    """
    Extract request_tokens/response_tokens from OpenRouter response if available.
    Fallback to (-1, -1) if absent.
    """
    usage = resp_json.get("usage") or {}
    return usage.get("prompt_tokens", -1), usage.get("completion_tokens", -1)


def parse_fields_from_json(obj: dict) -> Dict[str, Any]:
    """
    Pull expected fields if present.
    """
    return {
        "form_id": obj.get("form_id"),
        "criterion": obj.get("criterion"),
        "rating": obj.get("rating"),
        "justifications": obj.get("justifications"),
        "accuracy_labels": obj.get("accuracy_labels")
    }


# -------- Main --------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a single .txt file containing all prompts")
    ap.add_argument("--output", required=True, help="Where to write results JSONL")
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", ""), help="OpenRouter model name")
    ap.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", "You are a careful writing evaluator. Return JSON only."))
    ap.add_argument("--start", type=int, default=1, help="Start at this prompt index (1-based)")
    ap.add_argument("--max", type=int, default=-1, help="Max prompts to send (-1 for all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--append", action="store_true", help="Append to output file if exists")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    if not args.model:
        raise SystemExit("Model not set. Pass --model or set MODEL_NAME env var.")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = split_prompts(text)
    if not blocks:
        raise SystemExit("No prompts detected. Check the header format: '----- PROMPT i: title -----'.")

    # Filter by start/max
    filtered_blocks = []
    for b in blocks:
        try:
            if int(b[0]) >= args.start:
                filtered_blocks.append(b)
        except ValueError:
            # For non-integer indices like 'SC1', we include them
            # so they are not skipped by the --start filter.
            filtered_blocks.append(b)
    blocks = filtered_blocks

    if args.max > -1:
        blocks = blocks[:args.max]

    mode = "a" if args.append else "w"
    sent = 0

    with open(args.output, mode, encoding="utf-8") as out:
        for idx, title, body in blocks:
            try:
                resp_json = call_openrouter(
                    api_key=api_key,
                    model=args.model,
                    system_prompt=args.system,
                    user_content=body,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
                content = resp_json["choices"][0]["message"]["content"].strip()
                obj, parse_err = try_extract_json(content)
                req_toks, resp_toks = tokens_from_usage(resp_json)

                record = {
                    "idx": idx,
                    "title": title,
                    "request_tokens": req_toks,
                    "response_tokens": resp_toks,
                    "raw_text": content,
                    "parsed": parse_fields_from_json(obj) if obj else None,
                    "parse_error": parse_err,
                }
            except Exception as e:
                record = {
                    "idx": idx,
                    "title": title,
                    "error": str(e)
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            sent += 1
            print(f"[{sent}] Sent prompt {idx}: {title}")

    print(f"Done. Wrote {sent} results to {args.output}")


if __name__ == "__main__":
    main()
