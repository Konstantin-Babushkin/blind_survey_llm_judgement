#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Send ONLY selected prompts (by 1-based indices) from a .txt file to OpenRouter and save results.

Examples:
  python send_specific_prompts_openrouter.py \
    --input prompts_A_V2.txt \
    --output results_subset.jsonl \
    --model "anthropic/claude-3.5-sonnet" \
    --only "1,3,7-10"

  # Dry-run (show what will be sent, but don't call the API)
  python send_specific_prompts_openrouter.py --input prompts_A_V2.txt --only "2-4" --dry-run

Requirements:
  pip install python-dotenv pandas requests

Env vars:
  OPENROUTER_API_KEY=xxxx
  MODEL_NAME=anthropic/claude-3.5-sonnet (or pass --model)
  SYSTEM_PROMPT="You are a careful writing evaluator. Return JSON only."
"""

import os
import re
import json
import time
import argparse
import requests
from typing import List, Dict, Any, Tuple, Iterable, Set, Optional
from dotenv import load_dotenv

# --------- Parsing helpers ---------

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
            flush()
            current_idx = m.group(1)     # header index token (string)
            current_title = m.group(2)   # title text
        else:
            if current_idx is not None:
                current_body_lines.append(line)
    flush()
    return blocks


def parse_only_indices(expr: str, max_allowed: int) -> List[int]:
    """
    Parse a string like "1,3,7-10" into a sorted, de-duplicated list of 1-based integers.
    Raises ValueError for invalid tokens or out-of-range indices.
    """
    if not expr:
        raise ValueError("--only expression is empty")

    indices: List[int] = []
    seen: Set[int] = set()

    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            # range
            parts = token.split("-", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid range token '{token}'")
            a, b = parts[0].strip(), parts[1].strip()
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Range endpoints must be integers: '{token}'")
            start, end = int(a), int(b)
            if start < 1 or end < 1 or start > end:
                raise ValueError(f"Invalid range '{token}'")
            for i in range(start, end + 1):
                if i > max_allowed:
                    raise ValueError(f"Index {i} in range '{token}' exceeds available prompts ({max_allowed}).")
                if i not in seen:
                    seen.add(i)
                    indices.append(i)
        else:
            # single
            if not token.isdigit():
                raise ValueError(f"Invalid index '{token}' (must be 1-based integer)")
            i = int(token)
            if i < 1 or i > max_allowed:
                raise ValueError(f"Index {i} is out of range 1..{max_allowed}")
            if i not in seen:
                seen.add(i)
                indices.append(i)

    # keep user-specified order; no sort
    return indices


# --------- OpenRouter helpers ---------

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
    retries = 3
    last_exception = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 500, 502, 503, 504):
                last_exception = RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
                # retryable
            else:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            last_exception = e
        # backoff
        if attempt < retries - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    raise RuntimeError(f"Failed to call OpenRouter after {retries} attempts.") from last_exception


def try_extract_json(text: str) -> Tuple[Optional[dict], str]:
    """
    Try to parse the assistant content as JSON; fallback to first {...} block.
    Returns (obj or None, err_msg or "").
    """
    try:
        return json.loads(text), ""
    except Exception:
        pass
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
    usage = resp_json.get("usage") or {}
    return usage.get("prompt_tokens", -1), usage.get("completion_tokens", -1)


def parse_fields_from_json(obj: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    return {
        "form_id": obj.get("form_id"),
        "criterion": obj.get("criterion"),
        "rating": obj.get("rating"),
        "justifications": obj.get("justifications"),
        "accuracy_labels": obj.get("accuracy_labels")
    }


# --------- Main ---------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a single .txt file containing all prompts")
    ap.add_argument("--output", required=False, help="Where to write results JSONL (default: results_selected.jsonl)")
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", ""), help="OpenRouter model name")
    ap.add_argument("--system", default=os.getenv("SYSTEM_PROMPT", "You are a careful writing evaluator. Return JSON only."))
    ap.add_argument("--only", required=True, help='1-based indices to send, e.g. "1,3,7-10"')
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--append", action="store_true", help="Append to output file if exists")
    ap.add_argument("--dry-run", action="store_true", help="List selected prompts and exit without sending")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("OPENROUTER_API_KEY not set (and not a --dry-run).")

    if not args.model and not args.dry_run:
        raise SystemExit("Model not set. Pass --model or set MODEL_NAME env var (or use --dry-run).")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = split_prompts(text)
    if not blocks:
        raise SystemExit("No prompts detected. Check headers: '----- PROMPT i: title -----'.")

    # Convert 1-based indices to actual selection
    indices = parse_only_indices(args.only, max_allowed=len(blocks))

    # Build the exact subset in the specified order
    selected = []
    for i in indices:
        # our blocks are in order; we interpret "i" as 1-based positional index in this file
        idx, title, body = blocks[i - 1]
        selected.append((i, idx, title, body))  # keep both positional i and header idx for logging

    if args.dry_run:
        print("Dry-run: the following prompts would be sent (pos -> header_idx : title):")
        for pos, hdr_idx, title, _ in selected:
            print(f"  {pos} -> {hdr_idx} : {title}")
        return

    out_path = args.output or "results_selected.jsonl"
    mode = "a" if args.append else "w"
    sent = 0

    with open(out_path, mode, encoding="utf-8") as out:
        for pos, hdr_idx, title, body in selected:
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
                    # keep both positional order and header index for traceability
                    "pos": pos,              # 1-based position in the input file
                    "idx": hdr_idx,          # header index token from the file
                    "title": title,
                    "request_tokens": req_toks,
                    "response_tokens": resp_toks,
                    "raw_text": content,
                    "parsed": parse_fields_from_json(obj) if obj else None,
                    "parse_error": parse_err,
                }
            except Exception as e:
                record = {
                    "pos": pos,
                    "idx": hdr_idx,
                    "title": title,
                    "error": str(e)
                }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            sent += 1
            print(f"[{sent}/{len(selected)}] Sent prompt pos={pos} (header idx={hdr_idx}): {title}")

    print(f"Done. Wrote {sent} results to {out_path}")


if __name__ == "__main__":
    main()
