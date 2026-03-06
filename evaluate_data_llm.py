import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

ALLOWED_METHODS = {"baseline", "schema", "constrained"}
DEFAULT_MODELS = ["qwen3:8b", "llama3.1:8b", "ministral-3:8b"]
ISO_LANG_RE = re.compile(r"^[a-z]{2}$")

STRIP_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)

# Configuration
input_texts = "test_data.jsonl"
output_file = "llm_evaluation_results.jsonl"
word_output_file = "llm_word_results.jsonl"


@dataclass
class LLMResult:
    parsed: Optional[Dict[str, Any]]
    raw_text: str
    valid_json: bool
    parse_error: Optional[str]
    latency_ms: float
    retry_count: int


def build_prompt_main(text: str, method: str) -> str:
    base = (
        "Task: Identify the main language of the input text.\n"
        "Return JSON only with keys: main_language, confidence.\n"
        "main_language MUST be ISO-639-1 lowercase code (e.g., 'en', 'fr', 'es').\n"
        "confidence MUST be a float between 0 and 1 indicating the confidence level.\n"
    )
    if method in {"schema", "constrained"}:
        base += (
            'Output schema: {"main_language": "<ISO-639-1 code>", "confidence": <float between 0 and 1>}\n'
            'Example: {"main_language": "en", "confidence": 0.95}\n'
            "No markdown. No explanations.\n"
        )
    if method == "constrained":
        base += "If uncertain, return your best guess.\n"
    return f"{base}\nInput text: {text}\n"


def build_prompt_word(word: str, method: str) -> str:
    base = (
        "Task: Identify the language of the single token.\n"
        "Return JSON only with keys: word_language, confidence.\n"
        "word_language must be ISO-639-1 lowercase.\n"
        "confidence must be float in [0,1].\n"
    )
    if method in {"schema", "constrained"}:
        base += (
            'Output schema: {"word_language":"fr","confidence":0.87}\n'
            "No markdown. No explanations.\n"
        )
    return f"{base}\nToken:\n{word}"


def build_prompt_combined(
    text: str, token_list: List[Dict[str, Any]], method: str
) -> str:
    token_view = [{"index": t["index"], "token": t["token"]} for t in token_list]
    base = (
        "Task: Identify the main language of the text and any foreign-language words.\n"
        "Return JSON only with keys: main_language, main_confidence, foreign_words.\n"
        "foreign_words must be a list of objects with keys: index, token, language, confidence.\n"
        "language must be ISO-639-1 lowercase. confidence in [0,1].\n"
    )
    if method in {"schema", "constrained"}:
        base += (
            "No markdown/code fences.\n"
            'Output schema: {"main_language":"en","main_confidence":0.95,"foreign_words":[{"index":5,"token":"avec","language":"fr","confidence":0.9}]}\n'
        )
    if method == "constrained":
        base += (
            "Only use indices that appear in token list. Use [] if no foreign words.\n"
        )
    return (
        f"{base}\nText:\n{text}\n\nToken list with indices:\n"
        f"{json.dumps(token_view, ensure_ascii=False)}"
    )


def call_ollama(
    model: str,
    prompt: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
) -> Tuple[str, float]:
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    text = data.get("message", {}).get("content", "")
    return text, elapsed_ms


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    return json.loads(raw_text)


def call_with_retry(
    model: str,
    prompt: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    retries: int,
) -> LLMResult:
    total_latency = 0.0
    last_error = None
    last_raw = ""

    for attempt in range(retries + 1):
        raw, latency = call_ollama(model, prompt, ollama_url, temperature, timeout)
        total_latency += latency
        last_raw = raw
        try:
            parsed = parse_json_response(raw)
            return LLMResult(
                parsed=parsed,
                raw_text=raw,
                valid_json=True,
                parse_error=None,
                latency_ms=total_latency,
                retry_count=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                repair_prompt = (
                    "Your previous output was invalid JSON.\n"
                    "Return ONLY valid JSON matching the requested schema.\n"
                    f"Original output:\n{raw}"
                )
                prompt = repair_prompt

    return LLMResult(
        parsed=None,
        raw_text=last_raw,
        valid_json=False,
        parse_error=last_error,
        latency_ms=total_latency,
        retry_count=retries,
    )


samples = [json.loads(line) for line in open(input_texts, "r", encoding="utf-8")]

results = []


for model in DEFAULT_MODELS:
    for sample in samples:
        text = sample["text"]

        prompt = build_prompt_main(text, method="schema")
        llm_result = call_with_retry(
            model=model,
            prompt=prompt,
            ollama_url="http://localhost:11434",
            temperature=0.0,
            timeout=30,
            retries=2,
        )

        detected_lang = (
            llm_result.parsed.get("main_language")
            if llm_result.valid_json
            else "unknown"
        )
        confidence = (
            llm_result.parsed.get("confidence", 0.0) if llm_result.valid_json else 0.0
        )

        result = {
            "correct": sample["lang"] == detected_lang,
            "model": model,
            "source": sample["source"],
            "text": text,
            "text_length_chars": len(text),
            "text_length_words": len(text.split()),
            "true_lang": sample["lang"],
            "detected_lang": detected_lang,
            "confidence": float(confidence),
            "llm_valid_json": llm_result.valid_json,
            "llm_parse_error": llm_result.parse_error,
            "llm_latency_ms": llm_result.latency_ms,
            "llm_retry_count": llm_result.retry_count,
        }
        print(result)
        results.append(result)

# Write results to output file
with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(
    f"Processed {len(samples)} samples with {len(DEFAULT_MODELS)} models ({len(results)} total results)"
)
print(f"Results saved to {output_file}")
