# /// script
# dependencies = [
#     "pymupdf",
#     "requests",
# ]
# ///

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
import requests
from requests.exceptions import RequestException

ISO_LANG_RE = re.compile(r"^[a-z]{2}$")
STRIP_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

LANG_NAME_TO_ISO = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "turkish": "tr",
    "indonesian": "id",
    "ukrainian": "uk",
    "vietnamese": "vi",
    "dutch": "nl",
    "polish": "pl",
    "romanian": "ro",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
    "arabic": "ar",
    "hindi": "hi",
    "swedish": "sv",
}


@dataclass
class LLMResult:
    parsed: Optional[Dict[str, Any]]
    raw_text: str
    valid_json: bool
    parse_error: Optional[str]
    latency_ms: float
    retry_count: int


def safe_lang(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower()
    if ISO_LANG_RE.match(normalized):
        return normalized
    if "-" in normalized:
        prefix = normalized.split("-", maxsplit=1)[0]
        if ISO_LANG_RE.match(prefix):
            return prefix
    if normalized in LANG_NAME_TO_ISO:
        return LANG_NAME_TO_ISO[normalized]
    return "unknown"


def safe_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        return 0.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


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
        "format": "json",
        "options": {"temperature": temperature},
    }
    started_at = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", ""), elapsed_ms


def call_with_retry(
    model: str,
    prompt: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
) -> LLMResult:
    original_prompt = prompt
    total_latency = 0.0
    last_error = None
    last_raw = ""

    for attempt in range(retries + 1):
        attempt_started_at = time.perf_counter()
        try:
            raw_text, latency = call_ollama(
                model=model,
                prompt=prompt,
                ollama_url=ollama_url,
                temperature=temperature,
                timeout=timeout,
            )
            total_latency += latency
            last_raw = raw_text
        except RequestException as exc:
            total_latency += (time.perf_counter() - attempt_started_at) * 1000.0
            last_error = f"transport_error:{type(exc).__name__}: {exc}"
            if attempt < retries:
                sleep_seconds = max(0.0, retry_backoff_seconds) * (2**attempt)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue
            return LLMResult(
                parsed=None,
                raw_text=last_raw,
                valid_json=False,
                parse_error=last_error,
                latency_ms=total_latency,
                retry_count=attempt,
            )

        try:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON value must be an object")
            return LLMResult(
                parsed=parsed,
                raw_text=raw_text,
                valid_json=True,
                parse_error=None,
                latency_ms=total_latency,
                retry_count=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                prompt = (
                    f"{original_prompt}\n\n"
                    "The previous response was invalid JSON.\n"
                    "Return ONLY valid JSON matching the required schema.\n"
                    f"Original output:\n{raw_text}"
                )

    return LLMResult(
        parsed=None,
        raw_text=last_raw,
        valid_json=False,
        parse_error=last_error,
        latency_ms=total_latency,
        retry_count=retries,
    )


def build_prompt_main(text: str) -> str:
    return (
        "Task: Identify the main language of the input text.\n"
        "Return JSON only with keys: main_language, confidence.\n"
        "main_language MUST be ISO-639-1 lowercase code (e.g., 'en', 'fr', 'es').\n"
        "confidence MUST be a float between 0 and 1.\n"
        "No markdown. No explanations.\n"
        'Output schema: {"main_language":"en","confidence":0.95}\n'
        f"\nInput text:\n{text}"
    )


def build_prompt_foreign_batch(
    main_language: str, batch_tokens: List[Dict[str, Any]]
) -> str:
    token_view = [
        {"index": t["index"], "token": t["normalized_word"]} for t in batch_tokens
    ]
    return (
        "Task: Identify foreign-language tokens relative to the provided main language.\n"
        "Return JSON only with key: foreign_words.\n"
        "foreign_words MUST be a list of objects with keys: index, token, language, confidence.\n"
        "language MUST be ISO-639-1 lowercase. confidence MUST be in [0,1].\n"
        "Only include items where language differs from main_language.\n"
        "Only use indices present in the provided token list. Use [] if none.\n"
        "No markdown. No explanations.\n"
        'Output schema: {"foreign_words":[{"index":5,"token":"avec","language":"fr","confidence":0.91}]}\n'
        f"\nmain_language: {main_language}\n"
        f"token_list: {json.dumps(token_view, ensure_ascii=False)}"
    )


def chunk_list(
    items: List[Dict[str, Any]], chunk_size: int
) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def normalize_foreign_predictions(
    parsed: Optional[Dict[str, Any]],
    valid_indices: set[int],
) -> List[Dict[str, Any]]:
    if not parsed:
        return []

    foreign_words = parsed.get("foreign_words", [])
    if not isinstance(foreign_words, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in foreign_words:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index")
        if not isinstance(raw_index, int):
            continue
        if raw_index not in valid_indices:
            continue
        cleaned.append(
            {
                "index": raw_index,
                "token": str(item.get("token", "")),
                "language": safe_lang(item.get("language")),
                "confidence": safe_confidence(item.get("confidence")),
            }
        )

    dedup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for item in cleaned:
        key = (item["index"], item["language"])
        if key not in dedup or item["confidence"] > dedup[key]["confidence"]:
            dedup[key] = item
    return list(dedup.values())


def normalize_text_for_prompt(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def build_main_language_input(text: str, max_chars: int, slices: int) -> str:
    normalized = normalize_text_for_prompt(text)
    if len(normalized) <= max_chars:
        return normalized
    if slices <= 1:
        return normalized[:max_chars]

    per_slice = max(1, max_chars // slices)
    n = len(normalized)
    starts = [0]
    if slices >= 2:
        starts.append(max(0, (n // 2) - (per_slice // 2)))
    if slices >= 3:
        starts.append(max(0, n - per_slice))
    while len(starts) < slices:
        starts.append(max(0, (len(starts) * (n - per_slice)) // max(1, slices - 1)))

    used = set()
    segments: List[str] = []
    for start in starts:
        if start in used:
            continue
        used.add(start)
        segments.append(normalized[start : start + per_slice])

    assembled = "\n\n".join(segments)
    return assembled[:max_chars]


def detect_main_language(
    text_for_main: str,
    args: argparse.Namespace,
) -> Tuple[str, float, LLMResult]:
    main_prompt = build_prompt_main(text_for_main)
    main_result = call_with_retry(
        model=args.ollama_model,
        prompt=main_prompt,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    if main_result.valid_json and main_result.parsed:
        return (
            safe_lang(main_result.parsed.get("main_language")),
            safe_confidence(main_result.parsed.get("confidence")),
            main_result,
        )
    return "unknown", 0.0, main_result


def detect_main_language_with_fallback(
    text: str, args: argparse.Namespace
) -> Tuple[str, float, LLMResult]:
    text_for_main = build_main_language_input(
        text, max_chars=args.max_main_chars, slices=args.main_slices
    )
    doc_lang, doc_confidence, main_result = detect_main_language(text_for_main, args)
    if doc_lang != "unknown":
        return doc_lang, doc_confidence, main_result

    normalized = normalize_text_for_prompt(text)
    if not normalized:
        return "unknown", 0.0, main_result

    if args.debug:
        print(
            "[debug] Main detection failed; running chunked fallback vote on sampled slices."
        )

    chunks: List[str] = []
    fallback_slice_count = 3
    per_slice = max(1, min(args.max_main_chars // fallback_slice_count, len(normalized)))
    starts = [0, max(0, (len(normalized) // 2) - (per_slice // 2)), max(0, len(normalized) - per_slice)]
    for start in starts:
        chunks.append(normalized[start : start + per_slice])

    votes: List[str] = []
    confidences: List[float] = []
    for chunk in chunks:
        lang, conf, _ = detect_main_language(chunk, args)
        if lang == "unknown":
            continue
        votes.append(lang)
        confidences.append(conf)

    if not votes:
        return "unknown", 0.0, main_result

    winner, _ = Counter(votes).most_common(1)[0]
    winner_confidences = [c for v, c in zip(votes, confidences) if v == winner]
    avg_confidence = (
        sum(winner_confidences) / len(winner_confidences) if winner_confidences else 0.0
    )
    return winner, avg_confidence, main_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect document language and identify foreign words."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input document (text file or PDF).",
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "txt", "pdf"],
        help="Input format override. Defaults to auto-detect by extension.",
    )
    parser.add_argument(
        "--output",
        default="foreign_words.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Ollama model to use for both main-language and foreign-word detection.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base Ollama URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout per request in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries after transport errors or invalid JSON.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Base retry backoff seconds for transport failures.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Candidate token batch size for foreign-word detection calls.",
    )
    parser.add_argument(
        "--max-main-chars",
        type=int,
        default=8000,
        help="Maximum characters sent to main-language detection prompt.",
    )
    parser.add_argument(
        "--main-slices",
        type=int,
        default=3,
        help="Number of sampled slices (start/middle/end) for main-language prompt context.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug diagnostics for extraction and LLM parse issues.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional minimum confidence for word detections.",
    )

    return parser.parse_args()


def load_input_text(input_path, input_format):
    if input_format == "auto":
        if input_path.lower().endswith(".pdf"):
            input_format = "pdf"
        else:
            input_format = "txt"

    if input_format == "txt":
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    if input_format == "pdf":
        doc = pymupdf.open(input_path)
        text_parts = [page.get_text() for page in doc]
        return "\n\n".join(text_parts)

    raise ValueError(f"Unsupported input format '{input_format}'.")


def main():
    args = parse_args()
    text = load_input_text(args.input, args.input_format)
    if args.debug:
        preview = normalize_text_for_prompt(text)[:500]
        print(f"[debug] Extracted text chars: {len(text)}")
        print(f"[debug] Extracted preview: {preview}")

    doc_lang, doc_confidence, main_result = detect_main_language_with_fallback(text, args)
    if args.debug and (doc_lang == "unknown" or not main_result.valid_json):
        print(f"[debug] Main detection valid_json={main_result.valid_json}")
        print(f"[debug] Main detection parse_error={main_result.parse_error}")
        raw_preview = main_result.raw_text[:500].replace("\n", "\\n")
        print(f"[debug] Main detection raw response preview: {raw_preview}")

    print(
        f"Document language: {doc_lang} (confidence={float(doc_confidence):.4f}) using {args.ollama_model}"
    )

    clean_text = text.replace("\n", " ")
    words = clean_text.split()
    token_records = []
    for index, word in enumerate(words):
        normalized_word = STRIP_PUNCT_RE.sub("", word)
        if not normalized_word:
            continue
        if normalized_word.isdigit():
            continue
        token_records.append(
            {"index": index, "word": word, "normalized_word": normalized_word}
        )

    if doc_lang == "unknown":
        print("Document language unknown; no foreign-word detection performed.")
        foreign_records = []
    else:
        predictions: List[Dict[str, Any]] = []
        batches = chunk_list(token_records, args.batch_size)
        for batch in batches:
            foreign_prompt = build_prompt_foreign_batch(doc_lang, batch)
            foreign_result = call_with_retry(
                model=args.ollama_model,
                prompt=foreign_prompt,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            valid_indices = {item["index"] for item in batch}
            batch_predictions = normalize_foreign_predictions(
                parsed=foreign_result.parsed if foreign_result.valid_json else None,
                valid_indices=valid_indices,
            )
            predictions.extend(batch_predictions)

        dedup_by_index: Dict[int, Dict[str, Any]] = {}
        for item in predictions:
            if item["language"] == doc_lang or item["language"] == "unknown":
                continue
            if (
                args.confidence_threshold is not None
                and item["confidence"] < args.confidence_threshold
            ):
                continue
            index = item["index"]
            if (
                index not in dedup_by_index
                or item["confidence"] > dedup_by_index[index]["confidence"]
            ):
                dedup_by_index[index] = item

        token_map = {item["index"]: item for item in token_records}
        foreign_records = []
        for index in sorted(dedup_by_index.keys()):
            token_data = token_map.get(index)
            if token_data is None:
                continue
            item = dedup_by_index[index]
            foreign_records.append(
                {
                    "word": token_data["word"],
                    "normalized_word": token_data["normalized_word"],
                    "index": index,
                    "detected_lang": item["language"],
                    "confidence": float(item["confidence"]),
                    "document_lang": doc_lang,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        for record in foreign_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Foreign words written: {len(foreign_records)} to {args.output} using {args.ollama_model}"
    )


if __name__ == "__main__":
    main()
