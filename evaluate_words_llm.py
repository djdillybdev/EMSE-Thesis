import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException

ALLOWED_METHODS = {"baseline", "schema", "constrained"}
DEFAULT_MODELS = ["llama3.1:8b", "ministral-3:8b"]
ISO_LANG_RE = re.compile(r"^[a-z]{2}$")
STRIP_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)


class ProgressTracker:
    def __init__(self, total: int, label: str, log_every: int = 50) -> None:
        self.total = max(1, total)
        self.label = label
        self.log_every = max(1, log_every)
        self.count = 0
        self.started_at = time.perf_counter()

    def _render_bar(self, width: int = 28) -> str:
        ratio = min(1.0, self.count / self.total)
        filled = int(ratio * width)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _eta_seconds(self) -> float:
        if self.count <= 0:
            return 0.0
        elapsed = time.perf_counter() - self.started_at
        rate = self.count / elapsed if elapsed > 0 else 0.0
        if rate <= 0:
            return 0.0
        remaining = self.total - self.count
        return remaining / rate

    def update(self, status: str) -> None:
        self.count += 1
        elapsed = time.perf_counter() - self.started_at
        eta = self._eta_seconds()
        bar = self._render_bar()
        line = (
            f"\r{self.label} {bar} {self.count}/{self.total} "
            f"elapsed={elapsed:0.1f}s eta={eta:0.1f}s | {status}"
        )
        print(line, end="", flush=True)

        if self.count % self.log_every == 0 or self.count == self.total:
            print(
                (
                    f"\n[status] {self.label}: {self.count}/{self.total} complete "
                    f"({(self.count / self.total) * 100:0.1f}%)"
                ),
                flush=True,
            )

    def done(self) -> None:
        total_elapsed = time.perf_counter() - self.started_at
        print(
            f"\n[done] {self.label} completed in {total_elapsed:0.1f}s",
            flush=True,
        )


@dataclass
class LLMResult:
    parsed: Optional[Dict[str, Any]]
    raw_text: str
    valid_json: bool
    parse_error: Optional[str]
    latency_ms: float
    retry_count: int


def is_transport_error(parse_error: Optional[str]) -> bool:
    return isinstance(parse_error, str) and parse_error.startswith("transport_error:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLM language detection on single-word and injected-context tasks."
        )
    )
    parser.add_argument(
        "--input",
        default="test_data.jsonl",
        help="Input JSONL with fields: text, lang (or true_main_lang), source (optional).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama models to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["schema"],
        help="Prompt methods: baseline, schema, constrained.",
    )
    parser.add_argument(
        "--evaluation",
        choices=["words", "injected", "both"],
        default="both",
        help="Which evaluation to run: words, injected, or both.",
    )
    parser.add_argument(
        "--output-prefix",
        default="llm_word",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and injection.",
    )
    parser.add_argument(
        "--max-words-per-lang",
        type=int,
        default=200,
        help="Maximum number of unique normalized words sampled per language.",
    )
    parser.add_argument(
        "--inject-k",
        type=int,
        default=2,
        help="Number of foreign words injected per context sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of base samples used for injected-context task.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries after invalid JSON.",
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
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Base backoff between retries for transport failures.",
    )
    parser.add_argument(
        "--max-consecutive-transport-failures",
        type=int,
        default=12,
        help=(
            "Stop the current evaluation early after this many consecutive "
            "transport failures (0 disables early-stop)."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base Ollama URL.",
    )
    return parser.parse_args()


def validate_methods(methods: List[str]) -> List[str]:
    invalid = [m for m in methods if m not in ALLOWED_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported methods: {invalid}. Allowed: {sorted(ALLOWED_METHODS)}"
        )
    return methods


def load_samples(path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"Missing 'text' in row {idx}")
            lang = row.get("lang", row.get("true_main_lang"))
            if not isinstance(lang, str) or not ISO_LANG_RE.match(lang):
                raise ValueError(
                    f"Row {idx} has invalid language code '{lang}'. Expected ISO-639-1 lowercase."
                )
            row["lang"] = lang
            row.setdefault("source", "unknown")
            row.setdefault("id", f"sample-{idx:06d}")
            samples.append(row)
    if not samples:
        raise ValueError("Input dataset is empty.")
    return samples


def normalize_token(token: str) -> str:
    return STRIP_PUNCT_RE.sub("", token)


def build_word_pool(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pool_by_lang: Dict[str, List[Dict[str, Any]]] = {}
    seen_by_lang: Dict[str, set] = {}

    for sample in samples:
        lang = sample["lang"]
        pool_by_lang.setdefault(lang, [])
        seen_by_lang.setdefault(lang, set())

        tokens = sample["text"].replace("\n", " ").split()
        for index, token in enumerate(tokens):
            normalized = normalize_token(token)
            if not normalized:
                continue
            if normalized.isdigit():
                continue
            normalized_key = normalized.casefold()
            if normalized_key in seen_by_lang[lang]:
                continue

            seen_by_lang[lang].add(normalized_key)
            pool_by_lang[lang].append(
                {
                    "word": token,
                    "normalized_word": normalized,
                    "true_lang": lang,
                    "sample_id": sample["id"],
                    "source": sample["source"],
                    "source_index": index,
                }
            )

    return pool_by_lang


def sample_pure_words(
    pool_by_lang: Dict[str, List[Dict[str, Any]]],
    max_words_per_lang: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for lang in sorted(pool_by_lang.keys()):
        words = list(pool_by_lang[lang])
        rng.shuffle(words)
        selected.extend(words[:max_words_per_lang])
    return selected


def build_injected_samples(
    samples: List[Dict[str, Any]],
    pool_by_lang: Dict[str, List[Dict[str, Any]]],
    inject_k: int,
    max_samples: Optional[int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if inject_k <= 0:
        return []

    base_samples = list(samples)
    rng.shuffle(base_samples)
    if max_samples is not None:
        base_samples = base_samples[:max_samples]

    injected_cases: List[Dict[str, Any]] = []

    for sample in base_samples:
        base_lang = sample["lang"]
        foreign_langs = [
            lang for lang, words in pool_by_lang.items() if lang != base_lang and words
        ]
        if not foreign_langs:
            continue

        tokens: List[Dict[str, Any]] = [
            {"token": t, "injected": False, "language": base_lang}
            for t in sample["text"].replace("\n", " ").split()
        ]

        injected_target_count = min(inject_k, len(foreign_langs))
        chosen_langs = rng.sample(foreign_langs, k=injected_target_count)

        for lang in chosen_langs:
            candidate = rng.choice(pool_by_lang[lang])
            insert_idx = rng.randint(0, len(tokens))
            tokens.insert(
                insert_idx,
                {
                    "token": candidate["normalized_word"],
                    "injected": True,
                    "language": lang,
                },
            )

        final_tokens = [t["token"] for t in tokens]
        ground_truth = []
        for index, token_data in enumerate(tokens):
            if token_data["injected"]:
                ground_truth.append(
                    {
                        "index": index,
                        "token": token_data["token"],
                        "language": token_data["language"],
                    }
                )

        if not ground_truth:
            continue

        injected_cases.append(
            {
                "sample_id": sample["id"],
                "source": sample["source"],
                "base_lang": base_lang,
                "text": " ".join(final_tokens),
                "token_count": len(final_tokens),
                "ground_truth_foreign_words": ground_truth,
            }
        )

    return injected_cases


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
    if method == "constrained":
        base += "If uncertain, return your best guess.\n"
    return f"{base}\nToken:\n{word}"


def build_prompt_injected_context(text: str, method: str) -> str:
    base = (
        "Task: Identify the main language of the text and the foreign-language words.\n"
        "Return JSON only with keys: main_language, main_confidence, foreign_words.\n"
        "foreign_words must be a list of objects with keys: index, token, language, confidence.\n"
        "language must be ISO-639-1 lowercase and confidence must be in [0,1].\n"
    )
    if method in {"schema", "constrained"}:
        base += (
            "No markdown/code fences.\n"
            'Output schema: {"main_language":"en","main_confidence":0.95,"foreign_words":[{"index":5,"token":"avec","language":"fr","confidence":0.9}]}\n'
        )
    if method == "constrained":
        base += (
            "Only include words that are likely foreign relative to the main language. "
            "Use [] if none.\n"
        )
    return f"{base}\nText:\n{text}"


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


def call_with_retry(
    model: str,
    prompt: str,
    ollama_url: str,
    temperature: float,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
) -> LLMResult:
    total_latency = 0.0
    last_error = None
    last_raw = ""

    for attempt in range(retries + 1):
        attempt_started_at = time.perf_counter()
        try:
            raw_text, latency = call_ollama(
                model, prompt, ollama_url, temperature, timeout
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
                    "Your previous output was invalid JSON.\n"
                    "Return ONLY valid JSON matching the requested schema.\n"
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


def safe_lang(value: Any) -> str:
    if isinstance(value, str) and ISO_LANG_RE.match(value):
        return value
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


def normalize_foreign_predictions(
    parsed: Optional[Dict[str, Any]], token_count: int
) -> List[Dict[str, Any]]:
    if not parsed:
        return []

    foreign_words = parsed.get("foreign_words", [])
    if not isinstance(foreign_words, list):
        return []

    cleaned = []
    for item in foreign_words:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= token_count:
            continue

        lang = safe_lang(item.get("language"))
        token = item.get("token")
        conf = safe_confidence(item.get("confidence", 0.0))
        cleaned.append(
            {
                "index": idx,
                "token": token if isinstance(token, str) else "",
                "language": lang,
                "confidence": conf,
            }
        )

    # Keep only one prediction per (index, language) pair.
    dedup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for item in cleaned:
        key = (item["index"], item["language"])
        if key not in dedup or item["confidence"] > dedup[key]["confidence"]:
            dedup[key] = item

    return list(dedup.values())


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def aggregate_summary(
    pure_rows: List[Dict[str, Any]], injected_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "pure_word": {},
        "injected_context": {},
    }

    pure_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in pure_rows:
        key = (row["model"], row["method"])
        pure_groups.setdefault(key, []).append(row)

    for key, rows in pure_groups.items():
        model, method = key
        total = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        invalid_json = sum(1 for r in rows if not r["llm_valid_json"])
        avg_latency_ms = (
            sum(float(r["llm_latency_ms"]) for r in rows) / total if total else 0.0
        )
        summary["pure_word"].setdefault(model, {})[method] = {
            "samples": total,
            "accuracy": (correct / total) if total else 0.0,
            "invalid_json_rate": (invalid_json / total) if total else 0.0,
            "avg_latency_ms": avg_latency_ms,
        }

    injected_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in injected_rows:
        key = (row["model"], row["method"])
        injected_groups.setdefault(key, []).append(row)

    for key, rows in injected_groups.items():
        model, method = key
        total = len(rows)
        invalid_json = sum(1 for r in rows if not r["llm_valid_json"])
        total_tp = sum(int(r["foreign_tp"]) for r in rows)
        total_fp = sum(int(r["foreign_fp"]) for r in rows)
        total_fn = sum(int(r["foreign_fn"]) for r in rows)
        p, r, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
        avg_latency_ms = (
            sum(float(r["llm_latency_ms"]) for r in rows) / total if total else 0.0
        )
        main_accuracy = (
            sum(1 for row in rows if row["main_correct"]) / total if total else 0.0
        )
        summary["injected_context"].setdefault(model, {})[method] = {
            "samples": total,
            "main_accuracy": main_accuracy,
            "foreign_precision_micro": p,
            "foreign_recall_micro": r,
            "foreign_f1_micro": f1,
            "invalid_json_rate": (invalid_json / total) if total else 0.0,
            "avg_latency_ms": avg_latency_ms,
        }

    return summary


def evaluate_pure_words(
    words: List[Dict[str, Any]],
    models: List[str],
    methods: List[str],
    ollama_url: str,
    temperature: float,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
    max_consecutive_transport_failures: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rows: List[Dict[str, Any]] = []
    total_calls = len(models) * len(methods) * len(words)
    tracker = ProgressTracker(total=total_calls, label="pure_word", log_every=50)
    consecutive_transport_failures = 0

    for model in models:
        for method in methods:
            for item in words:
                prompt = build_prompt_word(item["normalized_word"], method)
                llm_result = call_with_retry(
                    model=model,
                    prompt=prompt,
                    ollama_url=ollama_url,
                    temperature=temperature,
                    timeout=timeout,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )

                predicted = (
                    safe_lang(llm_result.parsed.get("word_language"))
                    if llm_result.valid_json and llm_result.parsed
                    else "unknown"
                )
                confidence = (
                    safe_confidence(llm_result.parsed.get("confidence", 0.0))
                    if llm_result.valid_json and llm_result.parsed
                    else 0.0
                )

                rows.append(
                    {
                        "task": "pure_word",
                        "model": model,
                        "method": method,
                        "word": item["word"],
                        "normalized_word": item["normalized_word"],
                        "true_lang": item["true_lang"],
                        "detected_lang": predicted,
                        "confidence": confidence,
                        "correct": item["true_lang"] == predicted,
                        "sample_id": item["sample_id"],
                        "source": item["source"],
                        "source_index": item["source_index"],
                        "llm_valid_json": llm_result.valid_json,
                        "llm_parse_error": llm_result.parse_error,
                        "llm_latency_ms": llm_result.latency_ms,
                        "llm_retry_count": llm_result.retry_count,
                    }
                )

                if is_transport_error(llm_result.parse_error):
                    consecutive_transport_failures += 1
                else:
                    consecutive_transport_failures = 0

                tracker.update(
                    f"model={model} method={method} token={item['normalized_word'][:24]}"
                )

                if (
                    max_consecutive_transport_failures > 0
                    and consecutive_transport_failures
                    >= max_consecutive_transport_failures
                ):
                    tracker.done()
                    reason = (
                        "Exceeded max consecutive transport failures "
                        f"({max_consecutive_transport_failures}) in pure-word evaluation."
                    )
                    print(f"[status] {reason}")
                    return rows, reason

    tracker.done()
    return rows, None


def evaluate_injected_context(
    cases: List[Dict[str, Any]],
    models: List[str],
    methods: List[str],
    ollama_url: str,
    temperature: float,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
    max_consecutive_transport_failures: int,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    rows: List[Dict[str, Any]] = []
    total_calls = len(models) * len(methods) * len(cases)
    tracker = ProgressTracker(total=total_calls, label="injected_context", log_every=25)
    consecutive_transport_failures = 0

    for model in models:
        for method in methods:
            for case in cases:
                prompt = build_prompt_injected_context(case["text"], method)
                llm_result = call_with_retry(
                    model=model,
                    prompt=prompt,
                    ollama_url=ollama_url,
                    temperature=temperature,
                    timeout=timeout,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )

                if llm_result.valid_json and llm_result.parsed:
                    pred_main_lang = safe_lang(llm_result.parsed.get("main_language"))
                    predicted_foreign = normalize_foreign_predictions(
                        llm_result.parsed,
                        token_count=case["token_count"],
                    )
                else:
                    pred_main_lang = "unknown"
                    predicted_foreign = []

                truth_pairs = {
                    (item["index"], item["language"])
                    for item in case["ground_truth_foreign_words"]
                }
                pred_pairs = {
                    (item["index"], item["language"]) for item in predicted_foreign
                }

                tp = len(truth_pairs & pred_pairs)
                fp = len(pred_pairs - truth_pairs)
                fn = len(truth_pairs - pred_pairs)
                precision, recall, f1 = precision_recall_f1(tp, fp, fn)

                rows.append(
                    {
                        "task": "injected_context",
                        "model": model,
                        "method": method,
                        "sample_id": case["sample_id"],
                        "source": case["source"],
                        "text": case["text"],
                        "base_lang": case["base_lang"],
                        "pred_main_lang": pred_main_lang,
                        "main_correct": pred_main_lang == case["base_lang"],
                        "injected_count": len(case["ground_truth_foreign_words"]),
                        "predicted_count": len(predicted_foreign),
                        "foreign_tp": tp,
                        "foreign_fp": fp,
                        "foreign_fn": fn,
                        "foreign_precision": precision,
                        "foreign_recall": recall,
                        "foreign_f1": f1,
                        "ground_truth_foreign_words": case[
                            "ground_truth_foreign_words"
                        ],
                        "predicted_foreign_words": predicted_foreign,
                        "llm_valid_json": llm_result.valid_json,
                        "llm_parse_error": llm_result.parse_error,
                        "llm_latency_ms": llm_result.latency_ms,
                        "llm_retry_count": llm_result.retry_count,
                    }
                )

                if is_transport_error(llm_result.parse_error):
                    consecutive_transport_failures += 1
                else:
                    consecutive_transport_failures = 0

                tracker.update(
                    f"model={model} method={method} sample={case['sample_id']}"
                )

                if (
                    max_consecutive_transport_failures > 0
                    and consecutive_transport_failures
                    >= max_consecutive_transport_failures
                ):
                    tracker.done()
                    reason = (
                        "Exceeded max consecutive transport failures "
                        f"({max_consecutive_transport_failures}) in injected-context evaluation."
                    )
                    print(f"[status] {reason}")
                    return rows, reason

    tracker.done()
    return rows, None


def main() -> None:
    args = parse_args()
    methods = validate_methods(args.methods)
    rng = random.Random(args.seed)
    run_words = args.evaluation in {"words", "both"}
    run_injected = args.evaluation in {"injected", "both"}

    samples = load_samples(args.input)
    word_pool = build_word_pool(samples)
    pure_words: List[Dict[str, Any]] = []
    injected_cases: List[Dict[str, Any]] = []

    if run_words:
        pure_words = sample_pure_words(word_pool, args.max_words_per_lang, rng)
    if run_injected:
        injected_cases = build_injected_samples(
            samples=samples,
            pool_by_lang=word_pool,
            inject_k=args.inject_k,
            max_samples=args.max_samples,
            rng=rng,
        )

    print("[status] Dataset preparation complete.")
    print(f"[status] Selected evaluation: {args.evaluation}")
    print(
        f"Prepared {len(pure_words)} pure-word items across {len(word_pool)} languages."
    )
    print(f"Prepared {len(injected_cases)} injected-context items.")

    total_pure_calls = len(args.models) * len(methods) * len(pure_words)
    total_injected_calls = len(args.models) * len(methods) * len(injected_cases)
    total_calls = total_pure_calls + total_injected_calls
    print(
        "[status] Planned LLM calls: "
        f"pure_word={total_pure_calls}, injected_context={total_injected_calls}, total={total_calls}"
    )

    pure_rows: List[Dict[str, Any]] = []
    injected_rows: List[Dict[str, Any]] = []
    aborted_reason: Optional[str] = None

    if run_words:
        print("[status] Starting pure-word evaluation...")
        pure_rows, abort_reason = evaluate_pure_words(
            words=pure_words,
            models=args.models,
            methods=methods,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            max_consecutive_transport_failures=args.max_consecutive_transport_failures,
        )
        if abort_reason:
            aborted_reason = abort_reason
    else:
        print("[status] Skipping pure-word evaluation.")

    if run_injected and aborted_reason is None:
        print("[status] Starting injected-context evaluation...")
        injected_rows, abort_reason = evaluate_injected_context(
            cases=injected_cases,
            models=args.models,
            methods=methods,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            max_consecutive_transport_failures=args.max_consecutive_transport_failures,
        )
        if abort_reason:
            aborted_reason = abort_reason
    else:
        if aborted_reason is None:
            print("[status] Skipping injected-context evaluation.")
        else:
            print(
                "[status] Skipping injected-context evaluation due to earlier abort condition."
            )

    print("[status] Aggregating summary metrics...")
    summary = aggregate_summary(pure_rows, injected_rows)

    output_prefix = Path(args.output_prefix)
    pure_file = Path(f"{output_prefix}_pure_results.jsonl")
    injected_file = Path(f"{output_prefix}_injected_results.jsonl")
    summary_file = Path(f"{output_prefix}_summary.json")

    if run_words:
        write_jsonl(pure_file, pure_rows)
    if run_injected:
        write_jsonl(injected_file, injected_rows)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[status] Results written to disk.")
    if aborted_reason is not None:
        print(f"[status] Early stop triggered: {aborted_reason}")
        print("[status] Partial results were saved.")
    print(f"Done. Pure rows: {len(pure_rows)}, injected rows: {len(injected_rows)}")
    if run_words:
        print(f"Wrote {pure_file}")
    if run_injected:
        print(f"Wrote {injected_file}")
    print(f"Wrote {summary_file}")


if __name__ == "__main__":
    main()
