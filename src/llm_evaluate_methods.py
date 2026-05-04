import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evaluate_methods import (
    JsonlStreamWriter,
    classification_metrics,
    log,
    new_stats,
    parse_csv,
    should_save_raw_row,
    stats_mean,
    strip_token,
    tokenize,
    update_outcome_counts,
    update_stats,
    write_csv,
    write_json,
)
from prepare_data import (
    load_prepared_datasets_from_dir,
    prepared_manifest_summary,
)


PROMPT = """Find foreign-word tokens inside Spanish text.

Analyze only the text between <INPUT> and </INPUT>.
Assume the sentence is Spanish-context text.
Return JSON only.

Return this structure:
{
  "main_language": "es",
  "foreign_tokens": [
    {
      "token": "",
      "language": "",
      "confidence": 0.0
    }
  ]
}

Rules:
- Always set "main_language" to "es".
- Include only natural-language words that are not Spanish in "foreign_tokens".
- Exclude proper nouns, names, brands, titles, organizations, places, acronyms, initialisms, URLs, emails, hashtags, code, numbers, punctuation, and symbols.
- Exclude Spanish words and adapted loanwords commonly used in Spanish.
- Preserve each returned token exactly as written.
- Use ISO 639-1 language codes when clear; otherwise use "unknown".
- When uncertain, exclude the token.
- If none are found, return an empty "foreign_tokens" array.

Examples:
Apple presentó el nuevo iPhone. -> {"main_language":"es","foreign_tokens":[]}
La NASA publicó datos. -> {"main_language":"es","foreign_tokens":[]}
Me gusta hacer running. -> {"main_language":"es","foreign_tokens":[{"token":"running","language":"en","confidence":0.95}]}

<INPUT>
{{sentence}}
</INPUT>"""


MODEL_FAMILY = "ollama_llm"
# DEFAULT_MODELS = "llama3.2:latest,ministral-3:8b,qwen3.5:4b,gemma4"
DEFAULT_MODELS = "gemma4"
SPANISH = "es"
ISO_LANG_RE = re.compile(r"^[a-z]{2}$")
SPECIAL_TOKEN_LABELS = {"punctuation", "number", "proper_noun", "acronym", "unknown"}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "main_language": {"type": "string", "const": "es"},
        "foreign_tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "token": {"type": "string"},
                    "language": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["token", "language", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["main_language", "foreign_tokens"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LLMResult:
    parsed: Optional[Dict[str, Any]]
    raw_text: str
    valid_json: bool
    parse_error: Optional[str]
    latency_ms: float
    retry_count: int
    transport_retry_count: int
    json_retry_count: int
    total_duration_ms: float
    load_duration_ms: float
    prompt_eval_count: int
    prompt_eval_duration_ms: float
    eval_count: int
    eval_duration_ms: float


@dataclass(frozen=True)
class NormalizedTokenPrediction:
    token: str
    language: str
    is_foreign: bool
    confidence: float


@dataclass(frozen=True)
class NormalizedForeignPrediction:
    token: str
    language: str
    confidence: float


@dataclass(frozen=True)
class NormalizedResponse:
    main_language: str
    tokens: Tuple[NormalizedTokenPrediction, ...]
    foreign_tokens: Tuple[NormalizedForeignPrediction, ...]


@dataclass(frozen=True)
class AggregatePrediction:
    main_language: str
    token_predictions: Dict[int, Dict[str, Any]]
    foreign_predictions: Dict[int, Dict[str, Any]]
    valid_json: bool
    parse_error: Optional[str]
    latency_ms: float
    retry_count: int
    call_count: int
    total_duration_ms: float
    load_duration_ms: float
    prompt_eval_count: int
    prompt_eval_duration_ms: float
    eval_count: int
    eval_duration_ms: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Ollama LLM language identification using a prepared "
            "dataset profile created by src/prepare_data.py."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/flores_llm_run_gemma",
    )
    parser.add_argument(
        "--prepared-data-dir",
        required=True,
        help="Prepared dataset profile directory containing manifest and sample JSONLs.",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated Ollama models to evaluate.",
    )
    parser.add_argument("--skip-pure", action="store_true")
    parser.add_argument("--skip-injected", action="store_true")
    parser.add_argument("--skip-phrase-swaps", action="store_true")
    parser.add_argument(
        "--save-raw-level",
        choices=("errors_only", "token_full", "all_raw"),
        default="errors_only",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Transport retries per sample.",
    )
    parser.add_argument(
        "--json-retries",
        type=int,
        default=1,
        help="Retries for invalid JSON responses.",
    )
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-consecutive-transport-failures", type=int, default=12)
    parser.add_argument(
        "--keep-alive",
        default="10m",
        help="Ollama keep_alive value to keep the model loaded between requests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit each selected dataset to the first N samples. Use 0 for all.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Report dataset progress every N samples. Use 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--progress-min-seconds",
        type=float,
        default=30.0,
        help="Minimum seconds between periodic progress logs.",
    )
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="Log each rendered prompt and raw Ollama response.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.skip_pure and args.skip_injected and args.skip_phrase_swaps:
        raise ValueError("At least one evaluation dataset must run.")


def prompt_for_text(text):
    return PROMPT.replace("{{sentence}}", text)


def duration_ms_from_ns(value):
    try:
        return float(value) / 1_000_000.0
    except Exception:
        return 0.0


def format_transport_error(exc):
    message = f"transport_error:{type(exc).__name__}: {exc}"
    if not isinstance(exc, HTTPError):
        return message

    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if not body:
        return message
    return f"{message} body={body}"


def debug_log_llm_event(
    *,
    enabled,
    model,
    stage,
    prompt,
    sample_label=None,
    raw_text=None,
    error=None,
):
    if not enabled:
        return

    header = f"[{model}] LLM {stage}"
    if sample_label:
        header = f"{header} sample={sample_label}"
    log(header)
    log(f"[{model}] Prompt:\n{prompt}")
    if raw_text is not None:
        log(f"[{model}] Response:\n{raw_text}")
    if error is not None:
        log(f"[{model}] Error: {error}")


def call_ollama(
    model,
    prompt,
    ollama_url,
    temperature,
    timeout,
    keep_alive,
    debug_llm=False,
    sample_label=None,
    attempt_label=None,
):
    debug_log_llm_event(
        enabled=debug_llm,
        model=model,
        stage=attempt_label or "request",
        prompt=prompt,
        sample_label=sample_label,
    )
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature},
    }

    start = time.perf_counter()
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urlopen(request, timeout=timeout)
    latency_ms = (time.perf_counter() - start) * 1000.0
    data = json.loads(response.read().decode("utf-8"))
    raw_text = data.get("response", "")
    debug_log_llm_event(
        enabled=debug_llm,
        model=model,
        stage=(attempt_label or "request") + "_response",
        prompt=prompt,
        sample_label=sample_label,
        raw_text=raw_text,
    )
    return (
        raw_text,
        latency_ms,
        {
            "total_duration_ms": duration_ms_from_ns(data.get("total_duration")),
            "load_duration_ms": duration_ms_from_ns(data.get("load_duration")),
            "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
            "prompt_eval_duration_ms": duration_ms_from_ns(
                data.get("prompt_eval_duration")
            ),
            "eval_count": int(data.get("eval_count") or 0),
            "eval_duration_ms": duration_ms_from_ns(data.get("eval_duration")),
        },
    )


def call_with_retry(
    *,
    model,
    prompt,
    ollama_url,
    temperature,
    timeout,
    retries,
    json_retries,
    retry_backoff_seconds,
    keep_alive,
    debug_llm=False,
    sample_label=None,
):
    total_latency = 0.0
    last_error = None
    last_raw = ""
    usage = {
        "total_duration_ms": 0.0,
        "load_duration_ms": 0.0,
        "prompt_eval_count": 0,
        "prompt_eval_duration_ms": 0.0,
        "eval_count": 0,
        "eval_duration_ms": 0.0,
    }
    transport_retry_count = 0
    json_retry_count = 0

    for attempt in range(retries + 1):
        attempt_started_at = time.perf_counter()
        try:
            raw_text, latency, usage = call_ollama(
                model=model,
                prompt=prompt,
                ollama_url=ollama_url,
                temperature=temperature,
                timeout=timeout,
                keep_alive=keep_alive,
                debug_llm=debug_llm,
                sample_label=sample_label,
                attempt_label=f"transport_attempt_{attempt + 1}",
            )
            total_latency += latency
            last_raw = raw_text
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            total_latency += (time.perf_counter() - attempt_started_at) * 1000.0
            last_error = format_transport_error(exc)
            debug_log_llm_event(
                enabled=debug_llm,
                model=model,
                stage=f"transport_attempt_{attempt + 1}_error",
                prompt=prompt,
                sample_label=sample_label,
                error=last_error,
            )
            if attempt < retries:
                transport_retry_count += 1
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
                retry_count=transport_retry_count,
                transport_retry_count=transport_retry_count,
                json_retry_count=json_retry_count,
                total_duration_ms=usage["total_duration_ms"],
                load_duration_ms=usage["load_duration_ms"],
                prompt_eval_count=usage["prompt_eval_count"],
                prompt_eval_duration_ms=usage["prompt_eval_duration_ms"],
                eval_count=usage["eval_count"],
                eval_duration_ms=usage["eval_duration_ms"],
            )

        for json_attempt in range(json_retries + 1):
            try:
                parsed = json.loads(raw_text)
                if not isinstance(parsed, dict):
                    raise ValueError("Top-level JSON value must be an object.")
                return LLMResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    valid_json=True,
                    parse_error=None,
                    latency_ms=total_latency,
                    retry_count=transport_retry_count + json_retry_count,
                    transport_retry_count=transport_retry_count,
                    json_retry_count=json_retry_count,
                    total_duration_ms=usage["total_duration_ms"],
                    load_duration_ms=usage["load_duration_ms"],
                    prompt_eval_count=usage["prompt_eval_count"],
                    prompt_eval_duration_ms=usage["prompt_eval_duration_ms"],
                    eval_count=usage["eval_count"],
                    eval_duration_ms=usage["eval_duration_ms"],
                )
            except Exception as exc:
                last_error = str(exc)
                debug_log_llm_event(
                    enabled=debug_llm,
                    model=model,
                    stage=f"json_attempt_{json_attempt + 1}_error",
                    prompt=prompt,
                    sample_label=sample_label,
                    raw_text=raw_text,
                    error=last_error,
                )
                if json_attempt >= json_retries:
                    break
                json_retry_count += 1
                json_retry_started_at = time.perf_counter()
                try:
                    raw_text, latency, usage = call_ollama(
                        model=model,
                        prompt=prompt,
                        ollama_url=ollama_url,
                        temperature=temperature,
                        timeout=timeout,
                        keep_alive=keep_alive,
                        debug_llm=debug_llm,
                        sample_label=sample_label,
                        attempt_label=f"json_retry_{json_attempt + 1}",
                    )
                    total_latency += latency
                    last_raw = raw_text
                except (HTTPError, URLError, TimeoutError, OSError) as retry_exc:
                    total_latency += (
                        time.perf_counter() - json_retry_started_at
                    ) * 1000.0
                    last_error = format_transport_error(retry_exc)
                    debug_log_llm_event(
                        enabled=debug_llm,
                        model=model,
                        stage=f"json_retry_{json_attempt + 1}_error",
                        prompt=prompt,
                        sample_label=sample_label,
                        error=last_error,
                    )
                    return LLMResult(
                        parsed=None,
                        raw_text=last_raw,
                        valid_json=False,
                        parse_error=last_error,
                        latency_ms=total_latency,
                        retry_count=transport_retry_count + json_retry_count,
                        transport_retry_count=transport_retry_count,
                        json_retry_count=json_retry_count,
                        total_duration_ms=usage["total_duration_ms"],
                        load_duration_ms=usage["load_duration_ms"],
                        prompt_eval_count=usage["prompt_eval_count"],
                        prompt_eval_duration_ms=usage["prompt_eval_duration_ms"],
                        eval_count=usage["eval_count"],
                        eval_duration_ms=usage["eval_duration_ms"],
                    )

    return LLMResult(
        parsed=None,
        raw_text=last_raw,
        valid_json=False,
        parse_error=last_error,
        latency_ms=total_latency,
        retry_count=transport_retry_count + json_retry_count,
        transport_retry_count=transport_retry_count,
        json_retry_count=json_retry_count,
        total_duration_ms=usage["total_duration_ms"],
        load_duration_ms=usage["load_duration_ms"],
        prompt_eval_count=usage["prompt_eval_count"],
        prompt_eval_duration_ms=usage["prompt_eval_duration_ms"],
        eval_count=usage["eval_count"],
        eval_duration_ms=usage["eval_duration_ms"],
    )


def safe_lang(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if ISO_LANG_RE.match(normalized):
            return normalized
    return "unknown"


def safe_token_label(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if ISO_LANG_RE.match(normalized) or normalized in SPECIAL_TOKEN_LABELS:
            return normalized
    return "unknown"


def safe_confidence(value):
    try:
        confidence = float(value)
    except Exception:
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def normalize_response(parsed):
    if not isinstance(parsed, dict):
        return NormalizedResponse(
            main_language=SPANISH,
            tokens=(),
            foreign_tokens=(),
        )

    normalized_foreign = []
    for item in parsed.get("foreign_tokens", []):
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        if not isinstance(token, str):
            continue
        normalized_foreign.append(
            NormalizedForeignPrediction(
                token=token,
                language=safe_lang(item.get("language")),
                confidence=safe_confidence(item.get("confidence", 0.0)),
            )
        )

    main_language = safe_lang(parsed.get("main_language"))
    return NormalizedResponse(
        main_language=main_language if main_language != "unknown" else SPANISH,
        tokens=(),
        foreign_tokens=tuple(normalized_foreign),
    )


def alignment_key(value):
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    normalized, _start, _end = strip_token(stripped)
    candidate = normalized or stripped
    return candidate.casefold()


def align_token_predictions(predictions, scope_tokens):
    aligned = {}
    claimed_indexes = set()
    for prediction in predictions:
        key = alignment_key(prediction.token)
        if not key:
            continue
        for token in scope_tokens:
            if token.raw_index in claimed_indexes:
                continue
            if (
                alignment_key(token.raw) != key
                and alignment_key(token.normalized) != key
            ):
                continue
            aligned[token.raw_index] = {
                "token": token.raw,
                "normalized_token": token.normalized,
                "predicted_lang": prediction.language,
                "is_foreign": bool(prediction.is_foreign),
                "confidence": prediction.confidence,
            }
            claimed_indexes.add(token.raw_index)
            break
    return aligned


def align_foreign_token_predictions(predictions, scope_tokens):
    aligned = {}
    claimed_indexes = set()
    for prediction in predictions:
        key = alignment_key(prediction.token)
        if not key:
            continue
        for token in scope_tokens:
            if token.raw_index in claimed_indexes:
                continue
            if (
                alignment_key(token.raw) != key
                and alignment_key(token.normalized) != key
            ):
                continue
            aligned[token.raw_index] = {
                "token": token.raw,
                "normalized_token": token.normalized,
                "predicted_lang": prediction.language,
                "confidence": prediction.confidence,
            }
            claimed_indexes.add(token.raw_index)
            break
    return aligned


def build_text_row(
    *,
    dataset_name,
    sample_id,
    model,
    llm_mode,
    chunk_size,
    aggregate,
    base_extra,
    text,
    token_count,
    correct,
):
    return {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "model": model,
        "model_family": MODEL_FAMILY,
        "input_level": "text",
        "llm_mode": llm_mode,
        "chunk_size": chunk_size,
        "predicted_lang": aggregate.main_language,
        "text_length_chars": len(text),
        "text_length_words": token_count,
        "correct": correct,
        "llm_valid_json": aggregate.valid_json,
        "llm_parse_error": aggregate.parse_error,
        "llm_latency_ms": aggregate.latency_ms,
        "llm_retry_count": aggregate.retry_count,
        "llm_call_count": aggregate.call_count,
        "llm_total_duration_ms": aggregate.total_duration_ms,
        "llm_load_duration_ms": aggregate.load_duration_ms,
        "llm_prompt_eval_count": aggregate.prompt_eval_count,
        "llm_prompt_eval_duration_ms": aggregate.prompt_eval_duration_ms,
        "llm_eval_count": aggregate.eval_count,
        "llm_eval_duration_ms": aggregate.eval_duration_ms,
        **base_extra,
    }


def build_word_row(
    *,
    dataset_name,
    sample_id,
    model,
    llm_mode,
    chunk_size,
    token,
    predicted_lang,
    confidence,
    correct,
    is_foreign_ground_truth,
    is_foreign_predicted,
    base_extra,
):
    return {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "model": model,
        "model_family": MODEL_FAMILY,
        "input_level": "word",
        "llm_mode": llm_mode,
        "chunk_size": chunk_size,
        "token_index": token.raw_index,
        "token": token.raw,
        "normalized_token": token.normalized,
        "predicted_lang": predicted_lang,
        "confidence": confidence,
        "correct": correct,
        "is_foreign_ground_truth": is_foreign_ground_truth,
        "is_foreign_predicted": is_foreign_predicted,
        **base_extra,
    }


def predict_full_text(args, model, text, tokens, sample_label=None):
    llm_result = call_with_retry(
        model=model,
        prompt=prompt_for_text(text),
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        json_retries=args.json_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        keep_alive=args.keep_alive,
        debug_llm=args.debug_llm,
        sample_label=sample_label,
    )
    normalized = normalize_response(llm_result.parsed)
    foreign_predictions = align_foreign_token_predictions(
        normalized.foreign_tokens, tokens
    )
    token_predictions = {}
    for token in tokens:
        foreign_prediction = foreign_predictions.get(token.raw_index)
        if foreign_prediction is not None:
            token_predictions[token.raw_index] = {
                "token": token.raw,
                "normalized_token": token.normalized,
                "predicted_lang": foreign_prediction["predicted_lang"],
                "is_foreign": True,
                "confidence": foreign_prediction["confidence"],
            }
        else:
            token_predictions[token.raw_index] = {
                "token": token.raw,
                "normalized_token": token.normalized,
                "predicted_lang": SPANISH,
                "is_foreign": False,
                "confidence": 0.0,
            }
    return AggregatePrediction(
        main_language=normalized.main_language,
        token_predictions=token_predictions,
        foreign_predictions=foreign_predictions,
        valid_json=llm_result.valid_json,
        parse_error=llm_result.parse_error,
        latency_ms=llm_result.latency_ms,
        retry_count=llm_result.retry_count,
        call_count=1,
        total_duration_ms=llm_result.total_duration_ms,
        load_duration_ms=llm_result.load_duration_ms,
        prompt_eval_count=llm_result.prompt_eval_count,
        prompt_eval_duration_ms=llm_result.prompt_eval_duration_ms,
        eval_count=llm_result.eval_count,
        eval_duration_ms=llm_result.eval_duration_ms,
    )


def predicted_lang_for_token(aggregate, token_index):
    token_prediction = aggregate.token_predictions.get(token_index)
    if token_prediction is not None:
        return token_prediction["predicted_lang"], token_prediction["confidence"]
    foreign_prediction = aggregate.foreign_predictions.get(token_index)
    if foreign_prediction is not None:
        return foreign_prediction["predicted_lang"], foreign_prediction["confidence"]
    return SPANISH, 0.0


def init_pure_group():
    return {
        "total": 0,
        "correct": 0,
        "latency_ms": new_stats(),
        "retry_count": new_stats(),
        "invalid_json": 0,
        "call_count": new_stats(),
        "foreign_predicted": 0,
    }


def init_mixed_group():
    return {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "text_total": 0,
        "text_correct": 0,
        "latency_ms": new_stats(),
        "retry_count": new_stats(),
        "call_count": new_stats(),
        "invalid_json": 0,
    }


def finalize_pure_metric_groups(groups):
    metrics = []
    for (
        model,
        llm_mode,
        chunk_size,
        level,
        true_lang,
    ), group in sorted(groups.items()):
        metric = {
            "evaluation": f"pure_{level}",
            "model": model,
            "model_family": MODEL_FAMILY,
            "llm_mode": llm_mode,
            "chunk_size": chunk_size,
            "true_lang": true_lang,
            "total": group["total"],
            "accuracy": group["correct"] / group["total"] if group["total"] else 0.0,
            "latency_ms_mean": stats_mean(group["latency_ms"]),
            "retry_count_mean": stats_mean(group["retry_count"]),
            "llm_call_count_mean": stats_mean(group["call_count"]),
            "invalid_json_rate": (
                group["invalid_json"] / group["total"] if group["total"] else 0.0
            ),
        }
        if level == "word":
            metric["foreign_false_positive_rate"] = (
                group["foreign_predicted"] / group["total"] if group["total"] else 0.0
            )
        metrics.append(metric)
    return metrics


def finalize_mixed_metric_groups(groups, evaluation_name):
    metrics = []
    for (
        model,
        llm_mode,
        chunk_size,
        injected_lang,
    ), group in sorted(groups.items()):
        metric = classification_metrics(
            group["tp"], group["fp"], group["tn"], group["fn"]
        )
        metric.update(
            {
                "evaluation": evaluation_name,
                "model": model,
                "model_family": MODEL_FAMILY,
                "llm_mode": llm_mode,
                "chunk_size": chunk_size,
                "injected_lang": injected_lang,
                "main_accuracy": (
                    group["text_correct"] / group["text_total"]
                    if group["text_total"]
                    else 0.0
                ),
                "latency_ms_mean": stats_mean(group["latency_ms"]),
                "retry_count_mean": stats_mean(group["retry_count"]),
                "llm_call_count_mean": stats_mean(group["call_count"]),
                "invalid_json_rate": (
                    group["invalid_json"] / group["text_total"]
                    if group["text_total"]
                    else 0.0
                ),
            }
        )
        metrics.append(metric)
    return metrics


def write_metrics_bundle(output_dir, pure_metrics, injected_metrics, phrase_metrics):
    summary = {}
    if pure_metrics:
        write_csv(output_dir / "pure_foreign_detection_metrics.csv", pure_metrics)
        summary["pure"] = pure_metrics
    if injected_metrics:
        write_csv(output_dir / "injected_detection_metrics.csv", injected_metrics)
        summary["injected"] = injected_metrics
    if phrase_metrics:
        write_csv(output_dir / "phrase_detection_metrics.csv", phrase_metrics)
        summary["phrase"] = phrase_metrics

    write_json(output_dir / "metrics_summary.json", summary)


def record_transport_status(consecutive_failures, parse_error):
    if isinstance(parse_error, str) and parse_error.startswith("transport_error:"):
        return consecutive_failures + 1
    return 0


def format_seconds(value):
    if value <= 0:
        return "0s"
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(round(value)), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def log_periodic_progress(
    *,
    args,
    model,
    dataset_name,
    processed,
    total,
    started_at,
    last_logged_at,
    invalid_json_count,
    consecutive_transport_failures,
    latency_stats,
    prompt_eval_stats,
    eval_stats,
):
    if args.progress_every <= 0:
        return last_logged_at
    if processed <= 0 or processed >= total or processed % args.progress_every != 0:
        return last_logged_at

    now = time.monotonic()
    if last_logged_at is not None and (
        now - last_logged_at < max(0.0, args.progress_min_seconds)
    ):
        return last_logged_at

    elapsed = max(now - started_at, 0.001)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    percent = (processed / total * 100.0) if total else 100.0
    log(
        f"[{model}] {dataset_name}: {processed}/{total} ({percent:.1f}%) "
        f"elapsed={format_seconds(elapsed)} rate={rate:.2f}/s "
        f"eta={format_seconds(eta_seconds)} invalid_json={invalid_json_count} "
        f"transport_streak={consecutive_transport_failures} "
        f"mean_latency_ms={stats_mean(latency_stats):.1f} "
        f"mean_prompt_tokens={stats_mean(prompt_eval_stats):.1f} "
        f"mean_eval_tokens={stats_mean(eval_stats):.1f}"
    )
    return now


def log_dataset_start(model, dataset_name, total):
    log(f"[{model}] Starting {dataset_name} evaluation ({total} samples)")


def log_dataset_completion(
    *,
    model,
    dataset_name,
    total,
    started_at,
    invalid_json_count,
    latency_stats,
    prompt_eval_stats,
    eval_stats,
):
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = total / elapsed if total else 0.0
    mean_latency_ms = stats_mean(latency_stats)
    log(
        f"[{model}] Completed {dataset_name} evaluation in {format_seconds(elapsed)} "
        f"rate={rate:.2f}/s invalid_json={invalid_json_count} "
        f"mean_llm_latency_ms={mean_latency_ms:.1f} "
        f"mean_prompt_tokens={stats_mean(prompt_eval_stats):.1f} "
        f"mean_eval_tokens={stats_mean(eval_stats):.1f}"
    )


def evaluate_pure_dataset(
    *,
    args,
    model,
    samples,
    text_writer,
    word_writer,
    groups,
):
    consecutive_transport_failures = 0
    total_samples = len(samples)
    dataset_started_at = time.monotonic()
    last_progress_log_at = None
    invalid_json_count = 0
    latency_stats = new_stats()
    prompt_eval_stats = new_stats()
    eval_stats = new_stats()

    log_dataset_start(model, "pure", total_samples)

    for sample_index, sample in enumerate(samples, start=1):
        tokens = tokenize(sample.text)
        token_count = len(tokens)
        base_extra = {
            "row_index": sample.row_index,
            "flores_config": sample.flores_config,
            "true_lang": sample.lang,
        }
        full_prediction = predict_full_text(
            args,
            model,
            sample.text,
            tokens,
            sample_label=f"pure:{sample.sample_id}",
        )
        consecutive_transport_failures = record_transport_status(
            consecutive_transport_failures, full_prediction.parse_error
        )
        text_row = build_text_row(
            dataset_name="pure",
            sample_id=sample.sample_id,
            model=model,
            llm_mode="full_text",
            chunk_size=None,
            aggregate=full_prediction,
            base_extra=base_extra,
            text=sample.text,
            token_count=len(tokens),
            correct=full_prediction.main_language == sample.lang,
        )
        text_group = groups[(model, "full_text", None, "text", sample.lang)]
        text_group["total"] += 1
        text_group["correct"] += int(text_row["correct"])
        text_group["invalid_json"] += int(not full_prediction.valid_json)
        invalid_json_count += int(not full_prediction.valid_json)
        update_stats(text_group["latency_ms"], text_row["llm_latency_ms"])
        update_stats(text_group["retry_count"], text_row["llm_retry_count"])
        update_stats(text_group["call_count"], text_row["llm_call_count"])
        update_stats(latency_stats, full_prediction.latency_ms)
        update_stats(prompt_eval_stats, full_prediction.prompt_eval_count)
        update_stats(eval_stats, full_prediction.eval_count)
        if should_save_raw_row(text_row, args.save_raw_level):
            text_writer.write(text_row)

        foreign_prediction_indexes = set(full_prediction.foreign_predictions)
        for token in tokens:
            predicted_lang, confidence = predicted_lang_for_token(
                full_prediction, token.raw_index
            )
            is_foreign_predicted = token.raw_index in foreign_prediction_indexes
            word_row = build_word_row(
                dataset_name="pure",
                sample_id=sample.sample_id,
                model=model,
                llm_mode="full_text",
                chunk_size=None,
                token=token,
                predicted_lang=predicted_lang,
                confidence=confidence,
                correct=predicted_lang == sample.lang,
                is_foreign_ground_truth=False,
                is_foreign_predicted=is_foreign_predicted,
                base_extra=base_extra,
            )
            word_group = groups[(model, "full_text", None, "word", sample.lang)]
            word_group["total"] += 1
            word_group["correct"] += int(word_row["correct"])
            word_group["foreign_predicted"] += int(is_foreign_predicted)
            word_group["invalid_json"] += int(not full_prediction.valid_json)
            update_stats(word_group["latency_ms"], full_prediction.latency_ms)
            update_stats(word_group["retry_count"], full_prediction.retry_count)
            update_stats(word_group["call_count"], full_prediction.call_count)
            if should_save_raw_row(word_row, args.save_raw_level):
                word_writer.write(word_row)

        last_progress_log_at = log_periodic_progress(
            args=args,
            model=model,
            dataset_name="pure",
            processed=sample_index,
            total=total_samples,
            started_at=dataset_started_at,
            last_logged_at=last_progress_log_at,
            invalid_json_count=invalid_json_count,
            consecutive_transport_failures=consecutive_transport_failures,
            latency_stats=latency_stats,
            prompt_eval_stats=prompt_eval_stats,
            eval_stats=eval_stats,
        )

        if args.max_consecutive_transport_failures > 0 and (
            consecutive_transport_failures >= args.max_consecutive_transport_failures
        ):
            raise RuntimeError(
                "Exceeded max consecutive transport failures during pure full-text evaluation."
            )

    log_dataset_completion(
        model=model,
        dataset_name="pure",
        total=total_samples,
        started_at=dataset_started_at,
        invalid_json_count=invalid_json_count,
        latency_stats=latency_stats,
        prompt_eval_stats=prompt_eval_stats,
        eval_stats=eval_stats,
    )


def evaluate_mixed_dataset(
    *,
    args,
    model,
    dataset_name,
    evaluation_name,
    samples,
    text_writer,
    word_writer,
    groups,
):
    consecutive_transport_failures = 0
    total_samples = len(samples)
    dataset_started_at = time.monotonic()
    last_progress_log_at = None
    invalid_json_count = 0
    latency_stats = new_stats()
    prompt_eval_stats = new_stats()
    eval_stats = new_stats()

    log_dataset_start(model, dataset_name, total_samples)

    for sample_index, sample in enumerate(samples, start=1):
        tokens = tokenize(sample["text"])
        injected_indexes = {
            injection["token_index"]: injection for injection in sample["injections"]
        }
        base_extra = {
            "source_sample_id": sample["source_sample_id"],
            "row_index": sample["row_index"],
            "base_lang": sample["base_lang"],
            "injected_lang": sample["injected_lang"],
            "contamination_type": sample["contamination_type"],
        }

        aggregate = predict_full_text(
            args,
            model,
            sample["text"],
            tokens,
            sample_label=f"{dataset_name}:{sample['sample_id']}",
        )

        consecutive_transport_failures = record_transport_status(
            consecutive_transport_failures, aggregate.parse_error
        )
        text_row = build_text_row(
            dataset_name=dataset_name,
            sample_id=sample["sample_id"],
            model=model,
            llm_mode="full_text",
            chunk_size=None,
            aggregate=aggregate,
            base_extra=base_extra,
            text=sample["text"],
            token_count=len(tokens),
            correct=aggregate.main_language == sample["base_lang"],
        )
        if should_save_raw_row(text_row, args.save_raw_level):
            text_writer.write(text_row)

        metric_group = groups[(model, "full_text", None, sample["injected_lang"])]
        metric_group["text_total"] += 1
        metric_group["text_correct"] += int(text_row["correct"])
        metric_group["invalid_json"] += int(not aggregate.valid_json)
        invalid_json_count += int(not aggregate.valid_json)
        update_stats(metric_group["latency_ms"], aggregate.latency_ms)
        update_stats(metric_group["retry_count"], aggregate.retry_count)
        update_stats(metric_group["call_count"], aggregate.call_count)
        update_stats(latency_stats, aggregate.latency_ms)
        update_stats(prompt_eval_stats, aggregate.prompt_eval_count)
        update_stats(eval_stats, aggregate.eval_count)

        foreign_prediction_indexes = set(aggregate.foreign_predictions)
        for token in tokens:
            truth = token.raw_index in injected_indexes
            is_foreign_predicted = token.raw_index in foreign_prediction_indexes
            predicted_lang, confidence = predicted_lang_for_token(
                aggregate, token.raw_index
            )
            word_row = build_word_row(
                dataset_name=dataset_name,
                sample_id=sample["sample_id"],
                model=model,
                llm_mode="full_text",
                chunk_size=None,
                token=token,
                predicted_lang=predicted_lang,
                confidence=confidence,
                correct=truth == is_foreign_predicted,
                is_foreign_ground_truth=truth,
                is_foreign_predicted=is_foreign_predicted,
                base_extra=base_extra,
            )
            update_outcome_counts(metric_group, truth, is_foreign_predicted)
            if should_save_raw_row(word_row, args.save_raw_level):
                word_writer.write(word_row)

        last_progress_log_at = log_periodic_progress(
            args=args,
            model=model,
            dataset_name=dataset_name,
            processed=sample_index,
            total=total_samples,
            started_at=dataset_started_at,
            last_logged_at=last_progress_log_at,
            invalid_json_count=invalid_json_count,
            consecutive_transport_failures=consecutive_transport_failures,
            latency_stats=latency_stats,
            prompt_eval_stats=prompt_eval_stats,
            eval_stats=eval_stats,
        )

        if args.max_consecutive_transport_failures > 0 and (
            consecutive_transport_failures >= args.max_consecutive_transport_failures
        ):
            raise RuntimeError(
                f"Exceeded max consecutive transport failures during {evaluation_name}."
            )

    log_dataset_completion(
        model=model,
        dataset_name=dataset_name,
        total=total_samples,
        started_at=dataset_started_at,
        invalid_json_count=invalid_json_count,
        latency_stats=latency_stats,
        prompt_eval_stats=prompt_eval_stats,
        eval_stats=eval_stats,
    )


def write_run_metadata(
    args, output_dir, models, prepared_profile_dir, prepared_summary
):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "models": [{"name": model, "family": MODEL_FAMILY} for model in models],
        "prompt": PROMPT,
        "prepared_data": prepared_summary,
        "prepared_data_dir": str(prepared_profile_dir),
    }
    write_json(output_dir / "run_metadata.json", metadata)


def open_dataset_writers(output_dir, prefix):
    return {
        "text": JsonlStreamWriter(
            output_dir / f"{prefix}_text_predictions.jsonl",
            enabled=True,
        ),
        "word": JsonlStreamWriter(
            output_dir / f"{prefix}_word_predictions.jsonl",
            enabled=True,
        ),
    }


def close_dataset_writers(writer_bundle):
    for writer in writer_bundle.values():
        writer.close()


def limit_samples(samples, max_samples):
    if max_samples <= 0:
        return samples
    return samples[:max_samples]


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    models = parse_csv(args.models)
    if not models:
        raise RuntimeError("No Ollama models were provided.")

    prepared_manifest, prepared_profile_dir, prepared_datasets = (
        load_prepared_datasets_from_dir(
            args.prepared_data_dir,
            include_pure=not args.skip_pure,
            include_injected=not args.skip_injected,
            include_phrase=not args.skip_phrase_swaps,
        )
    )
    prepared_summary = prepared_manifest_summary(prepared_manifest)
    pure_samples = []
    if not args.skip_pure:
        pure_samples = limit_samples(prepared_datasets["pure"], args.max_samples)
        if not pure_samples:
            raise RuntimeError("No prepared pure samples were loaded for evaluation.")

    injected_samples = limit_samples(
        prepared_datasets.get("injected", []), args.max_samples
    )
    phrase_samples = limit_samples(
        prepared_datasets.get("phrase", []), args.max_samples
    )

    enabled_datasets = []
    if not args.skip_pure:
        enabled_datasets.append(f"pure={len(pure_samples)}")
    if not args.skip_injected:
        enabled_datasets.append(f"injected={len(injected_samples)}")
    if not args.skip_phrase_swaps:
        enabled_datasets.append(f"phrase={len(phrase_samples)}")

    log(
        "Starting LLM evaluation "
        f"profile={prepared_summary['profile_id']} "
        f"models={models} "
        f"datasets={', '.join(enabled_datasets)} "
        f"timeout={args.timeout}s retries={args.retries} "
        f"json_retries={args.json_retries} keep_alive={args.keep_alive} "
        f"debug_llm={args.debug_llm} "
        f"progress_every={args.progress_every} "
        f"progress_min_seconds={args.progress_min_seconds}"
    )

    write_run_metadata(
        args,
        output_dir,
        models,
        prepared_profile_dir,
        prepared_summary,
    )

    pure_groups = defaultdict(init_pure_group)
    injected_groups = defaultdict(init_mixed_group)
    phrase_groups = defaultdict(init_mixed_group)

    pure_writers = open_dataset_writers(output_dir, "pure")
    injected_writers = open_dataset_writers(output_dir, "injected")
    phrase_writers = open_dataset_writers(output_dir, "phrase")

    try:
        for model in models:
            model_started_at = time.monotonic()
            log(f"Starting Ollama evaluation for model {model}")
            if not args.skip_pure:
                evaluate_pure_dataset(
                    args=args,
                    model=model,
                    samples=pure_samples,
                    text_writer=pure_writers["text"],
                    word_writer=pure_writers["word"],
                    groups=pure_groups,
                )
            if not args.skip_injected:
                evaluate_mixed_dataset(
                    args=args,
                    model=model,
                    dataset_name="injected",
                    evaluation_name="injected_word_detection",
                    samples=injected_samples,
                    text_writer=injected_writers["text"],
                    word_writer=injected_writers["word"],
                    groups=injected_groups,
                )
            if not args.skip_phrase_swaps:
                evaluate_mixed_dataset(
                    args=args,
                    model=model,
                    dataset_name="phrase",
                    evaluation_name="phrase_word_detection",
                    samples=phrase_samples,
                    text_writer=phrase_writers["text"],
                    word_writer=phrase_writers["word"],
                    groups=phrase_groups,
                )
            log(
                f"Completed Ollama evaluation for model {model} in "
                f"{format_seconds(time.monotonic() - model_started_at)}"
            )
    finally:
        close_dataset_writers(pure_writers)
        close_dataset_writers(injected_writers)
        close_dataset_writers(phrase_writers)

    pure_metrics = finalize_pure_metric_groups(pure_groups)
    injected_metrics = finalize_mixed_metric_groups(
        injected_groups, "injected_word_detection"
    )
    phrase_metrics = finalize_mixed_metric_groups(
        phrase_groups, "phrase_word_detection"
    )
    write_metrics_bundle(output_dir, pure_metrics, injected_metrics, phrase_metrics)
    log(f"Completed LLM evaluation in {time.monotonic() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
