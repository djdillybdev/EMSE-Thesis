import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download
from lingua import Language, LanguageDetectorBuilder
from prepare_data import (
    load_prepared_datasets_from_dir,
    prepared_manifest_summary,
)


FASTTEXT_LABEL_PREFIX = "__label__"
SPANISH = "es"
NOT_SPANISH = "not_es"

DEFAULT_MODELS = (
    "spanish-binary-baseline,facebook-fasttext-language-identification,glotlid,"
    "lingua,lingua-spanish-only"
)

WHITESPACE_TOKEN_RE = re.compile(r"\S+")
URL_RE = re.compile(r"^(https?://|www\.)", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

ISO_639_3_TO_1 = {
    "eng": "en",
    "fra": "fr",
    "fre": "fr",
    "spa": "es",
    "deu": "de",
    "ger": "de",
    "ita": "it",
    "por": "pt",
    "cat": "ca",
    "eus": "eu",
    "baq": "eu",
    "nld": "nl",
    "dut": "nl",
    "pol": "pl",
    "swe": "sv",
    "rus": "ru",
    "tur": "tr",
    "ind": "id",
    "ukr": "uk",
    "vie": "vi",
    "ron": "ro",
    "rum": "ro",
    "jpn": "ja",
    "zho": "zh",
    "cmn": "zh",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
}

LINGUA_LANGUAGES = {
    "es": Language.SPANISH,
    "en": Language.ENGLISH,
    "pt": Language.PORTUGUESE,
    "it": Language.ITALIAN,
    "fr": Language.FRENCH,
    "de": Language.GERMAN,
    "ca": Language.CATALAN,
    "eu": Language.BASQUE,
}


@dataclass(frozen=True)
class Token:
    raw_index: int
    raw: str
    normalized: str
    start: int
    end: int
    normalized_start: int
    normalized_end: int
    leading: str
    trailing: str


@dataclass(frozen=True)
class Prediction:
    model: str
    model_family: str
    predicted_lang: str
    predicted_label: str
    confidence: float


@dataclass(frozen=True)
class WindowScore:
    model: str
    model_family: str
    predicted_lang: str
    predicted_label: str
    confidence: float
    main_lang: str
    main_lang_score: float
    foreign_score: float
    top_non_main_lang: str
    top_non_main_confidence: float


@dataclass(frozen=True)
class Sample:
    sample_id: str
    row_index: int
    flores_config: str
    lang: str
    text: str


def log(message):
    print(message, flush=True)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def strip_env_value(value):
    value = value.strip()
    if not value:
        return value
    if value[0] in {'"', "'"}:
        end_index = value.find(value[0], 1)
        return value[1:end_index] if end_index != -1 else value[1:]
    comment_index = value.find(" #")
    if comment_index != -1:
        value = value[:comment_index]
    return value.strip()


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return []

    loaded_keys = []
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = strip_env_value(value)
            loaded_keys.append(key)
    return loaded_keys


def normalize_lang_code(label):
    code = (label or "").replace(FASTTEXT_LABEL_PREFIX, "").strip()
    if not code:
        return "unknown"

    if "/" in code:
        code = code.rsplit("/", 1)[-1]
    if "_" in code:
        code = code.split("_", 1)[0]
    if "-" in code:
        code = code.split("-", 1)[0]

    lowered = code.lower()
    return ISO_639_3_TO_1.get(lowered, lowered)


def strip_token(raw):
    start = 0
    end = len(raw)
    while start < end and not raw[start].isalpha():
        start += 1
    while end > start and not raw[end - 1].isalpha():
        end -= 1
    return raw[start:end], start, end


def is_eligible_word(value):
    if not value:
        return False
    if URL_RE.match(value) or EMAIL_RE.match(value):
        return False
    if any(char.isdigit() for char in value):
        return False
    if "_" in value:
        return False
    return value.isalpha()


def tokenize(text):
    tokens = []
    for raw_index, match in enumerate(WHITESPACE_TOKEN_RE.finditer(text or "")):
        raw = match.group(0)
        normalized, local_start, local_end = strip_token(raw)
        if not is_eligible_word(normalized):
            continue
        tokens.append(
            Token(
                raw_index=raw_index,
                raw=raw,
                normalized=normalized,
                start=match.start(),
                end=match.end(),
                normalized_start=match.start() + local_start,
                normalized_end=match.start() + local_end,
                leading=raw[:local_start],
                trailing=raw[local_end:],
            )
        )
    return tokens


def get_lingua_code(language):
    iso_639_1 = getattr(language, "iso_code_639_1", None)
    if iso_639_1 is not None:
        name = getattr(iso_639_1, "name", None)
        if name:
            return name.lower()
        value = getattr(iso_639_1, "value", None)
        if value:
            return str(value).lower()

    iso_639_3 = getattr(language, "iso_code_639_3", None)
    if iso_639_3 is not None:
        name = getattr(iso_639_3, "name", None)
        if name:
            return normalize_lang_code(name)
        value = getattr(iso_639_3, "value", None)
        if value:
            return normalize_lang_code(str(value))

    return language.name.lower()


def selected_lingua_languages(prepared_manifest):
    configs = set(prepared_manifest["pure_configs"])
    configs.update(prepared_manifest["injection_configs"])
    language_codes = {SPANISH}
    for config in configs:
        language_codes.add(normalize_lang_code(config))

    return [
        LINGUA_LANGUAGES[code]
        for code in sorted(language_codes)
        if code in LINGUA_LANGUAGES
    ]


def build_lingua_detector(languages):
    return LanguageDetectorBuilder.from_languages(*languages).build()


def lingua_supported_langs():
    return {get_lingua_code(language) for language in Language.all()}


class ModelAdapter:
    def __init__(self, name, family, model):
        self.name = name
        self.family = family
        self.model = model

    def predict(self, text):
        raise NotImplementedError

    def score_against_main_lang(self, text, main_lang, top_k):
        raise NotImplementedError


class FastTextAdapter(ModelAdapter):
    def predict(self, text):
        clean_text = (text or "").replace("\n", " ").strip()
        if not clean_text:
            return Prediction(self.name, self.family, "unknown", "unknown", 0.0)
        labels, scores = self.model.predict(clean_text, k=1)
        label = labels[0].replace(FASTTEXT_LABEL_PREFIX, "")
        lang = normalize_lang_code(label)
        return Prediction(self.name, self.family, lang, label, float(scores[0]))

    def label_scores(self, text, top_k):
        clean_text = (text or "").replace("\n", " ").strip()
        if not clean_text:
            return []
        labels, scores = self.model.predict(clean_text, k=top_k)
        return [
            (
                label.replace(FASTTEXT_LABEL_PREFIX, ""),
                normalize_lang_code(label),
                float(score),
            )
            for label, score in zip(labels, scores)
        ]

    def score_against_main_lang(self, text, main_lang, top_k):
        label_scores = self.label_scores(text, top_k)
        return build_window_score(self, main_lang, label_scores)


class SpanishBinaryAdapter(FastTextAdapter):
    def predict(self, text):
        prediction = super().predict(text)
        label = prediction.predicted_label
        lang = SPANISH if label == SPANISH else NOT_SPANISH
        return Prediction(
            self.name,
            self.family,
            lang,
            label,
            prediction.confidence,
        )

    def label_scores(self, text, top_k):
        clean_text = (text or "").replace("\n", " ").strip()
        if not clean_text:
            return []
        labels, scores = self.model.predict(clean_text, k=2)
        output = []
        for label, score in zip(labels, scores):
            raw_label = label.replace(FASTTEXT_LABEL_PREFIX, "")
            lang = SPANISH if raw_label == SPANISH else NOT_SPANISH
            output.append((raw_label, lang, float(score)))
        return output


class LinguaAdapter(ModelAdapter):
    def predict(self, text):
        clean_text = (text or "").replace("\n", " ").strip()
        if not clean_text:
            return Prediction(self.name, self.family, "unknown", "unknown", 0.0)
        confidence_values = self.model.compute_language_confidence_values(clean_text)
        if not confidence_values:
            return Prediction(self.name, self.family, "unknown", "unknown", 0.0)
        top = confidence_values[0]
        lang = get_lingua_code(top.language)
        return Prediction(self.name, self.family, lang, lang, float(top.value))

    def score_against_main_lang(self, text, main_lang, top_k):
        clean_text = (text or "").replace("\n", " ").strip()
        if not clean_text:
            return build_window_score(self, main_lang, [])
        confidence_values = self.model.compute_language_confidence_values(clean_text)
        label_scores = [
            (
                get_lingua_code(value.language),
                get_lingua_code(value.language),
                float(value.value),
            )
            for value in confidence_values
        ]
        return build_window_score(self, main_lang, label_scores[:top_k])


class LinguaSpanishOnlyAdapter(ModelAdapter):
    def predict(self, text):
        clean_text = (text or "").replace("\n", " ").strip()
        if not clean_text:
            return Prediction(self.name, self.family, "unknown", "unknown", 0.0)

        detected = self.model.detect_language_of(clean_text)
        if detected == Language.SPANISH:
            return Prediction(self.name, self.family, SPANISH, SPANISH, 1.0)
        return Prediction(self.name, self.family, NOT_SPANISH, NOT_SPANISH, 1.0)

    def score_against_main_lang(self, text, main_lang, top_k):
        prediction = self.predict(text)
        main_lang_score = 1.0 if prediction.predicted_lang == main_lang else 0.0
        foreign_score = 1.0 - main_lang_score
        top_non_main_lang = (
            prediction.predicted_lang
            if prediction.predicted_lang != main_lang
            else "unknown"
        )
        return WindowScore(
            model=self.name,
            model_family=self.family,
            predicted_lang=prediction.predicted_lang,
            predicted_label=prediction.predicted_label,
            confidence=prediction.confidence,
            main_lang=main_lang,
            main_lang_score=main_lang_score,
            foreign_score=foreign_score,
            top_non_main_lang=top_non_main_lang,
            top_non_main_confidence=foreign_score,
        )


def build_window_score(model, main_lang, label_scores):
    if not label_scores:
        return WindowScore(
            model=model.name,
            model_family=model.family,
            predicted_lang="unknown",
            predicted_label="unknown",
            confidence=0.0,
            main_lang=main_lang,
            main_lang_score=0.0,
            foreign_score=1.0,
            top_non_main_lang="unknown",
            top_non_main_confidence=0.0,
        )

    predicted_label, predicted_lang, confidence = label_scores[0]
    main_lang_score = 0.0
    top_non_main_lang = "unknown"
    top_non_main_confidence = 0.0

    for label, lang, score in label_scores:
        if lang == main_lang:
            main_lang_score = max(main_lang_score, score)
        elif score > top_non_main_confidence:
            top_non_main_lang = lang
            top_non_main_confidence = score

    # For multi-language models, preserve both the total non-main mass and the
    # strongest explicit non-main alternative when constructing foreignness.
    foreign_score = max(1.0 - main_lang_score, top_non_main_confidence)
    foreign_score = max(0.0, min(1.0, foreign_score))
    return WindowScore(
        model=model.name,
        model_family=model.family,
        predicted_lang=predicted_lang,
        predicted_label=predicted_label,
        confidence=confidence,
        main_lang=main_lang,
        main_lang_score=main_lang_score,
        foreign_score=foreign_score,
        top_non_main_lang=top_non_main_lang,
        top_non_main_confidence=top_non_main_confidence,
    )


def discover_binary_models(model_root):
    root = Path(model_root)
    if not root.exists():
        return []
    model_paths = sorted(root.glob("*/model.bin"))
    return [(f"spanish-binary-{path.parent.name}", path) for path in model_paths]


def load_fasttext_hf_model(repo_id, filename):
    return fasttext.load_model(hf_hub_download(repo_id=repo_id, filename=filename))


def load_models(args, prepared_manifest):
    selected = set(parse_csv(args.models)) if args.models else None
    models = []
    fasttext.FastText.eprint = lambda _message: None

    for name, path in discover_binary_models(args.binary_model_root):
        if selected and name not in selected:
            continue
        log(f"Loading {name} from {path}")
        models.append(
            SpanishBinaryAdapter(
                name=name,
                family="spanish_binary",
                model=fasttext.load_model(str(path)),
            )
        )

    external_specs = [
        (
            "facebook-fasttext-language-identification",
            "fasttext_lid",
            args.facebook_fasttext_repo,
            args.facebook_fasttext_filename,
        ),
        ("glotlid", "glotlid", args.glotlid_repo, args.glotlid_filename),
    ]
    for name, family, repo_id, filename in external_specs:
        if selected and name not in selected:
            continue
        log(f"Loading {name} from Hugging Face repo {repo_id}/{filename}")
        models.append(
            FastTextAdapter(
                name=name,
                family=family,
                model=load_fasttext_hf_model(repo_id, filename),
            )
        )

    lingua_languages = selected_lingua_languages(prepared_manifest)
    if not selected or "lingua" in selected:
        log(
            "Building Lingua detector for "
            f"{', '.join(get_lingua_code(language) for language in lingua_languages)}"
        )
        models.append(
            LinguaAdapter(
                "lingua",
                "lingua",
                build_lingua_detector(lingua_languages),
            )
        )

    if not selected or "lingua-spanish-only" in selected:
        log("Building Lingua Spanish-only detector")
        models.append(
            LinguaSpanishOnlyAdapter(
                "lingua-spanish-only",
                "lingua_binary",
                build_lingua_detector([Language.SPANISH]),
            )
        )

    if selected:
        loaded = {model.name for model in models}
        missing = selected - loaded
        if missing:
            raise ValueError(
                f"Requested model(s) not loaded: {', '.join(sorted(missing))}"
            )

    if not models:
        raise RuntimeError("No models were loaded.")
    return models


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    log(f"Wrote {count} rows to {path}")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Wrote {len(rows)} rows to {path}")


class JsonlStreamWriter:
    def __init__(self, path, enabled=True, transform=None):
        self.path = Path(path)
        self.enabled = enabled
        self.transform = transform
        self.count = 0
        self.handle = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")

    def write(self, row):
        if not self.enabled:
            return
        if self.transform is not None:
            row = self.transform(row)
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self):
        if self.handle is not None:
            self.handle.close()
            log(f"Wrote {self.count} rows to {self.path}")
            self.handle = None


def should_save_raw_row(row, save_raw_level):
    if save_raw_level == "none":
        return False
    if save_raw_level in {"token_full", "all_raw"}:
        return True
    return not row.get("correct", True)


def should_save_prediction_rows(save_raw_level):
    return save_raw_level != "none"


def should_save_window_token_rows(save_window_raw):
    return save_window_raw in {"token_scores", "all"}


def should_save_window_rows(save_window_raw):
    return save_window_raw == "all"


def slim_prediction_row(row):
    slim = dict(row)
    for key in ("input_level", "row_index", "flores_config"):
        slim.pop(key, None)
    return slim


def slim_window_token_row(row, save_raw_level):
    slim = dict(row)
    for key in ("input_level", "row_index", "flores_config", "detection_method"):
        slim.pop(key, None)

    if save_raw_level != "all_raw":
        for key in (
            "token",
            "normalized_token",
            "original_normalized",
            "replacement_normalized",
            "span_id",
            "span_offset",
            "span_length",
        ):
            slim.pop(key, None)
    return slim


def new_stats():
    return {"count": 0, "sum": 0.0, "min": None, "max": None}


def update_stats(stats, value):
    if value is None:
        return
    value = float(value)
    stats["count"] += 1
    stats["sum"] += value
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)


def stats_mean(stats):
    return stats["sum"] / stats["count"] if stats["count"] else 0.0


def stats_min(stats):
    return stats["min"] if stats["min"] is not None else 0.0


def stats_max(stats):
    return stats["max"] if stats["max"] is not None else 0.0


def update_outcome_counts(group, truth, predicted):
    if truth and predicted:
        group["tp"] += 1
        return "tp"
    if not truth and predicted:
        group["fp"] += 1
        return "fp"
    if not truth and not predicted:
        group["tn"] += 1
        return "tn"
    group["fn"] += 1
    return "fn"


def finalize_pure_metrics(metrics_by_level):
    output = []
    for level, groups in metrics_by_level.items():
        for (model, family, true_lang), group in sorted(groups.items()):
            metric = {
                "evaluation": f"pure_{level}",
                "model": model,
                "model_family": family,
                "true_lang": true_lang,
                "total": group["total"],
                "accuracy": (
                    group["correct"] / group["total"] if group["total"] else 0.0
                ),
                "confidence_mean": stats_mean(group["confidence"]),
                "confidence_min": stats_min(group["confidence"]),
                "confidence_max": stats_max(group["confidence"]),
            }
            if level == "word":
                metric["foreign_false_positive_rate"] = (
                    group["foreign_predicted"] / group["total"]
                    if group["total"]
                    else 0.0
                )
            output.append(metric)
    return output


def finalize_injected_metrics(groups, evaluation_name):
    metrics = []
    for (model, family, injected_lang), group in sorted(groups.items()):
        metric = classification_metrics(
            group["tp"], group["fp"], group["tn"], group["fn"]
        )
        metric.update(
            {
                "evaluation": evaluation_name,
                "model": model,
                "model_family": family,
                "injected_lang": injected_lang,
                "tp_confidence_mean": stats_mean(group["tp_confidence"]),
                "fp_confidence_mean": stats_mean(group["fp_confidence"]),
                "tn_confidence_mean": stats_mean(group["tn_confidence"]),
                "fn_confidence_mean": stats_mean(group["fn_confidence"]),
                "text_confidence_mean": stats_mean(group["text_confidence"]),
            }
        )
        metrics.append(metric)
    return metrics


def finalize_pure_window_metrics(groups):
    metrics = []
    for (
        model,
        family,
        true_lang,
        window_size,
        decision_rule,
        threshold,
        shared_threshold,
    ), group in sorted(groups.items()):
        metrics.append(
            {
                "evaluation": "pure_window_token_detection",
                "model": model,
                "model_family": family,
                "true_lang": true_lang,
                "window_size": window_size,
                "window_decision_rule": decision_rule,
                "window_foreign_threshold": threshold,
                "window_shared_foreign_threshold": shared_threshold,
                "total": group["total"],
                "foreign_false_positive_rate": (
                    group["foreign_predicted"] / group["total"]
                    if group["total"]
                    else 0.0
                ),
                "foreign_probability_mean": stats_mean(group["foreign_probability"]),
                "foreign_probability_min": stats_min(group["foreign_probability"]),
                "foreign_probability_max": stats_max(group["foreign_probability"]),
                "main_lang_probability_mean": stats_mean(
                    group["main_lang_probability"]
                ),
            }
        )
    return metrics


def finalize_injected_window_metrics(groups, evaluation_name):
    metrics = []
    for (
        model,
        family,
        injected_lang,
        window_size,
        decision_rule,
        threshold,
        shared_threshold,
    ), group in sorted(groups.items()):
        metric = classification_metrics(
            group["tp"], group["fp"], group["tn"], group["fn"]
        )
        metric.update(
            {
                "evaluation": evaluation_name,
                "model": model,
                "model_family": family,
                "injected_lang": injected_lang,
                "window_size": window_size,
                "window_decision_rule": decision_rule,
                "window_foreign_threshold": threshold,
                "window_shared_foreign_threshold": shared_threshold,
                "tp_foreign_probability_mean": stats_mean(
                    group["tp_foreign_probability"]
                ),
                "fp_foreign_probability_mean": stats_mean(
                    group["fp_foreign_probability"]
                ),
                "tn_foreign_probability_mean": stats_mean(
                    group["tn_foreign_probability"]
                ),
                "fn_foreign_probability_mean": stats_mean(
                    group["fn_foreign_probability"]
                ),
                "main_lang_probability_mean": stats_mean(
                    group["main_lang_probability"]
                ),
            }
        )
        metrics.append(metric)
    return metrics


def prediction_record(prediction, extra):
    row = {
        "model": prediction.model,
        "model_family": prediction.model_family,
        "predicted_lang": prediction.predicted_lang,
        "predicted_label": prediction.predicted_label,
        "confidence": prediction.confidence,
    }
    row.update(extra)
    return row


def is_prediction_correct(model, true_lang, predicted_lang):
    if model.family in {"spanish_binary", "lingua_binary"}:
        expected = SPANISH if true_lang == SPANISH else NOT_SPANISH
        return predicted_lang == expected
    return predicted_lang == true_lang


def is_foreign_detection(model, text_lang, word_prediction):
    if model.family in {"spanish_binary", "lingua_binary"}:
        return word_prediction.predicted_lang == NOT_SPANISH
    return (
        text_lang != "unknown"
        and word_prediction.predicted_lang != "unknown"
        and word_prediction.predicted_lang != text_lang
    )


def parse_int_csv(value):
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value):
    return [float(item) for item in parse_csv(str(value))]


def parse_window_decision_modes(value):
    modes = parse_csv(value)
    if not modes:
        raise ValueError("--window-decision-modes must include at least one mode.")

    allowed = {"legacy_window", "contextual_hybrid"}
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise ValueError(
            "--window-decision-modes contains unsupported value(s): "
            + ", ".join(invalid)
        )
    return modes


def parse_window_foreign_thresholds(value):
    thresholds = parse_float_csv(value)
    if not thresholds:
        raise ValueError("--window-foreign-threshold must include at least one float.")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("--window-foreign-threshold must be in the range [0, 1].")
    return thresholds


def validate_probability_thresholds(values, flag_name):
    if not values:
        raise ValueError(f"{flag_name} must include at least one float.")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{flag_name} must be in the range [0, 1].")
    return values


def parse_window_contextual_thresholds(value):
    return validate_probability_thresholds(
        parse_float_csv(value), "--window-contextual-threshold"
    )


def parse_window_shared_foreign_thresholds(value):
    return validate_probability_thresholds(
        parse_float_csv(value), "--window-shared-foreign-threshold"
    )


def window_score_record(score, extra):
    row = {
        "model": score.model,
        "model_family": score.model_family,
        "predicted_lang": score.predicted_lang,
        "predicted_label": score.predicted_label,
        "confidence": score.confidence,
        "main_lang": score.main_lang,
        "main_lang_score": score.main_lang_score,
        "foreign_score": score.foreign_score,
        "top_non_main_lang": score.top_non_main_lang,
        "top_non_main_confidence": score.top_non_main_confidence,
        "detection_method": "window",
    }
    row.update(extra)
    return row


def build_window_rows_and_token_scores(
    *,
    sample,
    tokens,
    model,
    main_lang,
    window_sizes,
    thresholds,
    top_k,
    base_extra,
    token_truth,
    token_injections=None,
    include_window_rows=True,
    window_row_writer=None,
    decision_modes=("legacy_window",),
    contextual_thresholds=None,
    shared_foreign_thresholds=None,
    shared_foreign_min_window_count=1,
    shared_foreign_min_ratio=0.5,
):
    token_injections = token_injections or {}
    contextual_thresholds = contextual_thresholds or ()
    shared_foreign_thresholds = shared_foreign_thresholds or ()
    token_score_inputs = defaultdict(
        lambda: {"foreign_sum": 0.0, "main_sum": 0.0, "count": 0, "window_margins": []}
    )
    tokens_by_raw_index = {token.raw_index: token for token in tokens}
    token_positions = {
        token.raw_index: position for position, token in enumerate(tokens)
    }
    token_self_scores = {
        token.raw_index: model.score_against_main_lang(
            token.normalized, main_lang, top_k
        )
        for token in tokens
    }

    for window_size in window_sizes:
        if len(tokens) < window_size:
            continue
        for window_start in range(0, len(tokens) - window_size + 1):
            window_tokens = tokens[window_start : window_start + window_size]
            score = model.score_against_main_lang(
                " ".join(token.normalized for token in window_tokens),
                main_lang,
                top_k,
            )
            window_id = (
                f"{sample['sample_id'] if isinstance(sample, dict) else sample.sample_id}:"
                f"{model.name}:w{window_size}:{window_start}"
            )
            token_indexes = [token.raw_index for token in window_tokens]
            if include_window_rows:
                window_row_writer.write(
                    window_score_record(
                        score,
                        {
                            **base_extra,
                            "window_id": window_id,
                            "window_size": window_size,
                            "window_start": window_start,
                            "window_token_start": token_indexes[0],
                            "window_token_end": token_indexes[-1],
                            "window_token_indexes": token_indexes,
                            "window_text": " ".join(
                                token.normalized for token in window_tokens
                            ),
                        },
                    )
                )
            for token in window_tokens:
                aggregate = token_score_inputs[(window_size, token.raw_index)]
                aggregate["foreign_sum"] += score.foreign_score
                aggregate["main_sum"] += score.main_lang_score
                aggregate["count"] += 1
                aggregate["window_margins"].append(
                    score.foreign_score - score.main_lang_score
                )

    for (window_size, token_index), aggregate in sorted(token_score_inputs.items()):
        token_item = tokens_by_raw_index[token_index]
        foreign_probability = aggregate["foreign_sum"] / aggregate["count"]
        main_lang_probability = aggregate["main_sum"] / aggregate["count"]
        window_margin = foreign_probability - main_lang_probability
        self_score = token_self_scores[token_index]
        self_main_lang_probability = self_score.main_lang_score
        self_foreign_probability = self_score.foreign_score
        self_margin = self_foreign_probability - self_main_lang_probability
        neighbor_scores = build_neighbor_margin_scores(
            tokens=tokens,
            token_positions=token_positions,
            token_self_scores=token_self_scores,
            token_index=token_index,
            window_size=window_size,
        )
        neighbor_margin_baseline = weighted_median(neighbor_scores)
        contextual_margin_delta = self_margin - neighbor_margin_baseline
        neighbor_count = len(neighbor_scores)
        shared_window_weight_sum = sum(weight for _value, weight in neighbor_scores)
        injection = token_injections.get(token_index, {})
        truth = token_truth(token_index)
        base_row = {
            **base_extra,
            "model": model.name,
            "model_family": model.family,
            "detection_method": "window",
            "window_size": window_size,
            "token_index": token_item.raw_index,
            "token": token_item.raw,
            "normalized_token": token_item.normalized,
            "main_lang": main_lang,
            "main_lang_probability": main_lang_probability,
            "foreign_probability": foreign_probability,
            "window_count": aggregate["count"],
            "window_margin": window_margin,
            "self_main_lang_probability": self_main_lang_probability,
            "self_foreign_probability": self_foreign_probability,
            "self_margin": self_margin,
            "neighbor_margin_baseline": neighbor_margin_baseline,
            "contextual_margin_delta": contextual_margin_delta,
            "neighbor_count": neighbor_count,
            "shared_window_weight_sum": shared_window_weight_sum,
            "original_normalized": injection.get("original_normalized"),
            "replacement_normalized": injection.get("replacement_normalized"),
            "span_id": injection.get("span_id"),
            "span_offset": injection.get("span_offset"),
            "span_length": injection.get("span_length"),
        }
        for mode in decision_modes:
            for threshold in thresholds_for_mode(
                mode, thresholds, contextual_thresholds
            ):
                predicted = predict_window_token_foreign(
                    mode=mode,
                    main_lang_probability=main_lang_probability,
                    foreign_probability=foreign_probability,
                    contextual_margin_delta=contextual_margin_delta,
                    threshold=threshold,
                )

                row = {
                    **base_row,
                    "window_decision_rule": mode,
                    "window_foreign_threshold": threshold,
                    "window_shared_foreign_threshold": None,
                    "window_shared_foreign_min_window_count": None,
                    "window_shared_foreign_min_ratio": None,
                    "shared_foreign_window_count": 0,
                    "shared_foreign_window_ratio": 0.0,
                    "consensus_run_length": 0,
                    "is_foreign_ground_truth": truth,
                    "is_foreign_predicted": predicted,
                    "correct": truth == predicted,
                }
                yield row

            if mode == "contextual_hybrid":
                for threshold in contextual_thresholds:
                    for shared_threshold in shared_foreign_thresholds:
                        yield from build_contextual_hybrid_rows(
                            tokens=tokens,
                            tokens_by_raw_index=tokens_by_raw_index,
                            token_truth=token_truth,
                            token_injections=token_injections,
                            base_row=base_row,
                            base_extra=base_extra,
                            model=model,
                            main_lang=main_lang,
                            window_size=window_size,
                            token_score_inputs=token_score_inputs,
                            contextual_threshold=threshold,
                            shared_foreign_threshold=shared_threshold,
                            shared_foreign_min_window_count=shared_foreign_min_window_count,
                            shared_foreign_min_ratio=shared_foreign_min_ratio,
                        )
                break


def shared_window_count(sequence_length, window_size, left_index, right_index):
    min_start = max(0, max(left_index, right_index) - window_size + 1)
    max_start = min(min(left_index, right_index), sequence_length - window_size)
    if max_start < min_start:
        return 0
    return max_start - min_start + 1


def build_neighbor_margin_scores(
    *,
    tokens,
    token_positions,
    token_self_scores,
    token_index,
    window_size,
):
    sequence_length = len(tokens)
    source_position = token_positions[token_index]
    scores = []
    for neighbor in tokens:
        if neighbor.raw_index == token_index:
            continue
        neighbor_position = token_positions[neighbor.raw_index]
        weight = shared_window_count(
            sequence_length, window_size, source_position, neighbor_position
        )
        if weight <= 0:
            continue
        neighbor_score = token_self_scores[neighbor.raw_index]
        neighbor_margin = neighbor_score.foreign_score - neighbor_score.main_lang_score
        scores.append((neighbor_margin, weight))
    return scores


def weighted_median(values):
    if not values:
        return 0.0
    sorted_values = sorted(values, key=lambda item: item[0])
    total_weight = sum(weight for _value, weight in sorted_values)
    midpoint = total_weight / 2.0
    running_weight = 0.0
    for value, weight in sorted_values:
        running_weight += weight
        if running_weight >= midpoint:
            return value
    return sorted_values[-1][0]


def thresholds_for_mode(mode, legacy_thresholds, contextual_thresholds):
    if mode == "legacy_window":
        return legacy_thresholds
    if mode == "contextual_hybrid":
        return ()
    raise ValueError(f"Unsupported window decision mode: {mode}")


def predict_window_token_foreign(
    *,
    mode,
    main_lang_probability,
    foreign_probability,
    contextual_margin_delta,
    threshold,
):
    if mode == "legacy_window":
        return foreign_probability >= threshold
    if mode == "contextual_hybrid":
        return contextual_margin_delta >= threshold
    raise ValueError(f"Unsupported window decision mode: {mode}")


def build_contextual_hybrid_rows(
    *,
    tokens,
    tokens_by_raw_index,
    token_truth,
    token_injections,
    base_row,
    base_extra,
    model,
    main_lang,
    window_size,
    token_score_inputs,
    contextual_threshold,
    shared_foreign_threshold,
    shared_foreign_min_window_count,
    shared_foreign_min_ratio,
):
    per_token = []
    for token in tokens:
        aggregate = token_score_inputs[(window_size, token.raw_index)]
        foreign_probability = aggregate["foreign_sum"] / aggregate["count"]
        main_lang_probability = aggregate["main_sum"] / aggregate["count"]
        window_margin = foreign_probability - main_lang_probability
        shared_foreign_window_count = sum(
            1
            for margin in aggregate["window_margins"]
            if margin >= shared_foreign_threshold
        )
        shared_foreign_window_ratio = (
            shared_foreign_window_count / aggregate["count"]
            if aggregate["count"]
            else 0.0
        )
        per_token.append(
            {
                "token_index": token.raw_index,
                "window_margin": window_margin,
                "shared_foreign_window_count": shared_foreign_window_count,
                "shared_foreign_window_ratio": shared_foreign_window_ratio,
                "eligible_for_shared_foreign": (
                    shared_foreign_window_count >= shared_foreign_min_window_count
                    and shared_foreign_window_ratio >= shared_foreign_min_ratio
                ),
            }
        )

    run_lengths = {}
    run_start = 0
    while run_start < len(per_token):
        if not per_token[run_start]["eligible_for_shared_foreign"]:
            run_lengths[per_token[run_start]["token_index"]] = 0
            run_start += 1
            continue
        run_end = run_start
        while (
            run_end + 1 < len(per_token)
            and per_token[run_end + 1]["eligible_for_shared_foreign"]
        ):
            run_end += 1
        run_length = run_end - run_start + 1
        for item in per_token[run_start : run_end + 1]:
            run_lengths[item["token_index"]] = run_length
        run_start = run_end + 1

    token_lookup = {item["token_index"]: item for item in per_token}
    token_index = base_row["token_index"]
    token_state = token_lookup[token_index]
    predicted = base_row["contextual_margin_delta"] >= contextual_threshold or (
        base_row["contextual_margin_delta"] > -contextual_threshold
        and run_lengths[token_index] >= 2
    )
    truth = token_truth(token_index)
    injection = token_injections.get(token_index, {})

    yield {
        **base_extra,
        "model": model.name,
        "model_family": model.family,
        "detection_method": "window",
        "window_size": window_size,
        "token_index": token_index,
        "token": tokens_by_raw_index[token_index].raw,
        "normalized_token": tokens_by_raw_index[token_index].normalized,
        "main_lang": main_lang,
        "main_lang_probability": base_row["main_lang_probability"],
        "foreign_probability": base_row["foreign_probability"],
        "window_count": base_row["window_count"],
        "window_margin": token_state["window_margin"],
        "self_main_lang_probability": base_row["self_main_lang_probability"],
        "self_foreign_probability": base_row["self_foreign_probability"],
        "self_margin": base_row["self_margin"],
        "neighbor_margin_baseline": base_row["neighbor_margin_baseline"],
        "contextual_margin_delta": base_row["contextual_margin_delta"],
        "neighbor_count": base_row["neighbor_count"],
        "shared_window_weight_sum": base_row["shared_window_weight_sum"],
        "window_decision_rule": "contextual_hybrid",
        "window_foreign_threshold": contextual_threshold,
        "window_shared_foreign_threshold": shared_foreign_threshold,
        "window_shared_foreign_min_window_count": shared_foreign_min_window_count,
        "window_shared_foreign_min_ratio": shared_foreign_min_ratio,
        "shared_foreign_window_count": token_state["shared_foreign_window_count"],
        "shared_foreign_window_ratio": token_state["shared_foreign_window_ratio"],
        "consensus_run_length": run_lengths[token_index],
        "is_foreign_ground_truth": truth,
        "is_foreign_predicted": predicted,
        "correct": truth == predicted,
        "original_normalized": injection.get("original_normalized"),
        "replacement_normalized": injection.get("replacement_normalized"),
        "span_id": injection.get("span_id"),
        "span_offset": injection.get("span_offset"),
        "span_length": injection.get("span_length"),
    }


def run_pure_evaluation(args, models, pure_samples, supported_langs):
    output_dir = Path(args.output_dir)
    text_path = output_dir / "pure_text_predictions.jsonl"
    word_path = output_dir / "pure_word_predictions.jsonl"
    pure_window_path = output_dir / "pure_window_predictions.jsonl"
    pure_window_token_path = output_dir / "pure_window_token_scores.jsonl"
    only_window = args.only_window

    if not pure_samples:
        raise RuntimeError("No prepared pure samples were loaded.")

    pure_metrics_groups = {
        "text": defaultdict(
            lambda: {"total": 0, "correct": 0, "confidence": new_stats()}
        ),
        "word": defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "confidence": new_stats(),
                "foreign_predicted": 0,
            }
        ),
    }
    pure_window_groups = defaultdict(
        lambda: {
            "total": 0,
            "foreign_predicted": 0,
            "foreign_probability": new_stats(),
            "main_lang_probability": new_stats(),
        }
    )
    window_sizes = parse_int_csv(args.window_sizes)
    window_decision_modes = parse_window_decision_modes(args.window_decision_modes)
    window_thresholds = parse_window_foreign_thresholds(args.window_foreign_threshold)
    window_contextual_thresholds = parse_window_contextual_thresholds(
        args.window_contextual_threshold
    )
    window_shared_foreign_thresholds = parse_window_shared_foreign_thresholds(
        args.window_shared_foreign_threshold
    )
    text_writer = JsonlStreamWriter(
        text_path,
        enabled=not only_window and should_save_prediction_rows(args.save_raw_level),
        transform=slim_prediction_row,
    )
    word_writer = JsonlStreamWriter(
        word_path,
        enabled=not only_window and should_save_prediction_rows(args.save_raw_level),
        transform=slim_prediction_row,
    )
    window_writer = JsonlStreamWriter(
        pure_window_path,
        enabled=not args.skip_window and should_save_window_rows(args.save_window_raw),
        transform=slim_prediction_row,
    )
    window_token_writer = JsonlStreamWriter(
        pure_window_token_path,
        enabled=not args.skip_window
        and should_save_window_token_rows(args.save_window_raw),
        transform=lambda row: slim_window_token_row(row, args.save_raw_level),
    )

    try:
        for sample in pure_samples:
            tokens = tokenize(sample.text)
            for model in models:
                text_prediction = model.predict(sample.text)
                base_extra = {
                    "sample_id": sample.sample_id,
                    "row_index": sample.row_index,
                    "flores_config": sample.flores_config,
                    "true_lang": sample.lang,
                    "text_predicted_lang": text_prediction.predicted_lang,
                    "is_supported_target": is_supported_target(
                        supported_langs, model.name, sample.lang
                    ),
                }
                text_row = prediction_record(
                    text_prediction,
                    {
                        **base_extra,
                        "input_level": "text",
                        "text_length_chars": len(sample.text),
                        "text_length_words": len(tokens),
                        "correct": is_prediction_correct(
                            model, sample.lang, text_prediction.predicted_lang
                        ),
                    },
                )
                if not only_window:
                    text_group = pure_metrics_groups["text"][
                        (model.name, model.family, sample.lang)
                    ]
                    text_group["total"] += 1
                    text_group["correct"] += int(text_row["correct"])
                    update_stats(text_group["confidence"], text_row["confidence"])
                    if should_save_raw_row(text_row, args.save_raw_level):
                        text_writer.write(text_row)

                if not args.skip_window:
                    for window_token_row in build_window_rows_and_token_scores(
                        sample=sample,
                        tokens=tokens,
                        model=model,
                        main_lang=sample.lang,
                        window_sizes=window_sizes,
                        thresholds=window_thresholds,
                        top_k=args.window_top_k,
                        base_extra={
                            **base_extra,
                            "input_level": "window",
                        },
                        token_truth=lambda _token_index: False,
                        include_window_rows=should_save_window_rows(
                            args.save_window_raw
                        ),
                        window_row_writer=window_writer,
                        decision_modes=window_decision_modes,
                        contextual_thresholds=window_contextual_thresholds,
                        shared_foreign_thresholds=window_shared_foreign_thresholds,
                        shared_foreign_min_window_count=args.window_shared_foreign_min_window_count,
                        shared_foreign_min_ratio=args.window_shared_foreign_min_ratio,
                    ):
                        window_group = pure_window_groups[
                            (
                                window_token_row["model"],
                                window_token_row["model_family"],
                                window_token_row["true_lang"],
                                window_token_row["window_size"],
                                window_token_row["window_decision_rule"],
                                window_token_row["window_foreign_threshold"],
                                window_token_row["window_shared_foreign_threshold"],
                            )
                        ]
                        window_group["total"] += 1
                        window_group["foreign_predicted"] += int(
                            window_token_row["is_foreign_predicted"]
                        )
                        update_stats(
                            window_group["foreign_probability"],
                            window_token_row["foreign_probability"],
                        )
                        update_stats(
                            window_group["main_lang_probability"],
                            window_token_row["main_lang_probability"],
                        )
                        window_token_writer.write(window_token_row)

                if not only_window:
                    for token_item in tokens:
                        word_prediction = model.predict(token_item.normalized)
                        is_foreign = is_foreign_detection(
                            model, sample.lang, word_prediction
                        )
                        word_row = prediction_record(
                            word_prediction,
                            {
                                "sample_id": sample.sample_id,
                                "row_index": sample.row_index,
                                "flores_config": sample.flores_config,
                                "true_lang": sample.lang,
                                "input_level": "word",
                                "token_index": token_item.raw_index,
                                "token": token_item.raw,
                                "normalized_token": token_item.normalized,
                                "text_predicted_lang": text_prediction.predicted_lang,
                                "is_foreign_predicted": is_foreign,
                                "correct": is_prediction_correct(
                                    model,
                                    sample.lang,
                                    word_prediction.predicted_lang,
                                ),
                                "is_supported_target": is_supported_target(
                                    supported_langs, model.name, sample.lang
                                ),
                            },
                        )
                        word_group = pure_metrics_groups["word"][
                            (model.name, model.family, sample.lang)
                        ]
                        word_group["total"] += 1
                        word_group["correct"] += int(word_row["correct"])
                        word_group["foreign_predicted"] += int(
                            word_row["is_foreign_predicted"]
                        )
                        update_stats(word_group["confidence"], word_row["confidence"])
                        if should_save_raw_row(word_row, args.save_raw_level):
                            word_writer.write(word_row)
    finally:
        text_writer.close()
        word_writer.close()
        window_writer.close()
        window_token_writer.close()

    return (
        [] if only_window else finalize_pure_metrics(pure_metrics_groups)
    ), finalize_pure_window_metrics(pure_window_groups)


def run_mixed_evaluation(args, models, samples, output_prefix, evaluation_name):
    output_dir = Path(args.output_dir)
    only_window = args.only_window
    word_groups = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tp_confidence": new_stats(),
            "fp_confidence": new_stats(),
            "tn_confidence": new_stats(),
            "fn_confidence": new_stats(),
            "text_confidence": new_stats(),
        }
    )
    window_groups = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tp_foreign_probability": new_stats(),
            "fp_foreign_probability": new_stats(),
            "tn_foreign_probability": new_stats(),
            "fn_foreign_probability": new_stats(),
            "main_lang_probability": new_stats(),
        }
    )
    window_sizes = parse_int_csv(args.window_sizes)
    window_decision_modes = parse_window_decision_modes(args.window_decision_modes)
    window_thresholds = parse_window_foreign_thresholds(args.window_foreign_threshold)
    window_contextual_thresholds = parse_window_contextual_thresholds(
        args.window_contextual_threshold
    )
    window_shared_foreign_thresholds = parse_window_shared_foreign_thresholds(
        args.window_shared_foreign_threshold
    )
    word_writer = JsonlStreamWriter(
        output_dir / f"{output_prefix}_word_predictions.jsonl",
        enabled=not only_window and should_save_prediction_rows(args.save_raw_level),
        transform=slim_prediction_row,
    )
    window_writer = JsonlStreamWriter(
        output_dir / f"{output_prefix}_window_predictions.jsonl",
        enabled=not args.skip_window and should_save_window_rows(args.save_window_raw),
        transform=slim_prediction_row,
    )
    window_token_writer = JsonlStreamWriter(
        output_dir / f"{output_prefix}_window_token_scores.jsonl",
        enabled=not args.skip_window
        and should_save_window_token_rows(args.save_window_raw),
        transform=lambda row: slim_window_token_row(row, args.save_raw_level),
    )
    try:
        for sample in samples:
            sample_text = sample["text"]
            injected_indexes = {
                injection["token_index"]: injection
                for injection in sample["injections"]
            }
            tokens = tokenize(sample_text)
            for model in models:
                text_prediction = model.predict(sample_text)
                base_extra = {
                    "sample_id": sample["sample_id"],
                    "source_sample_id": sample["source_sample_id"],
                    "row_index": sample["row_index"],
                    "base_lang": SPANISH,
                    "injected_lang": sample["injected_lang"],
                    "contamination_type": sample.get(
                        "contamination_type", output_prefix
                    ),
                    "text_predicted_lang": text_prediction.predicted_lang,
                    "text_predicted_label": text_prediction.predicted_label,
                    "text_confidence": text_prediction.confidence,
                }
                if not args.skip_window:
                    for window_token_row in build_window_rows_and_token_scores(
                        sample=sample,
                        tokens=tokens,
                        model=model,
                        main_lang=SPANISH,
                        window_sizes=window_sizes,
                        thresholds=window_thresholds,
                        top_k=args.window_top_k,
                        base_extra={
                            **base_extra,
                            "input_level": "window",
                        },
                        token_truth=lambda token_index, indexes=injected_indexes: (
                            token_index in indexes
                        ),
                        token_injections=injected_indexes,
                        include_window_rows=should_save_window_rows(
                            args.save_window_raw
                        ),
                        window_row_writer=window_writer,
                        decision_modes=window_decision_modes,
                        contextual_thresholds=window_contextual_thresholds,
                        shared_foreign_thresholds=window_shared_foreign_thresholds,
                        shared_foreign_min_window_count=args.window_shared_foreign_min_window_count,
                        shared_foreign_min_ratio=args.window_shared_foreign_min_ratio,
                    ):
                        window_group = window_groups[
                            (
                                window_token_row["model"],
                                window_token_row["model_family"],
                                window_token_row["injected_lang"],
                                window_token_row["window_size"],
                                window_token_row["window_decision_rule"],
                                window_token_row["window_foreign_threshold"],
                                window_token_row["window_shared_foreign_threshold"],
                            )
                        ]
                        outcome = update_outcome_counts(
                            window_group,
                            window_token_row["is_foreign_ground_truth"],
                            window_token_row["is_foreign_predicted"],
                        )
                        update_stats(
                            window_group[f"{outcome}_foreign_probability"],
                            window_token_row["foreign_probability"],
                        )
                        update_stats(
                            window_group["main_lang_probability"],
                            window_token_row["main_lang_probability"],
                        )
                        window_token_writer.write(window_token_row)

                if not only_window:
                    for token_item in tokens:
                        word_prediction = model.predict(token_item.normalized)
                        truth = token_item.raw_index in injected_indexes
                        predicted = is_foreign_detection(
                            model, SPANISH, word_prediction
                        )
                        injection = injected_indexes.get(token_item.raw_index, {})
                        row = prediction_record(
                            word_prediction,
                            {
                                **base_extra,
                                "input_level": "word",
                                "token_index": token_item.raw_index,
                                "token": token_item.raw,
                                "normalized_token": token_item.normalized,
                                "is_foreign_ground_truth": truth,
                                "is_foreign_predicted": predicted,
                                "original_normalized": injection.get(
                                    "original_normalized"
                                ),
                                "replacement_normalized": injection.get(
                                    "replacement_normalized"
                                ),
                                "span_id": injection.get("span_id"),
                                "span_offset": injection.get("span_offset"),
                                "span_length": injection.get("span_length"),
                                "correct": truth == predicted,
                            },
                        )
                        word_group = word_groups[
                            (
                                row["model"],
                                row["model_family"],
                                row["injected_lang"],
                            )
                        ]
                        outcome = update_outcome_counts(
                            word_group,
                            row["is_foreign_ground_truth"],
                            row["is_foreign_predicted"],
                        )
                        update_stats(
                            word_group[f"{outcome}_confidence"],
                            row["confidence"],
                        )
                        update_stats(
                            word_group["text_confidence"], row["text_confidence"]
                        )
                        if should_save_raw_row(row, args.save_raw_level):
                            word_writer.write(row)
    finally:
        word_writer.close()
        window_writer.close()
        window_token_writer.close()

    return (
        [] if only_window else finalize_injected_metrics(word_groups, evaluation_name)
    ), finalize_injected_window_metrics(
        window_groups, f"{output_prefix}_window_token_detection"
    )


def run_injected_evaluation(args, models, injected_samples):
    return run_mixed_evaluation(
        args,
        models,
        injected_samples,
        "injected",
        "injected_word_detection",
    )


def run_phrase_evaluation(args, models, phrase_samples):
    return run_mixed_evaluation(
        args,
        models,
        phrase_samples,
        "phrase",
        "phrase_word_detection",
    )


def classification_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def write_metrics(
    output_dir,
    pure_metrics,
    injected_metrics,
    pure_window_metrics,
    injected_window_metrics,
    phrase_metrics,
    phrase_window_metrics,
):
    summary = {}
    if pure_metrics:
        write_csv(output_dir / "pure_foreign_detection_metrics.csv", pure_metrics)
        summary["pure"] = {
            "path": "pure_foreign_detection_metrics.csv",
            "rows": len(pure_metrics),
        }
    if injected_metrics:
        write_csv(output_dir / "injected_detection_metrics.csv", injected_metrics)
        summary["injected"] = {
            "path": "injected_detection_metrics.csv",
            "rows": len(injected_metrics),
        }
    if phrase_metrics:
        write_csv(output_dir / "phrase_detection_metrics.csv", phrase_metrics)
        summary["phrase"] = {
            "path": "phrase_detection_metrics.csv",
            "rows": len(phrase_metrics),
        }
    if pure_window_metrics:
        write_csv(output_dir / "pure_window_detection_metrics.csv", pure_window_metrics)
        summary["pure_window"] = {
            "path": "pure_window_detection_metrics.csv",
            "rows": len(pure_window_metrics),
        }
    if injected_window_metrics:
        write_csv(
            output_dir / "injected_window_detection_metrics.csv",
            injected_window_metrics,
        )
        summary["injected_window"] = {
            "path": "injected_window_detection_metrics.csv",
            "rows": len(injected_window_metrics),
        }
    if phrase_window_metrics:
        write_csv(
            output_dir / "phrase_window_detection_metrics.csv",
            phrase_window_metrics,
        )
        summary["phrase_window"] = {
            "path": "phrase_window_detection_metrics.csv",
            "rows": len(phrase_window_metrics),
        }

    write_json(output_dir / "metrics_summary.json", summary)


def build_supported_lang_map(models):
    lingua_langs = lingua_supported_langs()
    supported = {}
    for model in models:
        if model.family in {"spanish_binary", "lingua_binary"}:
            supported[model.name] = {SPANISH, NOT_SPANISH}
        elif model.family == "lingua":
            supported[model.name] = lingua_langs
        else:
            supported[model.name] = None
    return supported


def is_supported_target(supported_langs, model_name, lang):
    supported = supported_langs[model_name]
    if supported is None:
        return True
    return lang in supported


def write_run_metadata(
    args,
    models,
    output_dir,
    prepared_profile_dir,
    prepared_summary,
):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "window_decision_modes": parse_window_decision_modes(
            args.window_decision_modes
        ),
        "window_foreign_thresholds": parse_window_foreign_thresholds(
            args.window_foreign_threshold
        ),
        "window_contextual_thresholds": parse_window_contextual_thresholds(
            args.window_contextual_threshold
        ),
        "window_shared_foreign_thresholds": parse_window_shared_foreign_thresholds(
            args.window_shared_foreign_threshold
        ),
        "models": [{"name": model.name, "family": model.family} for model in models],
        "prepared_data": prepared_summary,
        "prepared_data_dir": str(prepared_profile_dir),
    }
    write_json(output_dir / "run_metadata.json", metadata)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate language ID models using a prepared dataset profile created "
            "by src/prepare_data.py."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/flores_foreign_words_run_1",
    )
    parser.add_argument(
        "--prepared-data-dir",
        required=True,
        help="Prepared dataset profile directory containing manifest and sample JSONLs.",
    )
    parser.add_argument("--binary-model-root", default="models/spanish_binary_runs")
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated model names to run. Defaults to the baseline "
            "Spanish binary model plus FastText, GlotLID, and Lingua."
        ),
    )
    parser.add_argument("--skip-pure", action="store_true")
    parser.add_argument("--skip-injected", action="store_true")
    parser.add_argument("--skip-phrase-swaps", action="store_true")
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument(
        "--only-window",
        action="store_true",
        help=(
            "Run only window-based prediction outputs and metrics. "
            "Internal text predictions still run where window scoring depends on them."
        ),
    )
    parser.add_argument(
        "--save-raw-level",
        choices=("none", "errors_only", "token_full", "all_raw"),
        default="none",
        help=(
            "How much raw prediction data to persist. "
            "'none' writes no text/word prediction JSONL files, "
            "'errors_only' keeps only misclassified non-window rows, "
            "'token_full' keeps all text/word rows, and 'all_raw' keeps everything."
        ),
    )
    parser.add_argument(
        "--save-window-raw",
        choices=("none", "token_scores", "all"),
        default="none",
        help=(
            "How much window-level raw output to persist. "
            "'none' writes no window raw JSONL files. "
            "'token_scores' keeps only aggregated per-token window scores."
        ),
    )
    parser.add_argument(
        "--window-sizes",
        default="2,3",
        help="Comma-separated sliding context window sizes.",
    )
    parser.add_argument(
        "--window-foreign-threshold",
        default="0.3,0.5,0.7",
        help=(
            "Comma-separated foreign probability thresholds for window token decisions."
        ),
    )
    parser.add_argument(
        "--window-decision-modes",
        default="legacy_window,contextual_hybrid",
        help="Comma-separated window token decision rules to run side by side.",
    )
    parser.add_argument(
        "--window-contextual-threshold",
        default="0.2",
        help="Comma-separated contextual outlier thresholds for contextual window mode.",
    )
    parser.add_argument(
        "--window-shared-foreign-threshold",
        default="0.15",
        help="Comma-separated shared foreign-margin thresholds for consensus fallback.",
    )
    parser.add_argument(
        "--window-shared-foreign-min-window-count",
        type=int,
        default=1,
        help="Minimum shared foreign window count required before consensus can fire.",
    )
    parser.add_argument(
        "--window-shared-foreign-min-ratio",
        type=float,
        default=0.5,
        help="Minimum shared foreign window ratio required before consensus can fire.",
    )
    parser.add_argument(
        "--window-top-k",
        type=int,
        default=10,
        help="Number of labels to request for window scoring.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--hf-token-env", default="HUGGING_FACE_TOKEN")
    parser.add_argument(
        "--facebook-fasttext-repo",
        default="facebook/fasttext-language-identification",
    )
    parser.add_argument("--facebook-fasttext-filename", default="model.bin")
    parser.add_argument("--glotlid-repo", default="cis-lmu/glotlid")
    parser.add_argument("--glotlid-filename", default="model.bin")
    return parser.parse_args()


def validate_args(args):
    parse_window_decision_modes(args.window_decision_modes)
    parse_window_foreign_thresholds(args.window_foreign_threshold)
    parse_window_contextual_thresholds(args.window_contextual_threshold)
    parse_window_shared_foreign_thresholds(args.window_shared_foreign_threshold)
    if args.window_top_k < 1:
        raise ValueError("--window-top-k must be at least 1.")
    if args.window_shared_foreign_min_window_count < 1:
        raise ValueError("--window-shared-foreign-min-window-count must be at least 1.")
    if (
        args.window_shared_foreign_min_ratio <= 0.0
        or args.window_shared_foreign_min_ratio > 1.0
    ):
        raise ValueError(
            "--window-shared-foreign-min-ratio must be in the range (0, 1]."
        )
    window_sizes = parse_int_csv(args.window_sizes)
    if not window_sizes:
        raise ValueError("--window-sizes must include at least one integer.")
    if any(window_size < 2 for window_size in window_sizes):
        raise ValueError("--window-sizes values must be at least 2.")
    if args.only_window and args.skip_window:
        raise ValueError("--only-window cannot be combined with --skip-window.")
    if args.skip_pure and args.skip_injected and args.skip_phrase_swaps:
        raise ValueError("At least one evaluation dataset must run.")


def main():
    args = parse_args()
    validate_args(args)

    start_time = time.monotonic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(args.env_file)
    token = os.getenv(args.hf_token_env)
    if token:
        log(f"Using Hugging Face token from ${args.hf_token_env}.")
    else:
        log(f"No Hugging Face token found in ${args.hf_token_env}.")

    prepared_manifest, prepared_profile_dir, prepared_datasets = (
        load_prepared_datasets_from_dir(
            args.prepared_data_dir,
            include_pure=not args.skip_pure,
            include_injected=not args.skip_injected,
            include_phrase=not args.skip_phrase_swaps,
        )
    )
    prepared_summary = prepared_manifest_summary(prepared_manifest)

    models = load_models(args, prepared_manifest)
    supported_langs = build_supported_lang_map(models)
    pure_samples = prepared_datasets.get("pure", [])
    injected_samples = prepared_datasets.get("injected", [])
    phrase_samples = prepared_datasets.get("phrase", [])

    pure_metrics = []
    pure_window_metrics = []
    injected_metrics = []
    injected_window_metrics = []
    phrase_metrics = []
    phrase_window_metrics = []
    write_run_metadata(
        args,
        models,
        output_dir,
        prepared_profile_dir,
        prepared_summary,
    )

    if not args.skip_pure:
        (
            pure_metrics,
            pure_window_metrics,
        ) = run_pure_evaluation(args, models, pure_samples, supported_langs)

    if not args.skip_injected:
        (
            injected_metrics,
            injected_window_metrics,
        ) = run_injected_evaluation(args, models, injected_samples)

    if not args.skip_phrase_swaps:
        (
            phrase_metrics,
            phrase_window_metrics,
        ) = run_phrase_evaluation(args, models, phrase_samples)

    write_metrics(
        output_dir,
        pure_metrics,
        injected_metrics,
        pure_window_metrics,
        injected_window_metrics,
        phrase_metrics,
        phrase_window_metrics,
    )
    log(f"Completed evaluation in {time.monotonic() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
