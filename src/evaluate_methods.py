import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fasttext
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lingua import Language, LanguageDetectorBuilder


FASTTEXT_LABEL_PREFIX = "__label__"
SPANISH = "es"
NOT_SPANISH = "not_es"

DEFAULT_FLORES_DATASET = "openlanguagedata/flores_plus"
DEFAULT_SPLIT = "devtest"
DEFAULT_SPANISH_CONFIG = "spa_Latn"
DEFAULT_INJECTION_CONFIGS = (
    "eng_Latn,por_Latn,ita_Latn,fra_Latn,deu_Latn,cat_Latn,eus_Latn"
)
DEFAULT_PURE_CONFIGS = f"{DEFAULT_SPANISH_CONFIG},{DEFAULT_INJECTION_CONFIGS}"
DEFAULT_MODELS = (
    "spanish-binary-baseline,"
    "facebook-fasttext-language-identification,"
    "glotlid,"
    "lingua"
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


def dataset_kwargs(token):
    return {"token": token} if token else {}


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


def config_to_lang(config):
    return normalize_lang_code(config)


def text_from_row(row):
    if "text" in row and row["text"] is not None:
        return str(row["text"])
    if "sentence" in row and row["sentence"] is not None:
        return str(row["sentence"])
    if "translation" in row and isinstance(row["translation"], dict):
        values = [value for value in row["translation"].values() if value]
        if values:
            return str(values[0])
    raise KeyError("Unable to find a text field in FLORES row.")


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


def build_lingua_detector():
    if hasattr(LanguageDetectorBuilder, "from_all_languages"):
        return LanguageDetectorBuilder.from_all_languages().build()
    return LanguageDetectorBuilder.from_languages(*Language.all()).build()


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
            (get_lingua_code(value.language), get_lingua_code(value.language), float(value.value))
            for value in confidence_values
        ]
        return build_window_score(self, main_lang, label_scores[:top_k])


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

    foreign_score = max(0.0, min(1.0, 1.0 - main_lang_score))
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


def load_models(args):
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

    if not selected or "lingua" in selected:
        log("Building Lingua detector")
        models.append(LinguaAdapter("lingua", "lingua", build_lingua_detector()))

    if selected:
        loaded = {model.name for model in models}
        missing = selected - loaded
        if missing:
            raise ValueError(f"Requested model(s) not loaded: {', '.join(sorted(missing))}")

    if not models:
        raise RuntimeError("No models were loaded.")
    return models


def load_flores_samples(dataset_name, config, split, token, limit=None):
    log(f"Loading FLORES {config}/{split}")
    dataset = load_dataset(
        dataset_name, config, split=split, **dataset_kwargs(token)
    )
    lang = config_to_lang(config)
    samples = []
    for index, row in enumerate(dataset):
        samples.append(
            Sample(
                sample_id=f"{config}:{split}:{index}",
                row_index=index,
                flores_config=config,
                lang=lang,
                text=text_from_row(row),
            )
        )
        if limit and len(samples) >= limit:
            break
    return samples


def resolve_pure_configs(args, token):
    if args.flores_configs:
        configs = parse_csv(args.flores_configs)
    else:
        configs = parse_csv(DEFAULT_PURE_CONFIGS)

    if args.limit_languages:
        wanted = set(parse_csv(args.limit_languages))
        configs = [
            config
            for config in configs
            if config in wanted or config_to_lang(config) in wanted
        ]
    return sorted(configs)


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
    if model.family == "spanish_binary":
        expected = SPANISH if true_lang == SPANISH else NOT_SPANISH
        return predicted_lang == expected
    return predicted_lang == true_lang


def is_foreign_detection(model, text_lang, word_prediction):
    if model.family == "spanish_binary":
        return word_prediction.predicted_lang == NOT_SPANISH
    return (
        text_lang != "unknown"
        and word_prediction.predicted_lang != "unknown"
        and word_prediction.predicted_lang != text_lang
    )


def parse_int_csv(value):
    return [int(item) for item in parse_csv(value)]


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
    threshold,
    top_k,
    base_extra,
    token_truth,
    token_injections=None,
):
    token_injections = token_injections or {}
    window_rows = []
    token_score_inputs = defaultdict(list)
    tokens_by_raw_index = {token.raw_index: token for token in tokens}

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
            window_rows.append(
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
                token_score_inputs[(window_size, token.raw_index)].append(score)

    token_rows = []
    for (window_size, token_index), scores in sorted(token_score_inputs.items()):
        token_item = tokens_by_raw_index[token_index]
        foreign_probability = sum(score.foreign_score for score in scores) / len(scores)
        main_lang_probability = (
            sum(score.main_lang_score for score in scores) / len(scores)
        )
        predicted = foreign_probability >= threshold
        injection = token_injections.get(token_index, {})
        truth = token_truth(token_index)
        token_rows.append(
            {
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
                "window_count": len(scores),
                "window_foreign_threshold": threshold,
                "is_foreign_ground_truth": truth,
                "is_foreign_predicted": predicted,
                "correct": truth == predicted,
                "original_normalized": injection.get("original_normalized"),
                "replacement_normalized": injection.get("replacement_normalized"),
                "span_id": injection.get("span_id"),
                "span_offset": injection.get("span_offset"),
                "span_length": injection.get("span_length"),
            }
        )

    return window_rows, token_rows


def run_pure_evaluation(args, models, token, supported_langs):
    output_dir = Path(args.output_dir)
    text_path = output_dir / "pure_text_predictions.jsonl"
    word_path = output_dir / "pure_word_predictions.jsonl"
    pure_window_path = output_dir / "pure_window_predictions.jsonl"
    pure_window_token_path = output_dir / "pure_window_token_scores.jsonl"

    configs = resolve_pure_configs(args, token)
    if not configs:
        raise RuntimeError("No FLORES configs selected for pure evaluation.")

    text_rows = []
    word_rows = []
    window_rows = []
    window_token_rows = []
    window_sizes = parse_int_csv(args.window_sizes)

    for config in configs:
        samples = load_flores_samples(
            args.flores_dataset,
            config,
            args.split,
            token,
            args.limit_samples_per_language,
        )
        for sample in samples:
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
                text_rows.append(
                    prediction_record(
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
                )

                if not args.skip_window:
                    model_window_rows, model_window_token_rows = (
                        build_window_rows_and_token_scores(
                            sample=sample,
                            tokens=tokens,
                            model=model,
                            main_lang=text_prediction.predicted_lang,
                            window_sizes=window_sizes,
                            threshold=args.window_foreign_threshold,
                            top_k=args.window_top_k,
                            base_extra={
                                **base_extra,
                                "input_level": "window",
                            },
                            token_truth=lambda _token_index: False,
                        )
                    )
                    window_rows.extend(model_window_rows)
                    window_token_rows.extend(model_window_token_rows)

                for token_item in tokens:
                    word_prediction = model.predict(token_item.normalized)
                    is_foreign = is_foreign_detection(
                        model, text_prediction.predicted_lang, word_prediction
                    )
                    word_rows.append(
                        prediction_record(
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
                                    model, sample.lang, word_prediction.predicted_lang
                                ),
                                "is_supported_target": is_supported_target(
                                    supported_langs, model.name, sample.lang
                                ),
                            },
                        )
                    )

    write_jsonl(text_path, text_rows)
    write_jsonl(word_path, word_rows)
    if not args.skip_window:
        write_jsonl(pure_window_path, window_rows)
        write_jsonl(pure_window_token_path, window_token_rows)
    return text_rows, word_rows, window_rows, window_token_rows


def nearest_token_by_relative_position(target_token, source_tokens, target_count):
    if not source_tokens or target_count <= 1:
        return source_tokens[0] if source_tokens else None
    relative = target_token.raw_index / max(target_count - 1, 1)
    source_index = round(relative * (len(source_tokens) - 1))
    source_index = max(0, min(len(source_tokens) - 1, source_index))
    return source_tokens[source_index]


def raw_text_parts(text):
    return [match.group(0) for match in WHITESPACE_TOKEN_RE.finditer(text)]


def build_injected_sample(spanish_sample, foreign_sample, injection_lang, ratio, rng):
    spanish_tokens = tokenize(spanish_sample.text)
    foreign_tokens = tokenize(foreign_sample.text)
    raw_parts = raw_text_parts(spanish_sample.text)
    if not spanish_tokens or not foreign_tokens or not raw_parts:
        return None

    injection_count = max(1, round(len(spanish_tokens) * ratio))
    candidate_tokens = spanish_tokens[:]
    rng.shuffle(candidate_tokens)

    replacements = []
    used_indexes = set()
    for spanish_token in candidate_tokens:
        if len(replacements) >= injection_count:
            break
        if spanish_token.raw_index in used_indexes:
            continue

        foreign_token = nearest_token_by_relative_position(
            spanish_token, foreign_tokens, len(raw_parts)
        )
        if foreign_token is None:
            continue
        if foreign_token.normalized.casefold() == spanish_token.normalized.casefold():
            continue
        if not is_eligible_word(foreign_token.normalized):
            continue

        replacement = (
            spanish_token.leading + foreign_token.normalized + spanish_token.trailing
        )
        raw_parts[spanish_token.raw_index] = replacement
        used_indexes.add(spanish_token.raw_index)
        replacements.append(
            {
                "token_index": spanish_token.raw_index,
                "original_token": spanish_token.raw,
                "original_normalized": spanish_token.normalized,
                "replacement_token": replacement,
                "replacement_normalized": foreign_token.normalized,
                "injected_lang": injection_lang,
                "foreign_source_index": foreign_token.raw_index,
            }
        )

    if not replacements:
        return None

    sample_id = f"{spanish_sample.sample_id}:inject:{injection_lang}"
    return {
        "sample_id": sample_id,
        "source_sample_id": spanish_sample.sample_id,
        "row_index": spanish_sample.row_index,
        "base_lang": SPANISH,
        "injected_lang": injection_lang,
        "contamination_type": "position_token",
        "injection_ratio": ratio,
        "requested_injections": injection_count,
        "actual_injections": len(replacements),
        "text": " ".join(raw_parts),
        "original_text": spanish_sample.text,
        "foreign_text": foreign_sample.text,
        "injections": replacements,
    }


def build_injected_samples(args, token):
    spanish_samples = load_flores_samples(
        args.flores_dataset,
        args.spanish_config,
        args.split,
        token,
        args.limit_samples_per_language,
    )
    spanish_by_row = {sample.row_index: sample for sample in spanish_samples}
    rng = random.Random(args.seed)
    injected = []

    for config in parse_csv(args.injection_configs):
        foreign_samples = load_flores_samples(
            args.flores_dataset,
            config,
            args.split,
            token,
            args.limit_samples_per_language,
        )
        injection_lang = config_to_lang(config)
        for foreign_sample in foreign_samples:
            spanish_sample = spanish_by_row.get(foreign_sample.row_index)
            if spanish_sample is None:
                continue
            injected_sample = build_injected_sample(
                spanish_sample,
                foreign_sample,
                injection_lang,
                args.injection_ratio,
                rng,
            )
            if injected_sample is not None:
                injected.append(injected_sample)

    return injected


def choose_relative_span(source_start, source_count, target_count, span_length):
    if target_count < span_length:
        return None
    if source_count <= 1:
        target_start = 0
    else:
        relative = source_start / max(source_count - span_length, 1)
        target_start = round(relative * max(target_count - span_length, 0))
    target_start = max(0, min(target_count - span_length, target_start))
    return target_start


def contiguous_token_span(tokens, start, span_length):
    span = tokens[start : start + span_length]
    if len(span) != span_length:
        return None
    expected_indexes = list(range(span[0].raw_index, span[0].raw_index + span_length))
    if [token.raw_index for token in span] != expected_indexes:
        return None
    return span


def build_phrase_sample(spanish_sample, foreign_sample, injection_lang, args, rng):
    spanish_tokens = tokenize(spanish_sample.text)
    foreign_tokens = tokenize(foreign_sample.text)
    raw_parts = raw_text_parts(spanish_sample.text)
    if not spanish_tokens or not foreign_tokens or not raw_parts:
        return None

    target_replacements = max(1, round(len(spanish_tokens) * args.phrase_replacement_ratio))
    candidate_starts = list(range(len(spanish_tokens)))
    rng.shuffle(candidate_starts)

    replacements = []
    used_indexes = set()
    replaced_count = 0
    for start in candidate_starts:
        if replaced_count >= target_replacements:
            break
        span_lengths = list(range(args.phrase_span_min, args.phrase_span_max + 1))
        rng.shuffle(span_lengths)

        for span_length in span_lengths:
            if replaced_count + span_length > target_replacements and replacements:
                continue
            spanish_span = contiguous_token_span(spanish_tokens, start, span_length)
            if spanish_span is None:
                continue
            span_indexes = {token.raw_index for token in spanish_span}
            if span_indexes & used_indexes:
                continue

            foreign_start = choose_relative_span(
                start,
                len(spanish_tokens),
                len(foreign_tokens),
                span_length,
            )
            if foreign_start is None:
                continue
            foreign_span = foreign_tokens[foreign_start : foreign_start + span_length]
            if len(foreign_span) != span_length:
                continue
            if any(not is_eligible_word(token.normalized) for token in foreign_span):
                continue

            same_count = sum(
                1
                for spanish_token, foreign_token in zip(spanish_span, foreign_span)
                if spanish_token.normalized.casefold()
                == foreign_token.normalized.casefold()
            )
            if same_count == span_length:
                continue

            span_id = len(replacements)
            for offset, (spanish_token, foreign_token) in enumerate(
                zip(spanish_span, foreign_span)
            ):
                replacement = (
                    spanish_token.leading
                    + foreign_token.normalized
                    + spanish_token.trailing
                )
                raw_parts[spanish_token.raw_index] = replacement
                used_indexes.add(spanish_token.raw_index)
                replacements.append(
                    {
                        "span_id": span_id,
                        "span_offset": offset,
                        "span_length": span_length,
                        "token_index": spanish_token.raw_index,
                        "original_token": spanish_token.raw,
                        "original_normalized": spanish_token.normalized,
                        "replacement_token": replacement,
                        "replacement_normalized": foreign_token.normalized,
                        "injected_lang": injection_lang,
                        "foreign_source_index": foreign_token.raw_index,
                    }
                )
            replaced_count += span_length
            break

    if not replacements:
        return None

    sample_id = f"{spanish_sample.sample_id}:phrase:{injection_lang}"
    return {
        "sample_id": sample_id,
        "source_sample_id": spanish_sample.sample_id,
        "row_index": spanish_sample.row_index,
        "base_lang": SPANISH,
        "injected_lang": injection_lang,
        "contamination_type": "phrase_span",
        "injection_ratio": args.phrase_replacement_ratio,
        "phrase_span_min": args.phrase_span_min,
        "phrase_span_max": args.phrase_span_max,
        "requested_injections": target_replacements,
        "actual_injections": len(replacements),
        "text": " ".join(raw_parts),
        "original_text": spanish_sample.text,
        "foreign_text": foreign_sample.text,
        "injections": replacements,
    }


def build_phrase_samples(args, token):
    spanish_samples = load_flores_samples(
        args.flores_dataset,
        args.spanish_config,
        args.split,
        token,
        args.limit_samples_per_language,
    )
    spanish_by_row = {sample.row_index: sample for sample in spanish_samples}
    rng = random.Random(args.seed)
    phrase_samples = []

    for config in parse_csv(args.injection_configs):
        foreign_samples = load_flores_samples(
            args.flores_dataset,
            config,
            args.split,
            token,
            args.limit_samples_per_language,
        )
        injection_lang = config_to_lang(config)
        for foreign_sample in foreign_samples:
            spanish_sample = spanish_by_row.get(foreign_sample.row_index)
            if spanish_sample is None:
                continue
            phrase_sample = build_phrase_sample(
                spanish_sample,
                foreign_sample,
                injection_lang,
                args,
                rng,
            )
            if phrase_sample is not None:
                phrase_samples.append(phrase_sample)

    return phrase_samples


def run_mixed_evaluation(args, models, sample_filename, output_prefix):
    output_dir = Path(args.output_dir)
    with (output_dir / sample_filename).open("r", encoding="utf-8") as f:
        mixed_samples = [json.loads(line) for line in f]

    rows = []
    window_rows = []
    window_token_rows = []
    window_sizes = parse_int_csv(args.window_sizes)
    for sample in mixed_samples:
        injected_indexes = {
            injection["token_index"]: injection for injection in sample["injections"]
        }
        tokens = tokenize(sample["text"])
        for model in models:
            text_prediction = model.predict(sample["text"])
            base_extra = {
                "sample_id": sample["sample_id"],
                "source_sample_id": sample["source_sample_id"],
                "row_index": sample["row_index"],
                "base_lang": SPANISH,
                "injected_lang": sample["injected_lang"],
                "contamination_type": sample.get("contamination_type", output_prefix),
                "text_predicted_lang": text_prediction.predicted_lang,
                "text_predicted_label": text_prediction.predicted_label,
                "text_confidence": text_prediction.confidence,
            }
            if not args.skip_window:
                model_window_rows, model_window_token_rows = (
                    build_window_rows_and_token_scores(
                        sample=sample,
                        tokens=tokens,
                        model=model,
                        main_lang=SPANISH,
                        window_sizes=window_sizes,
                        threshold=args.window_foreign_threshold,
                        top_k=args.window_top_k,
                        base_extra={
                            **base_extra,
                            "input_level": "window",
                        },
                        token_truth=lambda token_index, indexes=injected_indexes: (
                            token_index in indexes
                        ),
                        token_injections=injected_indexes,
                    )
                )
                window_rows.extend(model_window_rows)
                window_token_rows.extend(model_window_token_rows)

            for token_item in tokens:
                word_prediction = model.predict(token_item.normalized)
                truth = token_item.raw_index in injected_indexes
                predicted = is_foreign_detection(model, SPANISH, word_prediction)
                injection = injected_indexes.get(token_item.raw_index, {})
                rows.append(
                    prediction_record(
                        word_prediction,
                        {
                            **base_extra,
                            "input_level": "word",
                            "token_index": token_item.raw_index,
                            "token": token_item.raw,
                            "normalized_token": token_item.normalized,
                            "is_foreign_ground_truth": truth,
                            "is_foreign_predicted": predicted,
                            "original_normalized": injection.get("original_normalized"),
                            "replacement_normalized": injection.get(
                                "replacement_normalized"
                            ),
                            "span_id": injection.get("span_id"),
                            "span_offset": injection.get("span_offset"),
                            "span_length": injection.get("span_length"),
                            "correct": truth == predicted,
                        },
                    )
                )

    write_jsonl(output_dir / f"{output_prefix}_word_predictions.jsonl", rows)
    if not args.skip_window:
        write_jsonl(output_dir / f"{output_prefix}_window_predictions.jsonl", window_rows)
        write_jsonl(
            output_dir / f"{output_prefix}_window_token_scores.jsonl", window_token_rows
        )
    return rows, window_rows, window_token_rows


def run_injected_evaluation(args, models):
    return run_mixed_evaluation(
        args,
        models,
        "injected_samples.jsonl",
        "injected",
    )


def run_phrase_evaluation(args, models):
    return run_mixed_evaluation(
        args,
        models,
        "phrase_samples.jsonl",
        "phrase",
    )


def summarize_confidences(rows, key="confidence"):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


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


def make_pure_metrics(text_rows, word_rows):
    output = []

    for level, rows in (("text", text_rows), ("word", word_rows)):
        groups = defaultdict(list)
        for row in rows:
            groups[(row["model"], row["model_family"], row["true_lang"])].append(row)
        for (model, family, true_lang), group_rows in groups.items():
            correct = sum(1 for row in group_rows if row["correct"])
            confidence = summarize_confidences(group_rows)
            metric = {
                "evaluation": f"pure_{level}",
                "model": model,
                "model_family": family,
                "true_lang": true_lang,
                "total": len(group_rows),
                "accuracy": correct / len(group_rows) if group_rows else 0.0,
                "confidence_mean": confidence["mean"],
                "confidence_min": confidence["min"],
                "confidence_max": confidence["max"],
            }
            if level == "word":
                foreign_predicted = sum(
                    1 for row in group_rows if row["is_foreign_predicted"]
                )
                metric["foreign_false_positive_rate"] = (
                    foreign_predicted / len(group_rows) if group_rows else 0.0
                )
            output.append(metric)

    return output


def make_injected_metrics(rows, evaluation_name="injected_word_detection"):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["model_family"], row["injected_lang"])].append(row)

    metrics = []
    for (model, family, injected_lang), group_rows in groups.items():
        tp = sum(
            1
            for row in group_rows
            if row["is_foreign_ground_truth"] and row["is_foreign_predicted"]
        )
        fp = sum(
            1
            for row in group_rows
            if not row["is_foreign_ground_truth"] and row["is_foreign_predicted"]
        )
        tn = sum(
            1
            for row in group_rows
            if not row["is_foreign_ground_truth"] and not row["is_foreign_predicted"]
        )
        fn = sum(
            1
            for row in group_rows
            if row["is_foreign_ground_truth"] and not row["is_foreign_predicted"]
        )
        metric = classification_metrics(tp, fp, tn, fn)
        metric.update(
            {
                "evaluation": evaluation_name,
                "model": model,
                "model_family": family,
                "injected_lang": injected_lang,
            }
        )

        for outcome, predicate in (
            (
                "tp",
                lambda row: row["is_foreign_ground_truth"]
                and row["is_foreign_predicted"],
            ),
            (
                "fp",
                lambda row: not row["is_foreign_ground_truth"]
                and row["is_foreign_predicted"],
            ),
            (
                "tn",
                lambda row: not row["is_foreign_ground_truth"]
                and not row["is_foreign_predicted"],
            ),
            (
                "fn",
                lambda row: row["is_foreign_ground_truth"]
                and not row["is_foreign_predicted"],
            ),
        ):
            confidence = summarize_confidences(
                [row for row in group_rows if predicate(row)]
            )
            metric[f"{outcome}_confidence_mean"] = confidence["mean"]

        text_confidence = summarize_confidences(group_rows, key="text_confidence")
        metric["text_confidence_mean"] = text_confidence["mean"]
        metrics.append(metric)

    return metrics


def make_pure_window_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["model"],
                row["model_family"],
                row["true_lang"],
                row["window_size"],
            )
        ].append(row)

    metrics = []
    for (model, family, true_lang, window_size), group_rows in groups.items():
        foreign_predicted = sum(1 for row in group_rows if row["is_foreign_predicted"])
        foreign_probability = summarize_confidences(
            group_rows, key="foreign_probability"
        )
        main_lang_probability = summarize_confidences(
            group_rows, key="main_lang_probability"
        )
        metrics.append(
            {
                "evaluation": "pure_window_token_detection",
                "model": model,
                "model_family": family,
                "true_lang": true_lang,
                "window_size": window_size,
                "total": len(group_rows),
                "foreign_false_positive_rate": (
                    foreign_predicted / len(group_rows) if group_rows else 0.0
                ),
                "foreign_probability_mean": foreign_probability["mean"],
                "foreign_probability_min": foreign_probability["min"],
                "foreign_probability_max": foreign_probability["max"],
                "main_lang_probability_mean": main_lang_probability["mean"],
            }
        )

    return metrics


def make_injected_window_metrics(rows, evaluation_name="injected_window_token_detection"):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["model"],
                row["model_family"],
                row["injected_lang"],
                row["window_size"],
            )
        ].append(row)

    metrics = []
    for (model, family, injected_lang, window_size), group_rows in groups.items():
        tp = sum(
            1
            for row in group_rows
            if row["is_foreign_ground_truth"] and row["is_foreign_predicted"]
        )
        fp = sum(
            1
            for row in group_rows
            if not row["is_foreign_ground_truth"] and row["is_foreign_predicted"]
        )
        tn = sum(
            1
            for row in group_rows
            if not row["is_foreign_ground_truth"] and not row["is_foreign_predicted"]
        )
        fn = sum(
            1
            for row in group_rows
            if row["is_foreign_ground_truth"] and not row["is_foreign_predicted"]
        )
        metric = classification_metrics(tp, fp, tn, fn)
        metric.update(
            {
                "evaluation": evaluation_name,
                "model": model,
                "model_family": family,
                "injected_lang": injected_lang,
                "window_size": window_size,
            }
        )

        for outcome, predicate in (
            (
                "tp",
                lambda row: row["is_foreign_ground_truth"]
                and row["is_foreign_predicted"],
            ),
            (
                "fp",
                lambda row: not row["is_foreign_ground_truth"]
                and row["is_foreign_predicted"],
            ),
            (
                "tn",
                lambda row: not row["is_foreign_ground_truth"]
                and not row["is_foreign_predicted"],
            ),
            (
                "fn",
                lambda row: row["is_foreign_ground_truth"]
                and not row["is_foreign_predicted"],
            ),
        ):
            confidence = summarize_confidences(
                [row for row in group_rows if predicate(row)],
                key="foreign_probability",
            )
            metric[f"{outcome}_foreign_probability_mean"] = confidence["mean"]

        main_lang_probability = summarize_confidences(
            group_rows, key="main_lang_probability"
        )
        metric["main_lang_probability_mean"] = main_lang_probability["mean"]
        metrics.append(metric)

    return metrics


def write_metrics(
    output_dir,
    pure_text_rows,
    pure_word_rows,
    injected_rows,
    pure_window_token_rows,
    injected_window_token_rows,
    phrase_rows,
    phrase_window_token_rows,
):
    pure_metrics = make_pure_metrics(pure_text_rows, pure_word_rows)
    injected_metrics = make_injected_metrics(injected_rows)
    pure_window_metrics = make_pure_window_metrics(pure_window_token_rows)
    injected_window_metrics = make_injected_window_metrics(injected_window_token_rows)
    phrase_metrics = make_injected_metrics(phrase_rows, "phrase_word_detection")
    phrase_window_metrics = make_injected_window_metrics(
        phrase_window_token_rows,
        "phrase_window_token_detection",
    )

    write_csv(output_dir / "pure_foreign_detection_metrics.csv", pure_metrics)
    write_csv(output_dir / "injected_detection_metrics.csv", injected_metrics)
    write_csv(output_dir / "phrase_detection_metrics.csv", phrase_metrics)
    write_csv(output_dir / "pure_window_detection_metrics.csv", pure_window_metrics)
    write_csv(
        output_dir / "injected_window_detection_metrics.csv",
        injected_window_metrics,
    )
    write_csv(
        output_dir / "phrase_window_detection_metrics.csv",
        phrase_window_metrics,
    )
    write_json(
        output_dir / "metrics_summary.json",
        {
            "pure": pure_metrics,
            "injected": injected_metrics,
            "phrase": phrase_metrics,
            "pure_window": pure_window_metrics,
            "injected_window": injected_window_metrics,
            "phrase_window": phrase_window_metrics,
        },
    )


def build_supported_lang_map(models):
    lingua_langs = lingua_supported_langs()
    supported = {}
    for model in models:
        if model.family == "spanish_binary":
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


def write_run_metadata(args, models, output_dir, pure_configs):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "models": [
            {"name": model.name, "family": model.family}
            for model in models
        ],
        "pure_configs": pure_configs,
        "language_mappings": ISO_639_3_TO_1,
    }
    write_json(output_dir / "run_metadata.json", metadata)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate language ID models for FLORES foreign-word detection."
    )
    parser.add_argument("--output-dir", default="evaluation_results/flores_foreign_words")
    parser.add_argument("--binary-model-root", default="models/spanish_binary_runs")
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated model names to run. Defaults to the baseline "
            "Spanish binary model plus FastText, GlotLID, and Lingua."
        ),
    )
    parser.add_argument("--flores-dataset", default=DEFAULT_FLORES_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--flores-configs",
        default=None,
        help=(
            "Comma-separated FLORES configs for pure-language evaluation. "
            "Defaults to Spanish plus the injection target languages."
        ),
    )
    parser.add_argument("--limit-languages", default=None)
    parser.add_argument("--limit-samples-per-language", type=int, default=None)
    parser.add_argument("--spanish-config", default=DEFAULT_SPANISH_CONFIG)
    parser.add_argument("--injection-configs", default=DEFAULT_INJECTION_CONFIGS)
    parser.add_argument("--injection-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pure", action="store_true")
    parser.add_argument("--skip-injected", action="store_true")
    parser.add_argument("--skip-phrase-swaps", action="store_true")
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument(
        "--window-sizes",
        default="2,3",
        help="Comma-separated sliding context window sizes.",
    )
    parser.add_argument(
        "--window-foreign-threshold",
        type=float,
        default=0.5,
        help="Foreign probability threshold for window token decisions.",
    )
    parser.add_argument(
        "--window-top-k",
        type=int,
        default=10,
        help="Number of labels to request for window scoring.",
    )
    parser.add_argument(
        "--phrase-replacement-ratio",
        type=float,
        default=0.10,
        help="Approximate share of eligible Spanish tokens to replace with phrase spans.",
    )
    parser.add_argument(
        "--phrase-span-min",
        type=int,
        default=2,
        help="Minimum phrase span length for phrase swaps.",
    )
    parser.add_argument(
        "--phrase-span-max",
        type=int,
        default=4,
        help="Maximum phrase span length for phrase swaps.",
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
    if args.injection_ratio <= 0.0 or args.injection_ratio > 1.0:
        raise ValueError("--injection-ratio must be in the range (0, 1].")
    if args.phrase_replacement_ratio <= 0.0 or args.phrase_replacement_ratio > 1.0:
        raise ValueError("--phrase-replacement-ratio must be in the range (0, 1].")
    if args.phrase_span_min < 1:
        raise ValueError("--phrase-span-min must be at least 1.")
    if args.phrase_span_max < args.phrase_span_min:
        raise ValueError("--phrase-span-max must be >= --phrase-span-min.")
    if args.window_foreign_threshold < 0.0 or args.window_foreign_threshold > 1.0:
        raise ValueError("--window-foreign-threshold must be in the range [0, 1].")
    if args.window_top_k < 1:
        raise ValueError("--window-top-k must be at least 1.")
    window_sizes = parse_int_csv(args.window_sizes)
    if not window_sizes:
        raise ValueError("--window-sizes must include at least one integer.")
    if any(window_size < 2 for window_size in window_sizes):
        raise ValueError("--window-sizes values must be at least 2.")
    if args.limit_samples_per_language is not None and args.limit_samples_per_language < 1:
        raise ValueError("--limit-samples-per-language must be at least 1.")
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

    models = load_models(args)
    supported_langs = build_supported_lang_map(models)

    pure_text_rows = []
    pure_word_rows = []
    pure_window_rows = []
    pure_window_token_rows = []
    injected_rows = []
    injected_window_rows = []
    injected_window_token_rows = []
    phrase_rows = []
    phrase_window_rows = []
    phrase_window_token_rows = []
    pure_configs = []

    if not args.skip_pure:
        pure_configs = resolve_pure_configs(args, token)
        write_run_metadata(args, models, output_dir, pure_configs)
        (
            pure_text_rows,
            pure_word_rows,
            pure_window_rows,
            pure_window_token_rows,
        ) = run_pure_evaluation(args, models, token, supported_langs)
    else:
        write_run_metadata(args, models, output_dir, pure_configs)

    if not args.skip_injected:
        injected_samples = build_injected_samples(args, token)
        write_jsonl(output_dir / "injected_samples.jsonl", injected_samples)
        (
            injected_rows,
            injected_window_rows,
            injected_window_token_rows,
        ) = run_injected_evaluation(args, models)

    if not args.skip_phrase_swaps:
        phrase_samples = build_phrase_samples(args, token)
        write_jsonl(output_dir / "phrase_samples.jsonl", phrase_samples)
        (
            phrase_rows,
            phrase_window_rows,
            phrase_window_token_rows,
        ) = run_phrase_evaluation(args, models)

    write_metrics(
        output_dir,
        pure_text_rows,
        pure_word_rows,
        injected_rows,
        pure_window_token_rows,
        injected_window_token_rows,
        phrase_rows,
        phrase_window_token_rows,
    )
    log(f"Completed evaluation in {time.monotonic() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
