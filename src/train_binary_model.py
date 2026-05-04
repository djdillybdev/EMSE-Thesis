import argparse
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import fasttext
from datasets import load_dataset


LABEL_ES = "es"
LABEL_NOT_ES = "not_es"
FASTTEXT_LABEL_PREFIX = "__label__"

DEFAULT_WIKIPEDIA_NEGATIVE_LANGS = "en,fr,de,it,pt,nl,pl,sv"
DEFAULT_EUROPARL_PAIRS = "en-es,en-fr,de-en,en-it,en-pt,en-nl,en-pl,en-sv"
DEFAULT_FLORES_NEGATIVE_CONFIGS = (
    "eng_Latn,fra_Latn,deu_Latn,ita_Latn,por_Latn,nld_Latn,pol_Latn,swe_Latn"
)

WHITESPACE_RE = re.compile(r"\s+")
FASTTEXT_LABEL_RE = re.compile(r"__label__\S+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
URL_RE = re.compile(r"^(https?://|www\.)", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    source: str
    language: str


def log(message):
    print(message, flush=True)


def log_progress(prefix, count, limit, next_report):
    if count >= next_report:
        log(f"{prefix}: collected {count}/{limit}")
        return next_report + max(1, limit // 10)
    return next_report


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a fastText binary classifier for Spanish vs not-Spanish text."
    )
    parser.add_argument(
        "--output-model",
        default="models/spanish_binary.bin",
        help="Path for the trained fastText .bin model.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/spanish_binary_training",
        help="Directory for generated training files and metrics.",
    )
    parser.add_argument(
        "--positive-samples",
        type=int,
        default=5000,
        help="Number of Spanish training-source examples to collect.",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=5000,
        help="Number of non-Spanish training-source examples to collect.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum normalized text length to keep.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=600,
        help="Maximum normalized text length to keep.",
    )
    parser.add_argument(
        "--max-chunks-per-record",
        type=int,
        default=3,
        help="Maximum snippets to take from one source record.",
    )
    parser.add_argument(
        "--training-unit",
        choices=["sentence", "word"],
        default="sentence",
        help="Train on short text snippets or individual word tokens.",
    )
    parser.add_argument(
        "--min-word-chars",
        type=int,
        default=3,
        help="Minimum Unicode alphabetic token length for --training-unit word.",
    )
    parser.add_argument(
        "--max-tokens-per-record",
        type=int,
        default=50,
        help="Maximum word tokens to take from one source record in word mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of training-source examples used for training. The rest validate.",
    )
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also write a quantized .ftz model next to --output-model.",
    )
    parser.add_argument(
        "--dataset-token-env",
        "--hf-token-env",
        default="HUGGING_FACE_TOKEN",
        dest="dataset_token_env",
        help="Environment variable containing an optional Hugging Face token.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional .env file to load before reading --dataset-token-env.",
    )
    parser.add_argument(
        "--wikipedia-dataset",
        default="wikimedia/wikipedia",
        help="Hugging Face Wikipedia dataset id.",
    )
    parser.add_argument(
        "--wikipedia-snapshot",
        default="20231101",
        help="Wikipedia snapshot prefix used in configs like 20231101.es.",
    )
    parser.add_argument(
        "--wikipedia-positive-lang",
        default="es",
        help="Wikipedia language code for Spanish positives.",
    )
    parser.add_argument(
        "--wikipedia-negative-langs",
        default=DEFAULT_WIKIPEDIA_NEGATIVE_LANGS,
        help="Comma-separated Wikipedia language codes for non-Spanish negatives.",
    )
    parser.add_argument(
        "--europarl-dataset",
        default="Helsinki-NLP/europarl",
        help="Hugging Face Europarl dataset id.",
    )
    parser.add_argument(
        "--europarl-pairs",
        default=DEFAULT_EUROPARL_PAIRS,
        help="Comma-separated Europarl language-pair configs.",
    )
    parser.add_argument(
        "--flores-dataset",
        default="openlanguagedata/flores_plus",
        help="Hugging Face FLORES+ dataset id for held-out evaluation only.",
    )
    parser.add_argument(
        "--flores-positive-config",
        default="spa_Latn",
        help="FLORES+ config for Spanish evaluation examples.",
    )
    parser.add_argument(
        "--flores-negative-configs",
        default=DEFAULT_FLORES_NEGATIVE_CONFIGS,
        help="Comma-separated FLORES+ configs for held-out non-Spanish evaluation.",
    )
    parser.add_argument(
        "--flores-split",
        default="devtest",
        help="FLORES+ split used for held-out evaluation.",
    )
    parser.add_argument(
        "--flores-samples",
        type=int,
        default=1000,
        help="Total held-out FLORES+ examples to evaluate, split evenly by label.",
    )
    parser.add_argument(
        "--skip-flores-eval",
        action="store_true",
        help="Train and validate without running the held-out FLORES+ evaluation.",
    )
    return parser.parse_args()


def strip_env_value(value):
    value = value.strip()
    if not value:
        return value

    quote = value[0]
    if quote in {'"', "'"}:
        end_index = value.find(quote, 1)
        if end_index != -1:
            return value[1:end_index]
        return value[1:]

    comment_index = value.find(" #")
    if comment_index != -1:
        value = value[:comment_index]
    return value.strip()


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return []

    loaded_keys = []
    with open(env_path, "r", encoding="utf-8") as f:
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


def normalize_text(text, min_chars, max_chars):
    text = FASTTEXT_LABEL_RE.sub(" ", text or "")
    text = WHITESPACE_RE.sub(" ", text.replace("\t", " ").replace("\n", " ")).strip()
    if len(text) < min_chars or len(text) > max_chars:
        return None
    return text


def split_short_texts(text, min_chars, max_chars, max_chunks):
    candidates = []
    for paragraph in re.split(r"\n{2,}", text or ""):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            candidates.append(paragraph)
        else:
            candidates.extend(SENTENCE_SPLIT_RE.split(paragraph))

    normalized = []
    seen = set()
    for candidate in candidates:
        text = normalize_text(candidate, min_chars, max_chars)
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
        if len(normalized) >= max_chunks:
            break
    return normalized


def strip_surrounding_punctuation(token):
    start = 0
    end = len(token)

    while start < end and not token[start].isalpha():
        start += 1
    while end > start and not token[end - 1].isalpha():
        end -= 1

    return token[start:end]


def extract_word_tokens(text, min_word_chars, max_tokens):
    tokens = []
    seen = set()

    for raw_token in WHITESPACE_RE.split(text or ""):
        if not raw_token:
            continue
        if FASTTEXT_LABEL_RE.search(raw_token):
            continue
        if URL_RE.match(raw_token) or EMAIL_RE.match(raw_token):
            continue
        if any(char.isdigit() for char in raw_token):
            continue

        token = strip_surrounding_punctuation(raw_token).casefold()
        if len(token) < min_word_chars:
            continue
        if "_" in token:
            continue
        if not token.isalpha():
            continue
        if token in seen:
            continue

        tokens.append(token)
        seen.add(token)
        if len(tokens) >= max_tokens:
            break

    return tokens


def extract_training_units(
    text,
    training_unit,
    min_chars,
    max_chars,
    max_chunks_per_record,
    min_word_chars,
    max_tokens_per_record,
):
    if training_unit == "sentence":
        return split_short_texts(text, min_chars, max_chars, max_chunks_per_record)
    if training_unit == "word":
        return extract_word_tokens(text, min_word_chars, max_tokens_per_record)
    raise ValueError(f"Unsupported training unit '{training_unit}'.")


def dataset_kwargs(token):
    return {"token": token} if token else {}


def load_streaming_dataset(dataset_name, config, split, token):
    return load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
        **dataset_kwargs(token),
    )


def collect_wikipedia_examples(
    dataset_name,
    snapshot,
    lang,
    label,
    limit,
    training_unit,
    min_chars,
    max_chars,
    max_chunks_per_record,
    min_word_chars,
    max_tokens_per_record,
    token,
):
    if limit <= 0:
        return []

    config = f"{snapshot}.{lang}"
    log(
        f"Loading Wikipedia {config} for {label} {training_unit} examples; "
        f"target={limit}"
    )
    dataset = load_streaming_dataset(dataset_name, config, "train", token)
    examples = []
    seen = set()
    progress_prefix = f"Wikipedia {config}"
    next_report = max(1, min(100, limit // 10 or 1))

    for row in dataset:
        for text in extract_training_units(
            row.get("text", ""),
            training_unit,
            min_chars,
            max_chars,
            max_chunks_per_record,
            min_word_chars,
            max_tokens_per_record,
        ):
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            examples.append(
                Example(text=text, label=label, source="wikipedia", language=lang)
            )
            next_report = log_progress(
                progress_prefix, len(examples), limit, next_report
            )
            if len(examples) >= limit:
                log(f"Finished Wikipedia {config}: collected {len(examples)}/{limit}")
                return examples

    log(f"Finished Wikipedia {config}: collected {len(examples)}/{limit}")
    return examples


def collect_europarl_examples(
    dataset_name,
    pairs,
    label,
    target_languages,
    limit,
    training_unit,
    min_chars,
    max_chars,
    min_word_chars,
    max_tokens_per_record,
    token,
):
    if limit <= 0:
        return []

    examples = []
    seen = set()
    target_languages = set(target_languages)
    progress_prefix = (
        f"Europarl {','.join(pairs)} [{','.join(sorted(target_languages))}]"
    )
    next_report = max(1, min(100, limit // 10 or 1))

    for pair in pairs:
        log(
            f"Loading Europarl {pair} for {label} {training_unit} examples "
            f"from {', '.join(sorted(target_languages))}; "
            f"current={len(examples)}/{limit}"
        )
        dataset = load_streaming_dataset(dataset_name, pair, "train", token)
        for row in dataset:
            translation = row.get("translation", {})
            for lang, value in translation.items():
                if lang not in target_languages:
                    continue
                for text in extract_training_units(
                    value,
                    training_unit,
                    min_chars,
                    max_chars,
                    1,
                    min_word_chars,
                    max_tokens_per_record,
                ):
                    key = text.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    examples.append(
                        Example(
                            text=text,
                            label=label,
                            source="europarl",
                            language=lang,
                        )
                    )
                    next_report = log_progress(
                        progress_prefix, len(examples), limit, next_report
                    )
                    if len(examples) >= limit:
                        log(
                            "Finished Europarl collection: "
                            f"collected {len(examples)}/{limit}"
                        )
                        return examples

    log(f"Finished Europarl collection: collected {len(examples)}/{limit}")
    return examples


def collect_balanced_wikipedia_negatives(args, token, total):
    languages = parse_csv(args.wikipedia_negative_langs)
    per_language = split_quota(total, languages)
    examples = []
    for lang, limit in per_language.items():
        log(f"Collecting Wikipedia negatives for {lang}; target={limit}")
        examples.extend(
            collect_wikipedia_examples(
                args.wikipedia_dataset,
                args.wikipedia_snapshot,
                lang,
                LABEL_NOT_ES,
                limit,
                args.training_unit,
                args.min_chars,
                args.max_chars,
                args.max_chunks_per_record,
                args.min_word_chars,
                args.max_tokens_per_record,
                token,
            )
        )
    return examples


def collect_balanced_flores_examples(args, token, total, label, configs):
    per_config = split_quota(total, configs)
    examples = []
    for config, limit in per_config.items():
        log(
            f"Loading FLORES+ {config}/{args.flores_split} for {label} "
            f"examples; target={limit}"
        )
        dataset = load_streaming_dataset(
            args.flores_dataset, config, args.flores_split, token
        )
        lang = config.split("_")[0]
        count = 0
        progress_prefix = f"FLORES+ {config}"
        next_report = max(1, min(100, limit // 10 or 1))
        for row in dataset:
            for text in extract_training_units(
                row.get("text", ""),
                args.training_unit,
                args.min_chars,
                args.max_chars,
                args.max_chunks_per_record,
                args.min_word_chars,
                args.max_tokens_per_record,
            ):
                examples.append(
                    Example(text=text, label=label, source="flores", language=lang)
                )
                count += 1
                next_report = log_progress(progress_prefix, count, limit, next_report)
                if count >= limit:
                    break
            if count >= limit:
                break
        log(f"Finished FLORES+ {config}: collected {count}/{limit}")
    return examples


def split_quota(total, keys):
    keys = list(keys)
    if not keys:
        return {}
    base = total // len(keys)
    remainder = total % len(keys)
    return {
        key: base + (1 if index < remainder else 0) for index, key in enumerate(keys)
    }


def deduplicate_examples(examples):
    deduped = []
    seen = set()
    for example in examples:
        key = example.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def collect_training_examples(args, token):
    positive_wiki_target = args.positive_samples // 2
    positive_euro_target = args.positive_samples - positive_wiki_target
    negative_wiki_target = args.negative_samples // 2
    negative_euro_target = args.negative_samples - negative_wiki_target

    europarl_pairs = parse_csv(args.europarl_pairs)
    negative_languages = parse_csv(args.wikipedia_negative_langs)

    log(
        "Training-source targets: "
        f"unit={args.training_unit}, "
        f"Spanish={args.positive_samples} "
        f"(Wikipedia={positive_wiki_target}, Europarl={positive_euro_target}), "
        f"not-Spanish={args.negative_samples} "
        f"(Wikipedia={negative_wiki_target}, Europarl={negative_euro_target})"
    )

    positive_examples = []
    log("Collecting Spanish examples from Wikipedia...")
    positive_examples.extend(
        collect_wikipedia_examples(
            args.wikipedia_dataset,
            args.wikipedia_snapshot,
            args.wikipedia_positive_lang,
            LABEL_ES,
            positive_wiki_target,
            args.training_unit,
            args.min_chars,
            args.max_chars,
            args.max_chunks_per_record,
            args.min_word_chars,
            args.max_tokens_per_record,
            token,
        )
    )
    log("Collecting Spanish examples from Europarl...")
    positive_examples.extend(
        collect_europarl_examples(
            args.europarl_dataset,
            europarl_pairs,
            LABEL_ES,
            [args.wikipedia_positive_lang],
            positive_euro_target,
            args.training_unit,
            args.min_chars,
            args.max_chars,
            args.min_word_chars,
            args.max_tokens_per_record,
            token,
        )
    )

    negative_examples = []
    log("Collecting non-Spanish examples from Wikipedia...")
    negative_examples.extend(
        collect_balanced_wikipedia_negatives(args, token, negative_wiki_target)
    )
    log("Collecting non-Spanish examples from Europarl...")
    negative_examples.extend(
        collect_europarl_examples(
            args.europarl_dataset,
            europarl_pairs,
            LABEL_NOT_ES,
            negative_languages,
            negative_euro_target,
            args.training_unit,
            args.min_chars,
            args.max_chars,
            args.min_word_chars,
            args.max_tokens_per_record,
            token,
        )
    )

    positive_examples = deduplicate_examples(positive_examples)[: args.positive_samples]
    negative_examples = deduplicate_examples(negative_examples)[: args.negative_samples]

    log(
        f"After deduplication: Spanish={len(positive_examples)}, "
        f"not-Spanish={len(negative_examples)}"
    )

    if len(positive_examples) < args.positive_samples:
        print(
            f"Warning: collected {len(positive_examples)} Spanish examples, "
            f"requested {args.positive_samples}."
        )
    if len(negative_examples) < args.negative_samples:
        print(
            f"Warning: collected {len(negative_examples)} non-Spanish examples, "
            f"requested {args.negative_samples}."
        )

    balanced_count = min(len(positive_examples), len(negative_examples))
    if balanced_count == 0:
        raise RuntimeError("No balanced training data was collected.")

    return positive_examples[:balanced_count] + negative_examples[:balanced_count]


def collect_flores_evaluation_examples(args, token):
    positive_total = args.flores_samples // 2
    negative_total = args.flores_samples - positive_total
    try:
        positive = collect_balanced_flores_examples(
            args, token, positive_total, LABEL_ES, [args.flores_positive_config]
        )
        negative = collect_balanced_flores_examples(
            args,
            token,
            negative_total,
            LABEL_NOT_ES,
            parse_csv(args.flores_negative_configs),
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load FLORES+ evaluation data. The dataset may be gated; "
            f"authenticate with Hugging Face or set {args.dataset_token_env}, "
            "or rerun with --skip-flores-eval."
        ) from exc

    balanced_count = min(len(positive), len(negative))
    if balanced_count == 0:
        raise RuntimeError("No balanced FLORES evaluation data was collected.")
    return positive[:balanced_count] + negative[:balanced_count]


def fasttext_line(example):
    safe_text = example.text.replace("\n", " ")
    return f"{FASTTEXT_LABEL_PREFIX}{example.label} {safe_text}\n"


def write_fasttext_file(path, examples):
    log(f"Writing {len(examples)} fastText rows to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(fasttext_line(example))


def write_jsonl(path, examples):
    log(f"Writing {len(examples)} JSONL rows to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def split_train_validation(examples, train_ratio, seed):
    rng = random.Random(seed)
    train_examples = []
    validation_examples = []

    for label in (LABEL_ES, LABEL_NOT_ES):
        label_examples = [example for example in examples if example.label == label]
        rng.shuffle(label_examples)
        train_size = int(len(label_examples) * train_ratio)
        train_examples.extend(label_examples[:train_size])
        validation_examples.extend(label_examples[train_size:])

    rng.shuffle(train_examples)
    rng.shuffle(validation_examples)
    return train_examples, validation_examples


def predict_label(model, text):
    labels, _scores = model.predict(text.replace("\n", " "), k=1)
    return labels[0].replace(FASTTEXT_LABEL_PREFIX, "")


def evaluate_model(model, examples, training_unit):
    labels = [LABEL_ES, LABEL_NOT_ES]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}

    for example in examples:
        predicted = predict_label(model, example.text)
        if predicted not in labels:
            predicted = LABEL_NOT_ES
        confusion[example.label][predicted] += 1

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    metrics = {
        "training_unit": training_unit,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": confusion,
        "labels": {},
        "source_counts": Counter(example.source for example in examples),
        "language_counts": Counter(example.language for example in examples),
    }

    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        metrics["labels"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    metrics["source_counts"] = dict(metrics["source_counts"])
    metrics["language_counts"] = dict(metrics["language_counts"])
    return metrics


def write_json(path, data):
    log(f"Writing JSON to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_metrics(name, metrics):
    print(f"{name} accuracy: {metrics['accuracy']:.4f} ({metrics['total']} examples)")
    for label, label_metrics in metrics["labels"].items():
        print(
            f"  {label}: precision={label_metrics['precision']:.4f} "
            f"recall={label_metrics['recall']:.4f} f1={label_metrics['f1']:.4f} "
            f"support={label_metrics['support']}"
        )


def save_quantized_model(model, train_path, output_model):
    quantized_path = output_model.with_suffix(".ftz")
    log(f"Quantizing model using {train_path}...")
    model.quantize(input=str(train_path), retrain=True)
    model.save_model(str(quantized_path))
    log(f"Saved quantized model to {quantized_path}")


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.min_word_chars < 1:
        raise ValueError("--min-word-chars must be at least 1.")
    if args.max_tokens_per_record < 1:
        raise ValueError("--max-tokens-per-record must be at least 1.")

    if args.training_unit == "word":
        log(
            "Word-level mode enabled. Accented Unicode alphabetic words are preserved; "
            "for this mode, consider --word-ngrams 1 --dim 50 --epoch 20 --lr 0.3."
        )

    loaded_env_keys = load_env_file(args.env_file)
    token = os.getenv(args.dataset_token_env)
    if token and args.dataset_token_env in loaded_env_keys:
        log(f"Loaded Hugging Face token from {args.env_file}.")
    elif token:
        log(f"Using Hugging Face token from ${args.dataset_token_env}.")
    else:
        log(
            f"No Hugging Face token found in ${args.dataset_token_env}; "
            "public datasets may still work, gated datasets may fail."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    fasttext.FastText.eprint = lambda _message: None

    log("Collecting Wikipedia and Europarl training data...")
    start_time = time.monotonic()
    examples = collect_training_examples(args, token)
    log(
        f"Collected {len(examples)} balanced training-source examples "
        f"in {time.monotonic() - start_time:.1f}s."
    )

    train_examples, validation_examples = split_train_validation(
        examples, args.train_ratio, args.seed
    )
    log(
        f"Split examples: train={len(train_examples)}, "
        f"validation={len(validation_examples)}, train_ratio={args.train_ratio}"
    )

    train_path = output_dir / "train.fasttext.txt"
    validation_path = output_dir / "validation.fasttext.txt"
    write_fasttext_file(train_path, train_examples)
    write_fasttext_file(validation_path, validation_examples)
    write_jsonl(output_dir / "train_examples.jsonl", train_examples)
    write_jsonl(output_dir / "validation_examples.jsonl", validation_examples)

    data_summary = {
        "training_unit": args.training_unit,
        "train": {
            "total": len(train_examples),
            "labels": dict(Counter(example.label for example in train_examples)),
            "sources": dict(Counter(example.source for example in train_examples)),
            "languages": dict(Counter(example.language for example in train_examples)),
        },
        "validation": {
            "total": len(validation_examples),
            "labels": dict(Counter(example.label for example in validation_examples)),
            "sources": dict(Counter(example.source for example in validation_examples)),
            "languages": dict(
                Counter(example.language for example in validation_examples)
            ),
        },
    }
    write_json(output_dir / "data_summary.json", data_summary)

    log(
        f"Training fastText model with {len(train_examples)} examples "
        f"(epoch={args.epoch}, lr={args.lr}, wordNgrams={args.word_ngrams}, "
        f"dim={args.dim})..."
    )
    train_start_time = time.monotonic()
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        dim=args.dim,
    )
    model.save_model(str(output_model))
    log(f"Training completed in {time.monotonic() - train_start_time:.1f}s.")
    log(f"Saved model to {output_model}")

    log("Evaluating validation split...")
    validation_metrics = evaluate_model(model, validation_examples, args.training_unit)
    write_json(output_dir / "validation_metrics.json", validation_metrics)
    print_metrics("Validation", validation_metrics)

    if args.skip_flores_eval:
        log("Skipped FLORES+ held-out evaluation.")
        if args.quantize:
            save_quantized_model(model, train_path, output_model)
        return

    log("Collecting FLORES+ held-out evaluation data...")
    flores_start_time = time.monotonic()
    try:
        flores_examples = collect_flores_evaluation_examples(args, token)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    log(
        f"Collected {len(flores_examples)} FLORES+ evaluation examples "
        f"in {time.monotonic() - flores_start_time:.1f}s."
    )

    write_jsonl(output_dir / "flores_eval_examples.jsonl", flores_examples)
    log("Evaluating FLORES+ held-out split...")
    flores_metrics = evaluate_model(model, flores_examples, args.training_unit)
    write_json(output_dir / "flores_metrics.json", flores_metrics)
    print_metrics("FLORES+", flores_metrics)

    if args.quantize:
        save_quantized_model(model, train_path, output_model)


if __name__ == "__main__":
    main()
