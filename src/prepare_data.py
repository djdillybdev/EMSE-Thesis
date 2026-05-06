"""Prepare and load shared evaluation datasets derived from FLORES.

This script materializes one reusable prepared-data profile under
`data/prepared/...` (or another `--prepared-data-root`) based on the dataset
arguments. The output profile contains:

- `manifest.json`: the effective arguments and sample counts
- `pure_samples.jsonl`: plain FLORES samples used for pure-language evaluation
- `injected_samples.jsonl`: Spanish samples with token-level foreign injections
- `phrase_samples.jsonl`: Spanish samples with multi-token foreign phrase swaps

Both `src/evaluate_methods.py` and `src/llm_evaluate_methods.py` consume these
prepared files so the project does not regenerate or duplicate evaluation input
data per run.
"""

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


FASTTEXT_LABEL_PREFIX = "__label__"
SPANISH = "es"

DEFAULT_FLORES_DATASET = "openlanguagedata/flores_plus"
DEFAULT_SPLIT = "devtest"
DEFAULT_SPANISH_CONFIG = "spa_Latn"
DEFAULT_INJECTION_CONFIGS = (
    "eng_Latn,por_Latn,ita_Latn,fra_Latn,deu_Latn,cat_Latn,eus_Latn"
)
DEFAULT_PURE_CONFIGS = f"{DEFAULT_SPANISH_CONFIG},{DEFAULT_INJECTION_CONFIGS}"
DEFAULT_PREPARED_DATA_ROOT = "data/prepared"

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
class Sample:
    sample_id: str
    row_index: int
    flores_config: str
    lang: str
    text: str
    split: str


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
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def is_acronym_token(token):
    normalized = token.normalized if isinstance(token, Token) else str(token or "")
    return len(normalized) > 1 and normalized.isupper()


def is_eligible_injected_token(token, min_word_length):
    raw = token.raw if isinstance(token, Token) else str(token or "")
    normalized = token.normalized if isinstance(token, Token) else str(token or "")
    if not is_eligible_word(normalized):
        return False
    if any(char.isdigit() for char in raw):
        return False
    if len(normalized) < min_word_length:
        return False
    if is_acronym_token(token):
        return False
    if not normalized.islower():
        return False
    return True


def is_lowercase_token(token):
    normalized = token.normalized if isinstance(token, Token) else str(token or "")
    return bool(normalized) and normalized.islower()


def shuffle_tokens(tokens, rng):
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    return shuffled


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


def raw_text_parts(text):
    return [match.group(0) for match in WHITESPACE_TOKEN_RE.finditer(text)]


def iter_flores_samples(dataset_name, config, split, token, limit=None):
    log(f"Loading FLORES {config}/{split}")
    dataset = load_dataset(dataset_name, config, split=split, **dataset_kwargs(token))
    lang = config_to_lang(config)
    for index, row in enumerate(dataset):
        yield Sample(
            sample_id=f"{config}:{split}:{index}",
            row_index=index,
            flores_config=config,
            lang=lang,
            text=text_from_row(row),
            split=split,
        )
        if limit and index + 1 >= limit:
            break


def resolve_pure_configs(args):
    if getattr(args, "flores_configs", None):
        configs = parse_csv(args.flores_configs)
    else:
        configs = parse_csv(DEFAULT_PURE_CONFIGS)

    if getattr(args, "limit_languages", None):
        wanted = set(parse_csv(args.limit_languages))
        configs = [
            config
            for config in configs
            if config in wanted or config_to_lang(config) in wanted
        ]
    return sorted(configs)


def random_foreign_token(source_tokens, replacement_count):
    if not source_tokens:
        return None
    if replacement_count < len(source_tokens):
        return source_tokens[replacement_count]
    return source_tokens[replacement_count % len(source_tokens)]


def build_injected_sample(
    spanish_sample,
    foreign_sample,
    injection_lang,
    ratio,
    min_word_length,
    rng,
):
    spanish_tokens = tokenize(spanish_sample.text)
    foreign_tokens = tokenize(foreign_sample.text)
    raw_parts = raw_text_parts(spanish_sample.text)
    if not spanish_tokens or not foreign_tokens or not raw_parts:
        return None

    candidate_tokens = [
        token
        for token in spanish_tokens
        if is_eligible_injected_token(token, min_word_length)
    ]
    eligible_foreign_tokens = [
        token
        for token in foreign_tokens
        if is_eligible_injected_token(token, min_word_length)
    ]
    if not candidate_tokens or not eligible_foreign_tokens:
        return None

    injection_count = max(1, round(len(candidate_tokens) * ratio))
    candidate_tokens = shuffle_tokens(candidate_tokens, rng)
    eligible_foreign_tokens = shuffle_tokens(eligible_foreign_tokens, rng)

    replacements = []
    used_indexes = set()
    for spanish_token in candidate_tokens:
        if len(replacements) >= injection_count:
            break
        if spanish_token.raw_index in used_indexes:
            continue

        foreign_token = random_foreign_token(eligible_foreign_tokens, len(replacements))
        if foreign_token is None:
            continue
        if foreign_token.normalized.casefold() == spanish_token.normalized.casefold():
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

    return {
        "sample_id": f"{spanish_sample.sample_id}:inject:{injection_lang}",
        "source_sample_id": spanish_sample.sample_id,
        "row_index": spanish_sample.row_index,
        "source_config": spanish_sample.flores_config,
        "split": spanish_sample.split,
        "base_lang": SPANISH,
        "injected_lang": injection_lang,
        "contamination_type": "random_token",
        "injection_ratio": ratio,
        "requested_injections": injection_count,
        "actual_injections": len(replacements),
        "text": " ".join(raw_parts),
        "injections": replacements,
    }


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

    target_replacements = max(
        1, round(len(spanish_tokens) * args.phrase_replacement_ratio)
    )
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
            if any(
                not is_eligible_injected_token(token, args.injected_min_word_length)
                for token in spanish_span
            ):
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
            if any(
                not is_eligible_injected_token(token, args.injected_min_word_length)
                for token in foreign_span
            ):
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

    return {
        "sample_id": f"{spanish_sample.sample_id}:phrase:{injection_lang}",
        "source_sample_id": spanish_sample.sample_id,
        "row_index": spanish_sample.row_index,
        "source_config": spanish_sample.flores_config,
        "split": spanish_sample.split,
        "base_lang": SPANISH,
        "injected_lang": injection_lang,
        "contamination_type": "phrase_span",
        "injection_ratio": args.phrase_replacement_ratio,
        "phrase_span_min": args.phrase_span_min,
        "phrase_span_max": args.phrase_span_max,
        "requested_injections": target_replacements,
        "actual_injections": len(replacements),
        "text": " ".join(raw_parts),
        "injections": replacements,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    log(f"Wrote {count} rows to {path}")


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def prepared_profile_manifest(args):
    pure_configs = resolve_pure_configs(args)
    injection_configs = parse_csv(args.injection_configs)
    return {
        "flores_dataset": args.flores_dataset,
        "split": args.split,
        "pure_configs": pure_configs,
        "limit_languages": parse_csv(args.limit_languages)
        if getattr(args, "limit_languages", None)
        else [],
        "limit_samples_per_language": args.limit_samples_per_language,
        "spanish_config": args.spanish_config,
        "injection_configs": injection_configs,
        "seed": args.seed,
        "injection_ratio": args.injection_ratio,
        "injected_min_word_length": args.injected_min_word_length,
        "injected_exclude_acronyms": True,
        "injected_lowercase_only": True,
        "injected_exclude_proper_noun_like_tokens": True,
        "injected_contamination_type": "random_token",
        "phrase_replacement_ratio": args.phrase_replacement_ratio,
        "phrase_span_min": args.phrase_span_min,
        "phrase_span_max": args.phrase_span_max,
        "phrase_lowercase_only": True,
        "phrase_exclude_proper_noun_like_tokens": True,
    }


def prepared_profile_id(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def resolve_prepared_profile_dir(args):
    manifest = prepared_profile_manifest(args)
    dataset_slug = manifest["flores_dataset"].rsplit("/", 1)[-1].replace("-", "_")
    profile_id = prepared_profile_id(manifest)
    profile_name = f"{dataset_slug}_{manifest['split']}_{profile_id}"
    return Path(args.prepared_data_root) / profile_name, manifest, profile_id


def prepared_dataset_paths_from_dir(prepared_data_dir):
    profile_dir = Path(prepared_data_dir)
    return {
        "profile_dir": profile_dir,
        "manifest": profile_dir / "manifest.json",
        "pure": profile_dir / "pure_samples.jsonl",
        "injected": profile_dir / "injected_samples.jsonl",
        "phrase": profile_dir / "phrase_samples.jsonl",
    }


def ensure_prepared_data_dir(prepared_data_dir):
    paths = prepared_dataset_paths_from_dir(prepared_data_dir)
    missing = [
        label
        for label in ("manifest", "pure", "injected", "phrase")
        if not paths[label].exists()
    ]
    if missing:
        raise RuntimeError(
            "Prepared data directory is missing required files. "
            f"Expected under {paths['profile_dir']}: "
            "manifest.json, pure_samples.jsonl, injected_samples.jsonl, phrase_samples.jsonl."
        )
    return paths


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prepared_manifest_from_dir(prepared_data_dir):
    paths = ensure_prepared_data_dir(prepared_data_dir)
    return load_json(paths["manifest"]), paths["profile_dir"]


def load_prepared_rows(prepared_data_dir, dataset_key):
    paths = ensure_prepared_data_dir(prepared_data_dir)
    return list(iter_jsonl(paths[dataset_key]))


def load_prepared_pure_samples_from_dir(prepared_data_dir):
    return [
        Sample(
            sample_id=row["sample_id"],
            row_index=row["row_index"],
            flores_config=row["flores_config"],
            lang=row["lang"],
            text=row["text"],
            split=row["split"],
        )
        for row in load_prepared_rows(prepared_data_dir, "pure")
    ]


def load_prepared_injected_samples_from_dir(prepared_data_dir):
    return load_prepared_rows(prepared_data_dir, "injected")


def load_prepared_phrase_samples_from_dir(prepared_data_dir):
    return load_prepared_rows(prepared_data_dir, "phrase")


def prepared_manifest_summary(manifest):
    return {
        "profile_id": manifest["profile_id"],
        "pure_configs": manifest["pure_configs"],
        "injection_configs": manifest["injection_configs"],
        "sample_counts": manifest["sample_counts"],
    }


def load_prepared_datasets_from_dir(
    prepared_data_dir,
    *,
    include_pure=True,
    include_injected=True,
    include_phrase=True,
):
    manifest, profile_dir = load_prepared_manifest_from_dir(prepared_data_dir)
    datasets = {}
    if include_pure:
        datasets["pure"] = load_prepared_pure_samples_from_dir(prepared_data_dir)
    if include_injected:
        datasets["injected"] = load_prepared_injected_samples_from_dir(
            prepared_data_dir
        )
    if include_phrase:
        datasets["phrase"] = load_prepared_phrase_samples_from_dir(prepared_data_dir)
    return manifest, profile_dir, datasets


def build_sample_cache(args, token, manifests):
    sample_cache = {}
    for config in manifests:
        sample_cache[config] = list(
            iter_flores_samples(
                args.flores_dataset,
                config,
                args.split,
                token,
                args.limit_samples_per_language,
            )
        )
    return sample_cache


def prepare_datasets(args, token):
    profile_dir, manifest, profile_id = resolve_prepared_profile_dir(args)
    paths = prepared_dataset_paths_from_dir(profile_dir)

    required_configs = set(manifest["pure_configs"])
    required_configs.add(manifest["spanish_config"])
    required_configs.update(manifest["injection_configs"])
    sample_cache = build_sample_cache(args, token, sorted(required_configs))

    pure_rows = []
    for config in manifest["pure_configs"]:
        for sample in sample_cache.get(config, []):
            pure_rows.append(asdict(sample))

    spanish_by_row = {
        sample.row_index: sample
        for sample in sample_cache.get(manifest["spanish_config"], [])
    }
    injected_rng = random.Random(args.seed)
    phrase_rng = random.Random(args.seed)

    injected_rows = []
    phrase_rows = []
    for config in manifest["injection_configs"]:
        injection_lang = config_to_lang(config)
        for foreign_sample in sample_cache.get(config, []):
            spanish_sample = spanish_by_row.get(foreign_sample.row_index)
            if spanish_sample is None:
                continue
            injected_sample = build_injected_sample(
                spanish_sample,
                foreign_sample,
                injection_lang,
                args.injection_ratio,
                args.injected_min_word_length,
                injected_rng,
            )
            if injected_sample is not None:
                injected_rows.append(injected_sample)

            phrase_sample = build_phrase_sample(
                spanish_sample,
                foreign_sample,
                injection_lang,
                args,
                phrase_rng,
            )
            if phrase_sample is not None:
                phrase_rows.append(phrase_sample)

    profile_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["pure"], pure_rows)
    write_jsonl(paths["injected"], injected_rows)
    write_jsonl(paths["phrase"], phrase_rows)
    write_json(
        paths["manifest"],
        {
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "profile_id": profile_id,
            **manifest,
            "sample_counts": {
                "pure": len(pure_rows),
                "injected": len(injected_rows),
                "phrase": len(phrase_rows),
            },
        },
    )
    log(f"Prepared shared datasets under {profile_dir}")
    return profile_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download FLORES data and prepare shared pure, injected, and phrase "
            "evaluation datasets for reuse by both evaluation scripts."
        )
    )
    parser.add_argument(
        "--prepared-data-root",
        default=DEFAULT_PREPARED_DATA_ROOT,
        help=(
            "Base directory where prepared dataset profiles are stored. "
            "A hashed subdirectory is created under this root from the "
            "dataset-shaping arguments. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--flores-dataset",
        default=DEFAULT_FLORES_DATASET,
        help=(
            "Hugging Face dataset id to load FLORES-style text from. "
            "This is the source corpus used to build all prepared samples. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=(
            "Dataset split to prepare, for example `dev` or `devtest`. "
            "The same split is used for pure, injected, and phrase datasets. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--flores-configs",
        default=None,
        help=(
            "Comma-separated FLORES configs to include in the pure dataset. "
            "If omitted, the pure set uses Spanish plus the configured injection "
            "languages. Example: `spa_Latn,eng_Latn,fra_Latn`."
        ),
    )
    parser.add_argument(
        "--limit-languages",
        default=None,
        help=(
            "Optional comma-separated allowlist applied to the pure dataset "
            "configs. Values may be full FLORES configs like `spa_Latn` or "
            "normalized language codes like `es,en,fr`."
        ),
    )
    parser.add_argument(
        "--limit-samples-per-language",
        type=int,
        default=None,
        help=(
            "Optional cap on how many rows to load per FLORES config. "
            "Useful for small test runs or fast profile generation. "
            "If omitted, all rows from the selected split are used."
        ),
    )
    parser.add_argument(
        "--spanish-config",
        default=DEFAULT_SPANISH_CONFIG,
        help=(
            "FLORES config used as the Spanish base text when creating "
            "injected and phrase-mixed samples. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--injection-configs",
        default=DEFAULT_INJECTION_CONFIGS,
        help=(
            "Comma-separated FLORES configs that provide the foreign-language "
            "source text for token injections and phrase swaps. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--injection-ratio",
        type=float,
        default=0.2,
        help=(
            "Approximate share of eligible Spanish tokens to replace when "
            "building the token-level injected dataset. Must be in `(0, 1]`. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--injected-min-word-length",
        type=int,
        default=3,
        help=(
            "Minimum normalized word length for token-level injected samples. "
            "Applies to both the Spanish token being replaced and the foreign "
            "replacement token. Acronyms are excluded. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used to make token injection and phrase replacement "
            "selection deterministic. Changing this creates a different "
            "prepared profile. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--phrase-replacement-ratio",
        type=float,
        default=0.2,
        help=(
            "Approximate share of eligible Spanish tokens to cover with "
            "foreign phrase-span replacements in the phrase dataset. "
            "Must be in `(0, 1]`. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--phrase-span-min",
        type=int,
        default=2,
        help=(
            "Minimum number of contiguous tokens in a foreign phrase swap. "
            "Used only for the phrase dataset. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--phrase-span-max",
        type=int,
        default=4,
        help=(
            "Maximum number of contiguous tokens in a foreign phrase swap. "
            "Used only for the phrase dataset. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help=(
            "Path to an optional environment file to load before downloading "
            "from Hugging Face. This is mainly used to populate the token env "
            "var automatically. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--hf-token-env",
        default="HUGGING_FACE_TOKEN",
        help=(
            "Environment variable name that holds the Hugging Face access token. "
            "If the dataset is public you may not need it, but private or gated "
            "datasets do. Default: %(default)s."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.injection_ratio <= 0.0 or args.injection_ratio > 1.0:
        raise ValueError("--injection-ratio must be in the range (0, 1].")
    if args.injected_min_word_length < 1:
        raise ValueError("--injected-min-word-length must be at least 1.")
    if args.phrase_replacement_ratio <= 0.0 or args.phrase_replacement_ratio > 1.0:
        raise ValueError("--phrase-replacement-ratio must be in the range (0, 1].")
    if args.phrase_span_min < 1:
        raise ValueError("--phrase-span-min must be at least 1.")
    if args.phrase_span_max < args.phrase_span_min:
        raise ValueError("--phrase-span-max must be >= --phrase-span-min.")
    if (
        args.limit_samples_per_language is not None
        and args.limit_samples_per_language < 1
    ):
        raise ValueError("--limit-samples-per-language must be at least 1.")


def main():
    args = parse_args()
    validate_args(args)
    load_env_file(args.env_file)
    token = os.getenv(args.hf_token_env)
    if token:
        log(f"Using Hugging Face token from ${args.hf_token_env}.")
    else:
        log(f"No Hugging Face token found in ${args.hf_token_env}.")
    prepare_datasets(args, token)


if __name__ == "__main__":
    main()
