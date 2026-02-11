# /// script
# dependencies = [
#     "fasttext",
#     "numpy<2",
#     "huggingface_hub",
#     "lingua-language-detector",
#     "pymupdf",
# ]
# ///

import argparse
import json
import re

import fasttext
import pymupdf
from huggingface_hub import hf_hub_download
from lingua import Language, LanguageDetectorBuilder

# Suppress FastText warnings
fasttext.FastText.eprint = lambda x: None

# Mapping from ISO 639-3 to ISO 639-1 for common languages
ISO_639_3_TO_1 = {
    "eng": "en",
    "fra": "fr",
    "spa": "es",
    "deu": "de",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "tur": "tr",
    "ind": "id",
    "ukr": "uk",
    "vie": "vi",
    "nld": "nl",
    "pol": "pl",
    "ron": "ro",
    "jpn": "ja",
    "zho": "zh",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
    "swe": "sv",
}

SUPPORTED_MODELS = [
    "fasttext-lid.176",
    "fasttext-lid.176compressed",
    "glotlid",
    "lingua",
]

STRIP_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)


def build_lingua_detector():
    lingua_languages = [
        Language.ENGLISH,
        Language.FRENCH,
        Language.SPANISH,
        Language.ITALIAN,
        Language.GERMAN,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.TURKISH,
        Language.INDONESIAN,
        Language.UKRAINIAN,
        Language.VIETNAMESE,
        Language.DUTCH,
        Language.POLISH,
        Language.ROMANIAN,
    ]
    return LanguageDetectorBuilder.from_languages(*lingua_languages).build()


def load_model(model_name):
    if model_name == "fasttext-lid.176":
        return "fasttext", fasttext.load_model("models/lid.176.bin")
    if model_name == "fasttext-lid.176compressed":
        return "fasttext", fasttext.load_model("models/lid.176.ftz")
    if model_name == "glotlid":
        return "fasttext", fasttext.load_model(
            hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        )
    if model_name == "lingua":
        return "lingua", build_lingua_detector()

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}"
    )


def detect_language(text, model_type, model):
    clean_text = text.replace("\n", " ")

    if model_type == "fasttext":
        prediction = model.predict(clean_text, k=1)
        label = prediction[0][0]
        confidence = prediction[1][0]
        lang_code = label.replace("__label__", "")

        if "_" in lang_code:
            iso_639_3 = lang_code.split("_")[0]
            lang_code = ISO_639_3_TO_1.get(iso_639_3, iso_639_3)

    elif model_type == "lingua":
        confidence_values = model.compute_language_confidence_values(clean_text)
        if not confidence_values:
            return None, 0.0

        top_prediction = confidence_values[0]
        lang_code = top_prediction.language.iso_code_639_1.name.lower()
        confidence = top_prediction.value

    return lang_code, confidence


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
        "--doc-model",
        default="fasttext-lid.176",
        choices=SUPPORTED_MODELS,
        help="Model to use for document language detection.",
    )
    parser.add_argument(
        "--word-model",
        default="fasttext-lid.176",
        choices=SUPPORTED_MODELS,
        help="Model to use for word-level language detection.",
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
        return "".join(text_parts)

    raise ValueError(f"Unsupported input format '{input_format}'.")


def main():
    args = parse_args()

    doc_model_type, doc_model = load_model(args.doc_model)
    word_model_type, word_model = load_model(args.word_model)

    text = load_input_text(args.input, args.input_format)

    doc_lang, doc_confidence = detect_language(text, doc_model_type, doc_model)
    if doc_lang is None:
        doc_lang = "unknown"

    print(
        f"Document language: {doc_lang} (confidence={float(doc_confidence):.4f}) using {args.doc_model}"
    )

    clean_text = text.replace("\n", " ")
    words = clean_text.split()

    if doc_lang == "unknown":
        print("Document language unknown; no foreign-word detection performed.")
        foreign_records = []
    else:
        foreign_records = []
        for index, word in enumerate(words):
            normalized_word = STRIP_PUNCT_RE.sub("", word)
            if not normalized_word:
                continue
            if normalized_word.isdigit():
                continue

            detected_lang, confidence = detect_language(
                normalized_word, word_model_type, word_model
            )
            if detected_lang is None:
                detected_lang = "unknown"

            if detected_lang == doc_lang:
                continue

            if (
                args.confidence_threshold is not None
                and confidence < args.confidence_threshold
            ):
                continue

            foreign_records.append(
                {
                    "word": word,
                    "normalized_word": normalized_word,
                    "index": index,
                    "detected_lang": detected_lang,
                    "confidence": float(confidence),
                    "document_lang": doc_lang,
                }
            )

    with open(args.output, "w", encoding="utf-8") as f:
        for record in foreign_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Foreign words written: {len(foreign_records)} to {args.output} using {args.word_model}"
    )


if __name__ == "__main__":
    main()
