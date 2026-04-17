# EMSE-Thesis

Evaluate language detection models for identifying foreign words in documents. Compares four language detection models (FastText lid.176, FastText Compressed, GlotLID, and Lingua) at both document-level and word-level detection. Includes a GUI and CLI for interactive foreign word detection.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

## Installation

```bash
uv sync
```

## Model Setup

Four language detection models are supported. Some require manual download.

| Model | Flag value | Setup |
|---|---|---|
| FastText (full) | `fasttext-lid.176` | Download `lid.176.bin` from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html) and place in `models/` |
| FastText (compressed) | `fasttext-lid.176compressed` | Download `lid.176.ftz` from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html) and place in `models/` |
| GlotLID | `glotlid` | Downloaded automatically from Hugging Face on first run |
| Lingua | `lingua` | Built-in, no download needed |

Create the `models/` directory if it doesn't exist:

```bash
mkdir -p models
```

## Train Binary Spanish Classifier

Train a local fastText classifier that predicts whether short text is Spanish or not Spanish:

```bash
uv run python src/train_binary_model.py \
  --output-model models/spanish_binary.bin \
  --output-dir models/spanish_binary_training \
  --positive-samples 5000 \
  --negative-samples 5000 \
  --epoch 25 \
  --word-ngrams 2 \
  --quantize
```

The training script uses Hugging Face datasets for data loading only. It trains from `wikimedia/wikipedia` and `Helsinki-NLP/europarl`, then evaluates separately on FLORES+ as held-out benchmark data. FLORES+ is not used for training. FLORES+ is gated on Hugging Face, so set `HUGGING_FACE_TOKEN` in `.env`, authenticate globally, or pass `--skip-flores-eval` while testing the training path.

Outputs are written locally:

- `models/spanish_binary.bin`: fastText model
- `models/spanish_binary.ftz`: quantized model when `--quantize` is passed
- `models/spanish_binary_training/validation_metrics.json`: validation metrics from training-source data
- `models/spanish_binary_training/flores_metrics.json`: held-out FLORES+ metrics

By default, `src/train_binary_model.py` loads `.env` before reading `HUGGING_FACE_TOKEN`. Use `--env-file path/to/.env` for a different file, or `--hf-token-env HF_TOKEN` if your token is stored under another variable name.

Run a quick prediction:

```bash
python - <<'PY'
import fasttext

model = fasttext.load_model("models/spanish_binary.bin")
labels, scores = model.predict("Este es un texto de ejemplo.", k=1)
print(labels[0].replace("__label__", ""), float(scores[0]))
PY
```

Train a word-level model for detecting whether individual words are Spanish:

```bash
uv run python src/train_binary_model.py \
  --training-unit word \
  --output-model models/spanish_binary_words.bin \
  --output-dir models/spanish_binary_words_training \
  --positive-samples 20000 \
  --negative-samples 20000 \
  --min-word-chars 3 \
  --word-ngrams 1 \
  --dim 50 \
  --epoch 20 \
  --lr 0.3
```

Word-level mode preserves Unicode alphabetic Spanish words with accents and tildes, such as `acción`, `niño`, `vergüenza`, `está`, and `rápido`. It drops noisy tokens with digits, URLs, emails, underscores, or internal punctuation. Single-word language detection is ambiguous, so evaluate this model manually on Spanish documents with embedded foreign words before using it as the only signal.

## Hugging Face Setup

A Hugging Face account is needed for some features:

- **GlotLID auto-download**: The model repo is public, so a token is optional but recommended to avoid rate limits.
- **Data pipeline** (`fetch_data.py` and `src/train_binary_model.py`): A token may be required to download gated datasets or avoid rate limits.

To set up:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Either set `HUGGING_FACE_TOKEN` in a `.env` file, or run `huggingface-cli login` to authenticate globally

## GUI Usage

Launch the web-based GUI:

```bash
python gui.py
```

This opens a Gradio app in your browser with the following workflow:

1. **Select models** from the Document Model and Word Model dropdowns
2. **Upload** a `.txt` or `.pdf` file
3. **Adjust the confidence threshold** slider (0 = no filtering)
4. Click **Run Detection**

The results are shown in two tabs:

- **Table View** — browse detected words in a table; select a row to load it into the correction panel
- **Document View** — see the full document with foreign words highlighted by language; filter which languages are highlighted using the checkboxes

Both tabs include a **Correction Panel**: select a word, pick the correct language from the dropdown, and click "Apply Correction". This corrects all occurrences of that word.

Click **Save Results** to write the (corrected) results as JSONL to the output path.

## CLI Usage

Detect a document's main language, then identify words whose detected language differs from the document language.

```bash
python detect_foreign_words.py --input path/to/document.txt --output foreign_words.jsonl
python detect_foreign_words.py --input path/to/document.pdf --output foreign_words.jsonl
```

Optional arguments:

```
--doc-model {fasttext-lid.176,fasttext-lid.176compressed,glotlid,lingua}
--word-model {fasttext-lid.176,fasttext-lid.176compressed,glotlid,lingua}
--confidence-threshold FLOAT
--input-format {auto,txt,pdf}
```

## Output Format

The output is JSON Lines (one JSON object per foreign word) with these fields:

- `word`: the word token (whitespace split)
- `normalized_word`: the word with punctuation stripped
- `index`: zero-based index of the word in the document
- `detected_lang`: detected language of the word
- `confidence`: model confidence for the word detection
- `document_lang`: detected language of the document
