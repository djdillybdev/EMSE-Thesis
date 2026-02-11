# EMSE-Thesis

## Foreign word detection CLI

Detect a document’s main language, then identify words whose detected language differs from the document language.

### Usage

Run the script with a text file or PDF input and an output JSONL path.

```
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

### Output format

The output is JSON Lines (one JSON object per foreign word) with these fields:

- `word`: the word token (whitespace split)
- `index`: zero-based index of the word in the document
- `detected_lang`: detected language of the word
- `confidence`: model confidence for the word detection
- `document_lang`: detected language of the document
