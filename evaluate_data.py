# /// script
# dependencies = [
#     "fasttext",
#     "numpy<2",
#     "huggingface_hub",
# ]
# ///

import fasttext
import json
from huggingface_hub import hf_hub_download

# Configuration
input_texts = 'test_data.jsonl'
output_file = 'evaluation_results.jsonl'

# Load models
# Suppress FastText warnings
fasttext.FastText.eprint = lambda x: None

models = {
    'fasttext-lid.176': fasttext.load_model('models/lid.176.bin'),
    'glotlid': fasttext.load_model(hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin"))
}

# Mapping from ISO 639-3 to ISO 639-1 for common languages
ISO_639_3_TO_1 = {
    'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it',
    'por': 'pt', 'rus': 'ru', 'tur': 'tr', 'ind': 'id', 'ukr': 'uk',
    'vie': 'vi', 'nld': 'nl', 'pol': 'pl', 'ron': 'ro', 'jpn': 'ja',
    'zho': 'zh', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 'swe': 'sv',
}

def detect_language(text, model):
    # FastText assumes input is a single line.
    # Newlines can cause it to crash or behave unexpectedly.
    clean_text = text.replace('\n', ' ')

    # .predict returns a tuple: (labels, probabilities)
    # k=1 means return only the top 1 result
    prediction = model.predict(clean_text, k=1)

    # Extract the label and probability
    label = prediction[0][0]
    confidence = prediction[1][0]

    # FastText labels look like "__label__en".
    # We strip the "__label__" prefix to get just the code (e.g., "en")
    lang_code = label.replace("__label__", "")

    # Normalize GlotLID format (e.g., "eng_Latn" -> "en")
    # GlotLID uses ISO 639-3 codes with script notation
    if '_' in lang_code:
        iso_639_3 = lang_code.split('_')[0]  # Extract ISO 639-3 code
        lang_code = ISO_639_3_TO_1.get(iso_639_3, iso_639_3)  # Map to ISO 639-1, fallback to 639-3

    return lang_code, confidence

samples = [json.loads(line) for line in open(input_texts, 'r', encoding='utf-8')]

results = []

for sample in samples:
    text = sample['text']

    # Evaluate with each model
    for model_name, model in models.items():
        detected_lang, confidence = detect_language(text, model)

        result = {
            'correct': sample['lang'] == detected_lang,
            'model': model_name,
            'source': sample['source'],
            'text': text,
            'text_length_chars': len(text),
            'text_length_words': len(text.split()),
            'true_lang': sample['lang'],
            'detected_lang': detected_lang,
            'confidence': float(confidence),
        }
        results.append(result)

# Write results to output file
with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"Processed {len(samples)} samples with {len(models)} models ({len(results)} total results)")
print(f"Results saved to {output_file}")