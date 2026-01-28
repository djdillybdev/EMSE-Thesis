# /// script
# dependencies = [
#     "fasttext",
#     "numpy<2",
#     "huggingface_hub",
#     "lingua-language-detector",
# ]
# ///

import fasttext
import json
from huggingface_hub import hf_hub_download
from lingua import Language, LanguageDetectorBuilder

# Configuration
input_texts = 'test_data.jsonl'
output_file = 'evaluation_results.jsonl'
word_output_file = 'word_results.jsonl'

# Load models
# Suppress FastText warnings
fasttext.FastText.eprint = lambda x: None

# Initialize lingua detector with test data languages
lingua_languages = [
    Language.ENGLISH, Language.FRENCH, Language.SPANISH, Language.ITALIAN,
    Language.GERMAN, Language.PORTUGUESE, Language.RUSSIAN, Language.TURKISH,
    Language.INDONESIAN, Language.UKRAINIAN, Language.VIETNAMESE, Language.DUTCH,
    Language.POLISH, Language.ROMANIAN
]
lingua_detector = LanguageDetectorBuilder.from_languages(*lingua_languages).build()

models = {
    'fasttext-lid.176': ('fasttext', fasttext.load_model('models/lid.176.bin')),
    'fasttext-lid.176compressed': ('fasttext', fasttext.load_model('models/lid.176.ftz')),
    'glotlid': ('fasttext', fasttext.load_model(hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin"))),
    'lingua': ('lingua', lingua_detector)
}

# Mapping from ISO 639-3 to ISO 639-1 for common languages
ISO_639_3_TO_1 = {
    'eng': 'en', 'fra': 'fr', 'spa': 'es', 'deu': 'de', 'ita': 'it',
    'por': 'pt', 'rus': 'ru', 'tur': 'tr', 'ind': 'id', 'ukr': 'uk',
    'vie': 'vi', 'nld': 'nl', 'pol': 'pl', 'ron': 'ro', 'jpn': 'ja',
    'zho': 'zh', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi', 'swe': 'sv',
}

def detect_language(text, model_type, model):
    # Models assume input is a single line
    clean_text = text.replace('\n', ' ')
    if model_type == 'fasttext':

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

    elif model_type == 'lingua':

        # Get all confidence values
        confidence_values = model.compute_language_confidence_values(clean_text)

        if not confidence_values:
            # No detection possible - return None
            return None, 0.0

        # Get top prediction
        top_prediction = confidence_values[0]
        lang_code = top_prediction.language.iso_code_639_1.name.lower()
        confidence = top_prediction.value

    return lang_code, confidence

samples = [json.loads(line) for line in open(input_texts, 'r', encoding='utf-8')]

results = []

for sample in samples:
    text = sample['text']

    # Evaluate with each model
    for model_name, (model_type, model) in models.items():
        detected_lang, confidence = detect_language(text, model_type, model)

        # Handle cases where lingua returns None
        if detected_lang is None:
            detected_lang = 'unknown'

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

unique_words = set()
for sample in samples:
    text = sample['text']
    clean_text = text.replace('\n', ' ')
    words = clean_text.split()
    words = [(word, sample['lang']) for word in words]
    unique_words.update(words)

word_results = []

for word, lang in unique_words:
    for model_name, (model_type, model) in models.items():
        detected_lang, confidence = detect_language(word, model_type, model)

        # Handle cases where lingua returns None
        if detected_lang is None:
            detected_lang = 'unknown'

        result = {
            'word': word,
            'model': model_name,
            'true_lang': lang,
            'detected_lang': detected_lang,
            'confidence': float(confidence)
        }
        word_results.append(result)

# Write word results to output file
with open(word_output_file, 'w', encoding='utf-8') as f:
    for result in word_results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"Processed {len(unique_words)} unique words with {len(models)} models ({len(word_results)} total results)")
print(f"Results saved to {word_output_file}")