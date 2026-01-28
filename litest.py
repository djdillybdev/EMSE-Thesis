# /// script
# dependencies = [
#     "fasttext",
#     "numpy<2",
#     "pymupdf",
# ]
# ///

import fasttext
import pymupdf

# Path to the downloaded model file
model_path = 'models/lid.176.bin'

# Load the model
# specific warning suppression is optional but cleans up the output
fasttext.FastText.eprint = lambda x: None
model = fasttext.load_model(model_path)

def detect_language(text):
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

    return lang_code, confidence

# --- Example Usage ---

samples = [
    "Hello, how are you doing today?",
    "Je suis un ingénieur logiciel.",
    "Esto es un ejemplo de texto en español.",
    "Wie geht es Ihnen?",
    "今天是个好日子"
]

print(f"{'Text':<40} | {'Lang':<5} | {'Conf':<6}")
print("-" * 60)

for text in samples:
    lang, conf = detect_language(text)
    print(f"{text:<40} | {lang:<5} | {conf:.4f}")


# --- PDF Usage ---

doc = pymupdf.open("resources/aih_17_8_062.pdf")
text = ""
for page in doc:
  text += page.get_text()
lang, conf = detect_language(text)
print(f"{text:<40} | {lang:<5} | {conf:.4f}")