import nltk
import json
import os
from huggingface_hub import login
import requests
from bs4 import BeautifulSoup
import time
from datasets import load_dataset

try:
  nltk.data.find('corpora/udhr')
except LookupError:
  print("Downloading UDHR corpus...")
  nltk.download('udhr')
  print("UDHR download completed!")

from nltk.corpus import udhr

OUTPUT_FILE = "test_data.jsonl"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Compatible; AutomatedTest/1.0)'}

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or "YOUR_HUGGINGFACE_TOKEN_HERE"

print("Logging in to Hugging Face...")
login(token=HF_TOKEN, add_to_git_credential=False)

# Format: 'iso_code': 'url'
bbc_sources = {
  'en': 'https://www.bbc.com/news',
  'fr': 'https://www.bbc.com/afrique',
  'es': 'https://www.bbc.com/mundo',
  'pt': 'https://www.bbc.com/portuguese',
  'ru': 'https://www.bbc.com/russian',
  'tr': 'https://www.bbc.com/turkce',
  'id': 'https://www.bbc.com/indonesia',
  'uk': 'https://www.bbc.com/ukrainian',
  'vi': 'https://www.bbc.com/vietnamese'
}

# Format: 'iso_code': 'url'
euronews_sources = {
  'en': 'https://www.euronews.com/',
  'fr': 'https://fr.euronews.com/',
  'it': 'https://it.euronews.com/',
  'de': 'https://de.euronews.com/',
  'es': 'https://es.euronews.com/',
  'pt': 'https://pt.euronews.com/',
  'ru': 'https://ru.euronews.com/',
  'tr': 'https://tr.euronews.com/',
}

# Creative Commons (Civil Law / Contracts)
# Format: 'iso_code': 'url'
cc_sources = {
    'en': 'https://creativecommons.org/licenses/by/4.0/legalcode',
    'fr': 'https://creativecommons.org/licenses/by/4.0/legalcode.fr',
    'it': 'https://creativecommons.org/licenses/by/4.0/legalcode.it',
    'de': 'https://creativecommons.org/licenses/by/4.0/legalcode.de',
    'id': 'https://creativecommons.org/licenses/by/4.0/legalcode.id',
}

# Format: 'iso_code': 'NLTK_File_ID'
udhr_map = {
  'en': 'English-Latin1',
  'es': 'Spanish_Espanol-Latin1',
  'de': 'German_Deutsch-Latin1',
  'it': 'Italian_Italiano-Latin1',
  'fr': 'French_Francais-Latin1',
  'ru': 'Russian_Russky-UTF8',
  'vi': 'Vietnamese-UTF8',
  'pt': 'Portuguese_Portugues-Latin1'
}

flores_codes = {
    'en': 'eng_Latn',
    'fr': 'fra_Latn',
    'es': 'spa_Latn',
    'it': 'ita_Latn',
    'de': 'deu_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'tr': 'tur_Latn',
    'id': 'ind_Latn',
    'uk': 'ukr_Cyrl',
}

europarl_pairs = [
    ('en', 'fr'), ('en', 'es'), ('en', 'it'),
    ('de', 'en'), ('en', 'pt'), ('en', 'ro'),
    ('en', 'nl'), ('en', 'pl')
]

collected_data = []

def scrape_bbc(iso_code, url):
  """Scrape headlines and paragraphs from BBC articles"""
  print(f"Scraping BBC ({iso_code})[{url}]...")
  try:
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find paragraphs and headlines
    texts = [t.get_text().strip() for t in soup.find_all(['h3', 'p'])]

    # Filter out too short texts
    valid_texts = [t for t in texts if len(t.split()) > 6]

    for text in valid_texts:
      collected_data.append({
        'source': 'BBC',
        'lang': iso_code,
        'text': text
      })
  except Exception as e:
    print(f"Error scraping [{url}]: {e}")

def scrape_euronews(iso_code, url):
  """Scrape headlines and paragraphs from Euronews articles"""
  print(f"Scraping Euronews ({iso_code})[{url}]...")
  try:
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find paragraphs and headlines
    texts = [t.get_text().strip() for t in soup.find_all(['h1', 'h2', 'p'])]

    # Filter out too short texts
    valid_texts = [t for t in texts if len(t.split()) > 10 and 'Cookie' not in t]

    for text in valid_texts:
      collected_data.append({
        'source': 'Euronews',
        'lang': iso_code,
        'text': text
      })
  except Exception as e:
    print(f"Error scraping [{url}]: {e}")

def scrape_cc(iso_code, url):
  """Scrape Creative Commons for text examples"""
  print(f"Scraping CC ({iso_code})[{url}]...")
  try:
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find paragraphs and headlines
    texts = [t.get_text().strip() for t in soup.find_all(['p'])]

    # Filter out too short texts
    valid_texts = [t for t in texts if len(t.split()) > 10]

    for text in valid_texts:
      collected_data.append({
        'source': 'Legal(CC)',
        'lang': iso_code,
        'text': text
      })
  except Exception as e:
    print(f"Error scraping [{url}]: {e}")

def get_udhr(iso_code, file_id):
  """Get text from UDHR corpus"""
  try:
    raw_text = udhr.raw(file_id)
    # maybe consider grabbing just a snippet, but lets try the full text for now
    collected_data.append({
      'source': 'UDHR',
      'lang':iso_code,
      'text': raw_text
    })
  except Exception as e:
    print(f"Error reading UDHR {iso_code} {file_id}: {e}")

def get_flores(iso_code, iso_flores):
  print(f"Fetching Flores {iso_code} ({iso_flores})...")

  try:
    # Load streaming=True to avoid downloading terabytes of data
    ds = load_dataset("openlanguagedata/flores_plus", iso_flores, split="dev", streaming=True)

    count = 0
    for row in ds:
        text = row.get('text') or row.get('sentence')
        if text and len(text) > 20: # valid length
            collected_data.append({
                "source": "FLORES",
                "lang": iso_code,
                "text": text
            })
            count += 1
        if count >= 10:
            break
  except Exception as e:
    print(f"Error loading FLORES: {e}")

def get_europarl(src, target):
  print(f"Fetching Europarl Legal Text for {target}...")
  try:
    ds = load_dataset("Helsinki-NLP/europarl", f"{src}-{target}", split="train", streaming=True)
    count = 0
    for row in ds:
        # row['translation'] is {'en': '...', 'fr': '...'}
        text = row['translation'].get(target, "")

        # Filter for longer legal sentences
        if len(text.split()) > 20:
            collected_data.append({
                "source": "Europarl(Legal)",
                "lang": target,
                "text": text
            })
            count += 1
        if count >= 10:
            break
  except Exception as e:
      print(f"Skipping Europarl {target}: {e}")

# Gather BBC data
for lang, url in bbc_sources.items():
  scrape_bbc(lang, url)
  time.sleep(0.5)

# # Gather Euronews data
for lang, url in euronews_sources.items():
  scrape_euronews(lang, url)
  time.sleep(0.5)

# # Gather CC data
for lang, url in cc_sources.items():
  scrape_cc(lang, url)
  time.sleep(0.5)

# Gather UDHR data
for lang, file_id in udhr_map.items():
  get_udhr(lang, file_id)

# Gather Flores data
for iso_code, iso_flores in flores_codes.items():
  get_flores(iso_code, iso_flores)

# Gather Europarl data
for src, target in europarl_pairs:
  get_europarl(src, target)

# Save to file

print(f"Saving {len(collected_data)} text samples to {OUTPUT_FILE} . . .")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
  for entry in collected_data:
    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("Done.")