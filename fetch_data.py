import nltk
import json
import os
from huggingface_hub import login
import requests
from bs4 import BeautifulSoup
import time
from datasets import load_dataset

try:
    nltk.data.find("corpora/udhr")
except LookupError:
    print("Downloading UDHR corpus...")
    nltk.download("udhr")
    print("UDHR download completed!")

from nltk.corpus import udhr

OUTPUT_FILE = "test_data.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Compatible; AutomatedTest/1.0)"}

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN") or "YOUR_HUGGINGFACE_TOKEN_HERE"

print("Logging in to Hugging Face...")
login(token=HF_TOKEN, add_to_git_credential=False)

# Format: 'iso_code': 'url'
bbc_sources = {
    "en": "https://www.bbc.com/news",
    "fr": "https://www.bbc.com/afrique",
    "es": "https://www.bbc.com/mundo",
    "pt": "https://www.bbc.com/portuguese",
    "ru": "https://www.bbc.com/russian",
    "tr": "https://www.bbc.com/turkce",
    "id": "https://www.bbc.com/indonesia",
    "uk": "https://www.bbc.com/ukrainian",
    "vi": "https://www.bbc.com/vietnamese",
}

# Format: 'iso_code': 'url'
euronews_sources = {
    "en": "https://www.euronews.com/",
    "fr": "https://fr.euronews.com/",
    "it": "https://it.euronews.com/",
    "de": "https://de.euronews.com/",
    "es": "https://es.euronews.com/",
    "pt": "https://pt.euronews.com/",
    "ru": "https://ru.euronews.com/",
    "tr": "https://tr.euronews.com/",
}

# Creative Commons (Civil Law / Contracts)
# Format: 'iso_code': 'url'
cc_sources = {
    "en": "https://creativecommons.org/licenses/by/4.0/legalcode",
    "fr": "https://creativecommons.org/licenses/by/4.0/legalcode.fr",
    "it": "https://creativecommons.org/licenses/by/4.0/legalcode.it",
    "de": "https://creativecommons.org/licenses/by/4.0/legalcode.de",
    "id": "https://creativecommons.org/licenses/by/4.0/legalcode.id",
}

# Format: 'iso_code': 'NLTK_File_ID'
udhr_map = {
    "en": "English-Latin1",
    "es": "Spanish_Espanol-Latin1",
    "de": "German_Deutsch-Latin1",
    "it": "Italian_Italiano-Latin1",
    "fr": "French_Francais-Latin1",
    "ru": "Russian_Russky-UTF8",
    "vi": "Vietnamese-UTF8",
    "pt": "Portuguese_Portugues-Latin1",
}

flores_codes = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "id": "ind_Latn",
    "uk": "ukr_Cyrl",
}

europarl_pairs = [
    ("en", "fr"),
    ("en", "es"),
    ("en", "it"),
    ("de", "en"),
    ("en", "pt"),
    ("en", "ro"),
    ("en", "nl"),
    ("en", "pl"),
]

# Configuration for all data sources
DATA_SOURCES = [
    {
        "type": "web_scraper",
        "name": "BBC",
        "mappings": bbc_sources,
        "html_tags": ["h3", "p"],
        "min_words": 6,
        "delay": 0.5,
    },
    {
        "type": "web_scraper",
        "name": "Euronews",
        "mappings": euronews_sources,
        "html_tags": ["h1", "h2", "p"],
        "min_words": 10,
        "filter_text": "Cookie",
        "delay": 0.5,
    },
    {
        "type": "web_scraper",
        "name": "Legal(CC)",
        "mappings": cc_sources,
        "html_tags": ["p"],
        "min_words": 10,
        "delay": 0.5,
    },
    {"type": "nltk_corpus", "name": "UDHR", "mappings": udhr_map},
    {
        "type": "hf_dataset",
        "name": "FLORES",
        "dataset": "openlanguagedata/flores_plus",
        "mappings": flores_codes,
        "split": "dev",
        "text_keys": ["text", "sentence"],
        "min_length": 20,
        "limit": 15,
    },
    {
        "type": "hf_dataset_paired",
        "name": "Europarl(Legal)",
        "dataset": "Helsinki-NLP/europarl",
        "pairs": europarl_pairs,
        "min_words": 20,
        "limit": 100,
    },
]

collected_data = []


def fetch_web_scraper(source_config):
    """Generic web scraper that works from configuration"""
    name = source_config["name"]
    mappings = source_config["mappings"]
    html_tags = source_config["html_tags"]
    min_words = source_config["min_words"]
    delay = source_config.get("delay", 0)
    filter_text = source_config.get("filter_text")

    for iso_code, url in mappings.items():
        print(f"Scraping {name} ({iso_code})[{url}]...")
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find elements based on configured tags
            texts = [t.get_text().strip() for t in soup.find_all(html_tags)]

            # Filter based on configured criteria
            valid_texts = [t for t in texts if len(t.split()) > min_words]

            # Apply optional text filter
            if filter_text:
                valid_texts = [t for t in valid_texts if filter_text not in t]

            for text in valid_texts:
                collected_data.append({"source": name, "lang": iso_code, "text": text})
        except Exception as e:
            print(f"Error scraping [{url}]: {e}")

        if delay > 0:
            time.sleep(delay)


def fetch_nltk_corpus(source_config):
    """Generic NLTK corpus fetcher"""
    name = source_config["name"]
    mappings = source_config["mappings"]

    for iso_code, file_id in mappings.items():
        try:
            raw_text = udhr.raw(file_id)
            collected_data.append({"source": name, "lang": iso_code, "text": raw_text})
        except Exception as e:
            print(f"Error reading {name} {iso_code} {file_id}: {e}")


def fetch_hf_dataset(source_config):
    """Generic HuggingFace dataset fetcher with streaming and limits"""
    name = source_config["name"]
    dataset = source_config["dataset"]
    mappings = source_config["mappings"]
    split = source_config["split"]
    text_keys = source_config["text_keys"]
    min_length = source_config["min_length"]
    limit = source_config["limit"]

    for iso_code, dataset_code in mappings.items():
        print(f"Fetching {name} {iso_code} ({dataset_code})...")
        try:
            ds = load_dataset(dataset, dataset_code, split=split, streaming=True)

            count = 0
            for row in ds:
                # Try multiple text keys
                text = None
                for key in text_keys:
                    text = row.get(key)
                    if text:
                        break

                if text and len(text) > min_length:
                    collected_data.append(
                        {"source": name, "lang": iso_code, "text": text}
                    )
                    count += 1
                if count >= limit:
                    break
        except Exception as e:
            print(f"Error loading {name} {iso_code}: {e}")


def fetch_hf_dataset_paired(source_config):
    """Generic HuggingFace dataset fetcher for paired translation datasets"""
    name = source_config["name"]
    dataset = source_config["dataset"]
    pairs = source_config["pairs"]
    min_words = source_config["min_words"]
    limit = source_config["limit"]

    for src, target in pairs:
        print(f"Fetching {name} for {target}...")
        try:
            ds = load_dataset(dataset, f"{src}-{target}", split="train", streaming=True)
            count = 0
            for row in ds:
                text = row["translation"].get(target, "")

                if len(text.split()) > min_words:
                    collected_data.append(
                        {"source": name, "lang": target, "text": text}
                    )
                    count += 1
                if count >= limit:
                    break
        except Exception as e:
            print(f"Skipping {name} {target}: {e}")


def fetch_data_source(source_config):
    """Dispatcher that routes to the appropriate fetcher based on source type"""
    source_type = source_config["type"]

    if source_type == "web_scraper":
        fetch_web_scraper(source_config)
    elif source_type == "nltk_corpus":
        fetch_nltk_corpus(source_config)
    elif source_type == "hf_dataset":
        fetch_hf_dataset(source_config)
    elif source_type == "hf_dataset_paired":
        fetch_hf_dataset_paired(source_config)
    else:
        print(f"Unknown source type: {source_type}")


# Gather data from all configured sources
for source_config in DATA_SOURCES:
    fetch_data_source(source_config)

# Save to file

print(f"Saving {len(collected_data)} text samples to {OUTPUT_FILE} . . .")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in collected_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Done.")
