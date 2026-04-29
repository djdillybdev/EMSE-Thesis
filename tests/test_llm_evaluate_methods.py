import sys
import unittest
from types import SimpleNamespace

sys.modules.setdefault(
    "fasttext",
    SimpleNamespace(FastText=SimpleNamespace(eprint=lambda _message: None)),
)
sys.modules.setdefault("datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: []))
sys.modules.setdefault(
    "huggingface_hub",
    SimpleNamespace(hf_hub_download=lambda *args, **kwargs: ""),
)
sys.modules.setdefault(
    "lingua",
    SimpleNamespace(
        Language=SimpleNamespace(
            SPANISH="SPANISH",
            ENGLISH="ENGLISH",
            PORTUGUESE="PORTUGUESE",
            ITALIAN="ITALIAN",
            FRENCH="FRENCH",
            GERMAN="GERMAN",
            CATALAN="CATALAN",
            BASQUE="BASQUE",
            all=lambda: [],
        ),
        LanguageDetectorBuilder=SimpleNamespace(
            from_languages=lambda *args: SimpleNamespace(build=lambda: None)
        ),
    ),
)

from src.evaluate_methods import tokenize
from src.llm_evaluate_methods import (
    NormalizedForeignPrediction,
    NormalizedTokenPrediction,
    aggregate_main_language,
    align_foreign_token_predictions,
    align_token_predictions,
    build_nonoverlap_chunks,
)


class LlmEvaluateMethodsTests(unittest.TestCase):
    def test_build_nonoverlap_chunks_keeps_short_tail(self):
        tokens = tokenize("hola amigo bonito")

        chunks = build_nonoverlap_chunks("hola amigo bonito", tokens, 2)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "hola amigo")
        self.assertEqual(chunks[1]["text"], "bonito")
        self.assertEqual([token.raw_index for token in chunks[1]["tokens"]], [2])

    def test_build_nonoverlap_chunks_preserves_text_when_no_tokens(self):
        chunks = build_nonoverlap_chunks("123 !!!", [], 3)

        self.assertEqual(
            chunks,
            [{"chunk_index": 0, "chunk_size": 3, "tokens": [], "text": "123 !!!"}],
        )

    def test_aggregate_main_language_uses_first_chunk_tiebreak(self):
        lang = aggregate_main_language(
            [
                {"main_language": "en"},
                {"main_language": "es"},
                {"main_language": "en"},
                {"main_language": "es"},
            ]
        )
        self.assertEqual(lang, "en")

        lang = aggregate_main_language(
            [
                {"main_language": "fr"},
                {"main_language": "de"},
            ]
        )
        self.assertEqual(lang, "fr")

    def test_align_foreign_token_predictions_claims_duplicate_surface_forms_left_to_right(self):
        tokens = tokenize("bonjour bonjour hola")
        predictions = [
            NormalizedForeignPrediction(token="bonjour", language="fr", confidence=0.9),
            NormalizedForeignPrediction(token="bonjour", language="fr", confidence=0.8),
        ]

        aligned = align_foreign_token_predictions(predictions, tokens)

        self.assertEqual(sorted(aligned.keys()), [0, 1])
        self.assertEqual(aligned[0]["predicted_lang"], "fr")
        self.assertEqual(aligned[1]["predicted_lang"], "fr")

    def test_align_token_predictions_matches_stripped_forms(self):
        tokens = tokenize("hola, amigo")
        predictions = [
            NormalizedTokenPrediction(
                token="hola",
                language="es",
                is_foreign=False,
                confidence=0.9,
            ),
            NormalizedTokenPrediction(
                token="amigo",
                language="es",
                is_foreign=False,
                confidence=0.8,
            ),
        ]

        aligned = align_token_predictions(predictions, tokens)

        self.assertEqual(sorted(aligned.keys()), [0, 1])
        self.assertEqual(aligned[0]["token"], "hola,")
        self.assertEqual(aligned[1]["token"], "amigo")


if __name__ == "__main__":
    unittest.main()
