import random
import unittest
from types import SimpleNamespace
import sys

sys.modules.setdefault("datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: []))

from src.prepare_data import (
    Sample,
    build_injected_sample,
    build_phrase_sample,
    is_eligible_injected_token,
    prepared_profile_manifest,
    tokenize,
)


class PrepareDataTests(unittest.TestCase):
    def test_is_eligible_injected_token_requires_lowercase_alpha_words(self):
        token = tokenize("hola")[0]
        self.assertTrue(is_eligible_injected_token(token, 2))

        for text in ("Nova", "CEO", "QVC", "Ebola", "tipo1", "A320", "www.test.com"):
            with self.subTest(text=text):
                token = tokenize(text)
                if token:
                    self.assertFalse(is_eligible_injected_token(token[0], 2))

    def test_build_injected_sample_skips_titlecase_acronym_and_numeric_tokens(self):
        spanish_sample = Sample(
            sample_id="spa:1",
            row_index=1,
            flores_config="spa_Latn",
            lang="es",
            text="Madrid hola NASA amigo 2024 casa",
            split="devtest",
        )
        foreign_sample = Sample(
            sample_id="eng:1",
            row_index=1,
            flores_config="eng_Latn",
            lang="en",
            text="Nova alpha CEO beta gamma A320",
            split="devtest",
        )

        injected = build_injected_sample(
            spanish_sample,
            foreign_sample,
            "en",
            ratio=1.0,
            min_word_length=2,
            rng=random.Random(0),
        )

        self.assertIsNotNone(injected)
        replacements = injected["injections"]
        self.assertEqual(
            {item["original_normalized"] for item in replacements},
            {"hola", "amigo", "casa"},
        )
        self.assertEqual(
            {item["replacement_normalized"] for item in replacements},
            {"alpha", "beta", "gamma"},
        )
        self.assertEqual(injected["contamination_type"], "random_token")

    def test_build_injected_sample_returns_none_without_clean_foreign_tokens(self):
        spanish_sample = Sample(
            sample_id="spa:2",
            row_index=2,
            flores_config="spa_Latn",
            lang="es",
            text="hola amigo casa",
            split="devtest",
        )
        foreign_sample = Sample(
            sample_id="eng:2",
            row_index=2,
            flores_config="eng_Latn",
            lang="en",
            text="Nova CEO QVC 2024",
            split="devtest",
        )

        injected = build_injected_sample(
            spanish_sample,
            foreign_sample,
            "en",
            ratio=0.5,
            min_word_length=2,
            rng=random.Random(0),
        )

        self.assertIsNone(injected)

    def test_build_phrase_sample_uses_only_clean_lowercase_spans(self):
        spanish_sample = Sample(
            sample_id="spa:3",
            row_index=3,
            flores_config="spa_Latn",
            lang="es",
            text="Madrid hola amigo NASA casa verde",
            split="devtest",
        )
        foreign_sample = Sample(
            sample_id="eng:3",
            row_index=3,
            flores_config="eng_Latn",
            lang="en",
            text="Nova alpha beta CEO gamma delta",
            split="devtest",
        )
        args = SimpleNamespace(
            phrase_replacement_ratio=0.5,
            phrase_span_min=2,
            phrase_span_max=2,
            injected_min_word_length=2,
        )

        phrase = build_phrase_sample(
            spanish_sample,
            foreign_sample,
            "en",
            args,
            random.Random(0),
        )

        self.assertIsNotNone(phrase)
        replacements = phrase["injections"]
        self.assertEqual(
            [item["original_normalized"] for item in replacements],
            ["casa", "verde"],
        )
        self.assertEqual(
            [item["replacement_normalized"] for item in replacements],
            ["gamma", "delta"],
        )

    def test_prepared_profile_manifest_records_stricter_injection_policy(self):
        args = SimpleNamespace(
            flores_dataset="openlanguagedata/flores_plus",
            split="devtest",
            flores_configs="spa_Latn,eng_Latn",
            limit_languages=None,
            limit_samples_per_language=10,
            spanish_config="spa_Latn",
            injection_configs="eng_Latn",
            seed=1,
            injection_ratio=0.15,
            injected_min_word_length=3,
            phrase_replacement_ratio=0.2,
            phrase_span_min=2,
            phrase_span_max=3,
        )

        manifest = prepared_profile_manifest(args)

        self.assertTrue(manifest["injected_lowercase_only"])
        self.assertTrue(manifest["injected_exclude_proper_noun_like_tokens"])
        self.assertEqual(manifest["injected_contamination_type"], "random_token")
        self.assertTrue(manifest["phrase_lowercase_only"])
        self.assertTrue(manifest["phrase_exclude_proper_noun_like_tokens"])


if __name__ == "__main__":
    unittest.main()
