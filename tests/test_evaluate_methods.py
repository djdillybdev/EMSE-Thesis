import unittest
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

from src.evaluate_methods import (
    LinguaSpanishOnlyAdapter,
    WindowScore,
    build_window_rows_and_token_scores,
    build_window_score,
    run_pure_evaluation,
    tokenize,
)


class FakeWindowModel:
    def __init__(self, score_map):
        self.name = "fake-window-model"
        self.family = "test"
        self.score_map = score_map

    def score_against_main_lang(self, text, main_lang, top_k):
        main_score, foreign_score = self.score_map[text]
        predicted_lang = main_lang if main_score >= foreign_score else "fr"
        confidence = max(main_score, foreign_score)
        return WindowScore(
            model=self.name,
            model_family=self.family,
            predicted_lang=predicted_lang,
            predicted_label=predicted_lang,
            confidence=confidence,
            main_lang=main_lang,
            main_lang_score=main_score,
            foreign_score=foreign_score,
            top_non_main_lang="fr" if predicted_lang == main_lang else main_lang,
            top_non_main_confidence=foreign_score,
        )

    def predict(self, text):
        main_score, foreign_score = self.score_map[text]
        predicted_lang = "es" if main_score >= foreign_score else "fr"
        confidence = max(main_score, foreign_score)
        return SimpleNamespace(
            model=self.name,
            model_family=self.family,
            predicted_lang=predicted_lang,
            predicted_label=predicted_lang,
            confidence=confidence,
        )


class FakeLabelScoreModel:
    def __init__(self, label_score_map):
        self.name = "fake-label-score-model"
        self.family = "test"
        self.label_score_map = label_score_map

    def score_against_main_lang(self, text, main_lang, top_k):
        return build_window_score(self, main_lang, self.label_score_map[text][:top_k])


class FakeBinaryDetector:
    def __init__(self, detected_language):
        self.detected_language = detected_language

    def detect_language_of(self, _text):
        return self.detected_language


def collect_rows(text, score_map, **kwargs):
    tokens = tokenize(text)
    rows = list(
        build_window_rows_and_token_scores(
            sample={"sample_id": "sample-1"},
            tokens=tokens,
            model=FakeWindowModel(score_map),
            main_lang="es",
            window_sizes=kwargs.pop("window_sizes", [2]),
            thresholds=kwargs.pop("thresholds", [0.2]),
            top_k=5,
            base_extra={"true_lang": "es"},
            token_truth=kwargs.pop("token_truth", lambda _token_index: False),
            include_window_rows=False,
            window_row_writer=None,
            decision_modes=kwargs.pop(
                "decision_modes", ["legacy_window", "contextual_hybrid"]
            ),
            contextual_thresholds=kwargs.pop("contextual_thresholds", [0.5]),
            shared_foreign_thresholds=kwargs.pop("shared_foreign_thresholds", [0.3]),
            shared_foreign_min_window_count=kwargs.pop(
                "shared_foreign_min_window_count", 1
            ),
            shared_foreign_min_ratio=kwargs.pop("shared_foreign_min_ratio", 0.5),
            **kwargs,
        )
    )
    return rows


class WindowDecisionTests(unittest.TestCase):
    def test_build_window_score_uses_strongest_non_main_confidence(self):
        model = SimpleNamespace(name="fake-model", family="test")
        score = build_window_score(
            model,
            "es",
            [
                ("por", "pt", 0.9),
                ("spa", "es", 0.8),
                ("eng", "en", 0.05),
            ],
        )

        self.assertEqual(score.predicted_lang, "pt")
        self.assertEqual(score.main_lang_score, 0.8)
        self.assertEqual(score.top_non_main_confidence, 0.9)
        self.assertEqual(score.foreign_score, 0.9)

    def test_build_window_score_keeps_normalized_behavior_when_non_main_is_lower(self):
        model = SimpleNamespace(name="fake-model", family="test")
        score = build_window_score(
            model,
            "es",
            [
                ("spa", "es", 0.8),
                ("por", "pt", 0.15),
                ("eng", "en", 0.05),
            ],
        )

        self.assertEqual(score.main_lang_score, 0.8)
        self.assertEqual(score.top_non_main_confidence, 0.15)
        self.assertAlmostEqual(score.foreign_score, 0.2)

    def test_build_window_score_empty_input_is_unchanged(self):
        model = SimpleNamespace(name="fake-model", family="test")
        score = build_window_score(model, "es", [])

        self.assertEqual(score.predicted_lang, "unknown")
        self.assertEqual(score.main_lang_score, 0.0)
        self.assertEqual(score.foreign_score, 1.0)
        self.assertEqual(score.top_non_main_confidence, 0.0)

    def test_binary_adapter_score_against_main_lang_is_unchanged(self):
        adapter = LinguaSpanishOnlyAdapter(
            "binary-model",
            "lingua_binary",
            FakeBinaryDetector("SPANISH"),
        )
        score = adapter.score_against_main_lang("hola", "es", 5)

        self.assertEqual(score.main_lang_score, 1.0)
        self.assertEqual(score.foreign_score, 0.0)
        self.assertEqual(score.top_non_main_confidence, 0.0)

    def test_window_rows_use_updated_multiclass_foreign_score(self):
        tokens = tokenize("hola amigo")
        rows = list(
            build_window_rows_and_token_scores(
                sample={"sample_id": "sample-1"},
                tokens=tokens,
                model=FakeLabelScoreModel(
                    {
                        "hola": [("spa", "es", 0.9), ("por", "pt", 0.1)],
                        "amigo": [("spa", "es", 0.85), ("por", "pt", 0.15)],
                        "hola amigo": [("por", "pt", 0.9), ("spa", "es", 0.8)],
                    }
                ),
                main_lang="es",
                window_sizes=[2],
                thresholds=[0.2],
                top_k=5,
                base_extra={"true_lang": "es"},
                token_truth=lambda _token_index: False,
                include_window_rows=False,
                window_row_writer=None,
                decision_modes=["legacy_window"],
            )
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["foreign_probability"] == 0.9 for row in rows))

    def test_legacy_window_uses_foreign_probability_threshold(self):
        rows = collect_rows(
            "hola amigo",
            {
                "hola": (0.9, 0.1),
                "amigo": (0.85, 0.15),
                "hola amigo": (0.8, 0.2),
            },
            decision_modes=["legacy_window"],
            thresholds=[0.2],
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["is_foreign_predicted"] for row in rows))

    def test_contextual_hybrid_flags_isolated_foreign_token(self):
        rows = collect_rows(
            "hola bonjour amigo",
            {
                "hola": (0.9, 0.1),
                "bonjour": (0.1, 0.9),
                "amigo": (0.9, 0.1),
                "hola bonjour amigo": (0.45, 0.55),
            },
            window_sizes=[3],
            decision_modes=["contextual_hybrid"],
            contextual_thresholds=[0.5],
            shared_foreign_thresholds=[0.3],
        )

        predictions = {row["normalized_token"]: row["is_foreign_predicted"] for row in rows}
        self.assertEqual(predictions, {"hola": False, "bonjour": True, "amigo": False})

    def test_contextual_hybrid_uses_consensus_for_foreign_phrase(self):
        rows = collect_rows(
            "hola bonjour salut amigo",
            {
                "hola": (0.9, 0.1),
                "bonjour": (0.45, 0.55),
                "salut": (0.45, 0.55),
                "amigo": (0.9, 0.1),
                "hola bonjour": (0.6, 0.4),
                "bonjour salut": (0.2, 0.8),
                "salut amigo": (0.6, 0.4),
            },
            window_sizes=[2],
            decision_modes=["contextual_hybrid"],
            contextual_thresholds=[1.0],
            shared_foreign_thresholds=[0.2],
        )

        predictions = {row["normalized_token"]: row["is_foreign_predicted"] for row in rows}
        run_lengths = {row["normalized_token"]: row["consensus_run_length"] for row in rows}
        self.assertEqual(
            predictions,
            {"hola": False, "bonjour": True, "salut": True, "amigo": False},
        )
        self.assertEqual(run_lengths["bonjour"], 2)
        self.assertEqual(run_lengths["salut"], 2)

    def test_contextual_hybrid_keeps_main_language_token_inside_foreign_run(self):
        rows = collect_rows(
            "bonjour hola salut",
            {
                "bonjour": (0.1, 0.9),
                "hola": (0.9, 0.1),
                "salut": (0.1, 0.9),
                "bonjour hola salut": (0.2, 0.8),
            },
            window_sizes=[3],
            decision_modes=["contextual_hybrid"],
            contextual_thresholds=[0.5],
            shared_foreign_thresholds=[0.3],
        )

        predictions = {row["normalized_token"]: row["is_foreign_predicted"] for row in rows}
        self.assertEqual(predictions, {"bonjour": True, "hola": False, "salut": True})

    def test_contextual_hybrid_uses_real_supporting_window_counts(self):
        rows = collect_rows(
            "hola bonjour salut bonsoir amigo",
            {
                "hola": (0.9, 0.1),
                "bonjour": (0.45, 0.55),
                "salut": (0.45, 0.55),
                "bonsoir": (0.45, 0.55),
                "amigo": (0.9, 0.1),
                "hola bonjour salut": (0.2, 0.8),
                "bonjour salut bonsoir": (0.2, 0.8),
                "salut bonsoir amigo": (0.2, 0.8),
            },
            window_sizes=[3],
            decision_modes=["contextual_hybrid"],
            contextual_thresholds=[1.0],
            shared_foreign_thresholds=[0.2],
        )

        row_by_token = {row["normalized_token"]: row for row in rows}
        self.assertEqual(row_by_token["salut"]["shared_foreign_window_count"], 3)
        self.assertEqual(row_by_token["salut"]["shared_foreign_window_ratio"], 1.0)

    def test_run_pure_evaluation_uses_ground_truth_language_for_window_and_word(self):
        sample = SimpleNamespace(
            sample_id="sample-1",
            row_index=0,
            flores_config="fra_Latn",
            lang="fr",
            text="bonjour salut",
        )
        model = FakeWindowModel(
            {
                "bonjour salut": (0.8, 0.2),
                "bonjour": (0.2, 0.8),
                "salut": (0.2, 0.8),
            }
        )
        args = SimpleNamespace(
            output_dir="unused",
            only_window=False,
            skip_window=False,
            save_raw_level="none",
            save_window_raw="none",
            window_sizes="2",
            window_decision_modes="legacy_window",
            window_foreign_threshold="0.2",
            window_contextual_threshold="0.5",
            window_shared_foreign_threshold="0.3",
            window_shared_foreign_min_window_count=1,
            window_shared_foreign_min_ratio=0.5,
            window_top_k=5,
        )
        captured_main_langs = []
        original_builder = build_window_rows_and_token_scores

        def capture_builder(*args, **kwargs):
            captured_main_langs.append(kwargs["main_lang"])
            yield from original_builder(*args, **kwargs)

        with TemporaryDirectory() as tmpdir, patch(
            "src.evaluate_methods.build_window_rows_and_token_scores",
            side_effect=capture_builder,
        ):
            args.output_dir = tmpdir
            pure_metrics, pure_window_metrics = run_pure_evaluation(
                args,
                [model],
                [sample],
                supported_langs={model.name: None},
            )

        self.assertEqual(captured_main_langs, ["fr"])
        word_metric = next(
            row for row in pure_metrics if row["evaluation"] == "pure_word"
        )
        self.assertEqual(word_metric["foreign_false_positive_rate"], 0.0)
        self.assertEqual(len(pure_window_metrics), 1)


if __name__ == "__main__":
    unittest.main()
