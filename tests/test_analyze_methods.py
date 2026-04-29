import json
import tempfile
import unittest
from pathlib import Path

from src.analyze_methods import (
    analyze_run,
    build_candidate_summary,
    compact_config_label,
    compute_winners,
    display_model_name,
    filter_f1_plot_candidates,
    load_run_data,
    select_scatter_annotations,
    select_best_rows,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("rows must not be empty")
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row.get(header, "")) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AnalyzeMethodsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.run_dir = self.root / "flores_foreign_words_run_all"
        self.output_dir = self.root / "analysis_methods"
        self.run_dir.mkdir()
        self._create_run_fixture()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_run_fixture(self):
        write_json(
            self.run_dir / "run_metadata.json",
            {
                "models": [
                    {"name": "model_a", "family": "baseline"},
                    {"name": "model_b", "family": "baseline"},
                    {"name": "model_c", "family": "baseline"},
                ],
                "window_decision_modes": ["legacy_window", "contextual_hybrid"],
                "args": {"skip_pure": False, "skip_window": False, "only_window": False},
            },
        )
        write_csv(
            self.run_dir / "injected_detection_metrics.csv",
            [
                {
                    "model": "model_a",
                    "model_family": "baseline",
                    "injected_lang": "en",
                    "f1": 0.72,
                    "recall": 0.82,
                    "precision": 0.64,
                    "false_positive_rate": 0.15,
                    "accuracy": 0.79,
                },
                {
                    "model": "model_b",
                    "model_family": "baseline",
                    "injected_lang": "en",
                    "f1": 0.79,
                    "recall": 0.86,
                    "precision": 0.73,
                    "false_positive_rate": 0.12,
                    "accuracy": 0.83,
                },
                {
                    "model": "model_c",
                    "model_family": "baseline",
                    "injected_lang": "en",
                    "f1": 0.55,
                    "recall": 0.60,
                    "precision": 0.51,
                    "false_positive_rate": 0.22,
                    "accuracy": 0.68,
                },
            ],
        )
        write_csv(
            self.run_dir / "phrase_detection_metrics.csv",
            [
                {
                    "model": "model_a",
                    "model_family": "baseline",
                    "injected_lang": "en",
                    "f1": 0.69,
                    "recall": 0.78,
                    "precision": 0.62,
                    "false_positive_rate": 0.16,
                    "accuracy": 0.77,
                },
                {
                    "model": "model_b",
                    "model_family": "baseline",
                    "injected_lang": "en",
                    "f1": 0.77,
                    "recall": 0.84,
                    "precision": 0.71,
                    "false_positive_rate": 0.13,
                    "accuracy": 0.82,
                },
                {
                    "model": "model_c",
                    "model_family": "baseline",
                    "injected_lang": "en",
                    "f1": 0.52,
                    "recall": 0.58,
                    "precision": 0.47,
                    "false_positive_rate": 0.24,
                    "accuracy": 0.66,
                },
            ],
        )
        write_csv(
            self.run_dir / "injected_window_detection_metrics.csv",
            [
                {
                    "model": "model_a",
                    "model_family": "window",
                    "injected_lang": "en",
                    "f1": 0.82,
                    "recall": 0.88,
                    "precision": 0.77,
                    "false_positive_rate": 0.10,
                    "accuracy": 0.86,
                    "window_decision_rule": "legacy_window",
                    "window_size": 2,
                    "window_foreign_threshold": 0.7,
                    "window_shared_foreign_threshold": "",
                },
                {
                    "model": "model_a",
                    "model_family": "window",
                    "injected_lang": "fr",
                    "f1": 0.80,
                    "recall": 0.90,
                    "precision": 0.72,
                    "false_positive_rate": 0.09,
                    "accuracy": 0.85,
                    "window_decision_rule": "legacy_window",
                    "window_size": 3,
                    "window_foreign_threshold": 0.7,
                    "window_shared_foreign_threshold": "",
                },
                {
                    "model": "model_b",
                    "model_family": "window",
                    "injected_lang": "en",
                    "f1": 0.81,
                    "recall": 0.87,
                    "precision": 0.76,
                    "false_positive_rate": 0.10,
                    "accuracy": 0.85,
                    "window_decision_rule": "legacy_window",
                    "window_size": 2,
                    "window_foreign_threshold": 0.7,
                    "window_shared_foreign_threshold": "",
                },
                {
                    "model": "model_b",
                    "model_family": "window",
                    "injected_lang": "fr",
                    "f1": 0.83,
                    "recall": 0.86,
                    "precision": 0.80,
                    "false_positive_rate": 0.08,
                    "accuracy": 0.87,
                    "window_decision_rule": "contextual_hybrid",
                    "window_size": 3,
                    "window_foreign_threshold": 0.2,
                    "window_shared_foreign_threshold": 0.15,
                },
            ],
        )
        write_csv(
            self.run_dir / "phrase_window_detection_metrics.csv",
            [
                {
                    "model": "model_a",
                    "model_family": "window",
                    "injected_lang": "en",
                    "f1": 0.86,
                    "recall": 0.90,
                    "precision": 0.82,
                    "false_positive_rate": 0.09,
                    "accuracy": 0.88,
                    "window_decision_rule": "legacy_window",
                    "window_size": 3,
                    "window_foreign_threshold": 0.7,
                    "window_shared_foreign_threshold": "",
                },
                {
                    "model": "model_a",
                    "model_family": "window",
                    "injected_lang": "fr",
                    "f1": 0.84,
                    "recall": 0.92,
                    "precision": 0.77,
                    "false_positive_rate": 0.10,
                    "accuracy": 0.87,
                    "window_decision_rule": "legacy_window",
                    "window_size": 2,
                    "window_foreign_threshold": 0.7,
                    "window_shared_foreign_threshold": "",
                },
                {
                    "model": "model_b",
                    "model_family": "window",
                    "injected_lang": "en",
                    "f1": 0.84,
                    "recall": 0.92,
                    "precision": 0.78,
                    "false_positive_rate": 0.10,
                    "accuracy": 0.87,
                    "window_decision_rule": "legacy_window",
                    "window_size": 2,
                    "window_foreign_threshold": 0.7,
                    "window_shared_foreign_threshold": "",
                },
                {
                    "model": "model_b",
                    "model_family": "window",
                    "injected_lang": "fr",
                    "f1": 0.86,
                    "recall": 0.89,
                    "precision": 0.83,
                    "false_positive_rate": 0.07,
                    "accuracy": 0.89,
                    "window_decision_rule": "contextual_hybrid",
                    "window_size": 3,
                    "window_foreign_threshold": 0.2,
                    "window_shared_foreign_threshold": 0.15,
                },
            ],
        )
        write_csv(
            self.run_dir / "pure_foreign_detection_metrics.csv",
            [
                {
                    "model": "model_c",
                    "model_family": "baseline",
                    "true_lang": "es",
                    "foreign_false_positive_rate": 0.0,
                    "accuracy": 1.0,
                    "confidence_mean": 1.0,
                }
            ],
        )

    def test_load_run_data_separates_standard_window_and_pure(self):
        loaded = load_run_data(self.run_dir)
        self.assertEqual(set(loaded["standard_df"]["task_name"]), {"injected", "phrase"})
        self.assertEqual(set(loaded["window_df"]["task_name"]), {"injected", "phrase"})
        self.assertEqual(set(loaded["pure_df"]["scope"]), {"pure"})

    def test_display_model_name_shortens_known_models(self):
        self.assertEqual(
            display_model_name("facebook-fasttext-language-identification"), "fasttext"
        )
        self.assertEqual(display_model_name("lingua-spanish-only"), "lingua-es")
        self.assertEqual(display_model_name("custom-model"), "custom-model")

    def test_select_best_rows_keeps_best_window_config_per_model_per_task(self):
        loaded = load_run_data(self.run_dir)
        best = select_best_rows(loaded["window_df"])
        model_b_injected = best[(best["model"] == "model_b") & (best["task_name"] == "injected")].iloc[0]
        self.assertEqual(model_b_injected["window_decision_rule"], "contextual_hybrid")
        self.assertEqual(model_b_injected["window_size"], 3)

    def test_compute_winners_uses_injected_and_phrase_not_pure_metrics(self):
        loaded = load_run_data(self.run_dir)
        candidate_summary = build_candidate_summary(
            loaded["standard_df"], loaded["window_df"]
        )
        self.assertIn("candidate_display_label", candidate_summary.columns)
        self.assertIn("model_display", candidate_summary.columns)
        self.assertIn("candidate_axis_label", candidate_summary.columns)
        winners_df, overall_df = compute_winners(candidate_summary)

        injected_winner = winners_df[winners_df["winner_type"] == "injected"].iloc[0]
        phrase_winner = winners_df[winners_df["winner_type"] == "phrase"].iloc[0]
        overall_winner = winners_df[winners_df["winner_type"] == "overall"].iloc[0]

        self.assertEqual(injected_winner["model"], "model_b")
        self.assertEqual(injected_winner["candidate_scope"], "window")
        self.assertEqual(phrase_winner["model"], "model_a")
        self.assertEqual(phrase_winner["candidate_scope"], "window")
        self.assertEqual(overall_winner["model"], "model_b")
        self.assertEqual(overall_winner["candidate_scope"], "window")
        self.assertNotIn("model_c", set(overall_df.head(2)["model"]))

    def test_plot_helpers_reduce_density_and_format_configs(self):
        loaded = load_run_data(self.run_dir)
        candidate_summary = build_candidate_summary(
            loaded["standard_df"], loaded["window_df"]
        )
        winners_df, _ = compute_winners(candidate_summary)
        filtered = filter_f1_plot_candidates(candidate_summary, top_per_scope=1)
        self.assertLessEqual(filtered["candidate_key"].nunique(), 2)
        annotated = select_scatter_annotations(candidate_summary, winners_df)
        self.assertLessEqual(annotated["candidate_key"].nunique(), 4)
        self.assertEqual(
            compact_config_label("legacy_window, w=2, t=0.7"), "legacy w=2 t=0.7"
        )

    def test_analyze_run_writes_reduced_outputs_and_plots(self):
        result = analyze_run(self.run_dir, output_dir=self.output_dir)
        self.assertTrue((self.output_dir / "candidate_summary.csv").exists())
        self.assertTrue((self.output_dir / "best_techniques.csv").exists())
        self.assertTrue((self.output_dir / "report.md").exists())
        self.assertTrue((self.output_dir / "report.html").exists())
        self.assertTrue((self.output_dir / "plots" / "f1_comparison.png").exists())
        self.assertTrue((self.output_dir / "plots" / "precision_recall_scatter.png").exists())
        self.assertTrue((self.output_dir / "plots" / "winner_summary.png").exists())
        self.assertEqual(len(result["plots"]), 3)
        report = (self.output_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("model-a", report.lower().replace("_", "-"))


if __name__ == "__main__":
    unittest.main()
