import sys
import unittest
from types import SimpleNamespace
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

from src.evaluate_methods import tokenize
from src.llm_evaluate_methods import (
    NormalizedForeignPrediction,
    NormalizedTokenPrediction,
    align_foreign_token_predictions,
    align_token_predictions,
    call_ollama,
    call_with_retry,
)


class LlmEvaluateMethodsTests(unittest.TestCase):
    def test_call_ollama_uses_generate_endpoint(self):
        response_payload = {
            "response": '{"main_language":"es","foreign_tokens":[]}',
            "total_duration": 2_000_000,
            "load_duration": 500_000,
            "prompt_eval_count": 12,
            "prompt_eval_duration": 700_000,
            "eval_count": 8,
            "eval_duration": 900_000,
        }
        captured = {}

        class FakeResponse:
            def read(self):
                import json

                return json.dumps(response_payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            import json

            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with patch("src.llm_evaluate_methods.urlopen", side_effect=fake_urlopen):
            raw_text, _latency_ms, usage = call_ollama(
                model="llama3.2",
                prompt="Hola mundo",
                ollama_url="http://localhost:11434",
                temperature=0.0,
                timeout=45,
                keep_alive="10m",
            )

        self.assertEqual(captured["url"], "http://localhost:11434/api/generate")
        self.assertEqual(captured["timeout"], 45)
        self.assertEqual(captured["payload"]["prompt"], "Hola mundo")
        self.assertFalse(captured["payload"].get("stream"))
        self.assertNotIn("messages", captured["payload"])
        self.assertEqual(raw_text, response_payload["response"])
        self.assertEqual(usage["prompt_eval_count"], 12)
        self.assertEqual(usage["eval_count"], 8)

    def test_call_ollama_debug_logs_prompt_and_response(self):
        response_payload = {
            "response": '{"main_language":"es","foreign_tokens":[]}',
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }
        logged = []

        class FakeResponse:
            def read(self):
                import json

                return json.dumps(response_payload).encode("utf-8")

        with patch(
            "src.llm_evaluate_methods.urlopen", return_value=FakeResponse()
        ), patch("src.llm_evaluate_methods.log", side_effect=logged.append):
            call_ollama(
                model="llama3.2",
                prompt="Hola mundo",
                ollama_url="http://localhost:11434",
                temperature=0.0,
                timeout=45,
                keep_alive="10m",
                debug_llm=True,
                sample_label="pure:sample-1",
                attempt_label="transport_attempt_1",
            )

        combined_logs = "\n".join(logged)
        self.assertIn("LLM transport_attempt_1 sample=pure:sample-1", combined_logs)
        self.assertIn("Prompt:\nHola mundo", combined_logs)
        self.assertIn('Response:\n{"main_language":"es","foreign_tokens":[]}', combined_logs)

    def test_call_with_retry_retries_invalid_json(self):
        responses = iter(
            [
                ("not json", 10.0, {"total_duration_ms": 1.0, "load_duration_ms": 0.0, "prompt_eval_count": 1, "prompt_eval_duration_ms": 0.0, "eval_count": 1, "eval_duration_ms": 0.0}),
                ('{"main_language":"es","foreign_tokens":[]}', 12.0, {"total_duration_ms": 2.0, "load_duration_ms": 0.0, "prompt_eval_count": 2, "prompt_eval_duration_ms": 0.0, "eval_count": 2, "eval_duration_ms": 0.0}),
            ]
        )

        with patch(
            "src.llm_evaluate_methods.call_ollama",
            side_effect=lambda **_kwargs: next(responses),
        ):
            result = call_with_retry(
                model="llama3.2",
                prompt="Hola mundo",
                ollama_url="http://localhost:11434",
                temperature=0.0,
                timeout=45,
                retries=0,
                json_retries=1,
                retry_backoff_seconds=0.0,
                keep_alive="10m",
            )

        self.assertTrue(result.valid_json)
        self.assertEqual(result.json_retry_count, 1)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.parsed["main_language"], "es")
        self.assertEqual(result.prompt_eval_count, 2)

    def test_call_with_retry_logs_debug_errors(self):
        responses = iter(
            [
                ("not json", 10.0, {"total_duration_ms": 1.0, "load_duration_ms": 0.0, "prompt_eval_count": 1, "prompt_eval_duration_ms": 0.0, "eval_count": 1, "eval_duration_ms": 0.0}),
                ('{"main_language":"es","foreign_tokens":[]}', 12.0, {"total_duration_ms": 2.0, "load_duration_ms": 0.0, "prompt_eval_count": 2, "prompt_eval_duration_ms": 0.0, "eval_count": 2, "eval_duration_ms": 0.0}),
            ]
        )
        logged = []

        with patch(
            "src.llm_evaluate_methods.call_ollama",
            side_effect=lambda **_kwargs: next(responses),
        ), patch("src.llm_evaluate_methods.log", side_effect=logged.append):
            result = call_with_retry(
                model="llama3.2",
                prompt="Hola mundo",
                ollama_url="http://localhost:11434",
                temperature=0.0,
                timeout=45,
                retries=0,
                json_retries=1,
                retry_backoff_seconds=0.0,
                keep_alive="10m",
                debug_llm=True,
                sample_label="pure:sample-1",
            )

        self.assertTrue(result.valid_json)
        combined_logs = "\n".join(logged)
        self.assertIn("sample=pure:sample-1", combined_logs)
        self.assertIn("json_attempt_1_error", combined_logs)
        self.assertIn("Error:", combined_logs)

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
