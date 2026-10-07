"""Verify the V4.1 Flash API contract without contacting the provider."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from shopping_grpo.evaluation.model_client import (
    DEFAULT_FLASH_MODEL,
    DEFAULT_JUDGE_MODEL,
    OpenAIJSONClient,
    ModelResponseError,
)
from shopping_grpo.evaluation.rollout import client_from_env


class DeepSeekFlashClientTest(unittest.TestCase):
    def test_curator_and_judge_share_the_official_model_id(self):
        self.assertEqual(DEFAULT_FLASH_MODEL, "deepseek-flash")
        self.assertEqual(DEFAULT_JUDGE_MODEL, DEFAULT_FLASH_MODEL)

    def test_rollout_defaults_route_to_flash_without_model_or_url_overrides(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}, clear=True):
            client = client_from_env()
        self.assertEqual(client.model, DEFAULT_FLASH_MODEL)
        self.assertEqual(client.base_url, "https://api.deepseek.com/v1")

    def _complete(self, *, thinking):
        calls = []

        def transport(url, payload, headers, timeout):
            calls.append(payload)
            return {
                "model": "deepseek-flash",
                "choices": [{"message": {"content": '{"accepted": true}'}}],
            }

        client = OpenAIJSONClient(
            model=DEFAULT_FLASH_MODEL,
            base_url="https://provider.test/v1",
            api_key="test-api-key",
            thinking=thinking,
            reasoning_effort="max",
            transport=transport,
        )
        response = client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertNotIn("test-api-key", str(response))
        return calls[0], response

    def test_non_thinking_flash_explicitly_disables_provider_default(self):
        payload, response = self._complete(thinking=False)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(response["result"], {"accepted": True})

    def test_thinking_flash_preserves_effort_without_sampling_arguments(self):
        payload, response = self._complete(thinking=True)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(response["metadata"]["requested_model"], "deepseek-flash")


class JSONResponseRetryTest(unittest.TestCase):
    def _client(self, responses, *, retries=2):
        self.calls = []
        remaining = iter(responses)

        def transport(url, payload, headers, timeout):
            self.calls.append(deepcopy(payload))
            return next(remaining)

        return OpenAIJSONClient(
            model=DEFAULT_FLASH_MODEL,
            base_url="https://provider.test/v1",
            api_key="test-api-key",
            retries=retries,
            retry_delay_seconds=0,
            transport=transport,
        )

    @staticmethod
    def _response(content, *, finish_reason="stop"):
        return {"choices": [{"message": {"content": content},
                            "finish_reason": finish_reason}]}

    def test_malformed_json_retries_before_returning_valid_result(self):
        client = self._client([
            self._response('{"accepted": true trailing}'),
            self._response('{"accepted": true}'),
        ])
        result = client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertEqual(result["result"], {"accepted": True})
        self.assertEqual(result["metadata"]["attempts"], 2)
        self.assertEqual(len(result["metadata"]["retry_response_errors"]), 1)
        self.assertEqual(self.calls[0], self.calls[1])
        self.assertNotIn("test-api-key", str(result))
        self.assertNotIn("trailing", str(result))

    def test_invalid_envelope_and_root_retry_with_one_shared_budget(self):
        client = self._client([{}, self._response("[]"), self._response("{}")])
        result = client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertEqual(result["metadata"]["attempts"], 3)
        self.assertEqual(len(result["metadata"]["retry_response_errors"]), 2)

    def test_exhausted_json_retries_raise_without_fabricating_a_score(self):
        client = self._client([self._response("broken")] * 3)
        with self.assertRaisesRegex(ModelResponseError, "not strict JSON"):
            client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertEqual(len(self.calls), 3)

    def test_truncated_content_is_reported_without_returning_partial_json(self):
        client = self._client([
            self._response('{"accepted":', finish_reason="length"),
            self._response('{"accepted": true}'),
        ])
        result = client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertIn("finish_reason=length", result["metadata"]["retry_response_errors"][0])

    def test_response_retry_uses_exponential_backoff(self):
        client = self._client([self._response("bad"), self._response("bad"),
                               self._response("{}")])
        client.retry_delay_seconds = 2
        with patch("shopping_grpo.evaluation.model_client.time.sleep") as sleep:
            result = client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        self.assertEqual(result["metadata"]["retry_wait_seconds"], 6)

    def test_truncation_retries_even_when_partial_content_is_valid_json(self):
        client = self._client([self._response("{}", finish_reason="length"), self._response('{"ok":true}')])
        result = client.complete_json([{"role": "user", "content": "Return JSON."}])
        self.assertEqual(result["result"], {"ok": True})
        self.assertEqual(self.calls[1]["max_tokens"], 2 * self.calls[0]["max_tokens"])
        self.assertEqual(self.calls[0]["response_format"], {"type": "json_object"})

    def test_duplicate_keys_and_non_finite_numbers_are_not_strict_json(self):
        for content in ['{"score":0,"score":2}', '{"score":NaN}', '{"score":Infinity}']:
            with self.subTest(content=content):
                client = self._client([self._response(content)], retries=0)
                with self.assertRaises(ModelResponseError):
                    client.complete_json([{"role": "user", "content": "Return JSON."}])

    def test_schema_failures_share_the_bounded_retry_budget(self):
        def validate(value):
            if value.get("score") not in (0, 1, 2):
                raise ValueError("private invalid payload")
            return value
        client = self._client([self._response('{"score":9}'), self._response('{"score":2}')])
        result = client.complete_json([{"role": "user", "content": "Return JSON."}], validator=validate)
        self.assertEqual(result["result"], {"score": 2})
        self.assertEqual(result["metadata"]["attempts"], 2)
        self.assertNotIn("private invalid payload", str(result))


if __name__ == "__main__":
    unittest.main()
