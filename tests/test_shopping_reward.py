"""Reward v3 validation and strict terminal-success regression tests."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state, record_action_attempt, reward_breakdown, validate_reward,
)


def reward_detail(reward_type="gold_purchase", utility=1.0, valid=True):
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type, "termination_reason": reward_type,
        "reward_valid": valid, "terminal_utility": utility,
        "purchase_success": reward_type in {"gold_purchase", "valid_alternative_purchase"},
        "sampling_invalid": not valid, "hard_gates": {},
        "weighted_score": 1.0, "evidence_coverage": 1.0,
    }


def terminal_state(reward_type="gold_purchase", utility=1.0, valid=True):
    state = make_runtime_state(task_id=1, max_steps=35)
    detail = validate_reward(reward_detail(reward_type, utility, valid))
    state.update(detail)
    state.update(done=True, terminal_result={"done": True, "over": True},
                 final_reward=utility, reward_detail=detail)
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_gold_purchase_uses_environment_utility_without_extra_shaping(self):
        result = reward_breakdown(terminal_state())
        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertEqual(result["efficiency"], 0.0)
        self.assertFalse(result["sampling_invalid"])

    def test_alternative_purchase_is_not_strict_success(self):
        result = reward_breakdown(terminal_state("valid_alternative_purchase", 0.55))
        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["total"], 0.55)

    def test_incomplete_or_invalid_gold_terminal_cannot_count_as_success(self):
        for change in (
            {"done": False}, {"terminal_result": {"done": True, "over": False}},
            {"reward_valid": False}, {"reward_valid": None},
            {"infrastructure_invalid": True}, {"final_reward": float("nan")},
        ):
            with self.subTest(change=change):
                state = terminal_state()
                state.update(change)
                result = reward_breakdown(state)
                self.assertEqual(result["full"], 0.0)
                self.assertEqual(result["strict"], 0.0)
                self.assertEqual(result["purchase_success"], 0.0)
                self.assertEqual(result["total"], 0.0)

    def test_unverifiable_terminal_is_excluded_from_sampling(self):
        result = reward_breakdown(terminal_state("reward_unverifiable", 0.0, False))
        self.assertTrue(result["sampling_invalid"])
        self.assertFalse(result["infrastructure_invalid"])
        self.assertEqual(result["total"], 0.0)

    def test_missing_reward_v3_is_infrastructure_invalid(self):
        result = reward_breakdown(make_runtime_state(1, 35))
        self.assertTrue(result["infrastructure_invalid"])
        self.assertEqual(result["strict"], 0.0)

    def test_reward_v3_rejects_malformed_or_unsupported_details(self):
        for change in (
            {"reward_version": "unsupported"}, {"terminal_utility": float("nan")},
            {"weighted_score": 1.1}, {"reward_valid": "true"},
            {"sampling_invalid": True}, {"purchase_success": None},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_reward({**reward_detail(), **change})

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
