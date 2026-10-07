import json
import unittest

from shopping_grpo.collection.corrections import (
    CorrectiveReplayClient, generation_rollouts, mask_recovery_prefix, mine_training_failures,
)


def rollout(uid, index, task, utility, kind="gold_purchase", **extra):
    return {"uid": uid, "rollout_index": index, "task_id": task, "done": True,
            "reward_valid": True, "reward_version": "shopsimulator-reward-v3",
            "reward_type": kind, "termination_reason": kind, "infrastructure_invalid": False,
            "actions": [{"tool": "buy_now", "parameters": {}}],
            "reward": {"total": utility, "terminal_utility": utility,
                       "purchase_success": kind == "gold_purchase", "r_option": utility,
                       "sampling_invalid": False}, **extra}


class TrainingCorrectionsTest(unittest.TestCase):
    def test_mining_deduplicates_and_only_boosts_valid_signal(self):
        rows = [rollout("a", 0, 1, .1, "partial_alternative_purchase"),
                rollout("a", 1, 1, 1), rollout("b", 0, 2, 1), rollout("b", 1, 2, 1),
                rollout("c", 0, 3, 0, "wrong_purchase", infrastructure_invalid=True),
                rollout("c", 1, 3, 1), rollout("eval", 0, 9, .1, "wrong_purchase")]
        events = [{"event": "generation_batch", "rollouts": rows},
                  {"event": "optimizer_step", "rollouts": rows}]
        self.assertEqual(len(list(generation_rollouts(events))), len(rows))
        plans, weights = mine_training_failures(events, {1, 2, 3, 4}, {9})
        self.assertEqual(weights, {1: 3, 2: 1, 3: 1, 4: 1})
        self.assertEqual([p["task_id"] for p in plans], [1])
        with self.assertRaisesRegex(ValueError, "overlap"):
            mine_training_failures(events, {1, 9}, {9})

    def test_wrong_option_is_injected_only_on_same_visible_axis_and_masked(self):
        opening = {"tool": "open_product", "parameters": {"asin": "123456789012"}}
        correct = {"tool": "select_option", "parameters": {"value": "blue"}}
        wrong = {"tool": "select_option", "parameters": {"value": "red"}}
        plan = {"actions": [opening, correct], "failures": [
            {"tags": ["specification_matching"], "actions": [opening, wrong]}]}
        client = CorrectiveReplayClient(plan)
        messages = [{"role": "user", "content": "buy blue"}]
        messages.append(client.complete(messages, []))
        messages.append({"role": "tool", "content":
                         'available_options: {"color": ["blue", "red"]}\n\n'
                         '搜索功能是否可用: False\n可点击的按钮: ["red", "blue", "buy now"]'})
        bad = client.complete(messages, [])
        self.assertEqual(json.loads(bad["tool_calls"][0]["function"]["arguments"]), {"value": "red"})
        messages.append(bad)
        messages.append({"role": "tool", "content": "red selected"})
        messages.append(client.complete(messages, []))
        row = {"messages": messages}
        mask_recovery_prefix(row, client.injected_call_id)
        self.assertFalse(messages[1]["trainable"])
        self.assertFalse(messages[3]["trainable"])
        self.assertNotIn("trainable", messages[-1])
        self.assertEqual(json.loads(messages[-1]["tool_calls"][0]["function"]["arguments"]), {"value": "blue"})


if __name__ == "__main__":
    unittest.main()
