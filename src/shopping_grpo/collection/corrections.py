"""Training-only failure mining and environment-verified corrective replay."""

from collections import defaultdict
from copy import deepcopy
import json

from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.training.grpo.dynamic_sampling import (
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


def generation_rollouts(events):
    """Optimizer events repeat rollouts; count each generated rollout only once."""
    seen = set()
    for event in events:
        if event.get("event") != "generation_batch":
            continue
        for row in event.get("rollouts", []):
            key = (row["uid"], row["rollout_index"])
            if key not in seen:
                seen.add(key)
                yield row


def strict_success(row):
    return (
        row.get("done") is True
        and row.get("reward_valid") is True
        and row.get("reward_version") == "shopsimulator-reward-v3"
        and row.get("reward_type") == "gold_purchase"
        and row.get("termination_reason") == "gold_purchase"
        and bool(row.get("reward", {}).get("purchase_success"))
        and not row.get("infrastructure_invalid")
        and not row.get("reward_unverifiable")
    )


def failure_tags(row):
    if (row.get("infrastructure_invalid") or row.get("reward_unverifiable")
            or row.get("reward", {}).get("sampling_invalid")):
        return set()
    tags = set()
    if (row.get("reward_valid") is True
            and row.get("reward_type") in {"wrong_purchase", "partial_alternative_purchase"}
            and row.get("reward", {}).get("r_option", 1) < 1):
        tags.add("specification_matching")
    if (row.get("guard_rejections", 0) > 0
            or row.get("termination_reason") in {"repeat_loop", "too_many_guard_rejections"}):
        tags.add("page_state_management")
    return tags


def mine_training_failures(events, train_ids, excluded_ids):
    """Keep coverage; boost only tasks with demonstrated valid reward variation."""
    train_ids, excluded_ids = set(train_ids), set(excluded_ids)
    if train_ids & excluded_ids:
        raise ValueError("training tasks overlap held-out tasks")
    tasks = defaultdict(list)
    groups = defaultdict(list)
    for row in generation_rollouts(events):
        task_id = int(row["task_id"])
        if task_id in train_ids:
            tasks[task_id].append(row)
            groups[row["uid"]].append(row)
    effective = defaultdict(int)
    for uid, rows in groups.items():
        if len({r["task_id"] for r in rows}) != 1:
            raise ValueError("one rollout group contains different tasks")
        utility, success, invalid, reasons = extract_shopping_group_signals(rows)
        _, stats = select_reward_varying_groups(
            [uid] * len(rows), [r["reward"]["total"] for r in rows],
            terminal_utilities=utility, purchase_success=success,
            sampling_invalid=invalid, sampling_invalid_reasons=reasons,
        )
        effective[int(rows[0]["task_id"])] += stats["kept_group_count"]
    plans, weights = [], {}
    for task_id in sorted(train_ids):
        rows = tasks[task_id]
        failures = [r for r in rows if failure_tags(r)]
        tags = set().union(*(failure_tags(r) for r in failures))
        # Uniform floor preserves unseen/all-equal tasks; no rewards are altered.
        weights[task_id] = 1 + int(effective[task_id] > 0) * (1 + int(bool(tags)))
        successes = [r for r in rows if strict_success(r)]
        if not failures or not successes:
            continue
        success = min(successes, key=lambda r: len(r["actions"]))
        plans.append({
            "task_id": task_id, "tags": sorted(tags),
            "success_uid": success["uid"], "success_rollout_index": success["rollout_index"],
            "actions": deepcopy(success["actions"]),
            "failures": [{"uid": r["uid"], "rollout_index": r["rollout_index"],
                          "actions": r["actions"], "tags": sorted(failure_tags(r))}
                         for r in failures],
        })
    # Page-state cases are rarer; interleave them with specification cases.
    page = [p for p in plans if "page_state_management" in p["tags"]]
    spec = [p for p in plans if "page_state_management" not in p["tags"]]
    ordered = []
    while page or spec:
        for bucket in (page, spec):
            if bucket:
                ordered.append(bucket.pop(0))
    return ordered, weights


class CorrectiveReplayClient:
    """Replay successful public actions, optionally recover a real wrong option.

    A wrong option is inserted only when it was observed on the same product
    in a failed rollout and belongs to the same currently visible option axis.
    That action and its preceding context are masked from SFT supervision.
    No goal options or hidden targets are read to choose actions.
    """

    def __init__(self, plan):
        self.actions = deepcopy(plan["actions"])
        self.bad_options = defaultdict(set)
        for failure in plan["failures"]:
            if "specification_matching" not in failure["tags"]:
                continue
            asin = None
            for action in failure["actions"]:
                if action["tool"] == "open_product":
                    asin = action["parameters"]["asin"]
                elif action["tool"] == "back_to_search":
                    asin = None
                elif action["tool"] == "select_option" and asin:
                    self.bad_options[asin].add(action["parameters"]["value"])
        self.position = 0
        self.call_index = 0
        self.injected_call_id = None
        self.current_asin = None

    def complete(self, messages, tools):
        del tools
        if self.position >= len(self.actions):
            raise ValueError("verified action sequence exhausted without a terminal result")
        action = self.actions[self.position]
        observation = messages[-1].get("content", "")
        injection = None
        if action["tool"] == "select_option" and self.injected_call_id is None:
            option_line = next((line for line in observation.splitlines()
                                if line.startswith("available_options: ")), None)
            options = json.loads(option_line.split(": ", 1)[1]) if option_line else {}
            target = action["parameters"]["value"]
            for values in options.values():
                if not isinstance(values, list) or target not in values:
                    continue
                for wrong in sorted(self.bad_options[self.current_asin]):
                    if wrong != target and wrong in values:
                        candidate = {"tool": "select_option", "parameters": {"value": wrong}}
                        if action_reject_reason(candidate["tool"], candidate["parameters"], observation) is None:
                            injection = candidate
                            break
                if injection:
                    break
        self.call_index += 1
        call_id = f"repair_{self.call_index}"
        if injection:
            action = injection
            self.injected_call_id = call_id
        else:
            self.position += 1
        if action["tool"] == "open_product":
            self.current_asin = action["parameters"]["asin"]
        elif action["tool"] == "back_to_search":
            self.current_asin = None
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": action["tool"],
                         "arguments": json.dumps(action["parameters"], ensure_ascii=False)},
        }]}


def mask_recovery_prefix(row, injected_call_id):
    if injected_call_id is None:
        return
    found = False
    for message in row["messages"]:
        if message["role"] == "assistant":
            message["trainable"] = False
            if any(c.get("id") == injected_call_id for c in message.get("tool_calls", [])):
                found = True
                break
    if not found:
        raise ValueError("injected action missing from corrective SFT row")
