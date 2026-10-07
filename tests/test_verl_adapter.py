"""不依赖 veRL 安装的最小适配层单测。"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from shopping_grpo.environment.actions import action_reject_reason, clickable_buttons
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from shopping_grpo.training.grpo.adapter.tool_parser import ToolCallParseError


class ObservationProjectionBoundaryTest(unittest.TestCase):
    def _project_tool_response(self, raw):
        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.max_tool_response_length = 4096
            loop.observation_token_budget = 1536
            loop.observation_detail_token_budget = 4096
            loop.observation_generic_token_budget = 768
            loop.observation_search_top_k = 20
            # Four characters per token deliberately separates the two units.
            loop.tokenizer = SimpleNamespace(
                encode=lambda text, **kwargs: list(range((len(text) + 3) // 4))
            )
            state = make_runtime_state(1, 35)
            token = current_runtime_state.set(state)
            step = {"tool": "open_product"}

            async def parent_call(_loop, tool_call, tools_kwargs, agent_data):
                state["_pending_raw_observation"] = raw
                # veRL's character fallback runs before the adapter's projection.
                return SimpleNamespace(text=raw[:4096]), 0.0, step

            try:
                with patch.object(ToolAgentLoop, "_call_tool", parent_call):
                    response, reward, returned_step = await loop._call_tool(
                        SimpleNamespace(name="open_product", arguments="{}"),
                        {},
                        SimpleNamespace(),
                    )
                self.assertIs(returned_step, step)
                self.assertEqual(reward, 0.0)
                self.assertNotIn("_pending_raw_observation", state)
                return response.text, state, step
            finally:
                current_runtime_state.reset(token)

        return asyncio.run(run())

    def _detail(self, repeats):
        return (
            "Product [SEP] " + "detail " * repeats
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "buy now", "blue"]'
        )

    def test_detail_within_token_budget_survives_character_fallback(self):
        raw = self._detail(1000)
        visible, state, step = self._project_tool_response(raw)

        self.assertGreater(len(raw), 4096)
        self.assertEqual(visible, raw)
        self.assertLessEqual(step["projection"]["visible_tokens"], 4096)
        self.assertFalse(step["projection"]["truncated"])
        self.assertEqual(state["latest_observation"], visible)
        self.assertEqual(state["latest_observation_raw"], raw)
        self.assertFalse(state["terminate"])
        self.assertFalse(state["infrastructure_invalid"])
        self.assertEqual(state["observation_footer_failures"], 0)
        self.assertIsNone(action_reject_reason("select_option", {"value": "blue"}, visible))

    def test_over_budget_detail_is_projected_by_tokens_and_preserves_actions(self):
        raw = self._detail(4000)
        visible, state, step = self._project_tool_response(raw)

        self.assertGreater(len(visible), 4096)
        self.assertLessEqual((len(visible) + 3) // 4, 4096)
        self.assertGreater(step["projection"]["raw_tokens"], 4096)
        self.assertTrue(step["projection"]["truncated"])
        self.assertTrue(step["projection"]["critical_footer_preserved"])
        self.assertEqual(clickable_buttons(visible), clickable_buttons(raw))
        self.assertEqual(state["latest_observation"], visible)
        self.assertTrue(state["latest_observation_truncated"])
        self.assertFalse(state["terminate"])
        self.assertFalse(state["infrastructure_invalid"])

    def test_unprojectable_action_footer_still_invalidates_trajectory(self):
        raw = (
            "Product\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["buy now", "' + "x" * 17000 + '"]'
        )
        visible, state, step = self._project_tool_response(raw)

        self.assertTrue(state["terminate"])
        self.assertTrue(state["infrastructure_invalid"])
        self.assertEqual(state["termination_reason"], "observation_projection_failed")
        self.assertIn("critical footer exceeds", state["error"])
        self.assertEqual(state["observation_footer_failures"], 1)
        self.assertNotIn("projection", step)
        self.assertIn("trajectory is invalid", visible)


class ToolParseRecoveryTest(unittest.TestCase):
    def test_qwen_format_rules_reach_the_parent_before_tokenization(self):
        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.tool_parser_name = "qwen3_coder"
            messages = [{"role": "system", "content": "无参数工具必须传严格的 `{}`。"}]
            data = SimpleNamespace(messages=messages)
            async def pending(_loop, data, params):
                self.assertIn("<function=prev_page>", data.messages[0]["content"])
                self.assertNotIn("必须传严格的 `{}`", data.messages[0]["content"])
                return AgentState.GENERATING
            with patch.object(ToolAgentLoop, "_handle_pending_state", pending):
                self.assertEqual(await loop._handle_pending_state(data, {}), AgentState.GENERATING)
            self.assertIn("必须传严格的 `{}`", messages[0]["content"])
        asyncio.run(run())

    def _loop(self):
        loop = object.__new__(ShoppingToolAgentLoop)
        loop.context_compaction_enable = False
        loop.context_window_tokens = 16384
        loop.context_input_budget = 15360
        loop.context_generation_reserve_tokens = 512
        loop.context_safety_margin_tokens = 512
        loop.response_length = 12288
        loop.tokenizer = SimpleNamespace(decode=lambda ids: "<tool_call><function=search_products><parameter=query")
        async def template(*args, **kwargs):
            return [90, 91]
        loop.apply_chat_template = template
        return loop

    def test_generation_respects_input_budget_without_compaction(self):
        async def run():
            loop = self._loop()
            # A stricter input budget must be honored even with compaction off.
            loop.context_input_budget = 10000
            for size in (10000, 10001, 13904):
                state = make_runtime_state(1, 35)
                token = current_runtime_state.set(state)
                data = SimpleNamespace(prompt_ids=[1] * size, response_mask=[], response_logprobs=[])
                calls = []
                async def generate(_loop, data, params, **kwargs):
                    calls.append(params)
                    return AgentState.GENERATING
                try:
                    with patch.object(ToolAgentLoop, "_handle_generating_state", generate):
                        result = await loop._handle_generating_state(data, {})
                    if size == 10000:
                        self.assertEqual(result, AgentState.GENERATING)
                        self.assertEqual(len(calls), 1)
                    else:
                        self.assertEqual(result, AgentState.TERMINATED)
                        self.assertEqual(calls, [])
                        self.assertTrue(state["infrastructure_invalid"])
                        self.assertEqual(state["termination_reason"], "context_hard_limit_exceeded")
                finally:
                    current_runtime_state.reset(token)
        asyncio.run(run())

    def test_recovery_preserves_actor_tokens_and_masks_feedback_then_stops(self):
        async def run():
            loop = self._loop()
            data = SimpleNamespace(prompt_ids=[1], response_ids=[], response_mask=[], response_logprobs=[], messages=[], user_turns=0, tool_calls=[])
            state = make_runtime_state(1, 35)
            token = current_runtime_state.set(state)
            async def generate(_loop, data, params, **kwargs):
                self.assertEqual(params["max_tokens"], 512)
                data.response_ids = [2, 3]
                data.prompt_ids += [2, 3]
                data.response_mask += [1, 1]
                data.response_logprobs += [-0.2, -0.3]
                raise ToolCallParseError("incomplete_parameter")
            try:
                with patch.object(ToolAgentLoop, "_handle_generating_state", generate):
                    self.assertEqual(await loop._handle_generating_state(data, {}), AgentState.GENERATING)
                    self.assertEqual(data.prompt_ids, [1, 2, 3, 90, 91])
                    self.assertEqual(data.response_mask, [1, 1, 0, 0])
                    self.assertEqual(data.response_logprobs, [-0.2, -0.3, 0.0, 0.0])
                    self.assertIsNone(state["error"])
                    self.assertEqual(await loop._handle_generating_state(data, {}), AgentState.GENERATING)
                    self.assertEqual(await loop._handle_generating_state(data, {}), AgentState.TERMINATED)
                    self.assertEqual(data.user_turns, 2)
                    self.assertEqual(state["error"], "tool_call_parse_error_limit")
                    self.assertEqual(state["steps"], [])
                    self.assertTrue(reward_breakdown(state)["sampling_invalid"])
                    self.assertEqual(len(state["tool_parse_error_details"]), 3)
            finally:
                current_runtime_state.reset(token)
        asyncio.run(run())

    def test_valid_call_resets_consecutive_errors_and_respects_lower_token_limit(self):
        async def run():
            loop = self._loop()
            state = make_runtime_state(1, 35)
            state["consecutive_tool_parse_errors"] = 2
            token = current_runtime_state.set(state)
            async def generate(_loop, data, params, **kwargs):
                self.assertEqual(params["max_tokens"], 32)
                return AgentState.PROCESSING_TOOLS
            try:
                with patch.object(ToolAgentLoop, "_handle_generating_state", generate):
                    data = SimpleNamespace(prompt_ids=[1], response_mask=[], response_logprobs=[])
                    await loop._handle_generating_state(data, {"max_tokens": 32})
                self.assertEqual(state["consecutive_tool_parse_errors"], 0)
            finally:
                current_runtime_state.reset(token)
        asyncio.run(run())


def make_tool(name):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test-only {name} tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        tool_schema = schema
    else:
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, tool_schema)


class VerlAdapterRuntimeTest(unittest.TestCase):
    def test_agent_loop_preserves_real_verl_metrics_and_exports_shopping_diagnostics(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "shopsimulator-environment-v2.1",
                }

            def release(self):
                self.released = True

        async def fake_parent_run(_loop, sampling_params, **kwargs):
            state = current_runtime_state.get()
            state.update(
                {
                    "done": True,
                    "terminal_result": {"done": True, "over": True},
                    "termination_reason": "gold_purchase",
                    "final_reward": 1.0,
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {"key_options": 1.0},
                        "hard_gates": {
                            "category": {"passed": True},
                            "budget": {"passed": True},
                        },
                    },
                    "steps": [
                        {
                            "index": 0,
                            "tool": "search",
                            "parameters": {"query": "shoe"},
                            "done": False,
                            "reward": 0.0,
                        }
                    ],
                    "guard_rejection_reason_counts": {"asin_not_visible": 2},
                }
            )
            return AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
                reward_score=None,
                metrics=AgentLoopMetrics(generate_sequences=0.25),
                extra_fields={},
            )

        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.base_url = "http://shop.test"
            loop.timeout = 60
            loop.max_steps = 35
            loop.required_environment_version = "shopsimulator-environment-v2.1"
            loop.reward_mode = "constraint_aware"
            loop.env_factory = FakeEnv
            with patch.object(ToolAgentLoop, "run", fake_parent_run):
                return await ShoppingToolAgentLoop.run(
                    loop,
                    {},
                    extra_info={"task_id": 42},
                )

        output = asyncio.run(run())
        self.assertIsInstance(output.metrics, AgentLoopMetrics)
        self.assertEqual(
            output.metrics.model_dump(),
            {
                "generate_sequences": 0.25,
                "tool_calls": 0.0,
                "compute_score": 0.0,
                "num_preempted": -1,
            },
        )
        self.assertEqual(output.reward_score, 1.0)
        self.assertEqual(output.extra_fields["shopping"]["task_id"], 42)
        self.assertEqual(
            output.extra_fields["shopping"]["reward"]["terminal_utility"],
            1.0,
        )
        self.assertEqual(
            output.extra_fields["shopping"]["actions"],
            [{"tool": "search", "parameters": {"query": "shoe"}}],
        )
        self.assertEqual(
            output.extra_fields["shopping"]["guard_rejection_reasons"],
            {"asin_not_visible": 2},
        )
        self.assertTrue(created[0].released)

    def test_terminal_reward_only_uses_a_normal_environment_completion(self):
        done = make_runtime_state(task_id=1, max_steps=35)
        done.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 0.75})
        self.assertEqual(terminal_reward(done), 0.75)

        unfinished = make_runtime_state(task_id=1, max_steps=35)
        unfinished.update({"final_reward": 1.0, "terminal_result": {"done": False}})
        self.assertEqual(terminal_reward(unfinished), 0.0)

        errored = make_runtime_state(task_id=1, max_steps=35)
        errored.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "final_reward": 1.0,
                "error": "tool_error:timeout",
            }
        )
        self.assertEqual(terminal_reward(errored), 0.0)

    def test_context_state_is_task_local(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        token = current_runtime_state.set(state)
        try:
            self.assertIs(current_runtime_state.get(), state)
        finally:
            current_runtime_state.reset(token)

    def test_runtime_state_has_no_hidden_goal_fields(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        self.assertNotIn("goal", state)
        self.assertIsNone(state["reward_detail"])

    def test_task_id_is_read_from_verl_extra_info(self):
        self.assertEqual(task_id_from_kwargs({"extra_info": {"task_id": 42}}), 42)

    def test_task_id_accepts_numpy_style_scalar_container(self):
        class Scalar:
            def item(self):
                return {"task_id": 43}

        self.assertEqual(task_id_from_kwargs({"extra_info": Scalar()}), 43)

    def test_missing_task_id_fails_before_acquiring_an_environment(self):
        with self.assertRaisesRegex(ValueError, "task_id"):
            task_id_from_kwargs({"extra_info": {"split": "train"}})

    def test_terminal_observation_is_not_returned_to_the_model(self):
        class FakeEnv:
            def step(self, action):
                self.action = action
                return {
                    "instruction": "Goal: hidden answer\nReward: hidden breakdown",
                    "done": True,
                    "over": True,
                    "reward": 1.0,
                    "goal": {"secret": True},
                    "reward_detail": {"secret": True},
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("search_products").execute(
                    "tool-1", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(response.text, "Environment terminated.")
            self.assertTrue(state["terminate"])
            self.assertEqual(state["terminal_result"], {"done": True, "over": True})
            self.assertTrue(state["infrastructure_invalid"])
            self.assertIsNone(state["reward_detail"])
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_terminal_reward_v3_is_validated_without_entering_tool_observation(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "Goal: hidden answer",
                    "done": True,
                    "over": True,
                    "reward": 0.25,
                    "goal": {"secret": True},
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "partial_alternative_purchase",
                        "termination_reason": "partial_alternative_purchase",
                        "reward_valid": True,
                        "terminal_utility": 0.25,
                        "purchase_success": False,
                        "sampling_invalid": False,
                        "hard_gates": {},
                        "weighted_score": 0.5,
                        "hidden_answer": "do not retain",
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("search_products").execute(
                    "tool-1", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)

            self.assertEqual(response.text, "Environment terminated.")
            self.assertFalse(state["infrastructure_invalid"])
            self.assertEqual(
                state["reward_detail"]["reward_type"],
                "partial_alternative_purchase",
            )
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_terminal_reward_keeps_unverifiable_separate_from_infrastructure(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.0,
                    "termination_reason": "reward_unverifiable",
                    "reward_valid": False,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "reward_unverifiable",
                        "reward_valid": False,
                        "termination_reason": "reward_unverifiable",
                        "target_asin_match": False,
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                        "hard_gates": {
                            "category": {"status": "pass", "passed": True, "verifiable": True}
                        },
                        "weighted_score": 0.0,
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute(
                    "tool-v2", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertTrue(state["reward_unverifiable"])
            self.assertEqual(state["reward_type"], "reward_unverifiable")
            self.assertEqual(state["termination_reason"], "reward_unverifiable")

        asyncio.run(run())

    def test_reward_exposes_utility_success_and_sampling_validity_separately(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.55,
                    "termination_reason": "valid_alternative_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "valid_alternative_purchase",
                        "reward_valid": True,
                        "termination_reason": "valid_alternative_purchase",
                        "target_asin_match": False,
                        "terminal_utility": 0.55,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {
                            "brand": 0.0,
                            "model": 0.0,
                            "core_functions": 1.0,
                            "key_options": 1.0,
                        },
                        "hard_gates": {
                            "category": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                                "comparator": "category_leaf_ancestor_chain",
                                "source_field": "category",
                            }
                        },
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute(
                    "tool-v3",
                    {"query": "mug"},
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertFalse(state["reward_unverifiable"])
            self.assertEqual(
                state["reward_type"],
                "valid_alternative_purchase",
            )
            breakdown = reward_breakdown(state)
            self.assertEqual(breakdown["terminal_utility"], 0.55)
            self.assertEqual(breakdown["purchase_success"], 1.0)
            self.assertEqual(breakdown["r_att"], 1.0)
            self.assertEqual(breakdown["r_option"], 1.0)
            self.assertFalse(breakdown["sampling_invalid"])

        asyncio.run(run())

    def test_sync_environment_step_runs_off_the_event_loop_thread(self):
        main_thread = threading.get_ident()

        class FakeEnv:
            step_thread = None

            def step(self, action):
                self.step_thread = threading.get_ident()
                return {"instruction": "next", "done": False, "over": False, "reward": 0.0}

        async def run():
            env = FakeEnv()
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(env)
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute("tool-1", {"query": "mug"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertNotEqual(env.step_thread, main_thread)

        asyncio.run(run())

    def test_think_consumes_the_step_budget_and_terminates_at_the_exact_limit(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=1)
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("think").execute("tool-1", {"note": "plan"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(len(state["steps"]), 1)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "max_steps")
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_repeated_guard_rejections_terminate_instead_of_looping_forever(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "可点击的按钮: []"
            state["latest_observation_truncated"] = True
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                tool = make_tool("open_product")
                for index in range(3):
                    response, _, _ = await tool.execute(f"tool-{index}", {"asin": "123456789012"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "too_many_guard_rejections")
            self.assertEqual(state["steps"], [])
            self.assertEqual(state["action_attempt_count"], 3)
            self.assertEqual(state["repeat_action_count"], 2)
            self.assertEqual(state["guard_rejection_count"], 3)
            self.assertEqual(state["guard_rejection_after_truncation_count"], 3)
            self.assertEqual(state["action_attempt_after_truncation_count"], 3)
            self.assertEqual(
                state["guard_rejection_reason_counts"],
                {"click_not_in_previous_observation": 3},
            )
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_session_releases_its_environment_on_close(self):
        """无论正常终局还是异常路径，veRL lifecycle 都必须归还 ShopSimulator 租约。"""
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {"instruction": f"task {task_id}"}

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(max_steps=35, env_factory=FakeEnv)
            state = await session.start(task_id=8)
            state.update({"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 1.0})
            self.assertEqual(terminal_reward(state), 1.0)
            await session.close()

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_session_reset_and_release_run_off_the_event_loop_thread(self):
        main_thread = threading.get_ident()
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.reset_thread = None
                self.release_thread = None
                created.append(self)

            def reset(self, task_id):
                self.reset_thread = threading.get_ident()
                return {"instruction": f"task {task_id}"}

            def release(self):
                self.release_thread = threading.get_ident()

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            await session.close()

        asyncio.run(run())
        self.assertNotEqual(created[0].reset_thread, main_thread)
        self.assertNotEqual(created[0].release_thread, main_thread)

    def test_session_rejects_wrong_environment_version_and_releases(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "unsupported-environment",
                }

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(
                required_environment_version="shopsimulator-environment-v2.1",
                env_factory=FakeEnv,
            )
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                await session.start(1)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_reset_failure_still_releases_the_environment(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                raise RuntimeError("reset failed")

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                await session.start(task_id=8)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_release_failure_is_not_silently_hidden_or_forgotten(self):
        class FakeEnv:
            def __init__(self, **kwargs):
                pass

            def reset(self, task_id):
                return {"instruction": f"task {task_id}"}

            def release(self):
                raise RuntimeError("release failed")

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await session.close()
            self.assertEqual(session.state["error"], "release_error:RuntimeError:release failed")

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
