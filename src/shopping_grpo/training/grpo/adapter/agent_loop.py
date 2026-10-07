"""veRL 0.8 ToolAgentLoop 的 ShopSimulator 轨迹生命周期适配。

这个类不重新实现模型生成，而是在三个边界插入项目约束：上下文超长时压缩、工具
返回后投影 observation、trajectory 结束时计算奖励并释放环境。
"""

from __future__ import annotations

import json

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from shopping_grpo.environment.context import ContextBudgetError, compact_token_trajectory
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.training.grpo.adapter.runtime import (
    apply_reward_length_shaping,
    current_runtime_state,
    record_observation_projection,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.adapter.tool_parser import ShoppingToolParser, ToolCallParseError
from shopping_grpo.training.grpo.adapter.format_prompt import qwen_tool_messages, parse_error_feedback


class ShoppingToolAgentLoop(ToolAgentLoop):
    """Vanilla ToolAgentLoop with deterministic ShopSimulator termination and release."""

    def __init__(
        self,
        *args,
        base_url="http://127.0.0.1:5700",
        timeout=60,
        max_steps=35,
        required_environment_version=None,
        reward_mode="native",
        context_window_tokens=16384,
        context_generation_reserve_tokens=512,
        context_safety_margin_tokens=512,
        context_input_budget_tokens=15360,
        context_preserve_recent_groups=1,
        context_compaction_enable=False,
        observation_token_budget=1536,
        observation_detail_token_budget=4096,
        observation_generic_token_budget=768,
        observation_search_top_k=20,
        reward_length_shaping_enable=False,
        reward_length_soft_threshold=20,
        reward_length_penalty_per_step=0.01,
        reward_length_max_penalty=0.15,
        env_factory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.tool_parser_name == "qwen3_coder":
            self.tool_parser = ShoppingToolParser(self.tokenizer, self.tool_parser)
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.required_environment_version = required_environment_version
        self.reward_mode = str(reward_mode)
        self.context_window_tokens = int(context_window_tokens)
        self.context_generation_reserve_tokens = int(context_generation_reserve_tokens)
        self.context_safety_margin_tokens = int(context_safety_margin_tokens)
        self.context_input_budget_tokens = int(context_input_budget_tokens)
        self.context_preserve_recent_groups = int(context_preserve_recent_groups)
        self.context_compaction_enable = bool(context_compaction_enable)
        self.observation_token_budget = int(observation_token_budget)
        self.observation_detail_token_budget = int(observation_detail_token_budget)
        self.observation_generic_token_budget = int(observation_generic_token_budget)
        self.observation_search_top_k = int(observation_search_top_k)
        self.reward_length_shaping_enable = (
            reward_length_shaping_enable
            if isinstance(reward_length_shaping_enable, bool)
            else str(reward_length_shaping_enable).lower() == "true"
        )
        self.reward_length_soft_threshold = int(reward_length_soft_threshold)
        self.reward_length_penalty_per_step = float(reward_length_penalty_per_step)
        self.reward_length_max_penalty = float(reward_length_max_penalty)
        maximum_context_input = (
            self.context_window_tokens
            - self.context_generation_reserve_tokens
            - self.context_safety_margin_tokens
        )
        if not 0 < self.context_input_budget_tokens <= maximum_context_input:
            raise ValueError(
                "context_input_budget_tokens must be positive and fit the model context window"
            )
        self.context_input_budget = self.context_input_budget_tokens
        if self.context_preserve_recent_groups < 1:
            raise ValueError("context_preserve_recent_groups must be positive")
        if min(
            self.observation_token_budget,
            self.observation_detail_token_budget,
            self.observation_generic_token_budget,
        ) < 64:
            raise ValueError("all observation token budgets must be at least 64")
        if self.observation_search_top_k < 1:
            raise ValueError("observation_search_top_k must be positive")
        if self.reward_mode not in {"native", "constraint_aware"}:
            raise ValueError(f"unknown shopping reward mode: {self.reward_mode!r}")
        if self.reward_length_shaping_enable and self.reward_mode != "constraint_aware":
            raise ValueError("reward length shaping requires constraint_aware reward mode")
        if not 0 < self.reward_length_soft_threshold < self.max_steps:
            raise ValueError("reward_length_soft_threshold must be between 1 and max_steps")
        if min(self.reward_length_penalty_per_step, self.reward_length_max_penalty) < 0:
            raise ValueError("reward length penalties must be non-negative")
        self.env_factory = env_factory

    async def _handle_pending_state(self, agent_data, sampling_params):
        if self.tool_parser_name == "qwen3_coder":
            agent_data.messages = qwen_tool_messages(agent_data.messages)
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(
        self,
        agent_data,
        sampling_params,
        ignore_termination=False,
    ):
        """在每次生成前执行上下文预算检查，并限制本轮最大输出长度。"""
        runtime_state = current_runtime_state.get()
        current_input_tokens = len(agent_data.prompt_ids)
        if runtime_state is not None:
            runtime_state["context_max_input_tokens"] = max(
                runtime_state["context_max_input_tokens"],
                current_input_tokens,
            )
        if self.context_compaction_enable:
            try:
                prompt_ids, response_mask, response_logprobs, stats = compact_token_trajectory(
                    agent_data.prompt_ids,
                    agent_data.response_mask,
                    agent_data.response_logprobs,
                    max_input_tokens=self.context_input_budget,
                    preserve_recent_groups=self.context_preserve_recent_groups,
                )
            except ContextBudgetError as exc:
                if runtime_state is not None:
                    runtime_state["terminate"] = True
                    runtime_state["termination_reason"] = "context_budget_exhausted"
                    runtime_state["error"] = f"context_budget_exhausted:{exc}"
                    runtime_state["infrastructure_invalid"] = True
                return AgentState.TERMINATED
        else:
            prompt_ids = agent_data.prompt_ids
            response_mask = agent_data.response_mask
            response_logprobs = agent_data.response_logprobs
            stats = None
        if (
            not self.context_compaction_enable
            and current_input_tokens
            > self.context_input_budget
        ):
            if runtime_state is not None:
                runtime_state["terminate"] = True
                runtime_state["termination_reason"] = "context_hard_limit_exceeded"
                runtime_state["error"] = runtime_state["termination_reason"]
                runtime_state["infrastructure_invalid"] = True
            return AgentState.TERMINATED
        if stats is not None and stats.removed_tokens:
            # routed-experts 的额外状态无法随 token 一起安全裁剪，因此直接判为无效。
            if agent_data.routed_experts is not None:
                if runtime_state is not None:
                    runtime_state["terminate"] = True
                    runtime_state["termination_reason"] = "context_compaction_unsupported_routed_experts"
                    runtime_state["error"] = runtime_state["termination_reason"]
                    runtime_state["infrastructure_invalid"] = True
                return AgentState.TERMINATED
            agent_data.prompt_ids = prompt_ids
            agent_data.response_mask = response_mask
            agent_data.response_logprobs = response_logprobs
            if runtime_state is not None:
                runtime_state["context_compactions"] += 1
                runtime_state["context_tokens_removed"] += stats.removed_tokens
        bounded_sampling_params = dict(sampling_params)
        bounded_sampling_params["max_tokens"] = min(
            int(bounded_sampling_params.get("max_tokens", self.context_generation_reserve_tokens)),
            self.context_generation_reserve_tokens,
        )
        try:
            next_state = await super()._handle_generating_state(
                agent_data,
                bounded_sampling_params,
                ignore_termination=ignore_termination,
            )
        except ToolCallParseError as exc:
            if runtime_state is None:
                raise
            runtime_state["tool_parse_errors"] += 1
            runtime_state["consecutive_tool_parse_errors"] += 1
            reason = str(exc)
            counts = runtime_state["tool_parse_error_reasons"]
            counts[reason] = counts.get(reason, 0) + 1
            runtime_state["tool_parse_error_details"].append({
                "reason": reason,
                "generated_tokens": len(agent_data.response_ids),
                "generation_limit": bounded_sampling_params["max_tokens"],
                "at_generation_limit": len(agent_data.response_ids) >= bounded_sampling_params["max_tokens"],
                "response_tail": self.tokenizer.decode(agent_data.response_ids)[-1000:],
            })
            agent_data.tool_calls = []
            if runtime_state["consecutive_tool_parse_errors"] >= 3:
                runtime_state["terminate"] = True
                runtime_state["error"] = "tool_call_parse_error_limit"
                runtime_state["termination_reason"] = runtime_state["error"]
                return AgentState.TERMINATED
            # Preserve the sampled actor tokens and log-probabilities. Feedback
            # is observation text, so it must never receive policy loss.
            feedback = {"role": "tool", "content": parse_error_feedback(exc)}
            ids = await self.apply_chat_template([feedback], remove_system_prompt=True)
            if len(agent_data.response_mask) + len(ids) >= self.response_length:
                runtime_state["error"] = "tool_call_parse_error_budget"
                runtime_state["termination_reason"] = runtime_state["error"]
                runtime_state["terminate"] = True
                return AgentState.TERMINATED
            agent_data.messages.append(feedback)
            agent_data.prompt_ids += ids
            agent_data.response_mask += [0] * len(ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(ids)
            agent_data.user_turns += 1
            return AgentState.GENERATING
        if runtime_state is not None and next_state == AgentState.PROCESSING_TOOLS:
            runtime_state["consecutive_tool_parse_errors"] = 0
        return next_state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        """把工具适配器拿到的原始 observation 压缩成模型真正可见的版本。"""
        response, reward, step = await super()._call_tool(
            tool_call,
            tools_kwargs,
            agent_data,
        )
        state = current_runtime_state.get()
        if state is None:
            return response, reward, step
        raw_observation = state.pop("_pending_raw_observation", None)
        if raw_observation is None:
            return response, reward, step
        try:
            # 解析失败或投影破坏动作契约时，停止当前样本而不是把坏 observation
            # 继续喂给模型。
            parameters = json.loads(tool_call.arguments or "{}")
            visible_observation, projection = project_observation(
                tool_name=tool_call.name,
                observation=raw_observation,
                parameters=parameters,
                count_tokens=lambda text: len(
                    self.tokenizer.encode(text, add_special_tokens=False)
                ),
                token_budget=self.observation_token_budget,
                detail_token_budget=self.observation_detail_token_budget,
                generic_token_budget=self.observation_generic_token_budget,
                search_top_k=self.observation_search_top_k,
            )
        except Exception as exc:
            state["terminate"] = True
            state["termination_reason"] = "observation_projection_failed"
            state["error"] = f"observation_projection_failed:{exc.__class__.__name__}:{exc}"
            state["infrastructure_invalid"] = True
            state["observation_footer_failures"] += 1
            response.text = "Error: observation projection failed; trajectory is invalid."
            return response, 0.0, step

        projection_meta = projection.to_dict()
        state["latest_observation_raw"] = raw_observation
        state["latest_observation"] = visible_observation
        record_observation_projection(state, projection_meta)
        if isinstance(step, dict):
            step["projection"] = projection_meta
        # The parent has already applied veRL's character-based fallback. Replace
        # it with the projection of the saved raw observation: project_observation
        # enforces tokenizer-based budgets and preserves all actionable targets.
        # Do not reject or slice this result using the parent's character limit.
        response.text = visible_observation
        return response, reward, step

    async def _handle_processing_tools_state(self, agent_data):
        """强制每个 assistant 回合最多执行一个工具调用。"""
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and len(agent_data.tool_calls) > 1:
            runtime_state["terminate"] = True
            runtime_state["termination_reason"] = "parallel_tool_calls"
            runtime_state["error"] = "parallel_tool_calls"
            return AgentState.TERMINATED
        next_state = await super()._handle_processing_tools_state(agent_data)
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and runtime_state.get("terminate"):
            return AgentState.TERMINATED
        return next_state

    async def run(self, sampling_params, **kwargs):
        """启动 session、运行父类 AgentLoop，并在 finally 中释放环境租约。"""
        task_id = task_id_from_kwargs(kwargs)
        session = ShopSimulatorSession(
            base_url=self.base_url,
            timeout=self.timeout,
            max_steps=self.max_steps,
            required_environment_version=self.required_environment_version,
            env_factory=self.env_factory,
        )
        state = await session.start(task_id)
        try:
            output = await super().run(sampling_params, **kwargs)
            if not state["done"] and not state["error"]:
                state["error"] = "assistant_finished_without_environment_done"
                state["termination_reason"] = state["error"]
                state["terminate"] = True
            # 父类结束后统一从环境状态结算，避免把中途异常当作正常终局奖励。
            breakdown = apply_reward_length_shaping(
                reward_breakdown(state),
                state,
                enabled=getattr(self, "reward_length_shaping_enable", False),
                soft_threshold=getattr(self, "reward_length_soft_threshold", 20),
                penalty_per_step=getattr(self, "reward_length_penalty_per_step", 0.01),
                max_penalty=getattr(self, "reward_length_max_penalty", 0.15),
            )
            output.reward_score = (
                float(breakdown["total"])
                if self.reward_mode == "constraint_aware"
                else terminal_reward(state, mode=self.reward_mode)
            )
            output.extra_fields["shopping"] = {
                "task_id": task_id,
                "steps": len(state["steps"]),
                "actions": [
                    {"tool": step["tool"], "parameters": step["parameters"]}
                    for step in state["steps"]
                ],
                "done": bool(state["done"]),
                "termination_reason": state["termination_reason"],
                "error": state["error"],
                "tool_parse_errors": state["tool_parse_errors"],
                "tool_parse_error_reasons": dict(state["tool_parse_error_reasons"]),
                "tool_parse_error_details": list(state["tool_parse_error_details"]),
                "infrastructure_invalid": bool(state["infrastructure_invalid"]),
                "action_attempts": int(state["action_attempt_count"]),
                "repeat_actions": int(state["repeat_action_count"]),
                "overlong": bool(breakdown.get("overlong", False)),
                "reward_mode": self.reward_mode,
                "reward_version": state.get("reward_version"),
                "reward_type": state.get("reward_type"),
                "reward_valid": bool(state.get("reward_valid", True)),
                "reward_unverifiable": bool(state.get("reward_unverifiable")),
                "reward": breakdown,
                "context_compactions": int(state["context_compactions"]),
                "context_tokens_removed": int(state["context_tokens_removed"]),
                "context_max_input_tokens": int(state["context_max_input_tokens"]),
                "observation_projection_count": int(state["observation_projection_count"]),
                "observation_truncated_count": int(state["observation_truncated_count"]),
                "observation_raw_tokens": int(state["observation_raw_tokens"]),
                "observation_visible_tokens": int(state["observation_visible_tokens"]),
                "observation_max_raw_tokens": int(state["observation_max_raw_tokens"]),
                "observation_max_visible_tokens": int(
                    state["observation_max_visible_tokens"]
                ),
                "observation_visible_asin_count": int(
                    state["observation_visible_asin_count"]
                ),
                "observation_visible_button_count": int(
                    state["observation_visible_button_count"]
                ),
                "observation_any_truncated": bool(state["observation_any_truncated"]),
                "observation_footer_failures": int(state["observation_footer_failures"]),
                "guard_rejections": int(state["guard_rejection_count"]),
                "guard_rejection_reasons": dict(
                    state["guard_rejection_reason_counts"]
                ),
                "guard_rejections_after_truncation": int(
                    state["guard_rejection_after_truncation_count"]
                ),
                "action_attempts_after_truncation": int(
                    state["action_attempt_after_truncation_count"]
                ),
            }
            if session.trace_target is not None:
                output.extra_fields["trace_target"] = session.trace_target
            return output
        finally:
            await session.close()
