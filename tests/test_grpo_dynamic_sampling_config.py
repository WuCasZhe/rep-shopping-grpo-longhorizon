"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scripts.check_grpo_runtime import (
    PATCH_MARKER,
    compose_runtime_config,
    validate_dynamic_sampling,
    validate_training_memory_budget,
    validate_agent_context_budget,
)


class DynamicSamplingConfigTest(unittest.TestCase):
    def test_training_memory_budget_enforces_real_micro_batch_one(self):
        config = compose_runtime_config([])
        self.assertEqual(config.shopping_dynamic_sampling.max_consecutive_skipped_updates, 15)
        validate_training_memory_budget(config)
        self.assertEqual(config.data.max_response_length, 12288)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 16384)
        self.assertFalse(config.actor_rollout_ref.actor.use_dynamic_bsz)
        self.assertFalse(config.actor_rollout_ref.actor.calculate_entropy)
        self.assertEqual(
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(
            config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
        )
        self.assertEqual(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz)
        self.assertEqual(
            config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu, 1
        )

    def test_training_memory_budget_rejects_unsafe_overrides(self):
        unsafe_response = compose_runtime_config(["data.max_response_length=24576"])
        with self.assertRaisesRegex(SystemExit, "unsafe GRPO response budget"):
            validate_training_memory_budget(unsafe_response)

        dynamic_actor = compose_runtime_config(
            ["actor_rollout_ref.actor.use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(SystemExit, "actor.use_dynamic_bsz must be false"):
            validate_training_memory_budget(dynamic_actor)

        dynamic_rollout_log_prob = compose_runtime_config(
            ["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(
            SystemExit, "rollout.log_prob_use_dynamic_bsz must be false"
        ):
            validate_training_memory_budget(dynamic_rollout_log_prob)

    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = compose_runtime_config(
            [
                "shopping_dynamic_sampling.enable=true",
                "shopping_dynamic_sampling.metric=seq_reward",
                "shopping_dynamic_sampling.max_num_gen_batches=3",
                "shopping_dynamic_sampling.max_consecutive_skipped_updates=10",
                "shopping_dynamic_sampling.reward_tolerance=1e-8",
            ]
        )
        self.assertTrue(config.shopping_dynamic_sampling.enable)
        self.assertEqual(config.shopping_dynamic_sampling.metric, "seq_reward")
        self.assertEqual(config.shopping_dynamic_sampling.max_num_gen_batches, 3)
        self.assertEqual(
            config.shopping_dynamic_sampling.max_consecutive_skipped_updates, 10
        )
        self.assertEqual(config.shopping_dynamic_sampling.reward_tolerance, 1.0e-8)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertTrue(config.actor_rollout_ref.rollout.calculate_log_probs)

    def test_16k_protection_rejects_inconsistent_budgets_and_entropy(self):
        for override in (
            "data.max_response_length=16384",
            "data.max_prompt_length=4097",
            "actor_rollout_ref.rollout.max_model_len=12288",
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288",
            "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288",
            "shopping_trace.max_sequence_length=12288",
            "actor_rollout_ref.actor.calculate_entropy=true",
            "actor_rollout_ref.actor.entropy_coeff=0.01",
        ):
            with self.subTest(override=override), self.assertRaises(SystemExit):
                validate_training_memory_budget(compose_runtime_config([override]))

    def test_agent_budget_matches_training_and_rejects_overflow(self):
        config = compose_runtime_config([])
        agents = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/agent_loop.yaml")
        validate_agent_context_budget(config, agents)
        agents[0].context_window_tokens = 12288
        with self.assertRaisesRegex(SystemExit, "must match"):
            validate_agent_context_budget(config, agents)
        agents[0].context_window_tokens = 16384
        agents[0].context_input_budget_tokens = 15361
        with self.assertRaisesRegex(SystemExit, "must fit context window"):
            validate_agent_context_budget(config, agents)

    def test_enabled_config_requires_installed_patch_marker(self):
        config = compose_runtime_config(["shopping_dynamic_sampling.enable=true"])
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            trainer_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "patch marker is missing"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            trainer_source.write_text(f"# {PATCH_MARKER}\n", encoding="utf-8")
            validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
