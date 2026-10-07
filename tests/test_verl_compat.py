"""veRL 不应为了纯 padding 操作强制依赖 FlashAttention。"""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from shopping_grpo.training.grpo.compat import _with_dataloader_cleanup


class VerlCompatTest(unittest.TestCase):
    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def fit(self):
                return None

            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
                "verl.trainer": trainer,
                "verl.trainer.ppo": ppo,
                "verl.trainer.ppo.ray_trainer": ray_trainer,
            },
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()
            installed_fit = RayPPOTrainer.fit
            install_torch_padding_fallback()
            self.assertIs(RayPPOTrainer.fit, installed_fit)

        self.assertEqual(attention._get_attention_functions(), expected)
        self.assertTrue(RayPPOTrainer._update_actor._shopping_trace)
        self.assertTrue(RayPPOTrainer.fit._shopping_dataloader_cleanup)

    def test_fit_closes_both_loaders_after_early_return(self):
        train_shutdown, val_shutdown = Mock(), Mock()
        trainer = SimpleNamespace(
            train_dataloader=SimpleNamespace(
                _iterator=SimpleNamespace(_shutdown_workers=train_shutdown)
            ),
            val_dataloader=SimpleNamespace(
                _iterator=SimpleNamespace(_shutdown_workers=val_shutdown)
            ),
        )

        def fit(self, result):
            train_shutdown.assert_not_called()
            val_shutdown.assert_not_called()
            return result

        self.assertEqual(_with_dataloader_cleanup(fit)(trainer, result=200), 200)
        train_shutdown.assert_called_once_with()
        val_shutdown.assert_called_once_with()

    def test_cleanup_preserves_training_failure_and_closes_other_loader(self):
        failure = RuntimeError("training failed")
        shutdown = Mock()
        trainer = SimpleNamespace(
            train_dataloader=SimpleNamespace(
                _iterator=SimpleNamespace(_shutdown_workers=Mock(side_effect=OSError("cleanup")))
            ),
            val_dataloader=SimpleNamespace(
                _iterator=SimpleNamespace(_shutdown_workers=shutdown)
            ),
        )

        def fit(self):
            raise failure

        with self.assertLogs("shopping_grpo.training.grpo.compat", level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                _with_dataloader_cleanup(fit)(trainer)
        self.assertIs(caught.exception, failure)
        shutdown.assert_called_once_with()

    def test_cleanup_handles_unstarted_and_single_process_loaders(self):
        for trainer in (
            SimpleNamespace(),
            SimpleNamespace(train_dataloader=SimpleNamespace(_iterator=None)),
            SimpleNamespace(train_dataloader=SimpleNamespace(_iterator=object())),
        ):
            self.assertEqual(_with_dataloader_cleanup(lambda self: 7)(trainer), 7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
