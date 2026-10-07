# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train set: `data/grpo/train.parquet` (1,000 tasks)
- Validation set: `data/grpo/validation.parquet` (50 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v3

Hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh --dry-run
```

Train:

```bash
bash scripts/grpo.sh
```

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 16,384 (4,096 prompt + 12,288 response, including tool observations) |
| Maximum training steps | 200 |
| Save / validation frequency | 50 / 50 |
| Actor checkpoint retention | 4 (steps 50, 100, 150 and 200) |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement / loss | disabled / disabled |

The current memory protection caps actor, rollout, reference log-probability and
TRACE sequence budgets at 16,384 tokens. The agent uses a 15,360-token input
budget, a 512-token generation reserve and a 512-token safety margin. Runtime
preflight rejects inconsistent budgets and enabled entropy computation. Dynamic
batching remains disabled and each GPU micro-batch contains one trajectory;
the PPO token budget alone is not a trajectory truncation mechanism.

Dynamic sampling checks progress every three generation batches and permits at
most fifteen consecutive skipped windows (45 batches without an optimizer update).
Incomplete valid groups survive these windows with their original rollout log
probabilities because the policy has not changed. The cache holds fewer than
`train_batch_size` prompt groups, is consumed when the batch fills, and is cleared
before the next policy version. It is not a replay buffer across optimizer updates
or restarts. Invalid and constant-utility groups remain excluded.

Each actor turn is capped at 512 generated tokens, including when veRL omits
`max_tokens` from its sampling arguments. The project parser accepts only complete
Qwen tool calls matching tool schema v2. An incomplete call executes no action;
the actor receives format feedback and can retry within the existing trajectory
budget. Three consecutive parse failures terminate the trajectory as invalid.
Sampled actor tokens retain their original log probabilities, while feedback
tokens are masked out of policy loss. No terminal reward is fabricated.

ShopSimulator observations are projected from the saved raw tool response using
the model tokenizer and the per-page token budgets in `configs/agent_loop.yaml`.
This projection replaces veRL's earlier character-based response truncation;
the character limit is not reapplied to the projected text. Observations that
cannot fit their token budget while preserving actionable targets and the page
footer still invalidate the trajectory.
Structured detail and information pages retain complete variant state, price,
ASIN and page type; information pages use the detail budget. Search projection
retains exact prices and marks shortened text fields as snippets; full matching
evidence remains on the product's detail and information pages.

To prepare a training-only weighted task pool and verified SFT corrections from
existing diagnostics, use `scripts/prepare_training_repairs.py` as documented in
`docs/sft.md`. Pass its `grpo-train.parquet` through `train_grpo.py --train-data`.
Each source task keeps one copy; valid reward-varying tasks receive two copies,
or three when specification/page-state failures were observed. This is a bounded
sampling prior from an earlier policy, not a guarantee of future effective groups.

Before Qwen tokenization, the adapter clarifies that a no-argument tool has an
empty function body, with no parameter tag or empty-object placeholder. This
also adapts the system instruction cached in existing parquet prompts without
rewriting datasets. If a no-argument call is malformed, feedback names that
specific tool and shows its complete XML call. Invalid arguments are never
silently removed or executed by the parser.

`tool_parse_errors`, `tool_parse_error_reasons` and `tool_parse_error_details`
record failures, generated lengths, the output limit and the last 1,000 response
characters. `at_generation_limit` indicates a possible truncation, not proof of
the provider's finish reason. Successful recovery resets the consecutive counter.

Each run also appends `training_diagnostics.jsonl` under its output directory.
`generation_batch` records contain every generated rollout, its public tool
sequence, terminal result, reward breakdown, Guard rejection reasons and group
keep/drop decision. `optimizer_step` records preserve the scalar veRL metrics,
including PPO KL, clip fractions, response lengths and effective-group
rates. `skipped_update` records make zero-signal attempts visible even though
they do not advance the optimizer step.

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
The default logger is console. Optional `--logger swanlab` saves locally with
`SWANLAB_MODE=local`; it does not require a SwanLab API key or upload to the cloud.
Advanced overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The reported comparison uses step 100. Select checkpoints using validation
metrics rather than assuming that the final training step is best.
