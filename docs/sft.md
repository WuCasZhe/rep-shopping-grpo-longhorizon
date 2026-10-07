# LoRA SFT

This repository's preparation and default launcher use one single-stage SFT
recipe: `data/sft/train.jsonl` (800 rows) and `data/sft/validation.jsonl`
(200 rows), followed by GRPO and the held-out evaluation.
Training and validation task IDs must not overlap `data/evaluation/tasks.jsonl`.

Logs stay on local storage. SwanLab is disabled by default; when explicitly
enabled, its default mode is local and needs no API key.

## Run

After `bash scripts/setup.sh`, set `BASE_MODEL` to a local Qwen3.5-2B directory
or use the default `Qwen/Qwen3.5-2B`, then run:

```bash
bash scripts/sft.sh
```

This command executes training **and then merges the adapter**. It must only
be run when both operations are authorized. It writes the adapter to
`outputs/models/sft-lora` and the merged model to `outputs/models/sft-merged`.
Override these with `SFT_ADAPTER_DIR` and `SFT_MERGED_DIR` if needed.

## Recipe

| Setting | Value |
|---|---|
| Maximum sequence length | 16,384 |
| Epochs | 3 |
| Per-device train / validation batch | 1 / 1 |
| Gradient accumulation | 8 |
| Learning rate | `1e-4` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient checkpointing | enabled |
| Liger fused loss | enabled by the default launcher |
| Attention implementation | SDPA |
| Precision | automatic; BF16 preferred on supported CUDA devices |
| Saved checkpoint limit | 3 |

Only assistant tokens contribute to loss. User messages and environment
observations are masked. Examples exceeding the sequence limit are dropped,
not truncated; check the reported kept/dropped counts before interpreting the
training results. Validation loss measures training health, not final success.

With the current data and Qwen3.5-2B processor, the 16,384-token limit retains
797 of 800 training rows and all 200 validation rows. The source data remains
intact; the loader filters the three overlong training trajectories.

## GRPO and evaluation

The default GRPO launcher consumes `outputs/models/sft-merged`:

```bash
bash scripts/grpo.sh --dry-run
```

For another merged-model directory, pass `--model PATH` explicitly. The default
launcher does not read a `GRPO_MODEL_PATH` shell override.

Once execution is authorized, serve the merged model and evaluate through the
shared evaluation entry point. Strict success requires a complete
`gold_purchase` terminal result with `reward_valid=true`.

## Training failure corrections

Prepare corrections from GRPO training diagnostics without starting training:

```bash
python scripts/prepare_training_repairs.py \
  --diagnostics outputs/models/grpo/training_diagnostics.jsonl \
  --tokenizer outputs/models/sft-merged \
  --output-dir outputs/training-repairs \
  --replay-limit 32
```

The command excludes frozen evaluation, GRPO validation and SFT validation tasks,
deduplicates generation records, and replays successful action sequences for
training tasks with observed specification or page-state failures. When a failed
rollout chose another visible value on the same product's specification axis,
the replay can include that mistake followed by its correction. The mistaken
action and preceding assistant turns have `trainable: false`; the SFT loader
retains them as context with all labels masked. Only the recovery is supervised.
No hidden target is used to select replay actions.

Only complete, valid Reward v3 `gold_purchase` replays enter
`sft-corrections.jsonl`. Raw replays and source plans are audit artifacts and must
not be used as SFT inputs. Mix the correction rows into the existing training
split; `sft-train.jsonl` contains that mix and validation stays unchanged.
The accompanying `grpo-train.parquet` preserves
all source tasks with bounded multiplicities based on valid group variation.
