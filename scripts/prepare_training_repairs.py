#!/usr/bin/env python3
"""Mine GRPO failures, weight training tasks, and replay verified SFT corrections.

Does not train a model or run model evaluation. Replay uses recorded actions
against ShopSimulator, and accepts only complete, valid Reward v3 gold purchases.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from shopping_grpo.collection.corrections import (
    CorrectiveReplayClient, mask_recovery_prefix, mine_training_failures,
)
from shopping_grpo.collection.sft import (
    acceptance_reasons, build_sft_row, read_jsonl, task_ids_from_jsonl, write_jsonl,
)
from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT, collect_for_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, default=Path("data/grpo/train.parquet"))
    parser.add_argument("--sft-train", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--held-out", type=Path, nargs="+", default=[
        Path("data/evaluation/tasks.jsonl"), Path("data/grpo/validation.jsonl"),
        Path("data/sft/validation.jsonl"),
    ])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-limit", type=int, default=32)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_limit < 0:
        parser.error("--replay-limit must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be new or empty")

    import pandas as pd
    from transformers import AutoTokenizer
    from shopping_grpo.environment.projection import project_observation

    frame = pd.read_parquet(args.train_parquet)
    task_ids = [int(row["task_id"]) for row in frame["extra_info"]]
    if len(task_ids) != len(set(task_ids)):
        parser.error("source train parquet must have exactly one row per task")
    # The frozen evaluation set is mandatory even when --held-out is overridden.
    mandatory_evaluation = Path(__file__).resolve().parents[1] / "data/evaluation/tasks.jsonl"
    excluded = set()
    for path in [mandatory_evaluation, *args.held_out]:
        if not path.is_file():
            parser.error(f"required held-out file is missing: {path}")
        excluded.update(task_ids_from_jsonl(path))
    events = list(read_jsonl(args.diagnostics))
    plans, weights = mine_training_failures(events, set(task_ids), excluded)
    base_sft = list(read_jsonl(args.sft_train))
    if {int(row["task_id"]) for row in base_sft} & excluded:
        parser.error("source SFT training split overlaps held-out tasks")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    def project(tool_name, observation, parameters):
        text, meta = project_observation(
            tool_name, observation, parameters=parameters,
            count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )
        return text, meta.to_dict()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weighted_indices = [i for i, task_id in enumerate(task_ids) for _ in range(weights[task_id])]
    weighted = frame.iloc[weighted_indices].copy().reset_index(drop=True)
    # Refresh only the public system instructions; preserve every user request.
    weighted["prompt"] = weighted["prompt"].map(lambda prompt: [
        {**dict(m), "content": SYSTEM_PROMPT} if m["role"] == "system" else dict(m)
        for m in prompt
    ])
    weighted.to_parquet(args.output_dir / "grpo-train.parquet", index=False)
    write_jsonl(args.output_dir / "sampling_weights.jsonl", [
        {"task_id": task_id, "multiplicity": weight} for task_id, weight in weights.items()
    ])
    write_jsonl(args.output_dir / "correction_plans.jsonl", plans)
    accepted, rejected, categories = [], [], Counter()
    for plan in plans[:args.replay_limit]:
        client = CorrectiveReplayClient(plan)
        client.project_observation = project
        trajectory = collect_for_task(
            {"task_id": plan["task_id"]}, client=client, base_url=args.base_url, max_steps=35,
        )
        trajectory["correction_source"] = {
            "success_uid": plan["success_uid"], "tags": plan["tags"],
            "injected_call_id": client.injected_call_id,
        }
        # Raw simulator results remain audit-only, never model supervision.
        with (args.output_dir / "raw.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
        ok, reasons = acceptance_reasons(trajectory)
        if ok:
            row = build_sft_row(trajectory)
            mask_recovery_prefix(row, client.injected_call_id)
            accepted.append(row)
            categories.update(plan["tags"])
            categories["wrong_option_recovery"] += int(client.injected_call_id is not None)
        else:
            rejected.append({"task_id": plan["task_id"], "reasons": reasons})
        print(json.dumps({"task_id": plan["task_id"], "accepted": ok, "reasons": reasons}), flush=True)
    write_jsonl(args.output_dir / "sft-corrections.jsonl", accepted)
    write_jsonl(args.output_dir / "sft-train.jsonl", [*base_sft, *accepted])
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    summary = {
        "source_diagnostics_sha256": hashlib.sha256(args.diagnostics.read_bytes()).hexdigest(),
        "source_train_sha256": hashlib.sha256(args.train_parquet.read_bytes()).hexdigest(),
        "source_sft_train_sha256": hashlib.sha256(args.sft_train.read_bytes()).hexdigest(),
        "sft_base_rows": len(base_sft), "sft_mixed_rows": len(base_sft) + len(accepted),
        "train_tasks": len(task_ids), "weighted_rows": len(weighted_indices),
        "multiplicity_counts": dict(Counter(weights.values())),
        "correction_candidates": len(plans), "attempted": min(len(plans), args.replay_limit),
        "accepted": len(accepted), "rejected": len(rejected), "categories": dict(categories),
        "held_out_overlap": len(set(task_ids) & excluded),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
