"""CPU test at N=20.

Both precomputed candidates AND precomputed embeddings are required so the
runtime never has to load RoBERTa on CPU. The launcher asserts both before
delegating to `hullft.runner.run_from_config`.
"""

import argparse
import os

from hullft.config import apply_config_overrides, load_config
from hullft.reproducibility import set_global_determinism
from hullft.runner import run_from_config


def main():
    p = argparse.ArgumentParser(description="HullFT CPU test (N=20).")
    p.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override data.repeat (queries per run).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=None,
        help="Extra config overrides, e.g. selection.n_select=10.",
    )
    args = p.parse_args()

    config = load_config(args.config)
    if args.set:
        apply_config_overrides(config, args.set)

    pile_cfg = config.get("pile_index", {})
    if not pile_cfg.get("precomputed_candidates_dir"):
        raise ValueError(
            "cpu_test requires pile_index.precomputed_candidates_dir in config."
        )
    if not pile_cfg.get("precomputed_embeddings_dir"):
        raise ValueError(
            "cpu_test requires pile_index.precomputed_embeddings_dir. "
            "Run `python -m data.precompute_embeddings ...` first so RoBERTa is not loaded on CPU."
        )
    if config.get("selection", {}).get("device") != "cpu":
        raise ValueError("cpu_test requires selection.device: cpu in config.")

    seed = int(config.get("data", {}).get("seed", 42))
    set_global_determinism(seed)

    summary = run_from_config(
        config_path=args.config,
        config_dict=config,
        repeat_override=args.repeat,
    )

    finetune_secs = [
        m.get("finetune_seconds")
        for r in summary["results"]
        for m in (r.get("evaluation") or {}).values()
        if m.get("finetune_seconds") is not None
    ]
    selection_secs = [
        s
        for r in summary["results"]
        for s in (r.get("method_runtimes_seconds") or {}).values()
        if s is not None
    ]
    if finetune_secs:
        print(
            f"\nCPU timing (n={len(finetune_secs)} method-runs): "
            f"finetune mean={sum(finetune_secs) / len(finetune_secs):.2f}s, "
            f"selection mean={sum(selection_secs) / max(1, len(selection_secs)):.2f}s"
        )


if __name__ == "__main__":
    main()
