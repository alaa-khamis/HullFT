"""Main 1-to-50 sweep: HullFT (fw, integerization, gradient_refresh) vs kNN vs SIFT.

For each `N` in `--n_values`, the script overrides `selection.n_select=N`
and calls `hullft.runner.run_from_config` once. Results land under the
`output.results_dir` configured in `config.yaml` (one subdir per N), and a
top-level `summary.csv` is written alongside.
"""

import argparse
import copy
import csv
import os
from typing import Iterable

from hullft.config import apply_config_overrides, load_config
from hullft.reproducibility import set_global_determinism
from hullft.runner import run_from_config


def _parse_n_values(spec: str) -> list[int]:
    """Accept `"1-50"`, `"1,5,10"`, or a mix: `"1-5,10,15-20"`."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    seen = set()
    deduped = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def _flatten_results(n: int, summary: dict, rows: list[dict]):
    for r in summary["results"]:
        eval_dict = r.get("evaluation") or {}
        for method, m in eval_dict.items():
            rows.append(
                {
                    "n_select": n,
                    "method": method,
                    "query_idx": r["query_idx_test"],
                    "bpb": m.get("bpb"),
                    "perplexity": m.get("perplexity"),
                    "improvement_percent": m.get("improvement_percent"),
                    "finetune_seconds": m.get("finetune_seconds"),
                    "selection_seconds": r.get("method_runtimes_seconds", {}).get(
                        method
                    ),
                }
            )


def main():
    p = argparse.ArgumentParser(description="HullFT main 1-to-50 sweep.")
    p.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    p.add_argument(
        "--n_values",
        default="1-50",
        help="N values to sweep, e.g. '1-50' or '1,5,10,20,50'.",
    )
    p.add_argument(
        "--repeat", type=int, default=None, help="Override data.repeat (queries per N)."
    )
    p.add_argument(
        "--set",
        action="append",
        default=None,
        help="Extra config overrides, e.g. selection.k_preselect=400.",
    )
    args = p.parse_args()

    base_config = load_config(args.config)
    if args.set:
        apply_config_overrides(base_config, args.set)

    seed = int(base_config.get("data", {}).get("seed", 42))
    set_global_determinism(seed)

    n_values = _parse_n_values(args.n_values)
    print(f"Sweeping N in: {n_values}")

    results_dir = (
        base_config.get("output", {}).get("results_dir") or "outputs/main_sweep"
    )
    os.makedirs(results_dir, exist_ok=True)

    rows: list[dict] = []
    for n in n_values:
        print("\n" + "#" * 70)
        print(f"# N = {n}")
        print("#" * 70)
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("selection", {})["n_select"] = int(n)
        cfg.setdefault("output", {})["results_dir"] = os.path.join(
            results_dir, f"n_{n:02d}"
        )

        summary = run_from_config(
            config_path=args.config,
            config_dict=cfg,
            repeat_override=args.repeat,
        )
        _flatten_results(n, summary, rows)

    csv_path = os.path.join(results_dir, "summary.csv")
    fieldnames = [
        "n_select",
        "method",
        "query_idx",
        "bpb",
        "perplexity",
        "improvement_percent",
        "finetune_seconds",
        "selection_seconds",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSwept {len(n_values)} N values. Aggregate CSV: {csv_path}")


if __name__ == "__main__":
    main()
