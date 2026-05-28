"""SIFT and gradient-reuse variants (Fig 6).

Runs SIFT selection once at N=50, then evaluates the resulting selection with
each requested variant in `--variants`. Each variant is one finetune scheme:

- `global`      -> `sift_global_refresh` (deduplicate globally, then refresh every r)
- `consecutive` -> `sift_consecutive_refresh` (group consecutive duplicates only)

For each variant, the runner evaluates BPB at every r in
`finetune_approach.refresh_intervals` (default `[2, 3]`) in one pass.
"""

import argparse
import copy
import os

from hullft.config import apply_config_overrides, load_config
from hullft.reproducibility import set_global_determinism
from hullft.runner import run_from_config

_VARIANT_TO_METHOD = {
    "global": "sift_global_refresh",
    "consecutive": "sift_consecutive_refresh",
}


def main():
    p = argparse.ArgumentParser(description="SIFT and gradient-reuse variants.")
    p.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=["global", "consecutive"],
        choices=sorted(_VARIANT_TO_METHOD),
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override data.repeat (queries per variant).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=None,
        help="Extra config overrides, e.g. selection.n_select=20.",
    )
    args = p.parse_args()

    base_config = load_config(args.config)
    if args.set:
        apply_config_overrides(base_config, args.set)

    seed = int(base_config.get("data", {}).get("seed", 42))
    set_global_determinism(seed)

    results_root = (
        base_config.get("output", {}).get("results_dir")
        or "outputs/sift_gradient_reuse"
    )
    os.makedirs(results_root, exist_ok=True)

    summaries = {}
    for variant in args.variants:
        method_name = _VARIANT_TO_METHOD[variant]
        print("\n" + "#" * 70)
        print(f"# variant = {variant} ({method_name})")
        print("#" * 70)

        cfg = copy.deepcopy(base_config)
        cfg.setdefault("finetune_approach", {})["method"] = method_name
        cfg.setdefault("output", {})["results_dir"] = os.path.join(
            results_root, variant
        )
        cfg.setdefault("data", {})["cache_name"] = (
            base_config.get("data", {}).get("cache_name", "sift_gradient_reuse")
            + f"_{variant}"
        )

        summaries[variant] = run_from_config(
            config_path=args.config,
            config_dict=cfg,
            repeat_override=args.repeat,
        )

    print("\nVariants run:", list(summaries))


if __name__ == "__main__":
    main()
