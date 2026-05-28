"""CLI entrypoint: `python -m hullft.main --config <yaml> [--set key=value ...]`."""

import argparse
import json
import os

from .config import apply_config_overrides, load_config
from .reproducibility import set_global_determinism
from .runner import run_from_config


def _build_parser():
    p = argparse.ArgumentParser(
        description="Run a HullFT experiment from a YAML config."
    )
    p.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    p.add_argument("--output_dir", type=str, default=None, help="Override output dir.")
    p.add_argument(
        "--n_select", type=int, default=None, help="Override selection.n_select."
    )
    p.add_argument("--repeat", type=int, default=None, help="Override data.repeat.")
    p.add_argument(
        "--set",
        action="append",
        default=None,
        help="Override config field via dot path (repeatable), e.g. selection.n_select=25",
    )
    return p


def main():
    args = _build_parser().parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)
    overrides = []
    if args.set:
        overrides.extend(args.set)
    apply_config_overrides(config, overrides)

    seed = int(config.get("data", {}).get("seed", 42))
    set_global_determinism(seed)

    return run_from_config(
        config_path=args.config,
        config_dict=config,
        output_dir=args.output_dir,
        n_select_override=args.n_select,
        repeat_override=args.repeat,
    )


if __name__ == "__main__":
    main()
