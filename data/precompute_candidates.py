"""Precompute KNN candidate pools per subset.

For each subset, samples `--n_queries` test queries (seeded) and for each
query retrieves its `--k_candidates` nearest neighbors from the train index.
Writes `<output_dir>/<subset>.json` with the query text and candidate texts.
"""

import argparse
import json
import os

import numpy as np

from hullft.embedder import load_pile_test_texts
from data._index_loader import PileIndexLoader


def _subset_to_filename(subset_name):
    safe = (
        subset_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return f"{safe}.json"


def precompute_subset(
    pile_loader,
    test_dir,
    subset,
    n_queries,
    k_candidates,
    seed,
    min_text_chars,
    rng,
    out_path,
):
    print(f"\n{'=' * 60}\nProcessing subset: {subset}")
    texts = load_pile_test_texts(test_dir, subset_name=subset)
    print(f"  Found {len(texts)} test texts")

    valid = [
        (i, t) for i, t in enumerate(texts) if len((t or "").strip()) >= min_text_chars
    ]
    print(f"  Valid texts (>= {min_text_chars} chars): {len(valid)}")
    if not valid:
        print(f"  No valid texts for subset '{subset}', skipping")
        return

    n_q = min(n_queries, len(valid))
    chosen_positions = rng.choice(len(valid), size=n_q, replace=False)
    chosen = [valid[i] for i in sorted(chosen_positions)]
    print(f"  Selected {n_q} unique queries")

    query_data = []
    for qi, (orig_idx, query_text) in enumerate(chosen):
        if qi % 10 == 0:
            print(f"  Query {qi + 1}/{n_q}...", flush=True)
        _, _, candidate_texts = pile_loader.search(query_text, k=k_candidates)
        query_data.append(
            {
                "query_idx": orig_idx,
                "query_text": query_text,
                "candidates": candidate_texts,
            }
        )

    with open(out_path, "w") as f:
        json.dump(
            {
                "subset": subset,
                "n_queries": n_q,
                "k_candidates": k_candidates,
                "seed": seed,
                "queries": query_data,
            },
            f,
        )
    print(f"  Saved {n_q} queries x {k_candidates} candidates -> {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Precompute KNN candidates per pile subset."
    )
    p.add_argument("--train_index_dir", default="data/pile/train")
    p.add_argument("--test_dir", default="data/pile/test")
    p.add_argument("--output_dir", default="data/pile/precomputed")
    p.add_argument("--n_queries", type=int, default=150)
    p.add_argument("--k_candidates", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_text_chars", type=int, default=1)
    p.add_argument("--subsets", nargs="+", default=["ArXiv"])
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for subset in args.subsets:
        out_path = os.path.join(args.output_dir, _subset_to_filename(subset))
        if os.path.exists(out_path) and not args.overwrite:
            print(
                f"\nSkipping '{subset}' (already exists: {out_path}). Use --overwrite to recompute."
            )
            continue

        print(
            f"\nLoading train FAISS index from: {args.train_index_dir} (subset='{subset}')"
        )
        pile_loader = PileIndexLoader(
            args.train_index_dir,
            device="cuda",
            subset_filter=subset,
        )
        print(f"Train index loaded: {pile_loader.total_docs:,} docs")

        precompute_subset(
            pile_loader=pile_loader,
            test_dir=args.test_dir,
            subset=subset,
            n_queries=args.n_queries,
            k_candidates=args.k_candidates,
            seed=args.seed,
            min_text_chars=args.min_text_chars,
            rng=rng,
            out_path=out_path,
        )

    print("\nDone. Precomputed files:")
    for subset in args.subsets:
        out_path = os.path.join(args.output_dir, _subset_to_filename(subset))
        if os.path.exists(out_path):
            print(f"  {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
