"""Precompute query and candidate embeddings.

For each precomputed-candidates JSON in `--input_dir`, write a companion
`.npz` with:

- `query_idx`     : (N,) int64 original TEST index per query
- `query_emb`     : (N, D) float32 L2-normalized query embedding
- `candidate_emb` : (N, K, D) float32 L2-normalized candidate embeddings

The embedding calls here must match the runtime shapes exactly:
`embed_query(text)` per query and `embed_texts(candidates, batch_size=32)`
per query. fp32 GEMM is not shape-invariant, so changing batch shape will
drift outputs and break parity with the runtime path.
"""

import argparse
import json
import os

import numpy as np
import torch

from hullft.embedder import RobertaPileEmbedder


def _subset_to_filename(subset_name):
    safe = (
        subset_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return f"{safe}.json"


def precompute_for_subset(embedder, in_path, out_path, k_candidates, overwrite):
    if os.path.exists(out_path) and not overwrite:
        print(f"  Skip (exists): {out_path}")
        return

    print(f"  Loading candidates JSON: {in_path}")
    with open(in_path, "r") as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if not queries:
        print(f"  No queries in {in_path}, skipping")
        return

    k_min = min(len(q["candidates"]) for q in queries)
    k_use = k_min if k_candidates is None else min(k_candidates, k_min)
    print(f"  {len(queries)} queries, k_candidates={k_use} (json min={k_min})")

    query_emb_list, cand_emb_list = [], []
    for qi, q in enumerate(queries):
        if qi % 10 == 0:
            print(
                f"  Embedding query {qi + 1}/{len(queries)} with {k_use} candidates ...",
                flush=True,
            )
        # Zero-deviation rule: per-query embed_query and per-query embed_texts(batch_size=32).
        q_emb = embedder.embed_query(q["query_text"]).squeeze(0)
        c_emb = embedder.embed_texts(q["candidates"][:k_use], batch_size=32)
        query_emb_list.append(q_emb)
        cand_emb_list.append(c_emb)

    query_emb = np.stack(query_emb_list, axis=0).astype(np.float32)
    cand_emb = np.stack(cand_emb_list, axis=0).astype(np.float32)
    dim = int(query_emb.shape[1])
    assert cand_emb.shape == (len(queries), k_use, dim)

    query_idx = np.array([int(q["query_idx"]) for q in queries], dtype=np.int64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp.npz"
    np.savez(
        tmp_path,
        query_idx=query_idx,
        query_emb=query_emb,
        candidate_emb=cand_emb,
        subset=np.array(data.get("subset", ""), dtype=object),
        k_candidates=np.int64(k_use),
    )
    os.replace(tmp_path, out_path)
    print(f"  Saved -> {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(
        description="Precompute query and candidate embeddings."
    )
    p.add_argument(
        "--input_dir",
        default="data/pile/precomputed",
        help="Directory of precomputed candidate JSON files.",
    )
    p.add_argument(
        "--output_dir",
        default="data/pile/precomputed_embeddings",
        help="Where to save the per-subset .npz files.",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subset names to process (default: all JSONs in --input_dir).",
    )
    p.add_argument(
        "--k_candidates",
        type=int,
        default=None,
        help="Embed only the first K candidates per query. Defaults to all.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.subsets:
        in_files = []
        for subset in args.subsets:
            path = os.path.join(args.input_dir, _subset_to_filename(subset))
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Input JSON not found for subset '{subset}': {path}. "
                    "Run data/precompute_candidates.py first."
                )
            in_files.append(path)
    else:
        in_files = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".json")
        )
        if not in_files:
            raise FileNotFoundError(f"No .json files in {args.input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device} ...")
    embedder = RobertaPileEmbedder(device=device)

    print(f"\nProcessing {len(in_files)} subset file(s) -> {args.output_dir}")
    for in_path in in_files:
        fname = os.path.basename(in_path)
        out_path = os.path.join(args.output_dir, fname.replace(".json", ".npz"))
        print(f"\n{'=' * 60}\nSubset file: {fname}")
        precompute_for_subset(
            embedder,
            in_path=in_path,
            out_path=out_path,
            k_candidates=args.k_candidates,
            overwrite=args.overwrite,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
