"""Build a FAISS train index from monology/pile-uncopyrighted (or a custom JSONL).

Streams documents matching `--subsets` (pile_set_name filter), embeds them
with the pinned RoBERTa-Pile model, and writes IndexFlatL2 shards alongside
their JSONL and byte-offset index files to `--output_dir`.

Use `--input_jsonl` to point at any local JSONL with the same
`{"text", "meta"}` schema.
"""

import argparse
import json
import logging
import os
import struct
import time

import faiss
import numpy as np
from tqdm import tqdm

from hullft.embedder import EMBEDDING_MODEL, RobertaPileEmbedder

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024
DATASET_NAME = "monology/pile-uncopyrighted"

SIFT_SUBSETS = [
    "NIH ExPorter",
    "USPTO Backgrounds",
    "Github",
    "Enron Emails",
    "Wikipedia (en)",
    "PubMed Abstracts",
    "DM Mathematics",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build FAISS index from pile-uncopyrighted (or custom JSONL)."
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["ArXiv"],
        help="pile_set_name values to include, or 'all' for SIFT_SUBSETS.",
    )
    p.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional: read text and meta from a local JSONL instead of streaming HF.",
    )
    p.add_argument("--output_dir", type=str, default="data/pile/train")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--shard_size", type=int, default=500_000, help="Max docs per shard/index file."
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF split when --input_jsonl is not given.",
    )
    p.add_argument(
        "--max_docs_per_subset",
        type=int,
        default=None,
        help="Global cap on docs per subset.",
    )
    return p.parse_args()


def _stream_pile_subset(split, allowed_subsets, max_per_subset=None):
    from datasets import load_dataset

    logger.info("Loading %s split of %s (streaming)...", split, DATASET_NAME)
    if split in ("test", "val"):
        data_file = f"hf://datasets/{DATASET_NAME}/{split}.jsonl.zst"
        ds = load_dataset(
            "json", data_files={split: data_file}, split=split, streaming=True
        )
    else:
        ds = load_dataset(DATASET_NAME, split=split, streaming=True)

    counts = {}
    for example in ds:
        meta = example.get("meta", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        name = meta.get("pile_set_name", "")
        if name not in allowed_subsets:
            continue
        if max_per_subset is not None and counts.get(name, 0) >= max_per_subset:
            continue
        text = (example.get("text", "") or "").strip()
        if not text:
            continue
        counts[name] = counts.get(name, 0) + 1
        yield text, name


def _stream_local_jsonl(path, allowed_subsets, max_per_subset=None):
    counts = {}
    with open(path, "r") as f:
        for line in f:
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = example.get("meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            name = meta.get("pile_set_name", "")
            if allowed_subsets and name not in allowed_subsets:
                continue
            if max_per_subset is not None and counts.get(name, 0) >= max_per_subset:
                continue
            text = (example.get("text", "") or "").strip()
            if not text:
                continue
            counts[name] = counts.get(name, 0) + 1
            yield text, name


def _build_shard_index(texts, embedder, batch_size, dim):
    index = faiss.IndexFlatL2(dim)
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[start : start + batch_size]
        embs = embedder.embed_texts(batch, batch_size=batch_size)
        index.add(embs.astype(np.float32))
    return index


def _write_offsets(jsonl_path, offset_path):
    offsets = []
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    with open(offset_path, "wb") as f:
        for off in offsets:
            f.write(struct.pack("<Q", off))


def _flush_shard(shard_idx, texts, subsets, embedder, output_dir, batch_size, dim):
    shard_name = f"{shard_idx:02d}"
    jsonl_path = os.path.join(output_dir, f"{shard_name}.jsonl")
    index_path = os.path.join(output_dir, f"{shard_name}.jsonl.index")
    offset_path = os.path.join(output_dir, f"{shard_name}.jsonl.offsets")

    logger.info("Flushing shard %s: %d docs", shard_name, len(texts))
    with open(jsonl_path, "w") as f:
        for text, subset in zip(texts, subsets):
            f.write(
                json.dumps(
                    {"text": text, "meta": {"pile_set_name": subset}},
                    ensure_ascii=False,
                )
                + "\n"
            )

    index = _build_shard_index(texts, embedder, batch_size, dim)
    faiss.write_index(index, index_path)
    _write_offsets(jsonl_path, offset_path)


def build_index(args):
    allowed_subsets = (
        set(SIFT_SUBSETS) if args.subsets == ["all"] else set(args.subsets)
    )
    logger.info("Target subsets: %s", allowed_subsets)
    os.makedirs(args.output_dir, exist_ok=True)

    embedder = RobertaPileEmbedder(device=args.device)
    dim = embedder._model.config.hidden_size

    if args.input_jsonl:
        stream = _stream_local_jsonl(
            args.input_jsonl,
            allowed_subsets,
            max_per_subset=args.max_docs_per_subset,
        )
    else:
        stream = _stream_pile_subset(
            args.split,
            allowed_subsets,
            max_per_subset=args.max_docs_per_subset,
        )

    shard_idx = 0
    shard_texts, shard_subsets = [], []
    total_docs = 0
    t0 = time.time()

    for text, subset_name in stream:
        shard_texts.append(text)
        shard_subsets.append(subset_name)
        if len(shard_texts) >= args.shard_size:
            _flush_shard(
                shard_idx,
                shard_texts,
                shard_subsets,
                embedder,
                args.output_dir,
                args.batch_size,
                dim,
            )
            total_docs += len(shard_texts)
            shard_idx += 1
            shard_texts, shard_subsets = [], []

    if shard_texts:
        _flush_shard(
            shard_idx,
            shard_texts,
            shard_subsets,
            embedder,
            args.output_dir,
            args.batch_size,
            dim,
        )
        total_docs += len(shard_texts)

    elapsed = time.time() - t0
    logger.info(
        "Done. %d total docs in %d shards (%.1f min)",
        total_docs,
        shard_idx + 1,
        elapsed / 60,
    )

    manifest = {
        "total_docs": total_docs,
        "n_shards": shard_idx + 1,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": dim,
        "index_type": "IndexFlatL2",
        "normalized": True,
        "subsets": sorted(allowed_subsets),
        "split": args.split,
        "input_jsonl": args.input_jsonl,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    build_index(parse_args())
