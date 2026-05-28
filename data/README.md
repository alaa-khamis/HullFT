# `data/` — offline precompute scripts

These scripts are the only place in the repo that touches FAISS or the
HuggingFace Pile dataset. The runtime (`hullft/`) reads only the JSON / NPZ
files written here.

## Scripts

| Script | What it does |
|---|---|
| `ensure_test_jsonl.py` | Downloads `pile-uncopyrighted/test.jsonl.zst` and decompresses it. |
| `build_train_index.py` | Streams the pile train split (or any local JSONL), embeds with RoBERTa-Pile, writes IndexFlatL2 shards + JSONL + byte offsets. |
| `precompute_candidates.py` | For each subset, samples N queries and writes their top-K candidate texts (JSON). |
| `precompute_embeddings.py` | Embeds the precomputed queries + candidates (NPZ). Only needed for `cpu_test`. |
| `_index_loader.py` | (Internal) Multi-shard FAISS loader used by the two precompute scripts. |

Run any of them with `--help` for full flags.

## Custom dataset

`build_train_index.py` accepts `--input_jsonl PATH`. The JSONL must have one
record per line in the form

```json
{"text": "...", "meta": {"pile_set_name": "<subset name>"}}
```

After indexing, point the precompute scripts and the experiment configs at
`pile_set_name == "<subset name>"`.
