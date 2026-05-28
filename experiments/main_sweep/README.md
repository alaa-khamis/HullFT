# Main 1-to-50 sweep

Compares HullFT (`fw` + `integerization` + `gradient_refresh`) against `knn`
and `sift` baselines across selection budgets `N = 1..50` on a single Pile
subset.

## Run

```bash
python experiments/main_sweep/launch.py \
    --config experiments/main_sweep/config.yaml
```

Optional flags:

- `--n_values "1,5,10,20,50"` — sweep a custom subset of N values
- `--repeat 30` — override the per-N query count (default 150)
- `--set selection.k_preselect=400` — pass any dot-path override

## Prerequisites

```bash
python -m data.ensure_test_jsonl --test_dir data/pile/test
python -m data.build_train_index --subsets "ArXiv" --output_dir data/pile/train
python -m data.precompute_candidates \
    --train_index_dir data/pile/train \
    --test_dir data/pile/test \
    --output_dir data/pile/precomputed \
    --subsets "ArXiv" --n_queries 150 --k_candidates 200 --seed 42
```

## Outputs

- `outputs/main_sweep_arxiv/n_{N:02d}/` — per-N run directory (one
  `summary.json` per N with full per-query results)
- `outputs/main_sweep_arxiv/summary.csv` — flat CSV across all N:
  `n_select, method, query_idx, bpb, perplexity, improvement_percent,
  finetune_seconds, selection_seconds`

## Swapping subset / dataset

To run on a different Pile subset (e.g. `Github`):

1. Build the train index for it: `python -m data.build_train_index --subsets "Github"`
2. Precompute candidates: `python -m data.precompute_candidates --subsets "Github"`
3. Edit `config.yaml` -> `pile_index.subset_filter: Github`

To run on an arbitrary dataset, point
`python -m data.build_train_index --input_jsonl PATH` at any JSONL whose
records have `{"text": ..., "meta": {"pile_set_name": ...}}`.
