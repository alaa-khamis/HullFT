# CPU test (N=20)

Runs the full selection + finetune pipeline on CPU at `N=20`. The point of this
experiment is **timing**, not accuracy — so the launcher refuses to start unless
both the candidate JSON *and* the precomputed embedding NPZ exist, ensuring
RoBERTa is never loaded on CPU at runtime.

## Run

```bash
python experiments/cpu_test/launch.py \
    --config experiments/cpu_test/config.yaml
```

Optional flags:

- `--repeat 30` — override the query count
- `--set selection.n_select=10` — pass any dot-path override

## Prerequisites

```bash
python -m data.ensure_test_jsonl --test_dir data/pile/test
python -m data.build_train_index --subsets "ArXiv" --output_dir data/pile/train
python -m data.precompute_candidates --subsets "ArXiv"   # writes data/pile/precomputed/
python -m data.precompute_embeddings --subsets "ArXiv"   # writes data/pile/precomputed_embeddings/
```

## Outputs

- `outputs/cpu_test_arxiv/cpu_test_arxiv_<timestamp>_<id>/summary.json`

Each `summary.results[i].evaluation[method]` carries `finetune_seconds` and
each `summary.results[i].method_runtimes_seconds[method]` carries the per-method
selection time — these two numbers are what the CPU test reports.

## Swapping subset / dataset

To run on a different Pile subset (e.g. `Github`):

1. `python -m data.build_train_index --subsets "Github"`
2. `python -m data.precompute_candidates --subsets "Github"`
3. `python -m data.precompute_embeddings --subsets "Github"`
4. Edit `config.yaml` -> `pile_index.subset_filter: Github`
