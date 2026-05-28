# SIFT + gradient reuse (Fig 6)

Runs SIFT selection once at `N=50` and evaluates the resulting selection
under multiple gradient-reuse variants and refresh intervals `r`. By default
runs both variants, each at `r ∈ {2, 3}`, in a single pass.

## Variants

- `global` (`sift_global_refresh`) — globally deduplicate the selection, then
  recompute gradient every `r` steps per unique text.
- `consecutive` (`sift_consecutive_refresh`) — group only *consecutive*
  duplicates, preserving SIFT's natural re-appearances.

Result keys land as `sift_global_refresh_r{r}` and
`sift_consecutive_refresh_r{r}` inside each `summary.json`.

## Run

```bash
python experiments/sift_gradient_reuse/launch.py \
    --config experiments/sift_gradient_reuse/config.yaml
```

Optional flags:

- `--variants global` — run only one variant
- `--repeat 30` — override the query count
- `--set finetune_approach.refresh_intervals='[2,3,5]'` — sweep more r values

## Prerequisites

```bash
python -m data.ensure_test_jsonl --test_dir data/pile/test
python -m data.build_train_index --subsets "ArXiv" --output_dir data/pile/train
python -m data.precompute_candidates --subsets "ArXiv"
```

## Outputs

- `outputs/sift_gradient_reuse_arxiv/global/.../summary.json`
- `outputs/sift_gradient_reuse_arxiv/consecutive/.../summary.json`

## Swapping subset / dataset

1. `python -m data.build_train_index --subsets "<Name>"`
2. `python -m data.precompute_candidates --subsets "<Name>"`
3. Edit `config.yaml` -> `pile_index.subset_filter: <Name>`
