"""Orchestration for a single experiment run.

Precomputed-only: every query and candidate text comes from
`data/precompute_candidates.py` JSON output. When precomputed embeddings
are also available, RoBERTa is skipped entirely.

Public entry point: `run_from_config(config_path, config_dict=None)`.
"""

import copy
import gc
import json
import os
import time
import uuid
from datetime import datetime

import numpy as np
import torch

from .config import load_config, save_config_to_dir
from .evaluation import compare_bpb_results, evaluate_selections
from .fill import integerize_by_geometry, pad_by_weights_deterministic
from .reproducibility import set_global_determinism
from .selector import get_selector

_CONVEX_METHODS = {"fw"}


def _subset_to_filename(subset_name):
    safe = (
        subset_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return f"{safe}.json"


def _build_run_output_dir(base_dir, cache_name):
    safe_cache_name = (cache_name or "run").replace(" ", "_")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    identifier = uuid.uuid4().hex[:6]
    return os.path.join(base_dir, f"{safe_cache_name}_{timestamp}_{identifier}")


def _run_selection_methods(
    methods,
    method_params,
    pool_embeddings,
    query_embedding,
    pool_global_indices,
    n_select,
    convex_hull_config,
):
    """Run each selector on the candidate pool. Apply fill strategies to convex methods."""
    convex_hull_config = convex_hull_config or {}
    fill_strategies = convex_hull_config.get("fill_strategies", ["none"])
    fill_strategies = [str(s).lower() for s in fill_strategies]
    swap_passes = int(convex_hull_config.get("swap_passes", 2))
    force_fill = bool(convex_hull_config.get("force_fill", False))

    query_target = query_embedding.astype("float64")
    if query_target.ndim == 1:
        query_target = query_target.reshape(1, -1)
    query_target = query_target.mean(axis=0)

    selections_global = {}
    selections_weights = {}
    method_runtimes = {}

    for method_name in methods:
        is_convex = method_name in _CONVEX_METHODS
        current_strategies = fill_strategies if is_convex else ["none"]

        for strategy in current_strategies:
            if strategy == "none":
                display_name = method_name
            elif strategy == "pad_by_weights":
                display_name = f"{method_name}_padded"
            elif strategy == "integerization":
                display_name = f"{method_name}_integerized"
            else:
                display_name = f"{method_name}_{strategy}"

            print(f"\nRunning {display_name}...")
            params = (method_params or {}).get(method_name, {}).copy()
            selector = get_selector(method_name, **params)

            start_time = time.perf_counter()
            indices, weights = selector.select(
                pool_embeddings, query_embedding, n_select
            )
            indices = np.asarray(indices).reshape(-1)

            if (
                is_convex
                and weights is not None
                and len(weights) > 0
                and (len(indices) < n_select or force_fill)
            ):
                if strategy == "pad_by_weights":
                    indices, weights = pad_by_weights_deterministic(
                        indices, weights, n_select
                    )
                elif strategy == "integerization":
                    support_embeddings = pool_embeddings.astype("float64")[indices]
                    indices, weights = integerize_by_geometry(
                        indices,
                        weights,
                        support_embeddings,
                        query_target,
                        n_select,
                        swap_passes=swap_passes,
                    )

            if is_convex and weights is not None:
                w = np.asarray(weights).reshape(-1)
                if len(w) == len(indices) and len(indices) > 0:
                    order = np.argsort(-w, kind="stable")
                    indices = indices[order]
                    weights = w[order]

            if not is_convex:
                weights = None

            selections_global[display_name] = pool_global_indices[indices]
            selections_weights[display_name] = weights
            method_runtimes[display_name] = time.perf_counter() - start_time
            print(f"  Selected {len(indices)} points")

    if not selections_global:
        raise ValueError("No methods ran successfully.")

    return selections_global, selections_weights, method_runtimes


def _ensure_query_embeddings(precomputed_entry, embedder, k_preselect):
    """Return (query_text, query_emb, pool_global_idx, pool_emb, pool_texts).

    Embeds via `embedder` only if the precomputed entry lacks embeddings.
    """
    query_text = precomputed_entry["query_text"]
    candidate_texts = precomputed_entry["candidates"]
    query_emb = precomputed_entry.get("query_embedding")
    pool_emb = precomputed_entry.get("candidate_embeddings")

    if pool_emb is not None and len(pool_emb) < len(candidate_texts):
        candidate_texts = candidate_texts[: len(pool_emb)]
    if k_preselect is not None and k_preselect < len(candidate_texts):
        candidate_texts = candidate_texts[:k_preselect]
        if pool_emb is not None and len(pool_emb) > k_preselect:
            pool_emb = pool_emb[:k_preselect]

    if "candidate_indices" in precomputed_entry:
        pool_global_idx = np.array(precomputed_entry["candidate_indices"], dtype=int)
        if k_preselect is not None and k_preselect < len(pool_global_idx):
            pool_global_idx = pool_global_idx[:k_preselect]
    else:
        pool_global_idx = np.arange(len(candidate_texts))

    if query_emb is None:
        if embedder is None:
            raise RuntimeError(
                "Precomputed entry has no query embedding and no embedder was provided."
            )
        query_emb = embedder.embed_query(query_text).squeeze(0)
    else:
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)

    if pool_emb is None:
        if embedder is None:
            raise RuntimeError(
                "Precomputed entry has no candidate embeddings and no embedder was provided."
            )
        pool_emb = embedder.embed_texts(candidate_texts, batch_size=32)
    else:
        pool_emb = np.asarray(pool_emb, dtype=np.float32)

    return query_text, query_emb, pool_global_idx, pool_emb, candidate_texts


def _run_one_query(
    precomputed_entry,
    embedder,
    n_select,
    methods,
    method_params,
    convex_hull_config,
    k_preselect,
    eval_cfg,
    eval_model_obj,
    eval_tokenizer_obj,
    finetune_approach_config,
    seed,
    eval_device,
):
    """Run selection and BPB evaluation for a single query."""
    run_start = time.perf_counter()

    query_text, query_emb, pool_global_idx, pool_emb, candidate_texts = (
        _ensure_query_embeddings(precomputed_entry, embedder, k_preselect)
    )

    stage_start = time.perf_counter()
    selections_global, selections_weights, method_runtimes = _run_selection_methods(
        methods,
        method_params,
        pool_emb,
        query_emb,
        pool_global_idx,
        n_select,
        convex_hull_config,
    )
    selection_seconds = time.perf_counter() - stage_start

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bpb_results = None
    eval_metadata = {"finetune_warmup_seconds": None}
    if eval_cfg.get("enabled", False) and eval_cfg.get("model_name"):
        bpb_results, eval_metadata = evaluate_selections(
            eval_cfg["model_name"],
            selections_global,
            candidate_texts,
            [query_text],
            batch_size=eval_cfg.get("batch_size", 8),
            max_length=eval_cfg.get("max_length", 1024),
            device=eval_device,
            finetune_batch_size=eval_cfg.get(
                "finetune_batch_size", eval_cfg.get("batch_size", 8)
            ),
            finetune_max_length=eval_cfg.get(
                "finetune_max_length", eval_cfg.get("max_length", 1024)
            ),
            finetune_lr=eval_cfg.get("finetune_lr", 5e-5),
            model=eval_model_obj,
            tokenizer=eval_tokenizer_obj,
            seed=seed,
            return_metadata=True,
            finetune_approach_config=finetune_approach_config,
        )
        compare_bpb_results(bpb_results)

    return {
        "query_idx_test": int(precomputed_entry["query_idx"]),
        "query_text_bytes": len(query_text.encode("utf-8")),
        "candidate_pool_indices": pool_global_idx.tolist(),
        "method_selections": {
            m: np.asarray(idx).reshape(-1).tolist()
            for m, idx in selections_global.items()
        },
        "method_selection_weights": {
            m: (np.asarray(w).reshape(-1).tolist() if w is not None else None)
            for m, w in selections_weights.items()
        },
        "method_runtimes_seconds": method_runtimes,
        "method_selection_counts": {
            m: len(idx) for m, idx in selections_global.items()
        },
        "selection_seconds": selection_seconds,
        "evaluation": bpb_results,
        "finetune_warmup_seconds": eval_metadata.get("finetune_warmup_seconds"),
        "total_run_seconds": time.perf_counter() - run_start,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def run_from_config(
    config_path: str,
    config_dict=None,
    output_dir=None,
    n_select_override=None,
    repeat_override=None,
):
    """Run an experiment from a YAML config. Returns a list of per-query result dicts."""
    config = config_dict if config_dict is not None else load_config(config_path)

    data_cfg = config.get("data", {})
    select_cfg = config.get("selection", {})
    method_params = config.get("method_params", {})
    eval_cfg = config.get("evaluation", {})
    output_cfg = config.get("output", {})
    convex_hull_cfg = select_cfg.get("convex_hull", {})
    pile_cfg = config.get("pile_index", {})
    finetune_approach_cfg = config.get("finetune_approach", {})

    seed = int(data_cfg.get("seed", 42))
    set_global_determinism(seed)

    eval_device = eval_cfg.get("device") or select_cfg.get("device")

    n_select = int(
        n_select_override
        if n_select_override is not None
        else select_cfg.get("n_select", 20)
    )
    n_repeats = int(
        repeat_override
        if repeat_override is not None
        else (data_cfg.get("repeat", 1) or 1)
    )
    methods = select_cfg.get("methods", ["fw"])
    k_preselect = int(select_cfg.get("k_preselect", 200))
    subset_filter = pile_cfg.get("subset_filter")
    if not subset_filter:
        raise ValueError("pile_index.subset_filter is required.")

    precomputed_dir = pile_cfg.get("precomputed_candidates_dir")
    if not precomputed_dir:
        raise ValueError("pile_index.precomputed_candidates_dir is required.")
    subset_file = os.path.join(precomputed_dir, _subset_to_filename(subset_filter))
    if not os.path.exists(subset_file):
        raise FileNotFoundError(
            f"Precomputed file not found: {subset_file}. Run data/precompute_candidates.py first."
        )
    with open(subset_file) as f:
        precomputed_data = json.load(f)
    queries = precomputed_data.get("queries", [])
    if not queries:
        raise ValueError(f"No queries in {subset_file}.")

    # Attach precomputed embeddings if available.
    precomputed_emb_dir = pile_cfg.get("precomputed_embeddings_dir")
    if precomputed_emb_dir:
        emb_file = os.path.join(
            precomputed_emb_dir,
            _subset_to_filename(subset_filter).replace(".json", ".npz"),
        )
        if not os.path.exists(emb_file):
            raise FileNotFoundError(
                f"precomputed_embeddings_dir is set but file not found: {emb_file}. "
                "Run data/precompute_embeddings.py first."
            )
        npz = np.load(emb_file, allow_pickle=True)
        q_emb = npz["query_emb"]
        c_emb = npz["candidate_emb"]
        q_idx_list = npz["query_idx"].tolist()
        idx_to_row = {int(q): i for i, q in enumerate(q_idx_list)}
        attached = 0
        for q in queries:
            row = idx_to_row.get(int(q["query_idx"]))
            if row is None:
                continue
            q["query_embedding"] = np.asarray(q_emb[row], dtype=np.float32)
            q["candidate_embeddings"] = np.asarray(c_emb[row], dtype=np.float32)
            attached += 1
        print(
            f"Loaded precomputed embeddings: {emb_file} "
            f"(attached {attached}/{len(queries)})"
        )

    n_repeats = min(n_repeats, len(queries))
    queries = queries[:n_repeats]

    all_have_embeddings = all(
        q.get("query_embedding") is not None
        and q.get("candidate_embeddings") is not None
        for q in queries
    )

    embedder = None
    if not all_have_embeddings:
        from .embedder import RobertaPileEmbedder

        embedder_device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = RobertaPileEmbedder(device=embedder_device)

    eval_model_obj = None
    eval_tokenizer_obj = None
    if eval_cfg.get("enabled", False):
        model_name = eval_cfg["model_name"]
        print(f"\nPre-loading evaluation model: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        eval_tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
        if eval_tokenizer_obj.pad_token is None:
            eval_tokenizer_obj.pad_token = eval_tokenizer_obj.eos_token
        eval_model_obj = AutoModelForCausalLM.from_pretrained(model_name)

    cache_name = data_cfg.get("cache_name", "experiment")
    base_output_dir = (
        output_dir
        or output_cfg.get("results_dir")
        or os.path.join(os.getcwd(), "outputs", cache_name)
    )
    run_output_dir = _build_run_output_dir(base_output_dir, cache_name)
    os.makedirs(run_output_dir, exist_ok=True)
    save_config_to_dir(copy.deepcopy(config), run_output_dir)

    results = []
    for run_idx, entry in enumerate(queries, start=1):
        print("\n" + "=" * 60)
        print(f"Query {run_idx}/{len(queries)} (test idx {entry['query_idx']})")
        print("=" * 60)
        result = _run_one_query(
            precomputed_entry=entry,
            embedder=embedder,
            n_select=n_select,
            methods=methods,
            method_params=method_params,
            convex_hull_config=convex_hull_cfg,
            k_preselect=k_preselect,
            eval_cfg=eval_cfg,
            eval_model_obj=eval_model_obj,
            eval_tokenizer_obj=eval_tokenizer_obj,
            finetune_approach_config=finetune_approach_cfg,
            seed=seed,
            eval_device=eval_device,
        )
        result["repeat_index"] = run_idx
        result["repeat_total"] = len(queries)
        results.append(result)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "config_path": config_path,
        "subset_filter": subset_filter,
        "seed": seed,
        "n_select": n_select,
        "n_queries": len(queries),
        "methods": methods,
        "query_indices": [int(q["query_idx"]) for q in queries],
        "results": results,
    }
    summary_path = os.path.join(run_output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nSummary saved to {summary_path}")

    return summary


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
