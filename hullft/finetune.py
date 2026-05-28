"""Test-time finetuning on selected texts."""

import time
from collections import OrderedDict

import torch


def finetune_on_texts(
    model,
    tokenizer,
    texts,
    batch_size=1,
    max_length=1024,
    lr=5e-5,
    device=None,
):
    """Finetune a causal LM on the provided texts with one Adam step per (mini-)batch."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = float(lr)
    batch_size = int(batch_size)
    max_length = int(max_length)

    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, eps=1e-8)

    steps = 0
    start_time = time.perf_counter()

    if len(texts) == 0:
        return {"finetune_steps": 0, "finetune_seconds": 0.0}

    if batch_size != 1:
        print(
            f"Warning: finetune_batch_size={batch_size} != 1. "
            "This does NOT match the paper's 'one gradient step per selected sequence'."
        )

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels

        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1

    finetune_seconds = time.perf_counter() - start_time
    model.eval()

    return {
        "finetune_steps": steps,
        "finetune_seconds": finetune_seconds,
    }


def _deduplicate_texts(texts):
    """Collapse duplicate texts globally, preserving first-seen order."""
    seen = OrderedDict()
    unique_texts = []
    counts = []
    for t in texts:
        if t in seen:
            counts[seen[t]] += 1
        else:
            seen[t] = len(unique_texts)
            unique_texts.append(t)
            counts.append(1)
    return unique_texts, counts


def _group_consecutive_texts(texts):
    """Collapse only consecutive duplicates into (run_texts, run_counts)."""
    run_texts = []
    run_counts = []
    for t in texts:
        if run_texts and run_texts[-1] == t:
            run_counts[-1] += 1
        else:
            run_texts.append(t)
            run_counts.append(1)
    return run_texts, run_counts


def _tokenize_single(tokenizer, text, max_length, device):
    inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return inputs


def _finetune_gradient_refresh(
    model,
    tokenizer,
    texts,
    counts,
    batch_size,
    max_length,
    lr,
    device,
    refresh_interval=2,
):
    """For each unique text with count J_k, recompute gradient every R steps, reuse otherwise."""
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, eps=1e-8)
    steps = 0
    R = max(1, int(refresh_interval))

    for text, count in zip(texts, counts):
        inputs = _tokenize_single(tokenizer, text, max_length, device)
        cached_grads = None

        for j in range(count):
            if j % R == 0:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                cached_grads = {
                    name: p.grad.clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                if cached_grads is not None:
                    for name, p in model.named_parameters():
                        if name in cached_grads:
                            p.grad = cached_grads[name].clone()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            steps += 1

    model.eval()
    return steps


_APPROACH_METHODS = {
    "standard",
    "gradient_refresh",
    "sift_consecutive_refresh",
    "sift_global_refresh",
}


def finetune_with_approach(
    model,
    tokenizer,
    texts,
    approach_config,
    batch_size=1,
    max_length=1024,
    lr=5e-5,
    device=None,
):
    """Dispatch to the requested finetune method (see `_APPROACH_METHODS`)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = float(lr)
    max_length = int(max_length)

    if approach_config is None:
        approach_config = {}
    method = approach_config.get("method", "standard")

    if method == "standard":
        result = finetune_on_texts(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            max_length=max_length,
            lr=lr,
            device=device,
        )
        result["finetune_approach"] = "standard"
        return result

    if len(texts) == 0:
        return {
            "finetune_steps": 0,
            "finetune_seconds": 0.0,
            "finetune_approach": method,
        }

    start_time = time.perf_counter()

    if method == "gradient_refresh":
        unique_texts, counts = _deduplicate_texts(texts)
        n_unique = len(unique_texts)
        n_total = sum(counts)
        print(
            f"  Finetune approach '{method}': {n_total} texts -> {n_unique} unique, "
            f"counts={counts[:10]}{'...' if len(counts) > 10 else ''}"
        )
        refresh_interval = int(approach_config.get("refresh_interval", 2))
        steps = _finetune_gradient_refresh(
            model,
            tokenizer,
            unique_texts,
            counts,
            batch_size,
            max_length,
            lr,
            device,
            refresh_interval=refresh_interval,
        )
        finetune_unique_texts = n_unique
    elif method == "sift_global_refresh":
        unique_texts, counts = _deduplicate_texts(texts)
        n_unique = len(unique_texts)
        n_total = sum(counts)
        print(
            f"  Finetune approach '{method}': {n_total} texts -> {n_unique} unique, "
            f"counts={counts[:10]}{'...' if len(counts) > 10 else ''}"
        )
        refresh_interval = int(approach_config.get("refresh_interval", 2))
        steps = _finetune_gradient_refresh(
            model,
            tokenizer,
            unique_texts,
            counts,
            batch_size,
            max_length,
            lr,
            device,
            refresh_interval=refresh_interval,
        )
        finetune_unique_texts = n_unique
    elif method == "sift_consecutive_refresh":
        run_texts, run_counts = _group_consecutive_texts(texts)
        n_runs = len(run_texts)
        n_total = sum(run_counts)
        print(
            f"  Finetune approach '{method}': {n_total} texts -> {n_runs} consecutive runs, "
            f"counts={run_counts[:10]}{'...' if len(run_counts) > 10 else ''}"
        )
        refresh_interval = int(approach_config.get("refresh_interval", 2))
        steps = _finetune_gradient_refresh(
            model,
            tokenizer,
            run_texts,
            run_counts,
            batch_size,
            max_length,
            lr,
            device,
            refresh_interval=refresh_interval,
        )
        finetune_unique_texts = n_runs
    else:
        raise ValueError(
            f"Unknown finetune approach '{method}'. Valid: {sorted(_APPROACH_METHODS)}"
        )

    finetune_seconds = time.perf_counter() - start_time
    return {
        "finetune_steps": steps,
        "finetune_seconds": finetune_seconds,
        "finetune_approach": method,
        "finetune_unique_texts": finetune_unique_texts,
        "finetune_total_texts": n_total,
    }
