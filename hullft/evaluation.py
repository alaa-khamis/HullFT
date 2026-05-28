"""BPB evaluation and method comparison."""

import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .finetune import finetune_on_texts, finetune_with_approach


def calculate_bpb(model, tokenizer, texts, batch_size=1, max_length=1024, device=None):
    """Bits-per-byte on `texts` using the model's own causal LM loss."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    total_log_likelihood = 0.0
    total_bytes = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating BPB"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            token_losses = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            token_log_probs = -token_losses
            token_log_probs = token_log_probs * shift_mask

            total_log_likelihood += token_log_probs.sum().item()

            for text in batch_texts:
                total_bytes += len(text.encode("utf-8"))

    log_likelihood_bits = total_log_likelihood / np.log(2)
    bpb = -log_likelihood_bits / total_bytes
    return {
        "bpb": bpb,
        "total_log_likelihood": total_log_likelihood,
        "perplexity": float(np.exp(-total_log_likelihood / total_bytes)),
    }


_CONVEX_METHODS = {"fw"}


def _is_convex_method(method_name: str) -> bool:
    return method_name in _CONVEX_METHODS or any(
        method_name.startswith(base + "_") for base in _CONVEX_METHODS
    )


def evaluate_selections(
    model_name,
    selections_dict,
    train_texts,
    eval_texts,
    batch_size=8,
    max_length=1024,
    device=None,
    finetune_batch_size=4,
    finetune_max_length=1024,
    finetune_lr=5e-5,
    model=None,
    tokenizer=None,
    seed=None,
    return_metadata=False,
    finetune_approach_config=None,
):
    """Evaluate each method by finetuning on its selected train texts and scoring BPB on eval_texts."""
    if tokenizer is None:
        print(f"Loading tokenizer for evaluation: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        print(f"Loading model for evaluation: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        print(f"Using pre-loaded model for evaluation: {model_name}")

    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    results = {}
    return_value = None
    try:
        print("\nEvaluating Baseline (no fine-tuning)...")
        baseline_result = calculate_bpb(
            model,
            tokenizer,
            eval_texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        baseline_result.update({"finetune_steps": 0, "finetune_seconds": 0.0})
        results["Baseline"] = baseline_result
        print(f"  BPB: {baseline_result['bpb']:.4f}")
        print(f"  Perplexity: {baseline_result['perplexity']:.4f}")

        # Untimed warmup so the first real method isn't charged for cold-start overhead.
        finetune_warmup_enabled = True
        finetune_warmup_seconds = None
        warmup_text = next(
            (t for t in train_texts if isinstance(t, str) and t.strip()),
            None,
        )
        if warmup_text is not None:
            model.load_state_dict(initial_state, strict=True)
            warmup_start = time.perf_counter()
            _ = finetune_on_texts(
                model,
                tokenizer,
                [warmup_text],
                batch_size=finetune_batch_size,
                max_length=finetune_max_length,
                lr=finetune_lr,
                device=device,
            )
            finetune_warmup_seconds = time.perf_counter() - warmup_start
            model.load_state_dict(initial_state, strict=True)
        else:
            finetune_warmup_enabled = False

        for method_name, indices in selections_dict.items():
            print(f"\nEvaluating {method_name}...")
            model.load_state_dict(initial_state, strict=True)
            selected_texts = [train_texts[i] for i in indices]
            finetune_texts = selected_texts

            configured_method = (finetune_approach_config or {}).get(
                "method", "standard"
            )
            allow_sift_multiplicity = method_name == "sift" and configured_method in (
                "sift_consecutive_refresh",
                "sift_global_refresh",
            )
            effective_approach_config = (
                finetune_approach_config
                if (_is_convex_method(method_name) or allow_sift_multiplicity)
                else None
            )
            approach_method = (effective_approach_config or {}).get(
                "method", "standard"
            )

            # Multi-r SIFT path: a single SIFT selection scored under multiple r values.
            refresh_intervals_cfg = (effective_approach_config or {}).get(
                "refresh_intervals"
            )
            multi_r_sift = (
                method_name == "sift"
                and approach_method
                in ("sift_global_refresh", "sift_consecutive_refresh")
                and isinstance(refresh_intervals_cfg, (list, tuple))
                and len(refresh_intervals_cfg) > 0
            )

            if multi_r_sift:
                for r_value in refresh_intervals_cfg:
                    r_int = int(r_value)
                    model.load_state_dict(initial_state, strict=True)
                    per_r_config = {
                        **effective_approach_config,
                        "method": approach_method,
                        "refresh_interval": r_int,
                    }
                    finetune_stats = finetune_with_approach(
                        model,
                        tokenizer,
                        finetune_texts,
                        approach_config=per_r_config,
                        batch_size=finetune_batch_size,
                        max_length=finetune_max_length,
                        lr=finetune_lr,
                        device=device,
                    )
                    finetune_stats["finetune_selected_count"] = len(selected_texts)
                    finetune_stats["finetune_refresh_interval"] = r_int

                    result = calculate_bpb(
                        model,
                        tokenizer,
                        eval_texts,
                        batch_size=batch_size,
                        max_length=max_length,
                        device=device,
                    )
                    result.update(finetune_stats)

                    result_key = f"{approach_method}_r{r_int}"
                    results[result_key] = result
                    print(f"  [{result_key}] BPB: {result['bpb']:.4f}")
                    print(f"  [{result_key}] Perplexity: {result['perplexity']:.4f}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue

            if approach_method != "standard" and effective_approach_config is not None:
                finetune_stats = finetune_with_approach(
                    model,
                    tokenizer,
                    finetune_texts,
                    approach_config=effective_approach_config,
                    batch_size=finetune_batch_size,
                    max_length=finetune_max_length,
                    lr=finetune_lr,
                    device=device,
                )
            else:
                finetune_stats = finetune_on_texts(
                    model,
                    tokenizer,
                    finetune_texts,
                    batch_size=finetune_batch_size,
                    max_length=finetune_max_length,
                    lr=finetune_lr,
                    device=device,
                )
            finetune_stats["finetune_selected_count"] = len(selected_texts)

            result = calculate_bpb(
                model,
                tokenizer,
                eval_texts,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
            result.update(finetune_stats)
            results[method_name] = result
            print(f"  BPB: {result['bpb']:.4f}")
            print(f"  Perplexity: {result['perplexity']:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Restoring model state...")
        model.load_state_dict(initial_state, strict=True)

        metadata = {
            "determinism_mode": "global_seed_once",
            "finetune_warmup_enabled": bool(finetune_warmup_enabled),
            "finetune_warmup_seconds": finetune_warmup_seconds,
        }
        if seed is not None:
            metadata["seed"] = int(seed)

        return_value = (results, metadata) if return_metadata else results
    finally:
        if model is not None:
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return return_value


def compare_bpb_results(results):
    """Print a per-method comparison and annotate each entry with improvement_percent."""
    print("\n" + "=" * 80)
    print("BPB Comparison")
    print("=" * 80)
    print(f"{'Method':<25} {'BPB':<10} {'Perplexity':<12} {'Improv':<15}")
    print("-" * 80)

    reference_method = "Baseline"
    reference_bpb = results.get(reference_method, {}).get("bpb")

    for method, result in results.items():
        result["improvement_reference"] = reference_method
        if reference_bpb is not None and reference_bpb > 0:
            if method == "Baseline":
                result["improvement_percent"] = 100.0
            else:
                result["improvement_percent"] = float(
                    (result["bpb"] / reference_bpb) * 100
                )
        else:
            result["improvement_percent"] = None

    for method, result in sorted(results.items(), key=lambda x: x[1]["bpb"]):
        improv = result.get("improvement_percent")
        if improv is None:
            improv_str = "-"
        elif method == "Baseline":
            improv_str = "100.00% (ref)"
        else:
            improv_str = f"{float(improv):.2f}%"
        print(
            f"{method:<25} {result['bpb']:<10.4f} {result['perplexity']:<12.2f} {improv_str:<15}"
        )

    print("=" * 80)
    print(
        "Lower BPB is better. Improvement % = (method_BPB / Baseline_BPB) x 100; lower % is better."
    )
