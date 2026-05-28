"""Expand a sparse Frank-Wolfe selection to exactly n_select instances."""

import numpy as np


def pad_by_weights_deterministic(indices, weights, n_select):
    """Largest-remainder padding to exactly n_select instances."""
    indices = np.asarray(indices)
    weights = np.asarray(weights, dtype=np.float64)

    if len(indices) == 0:
        return np.array([], dtype=indices.dtype), np.array([], dtype=np.float64)

    if len(indices) >= n_select:
        return indices[:n_select].copy(), weights[:n_select].copy()

    weight_sum = weights.sum()
    if weight_sum <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weight_sum

    target_counts = weights * n_select
    floor_counts = np.floor(target_counts).astype(int)
    remainders = target_counts - floor_counts

    remaining_slots = n_select - floor_counts.sum()
    if remaining_slots > 0:
        bonus_indices = np.argsort(-remainders)[:remaining_slots]
        floor_counts[bonus_indices] += 1

    expanded_indices = []
    expanded_weights = []
    for idx, orig_weight, count in zip(indices, weights, floor_counts):
        for _ in range(count):
            expanded_indices.append(idx)
            expanded_weights.append(orig_weight)

    return (
        np.array(expanded_indices, dtype=indices.dtype),
        np.array(expanded_weights, dtype=np.float64),
    )


def _repeat_indices_from_counts(indices, counts):
    out = []
    for idx, c in zip(indices, counts):
        if c > 0:
            out.extend([int(idx)] * int(c))
    return np.array(out, dtype=int)


def _integerize_support_geometry(
    support_points,
    target,
    support_weights,
    n_points,
    swap_passes=2,
):
    """Integer allocation that minimises reconstruction error of the query.

    Steps:
      1. Floor allocation: c_i = floor(N * w_i).
      2. Greedy remainder fill (pick index whose +1 reduces loss the most).
      3. Local swap refinement for `swap_passes` passes.
    """
    Xs = np.asarray(support_points, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(support_weights, dtype=np.float64).copy()
    m = len(w)

    if n_points <= 0 or m == 0:
        return np.zeros(m, dtype=int)
    if m == 1:
        return np.array([int(n_points)], dtype=int)

    w[w < 0] = 0.0
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / m
    else:
        w /= s

    c = np.floor(n_points * w).astype(int)
    rem = int(n_points - c.sum())

    def _loss(cvec):
        q = cvec.astype(np.float64) / float(n_points)
        recon = q @ Xs
        diff = t - recon
        return float(np.dot(diff, diff))

    for _ in range(rem):
        best_i, best_v = None, np.inf
        for i in range(m):
            cc = c.copy()
            cc[i] += 1
            v = _loss(cc)
            if v < best_v:
                best_v = v
                best_i = i
        c[best_i] += 1

    for _ in range(max(0, int(swap_passes))):
        improved = False
        cur = _loss(c)
        src = np.where(c > 0)[0]
        for i in src:
            for j in range(m):
                if i == j:
                    continue
                cc = c.copy()
                cc[i] -= 1
                cc[j] += 1
                v = _loss(cc)
                if v + 1e-14 < cur:
                    c = cc
                    cur = v
                    improved = True
        if not improved:
            break

    diff = int(n_points - c.sum())
    if diff > 0:
        c[np.argmax(w)] += diff
    elif diff < 0:
        for _ in range(-diff):
            c[int(np.argmax(c))] -= 1

    return c


def integerize_by_geometry(
    indices, weights, support_embeddings, target, n_select, swap_passes=2
):
    """Expand a sparse FW selection to exactly n_select instances via geometry-aware integerization."""
    indices = np.asarray(indices, dtype=int)
    weights = np.asarray(weights, dtype=np.float64)
    m = len(indices)

    if n_select <= 0 or m == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float64)

    weights = weights.copy()
    weights[weights < 0] = 0.0
    s = weights.sum()
    if s <= 0:
        weights[:] = 1.0 / m
    else:
        weights /= s

    counts = _integerize_support_geometry(
        support_points=support_embeddings,
        target=target,
        support_weights=weights,
        n_points=n_select,
        swap_passes=swap_passes,
    )

    expanded_idx = _repeat_indices_from_counts(indices, counts)

    if expanded_idx.size < n_select:
        pad = np.full(
            n_select - expanded_idx.size, int(indices[np.argmax(weights)]), dtype=int
        )
        expanded_idx = np.concatenate([expanded_idx, pad])
    elif expanded_idx.size > n_select:
        expanded_idx = expanded_idx[:n_select]

    expanded_w = np.ones(len(expanded_idx), dtype=np.float64) / max(
        1, len(expanded_idx)
    )
    return expanded_idx, expanded_w
