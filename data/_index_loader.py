"""Multi-shard FAISS index with JSONL text lookup (offline precompute only).

This is NOT imported at runtime. It is used by `precompute_candidates.py`
and `precompute_embeddings.py` to search the offline FAISS index and read
back per-document text. Runtime experiments operate on the JSON and NPZ
outputs those scripts produce.
"""

import json
import logging
import os
import struct

import faiss
import numpy as np

from hullft.embedder import EMBEDDING_MODEL, RobertaPileEmbedder

logger = logging.getLogger(__name__)


class PileIndexLoader:
    """Multi-shard FAISS index with JSONL text lookup."""

    def __init__(
        self,
        index_dir,
        device="cuda",
        load_model=True,
        load_shards=True,
        subset_filter=None,
    ):
        self.index_dir = index_dir
        self.device = device
        self.subset_filter = subset_filter
        self.shards = []
        self.subset_groups = {}
        self.grouped_mode = False
        self.metric_type = faiss.METRIC_INNER_PRODUCT
        if load_shards:
            self._load_shards()

        self._embedder = None
        if load_model:
            self._load_embedder()

    @staticmethod
    def _subset_to_slug(subset_name):
        return (
            subset_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

    def _load_shards(self):
        entries = sorted(os.listdir(self.index_dir))
        direct_index_files = [f for f in entries if f.endswith(".jsonl.index")]

        group_specs = []
        if direct_index_files:
            group_specs.append(("all", self.index_dir, sorted(direct_index_files)))
        else:
            if not self.subset_filter:
                available = [
                    name
                    for name in entries
                    if os.path.isdir(os.path.join(self.index_dir, name))
                    and any(
                        f.endswith(".jsonl.index")
                        for f in os.listdir(os.path.join(self.index_dir, name))
                    )
                ]
                raise ValueError(
                    f"subset_filter is required when using a grouped index directory "
                    f"(multiple subset subdirs detected under {self.index_dir}). "
                    f"Available subsets: {available}."
                )
            allowed_slug = self._subset_to_slug(self.subset_filter)
            for name in entries:
                if name != allowed_slug:
                    continue
                group_dir = os.path.join(self.index_dir, name)
                if not os.path.isdir(group_dir):
                    continue
                index_files = sorted(
                    f for f in os.listdir(group_dir) if f.endswith(".jsonl.index")
                )
                if index_files:
                    group_specs.append((name, group_dir, index_files))

        if not group_specs:
            if self.subset_filter:
                slug = self._subset_to_slug(self.subset_filter)
                raise FileNotFoundError(
                    f"No .jsonl.index files found for subset '{self.subset_filter}' "
                    f"(expected dir: {os.path.join(self.index_dir, slug)}). Build the index first."
                )
            raise FileNotFoundError(
                f"No .jsonl.index files in {self.index_dir} (or its subset subdirectories)"
            )

        self.grouped_mode = len(group_specs) > 1 or (
            len(group_specs) == 1 and group_specs[0][1] != self.index_dir
        )

        cumulative = 0
        for group_name, group_dir, index_files in group_specs:
            shard_ids = []
            for idx_file in index_files:
                base = idx_file.replace(".index", "")
                jsonl_path = os.path.join(group_dir, base)
                index_path = os.path.join(group_dir, idx_file)
                offset_path = os.path.join(group_dir, base + ".offsets")

                logger.info("Loading shard: %s/%s", group_name, idx_file)
                index = faiss.read_index(index_path)
                offsets = (
                    _load_offsets(offset_path) if os.path.exists(offset_path) else None
                )

                self.shards.append(
                    {
                        "name": f"{group_name}/{base}",
                        "group": group_name,
                        "root_dir": group_dir,
                        "index": index,
                        "jsonl_path": jsonl_path,
                        "offsets": offsets,
                        "global_offset": cumulative,
                        "n_docs": index.ntotal,
                    }
                )
                shard_ids.append(len(self.shards) - 1)
                cumulative += index.ntotal
            self.subset_groups[group_name] = shard_ids

        self.total_docs = cumulative
        if self.shards:
            self.metric_type = getattr(
                self.shards[0]["index"], "metric_type", faiss.METRIC_INNER_PRODUCT
            )
        logger.info(
            "Loaded %d shards, %d total docs", len(self.shards), self.total_docs
        )

    def _load_embedder(self):
        manifest_path = os.path.join(self.index_dir, "manifest.json")
        model_name = EMBEDDING_MODEL
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                model_name = json.load(f).get("embedding_model", EMBEDDING_MODEL)
        else:
            roots = sorted(
                {s.get("root_dir") for s in self.shards if s.get("root_dir")}
            )
            for root in roots:
                root_manifest = os.path.join(root, "manifest.json")
                if os.path.exists(root_manifest):
                    with open(root_manifest) as f:
                        model_name = json.load(f).get(
                            "embedding_model", EMBEDDING_MODEL
                        )
                    break
        self._embedder = RobertaPileEmbedder(model_name=model_name, device=self.device)

    def embed_query(self, text):
        return self._embedder.embed_query(text)

    def embed_texts(self, texts, batch_size=32):
        return self._embedder.embed_texts(texts, batch_size=batch_size)

    def to(self, device):
        self.device = str(device)
        if self._embedder is not None:
            self._embedder.to(device)
        return self

    def search(self, query_text, k=200):
        return self.search_by_embedding(self.embed_query(query_text), k)

    def search_by_embedding(self, query_emb, k=200):
        query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        if len(self.shards) == 0 or k <= 0:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32), []

        if self.grouped_mode:
            grouped_scores, grouped_local, grouped_shards = [], [], []
            for shard_ids in self.subset_groups.values():
                g_scores, g_local, g_shards = self._search_shard_group(
                    query_emb, shard_ids, k
                )
                if len(g_scores) == 0:
                    continue
                grouped_scores.append(g_scores)
                grouped_local.append(g_local)
                grouped_shards.append(g_shards)

            if not grouped_scores:
                return (
                    np.array([], dtype=np.int64),
                    np.empty((0, self.shards[0]["index"].d), dtype=np.float32),
                    [],
                )

            scores = np.concatenate(grouped_scores)
            local_indices = np.concatenate(grouped_local)
            shard_ids_arr = np.concatenate(grouped_shards)

            top_k = min(k, len(scores))
            top_order = self._top_order(scores, top_k)
            result_scores = scores[top_order]
            result_local = local_indices[top_order]
            result_shards = shard_ids_arr[top_order]
        else:
            scores, result_local, result_shards = self._search_shard_group(
                query_emb, list(range(len(self.shards))), k
            )
            top_k = len(scores)
            result_scores = scores  # noqa: F841

        global_indices = np.array(
            [
                self.shards[sid]["global_offset"] + lid
                for sid, lid in zip(result_shards, result_local)
            ],
            dtype=np.int64,
        )

        dim = self.shards[0]["index"].d
        embeddings = np.empty((top_k, dim), dtype=np.float32)
        for i, (sid, lid) in enumerate(zip(result_shards, result_local)):
            embeddings[i] = self.shards[sid]["index"].reconstruct(int(lid))

        texts = [
            self._read_text(sid, int(lid))
            for sid, lid in zip(result_shards, result_local)
        ]
        return global_indices, embeddings, texts

    def _search_shard_group(self, query_emb, shard_ids, k):
        all_scores, all_local_indices, all_shard_ids = [], [], []
        for sid in shard_ids:
            shard = self.shards[sid]
            n = shard["index"].ntotal
            per_shard_k = min(k, n)
            if per_shard_k == 0:
                continue
            scores, local_idx = shard["index"].search(query_emb, per_shard_k)
            all_scores.append(scores[0])
            all_local_indices.append(local_idx[0])
            all_shard_ids.append(np.full(per_shard_k, sid, dtype=np.int32))

        if not all_scores:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int32),
            )

        scores = np.concatenate(all_scores)
        local_indices = np.concatenate(all_local_indices)
        shard_ids_array = np.concatenate(all_shard_ids)
        top_k = min(k, len(scores))
        top_order = self._top_order(scores, top_k)
        return scores[top_order], local_indices[top_order], shard_ids_array[top_order]

    def _top_order(self, scores, k):
        if self.metric_type == faiss.METRIC_L2:
            return np.argsort(scores)[:k]
        return np.argsort(-scores)[:k]

    def _read_text(self, shard_id, local_idx):
        shard = self.shards[shard_id]
        offsets = shard.get("offsets")
        if offsets is not None and local_idx < len(offsets):
            with open(shard["jsonl_path"], "r") as f:
                f.seek(offsets[local_idx])
                line = f.readline()
        else:
            with open(shard["jsonl_path"], "r") as f:
                for i, line in enumerate(f):
                    if i == local_idx:
                        break
                else:
                    return ""
        try:
            return json.loads(line).get("text", "")
        except (json.JSONDecodeError, AttributeError):
            return ""

    def read_texts_by_global_indices(self, global_indices):
        return [self._read_text(*self._global_to_local(int(g))) for g in global_indices]

    def _global_to_local(self, global_idx):
        for sid, shard in enumerate(self.shards):
            if global_idx < shard["global_offset"] + shard["n_docs"]:
                return sid, global_idx - shard["global_offset"]
        raise IndexError(
            f"Global index {global_idx} out of range (total={self.total_docs})"
        )


def _load_offsets(path):
    file_size = os.path.getsize(path)
    n = file_size // 8
    offsets = []
    with open(path, "rb") as f:
        for _ in range(n):
            offsets.append(struct.unpack("<Q", f.read(8))[0])
    return offsets
