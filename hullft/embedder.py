"""RoBERTa embedder for Pile text and helpers for the test JSONL.

Defines RobertaPileEmbedder, which produces L2-normalized mean-pooled
embeddings, and small utilities to download and read the pile-uncopyrighted
test split.
"""

import json
import logging
import os

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "socialfoundations/roberta-large-pile-lr2e-5-bs16-8gpu-1700000"


class RobertaPileEmbedder:
    """RoBERTa-large fine-tuned on Pile; mean-pooled, L2-normalized."""

    def __init__(self, model_name=EMBEDDING_MODEL, device="cuda"):
        self.model_name = model_name
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info("Loading embedding model: %s on %s", model_name, dev)
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._model = (
            RobertaModel.from_pretrained(model_name, revision="refs/pr/1")
            .to(dev)
            .eval()
        )
        self._dev = dev

    def to(self, device):
        target = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(target).eval()
        self._dev = target
        return self

    @torch.no_grad()
    def embed_texts(self, texts, batch_size=32):
        """Embed a list of texts.

        Input: list of strings, optional batch size.
        Output: (n, dim) float32 array of L2-normalized mean-pooled embeddings.

        Tokenizes each batch, runs RoBERTa, mean-pools over the attention
        mask, then L2-normalizes each row.
        """
        if not texts:
            return np.empty((0, self._model.config.hidden_size), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self._tokenizer(
                batch,
                truncation=True,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = tokens["input_ids"].to(self._dev)
            attention_mask = tokens["attention_mask"].to(self._dev)

            output = self._model(input_ids=input_ids, attention_mask=attention_mask)
            state = output["last_hidden_state"]
            emb = (state * attention_mask[:, :, None]).sum(dim=1)
            emb /= attention_mask[:, :, None].sum(dim=1).clamp(min=1e-9)
            emb = emb.cpu().numpy().astype(np.float32)

            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norm, 1e-9)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def embed_query(self, text):
        """Embed a single query string.

        Input: one text string.
        Output: (1, dim) float32 array, L2-normalized.
        """
        tokens = self._tokenizer(
            [text],
            truncation=True,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].to(self._dev)
        attention_mask = tokens["attention_mask"].to(self._dev)

        output = self._model(input_ids=input_ids, attention_mask=attention_mask)
        state = output["last_hidden_state"]
        emb = (state * attention_mask[:, :, None]).sum(dim=1)
        emb /= attention_mask[:, :, None].sum(dim=1).clamp(min=1e-9)
        emb = emb.cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-9)
        return emb


def ensure_test_jsonl(test_dir, filename="test.jsonl"):
    """Make sure the pile-uncopyrighted test JSONL exists on disk.

    Input: directory to place the file in, optional filename.
    Output: path to the decompressed JSONL.

    If the file is missing, downloads the .zst from HuggingFace, streams
    it to disk, decompresses to JSONL, and removes the .zst.
    """
    jsonl_path = os.path.join(test_dir, filename)
    if os.path.exists(jsonl_path):
        return jsonl_path

    os.makedirs(test_dir, exist_ok=True)

    import requests
    import zstandard as zstd

    url = (
        "https://huggingface.co/datasets/monology/pile-uncopyrighted"
        "/resolve/main/test.jsonl.zst"
    )
    zst_path = jsonl_path + ".zst"

    print(f"[test-data] Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    downloaded = 0
    with open(zst_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(
                    f"\r[test-data]  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct}%)",
                    end="",
                    flush=True,
                )
    print()

    print(f"[test-data] Decompressing to {jsonl_path} ...")
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fin, open(jsonl_path, "wb") as fout:
        dctx.copy_stream(fin, fout)
    os.remove(zst_path)
    print(f"[test-data] Done. {os.path.getsize(jsonl_path) / 1e6:.0f} MB")
    return jsonl_path


def load_pile_test_texts(test_dir, subset_name=None):
    """Load test texts from the pile-uncopyrighted JSONL.

    Input: directory containing test.jsonl, optional pile_set_name filter.
    Output: list of texts when subset_name is given, otherwise list of
    (text, pile_set_name) tuples.
    """
    jsonl_path = ensure_test_jsonl(test_dir)
    texts = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (record.get("text", "") or "").strip()
            if not text:
                continue
            meta = record.get("meta", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            name = meta.get("pile_set_name", "")
            if subset_name is not None and name != subset_name:
                continue
            if subset_name is not None:
                texts.append(text)
            else:
                texts.append((text, name))
    return texts
