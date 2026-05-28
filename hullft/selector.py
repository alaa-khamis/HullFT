"""Selection methods used by the pipeline: KNN, SIFT, and Frank-Wolfe.

Each selector takes a pool of embeddings and a query embedding, and returns
the indices of selected pool points along with optional weights. Frank-Wolfe
returns a sparse convex combination whose support may be smaller than the
requested count; the fill module in hullft.fill expands that support to the
exact target count (when requested).
"""

import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Selector:
    """Base class for data point selectors."""

    name = "base"

    def select(self, embeddings, query_embedding, n_points):
        raise NotImplementedError

    def _prepare_query(self, query_embedding, dtype="float64"):
        query = query_embedding.astype(dtype)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return query.mean(axis=0)

    def _frank_wolfe(self, local_embeddings, target, n_points, max_iter, tol):
        """Sparse Frank-Wolfe approximation of target by pool embeddings.

        Input: pool embeddings, target vector, max support size, max
        iterations, residual tolerance.
        Output: support indices and matching weights.

        Initializes the support with the most similar pool point, then at
        each step picks the pool point most correlated with the residual
        and takes the optimal step toward it. Stops when the support
        reaches n_points or the residual norm falls below tol.
        """
        n_local = len(local_embeddings)

        similarities = local_embeddings @ target
        best_idx = np.argmax(similarities)

        weights = np.zeros(n_local)
        weights[best_idx] = 1.0
        current = local_embeddings[best_idx].copy()
        selected_set = {int(best_idx)}

        for _ in range(max_iter):
            residual = target - current
            if np.linalg.norm(residual) < tol:
                break

            correlations = local_embeddings @ residual
            s_idx = np.argmax(correlations)
            s = local_embeddings[s_idx]

            d_vec = s - current
            denom = np.dot(d_vec, d_vec)
            if denom < 1e-12:
                break

            gamma = np.clip(np.dot(residual, d_vec) / denom, 0, 1)
            weights = (1 - gamma) * weights
            weights[s_idx] += gamma
            current = current + gamma * d_vec
            selected_set.add(int(s_idx))

            if len(selected_set) >= n_points:
                break

        support_local = np.array(sorted(selected_set), dtype=int)
        support_weights = weights[support_local]
        if support_local.size > 0:
            order = np.argsort(-support_weights)
            support_local = support_local[order]
            support_weights = support_weights[order]

        return support_local[:n_points].copy(), support_weights[:n_points].copy()


class KNNSelector(Selector):
    """K-nearest-neighbour selection backed by FAISS or sklearn."""

    name = "knn"

    def __init__(self, use_faiss=True):
        self.use_faiss = use_faiss

    def select(self, embeddings, query_embedding, n_points):
        """Return the top n_points pool indices nearest to the query.

        Input: pool embeddings, query embedding, number of points.
        Output: selected indices and their relative weights.
        """
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        query = query_embedding.astype("float32")

        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[0] > 1:
            query = query.mean(axis=0, keepdims=True)

        if self.use_faiss:
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            distances, indices = index.search(query, n_points)
            return indices[0], distances[0]

        nn = NearestNeighbors(n_neighbors=n_points, metric="cosine")
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(query)
        return indices[0], 1 - distances[0]


class SIFTSelector(Selector):
    """SIFT selection via the activeft Retriever."""

    name = "sift"

    def __init__(self, llambda=0.01, fast=False):
        self.llambda = float(llambda)
        self.fast = bool(fast)
        from activeft.sift import Retriever  # type: ignore

        self._Retriever = Retriever

    def select(self, embeddings, query_embedding, n_points):
        """Run SIFT retrieval against the pool.

        Input: pool embeddings, query embedding, number of points.
        Output: selected indices and their SIFT scores.
        """
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        query = query_embedding.astype("float32")

        if query.ndim == 1:
            query = query.reshape(1, -1)

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        retriever = self._Retriever(
            index, llambda=self.llambda, fast=self.fast, only_faiss=False
        )
        values, indices, _, _ = retriever.search(query, N=n_points, K=None)
        return indices, values


class FWSelector(Selector):
    """Frank-Wolfe sparse convex combination selector (HullFT base selection)."""

    name = "fw"

    def __init__(self, max_iter=100, tol=1e-4):
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def select(self, embeddings, query_embedding, n_points):
        """Run Frank-Wolfe against the pool.

        Input: pool embeddings, query embedding, max support size.
        Output: support indices and matching weights (support may be
        smaller than n_points).
        """
        embeddings = embeddings.astype("float64")
        target = self._prepare_query(query_embedding, dtype="float64")
        return self._frank_wolfe(embeddings, target, n_points, self.max_iter, self.tol)


_SELECTORS = {
    "knn": KNNSelector,
    "sift": SIFTSelector,
    "fw": FWSelector,
}


def get_selector(name: str, **kwargs) -> Selector:
    """Instantiate a selector by name."""
    kwargs.pop("device", None)
    if name not in _SELECTORS:
        raise ValueError(
            f"Unknown selector: {name!r}. Available: {list(_SELECTORS.keys())}"
        )
    return _SELECTORS[name](**kwargs)
