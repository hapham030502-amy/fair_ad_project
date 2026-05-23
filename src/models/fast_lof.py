from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree


class FastKDTreeLOF:
    """LOF scoring optimized for the project datasets using KDTree.

    Quy ước tương thích với src.run_all_models.classical_scores:
    decision_function(X) trả về giá trị càng CAO càng bình thường.
    Vì classical_scores trả về -decision_function(X), anomaly score cuối cùng sẽ là LOF,
    tức giá trị càng cao càng bất thường.
    """

    def __init__(self, n_neighbors: int = 20, leaf_size: int = 40, eps: float = 1e-12):
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.eps = float(eps)
        self._tree = None
        self._X_train = None
        self._k_distance_train = None
        self._lrd_train = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X phải là ma trận 2 chiều")
        if len(X) <= self.n_neighbors:
            raise ValueError("Số mẫu train phải lớn hơn n_neighbors")

        self._X_train = X
        self._tree = KDTree(X, leaf_size=self.leaf_size)

        # Query k+1 vì điểm gần nhất của chính nó có khoảng cách 0; bỏ cột self-neighbor.
        dist, ind = self._tree.query(X, k=self.n_neighbors + 1)
        dist = dist[:, 1:]
        ind = ind[:, 1:]

        self._k_distance_train = dist[:, -1]
        reach = np.maximum(dist, self._k_distance_train[ind])
        self._lrd_train = 1.0 / (reach.mean(axis=1) + self.eps)
        return self

    def _lof_score(self, X):
        if self._tree is None or self._lrd_train is None or self._k_distance_train is None:
            raise RuntimeError("Cần gọi fit(X_train) trước khi score")
        X = np.asarray(X, dtype=np.float32)
        dist, ind = self._tree.query(X, k=self.n_neighbors)
        reach = np.maximum(dist, self._k_distance_train[ind])
        lrd_query = 1.0 / (reach.mean(axis=1) + self.eps)
        lof = self._lrd_train[ind].mean(axis=1) / (lrd_query + self.eps)
        return lof.astype(float)

    def decision_function(self, X):
        # Giá trị càng lớn càng bình thường; anomaly score bên ngoài = -decision_function.
        return -self._lof_score(X)

    def score_samples(self, X):
        return self.decision_function(X)
