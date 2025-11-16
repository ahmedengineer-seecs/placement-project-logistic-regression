import numpy as np


class KMeans:
    """
    Simple K-Means implementation with k-means++ init.
    Usage:
      km = KMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=0)
      km.fit(X)
      labels = km.predict(X)
      centers = km.cluster_centers_
      inertia = km.inertia_
    """
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, init="kmeans++", random_state=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init = init
        self.random_state = None if random_state is None else int(random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _check_array(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        return X

    def _init_centers_random(self, X, rng):
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_centers_kmeanspp(self, X, rng):
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=float)
        # choose first center uniformly
        first_idx = rng.integers(0, n_samples)
        centers[0] = X[first_idx]
        # distances squared to nearest center
        closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for c in range(1, self.n_clusters):
            probs = closest_dist_sq / closest_dist_sq.sum()
            idx = rng.choice(n_samples, p=probs)
            centers[c] = X[idx]
            dist_sq = np.sum((X - centers[c]) ** 2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
        return centers

    def fit(self, X):
        X = self._check_array(X)
        n_samples, n_features = X.shape
        if self.n_clusters <= 0 or self.n_clusters > n_samples:
            raise ValueError("n_clusters must be > 0 and <= n_samples")

        rng = np.random.default_rng(self.random_state)

        if self.init == "kmeans++":
            centers = self._init_centers_kmeanspp(X, rng)
        elif self.init == "random":
            centers = self._init_centers_random(X, rng)
        else:
            raise ValueError("Unknown init method")

        labels = np.full(n_samples, -1, dtype=int)
        for it in range(1, self.max_iter + 1):
            # assign labels
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # shape (n_samples, n_clusters)
            new_labels = np.argmin(dists, axis=1)

            # compute inertia
            inertia = np.sum(dists[np.arange(n_samples), new_labels])

            # recompute centers
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                members = X[new_labels == k]
                if len(members) == 0:
                    # reinitialize empty cluster to a random sample
                    new_centers[k] = X[rng.integers(0, n_samples)]
                else:
                    new_centers[k] = members.mean(axis=0)

            # check for convergence (centers movement)
            center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1)).max()
            centers = new_centers
            labels = new_labels
            self.n_iter_ = it
            if center_shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(inertia)
        return self

    def predict(self, X):
        X = self._check_array(X)
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
        dists = np.sum((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    # Quick demo using synthetic data
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    km = KMeans(n_clusters=4, random_state=42, init="kmeans++")
    km.fit(X)
    print("Centers:\n", km.cluster_centers_)
    print("Inertia:", km.inertia_)
    # optional: compare with sklearn KMeans
    try:
        from sklearn.cluster import KMeans as SKKMeans
        sk = SKKMeans(n_clusters=4, random_state=42).fit(X)
        print("sklearn inertia:", sk.inertia_)
    except Exception:
        pass