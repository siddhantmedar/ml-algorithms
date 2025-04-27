import numpy as np

np.random.seed(313)


n_samples = 2048
n_features = 2

X = np.random.uniform(-5, 5, (n_samples, n_features))
y = 3 * X[:, 0] + np.sin(X[:, 1]) + np.random.randn(n_samples)

split_ratio = int(0.9 * X.shape[0])

X_train, y_train = X[:split_ratio], y[:split_ratio]
X_query, y_query = X[split_ratio:], y[split_ratio:]

print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test set: X_query shape: {X_query.shape}, y_query shape: {y_query.shape}")


class KNN:
    def __init__(self, k=5):
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_query):
        y_preds = []
        distances = np.array([self._get_distances(x) for x in X_query])
        all_indices = np.array(
            [np.argpartition(distance, self.k)[: self.k] for distance in distances]
        )
        all_neighbors = np.array([self.y[indices] for indices in all_indices])
        return np.array([np.mean(neighbors) for neighbors in all_neighbors])

    def _get_distances(self, x):
        return np.linalg.norm((x - self.X), axis=1)


def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    knn = KNN()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_query)

    loss = compute_loss(y_query, y_pred)

    print(f"Mean Squared Error on Test Set: {loss}")
