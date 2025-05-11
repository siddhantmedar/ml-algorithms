import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def get_dataset(n_samples, centers=None, shuffle=False):
    if centers is None:
        centers = [(-3, -3), (3, 3)]

    X, y = make_blobs(
        n_samples=n_samples, centers=centers, random_state=313, cluster_std=1.2
    )
    y = 2 * y - 1

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    return X, y


X, y = get_dataset(n_samples=1000, shuffle=True)

split_ratio = int(0.9 * X.shape[0])

X_train, y_train = X[:split_ratio], y[:split_ratio]
X_test, y_test = X[split_ratio:], y[split_ratio:]

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")


class SVMClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.w = None
        self.b = 0.0
        self.c = 1.0
        self.epochs = epochs
        self.learning_rate = learning_rate

    def compute_loss(self, X, y):
        hinge_loss = 0.0

        n_samples, n_features = X.shape

        lambda_ = 1 / self.c

        for x_i, y_i in zip(X, y):
            z_i = 1 - y_i * (x_i.dot(self.w) + self.b)
            hinge_loss += max(0, z_i)

        hinge_loss = (1 / n_samples) * hinge_loss

        reg_loss = lambda_ * self.w.dot(self.w)

        return hinge_loss + reg_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.w is None:
            self.w = np.random.randn(n_features)

        prev_loss = float("inf")
        tol = 1e-4

        for epoch in range(self.epochs):
            dw = np.zeros_like(self.w)
            db = 0.0

            for x_i, y_i in zip(X, y):
                z_i = 1 - y_i * (x_i.dot(self.w) + self.b)

                if z_i > 0:
                    dw += -(y_i * x_i)
                    db += -y_i

            dw = (2 * (1 / self.c) * self.w) + ((1 / n_samples) * dw)
            db = (1 / n_samples) * db

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            loss = self.compute_loss(X, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            if abs(loss - prev_loss) < tol:
                print(f"Converged at epoch {epoch}, Loss: {loss:.4f}")
                break

            prev_loss = loss

    def predict(self, X):
        y_pred = X.dot(self.w) + self.b
        return np.where(y_pred > 0, 1, -1)

    def predict_proba(self, X):
        return X.dot(self.w) + self.b


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


if __name__ == "__main__":
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # visualize the decision boundary
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", alpha=0.7)
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
    plt.title(f"SVM Decision Boundary (Test Accuracy: {acc:.2f})")
    plt.show()
