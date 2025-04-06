import numpy as np
from sklearn.metrics import confusion_matrix

np.random.seed(313)


n_samples, n_features = 1000, 3
X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

split_ratio = int(0.90 * n_samples)

X_train, y_train = X[:split_ratio], y[:split_ratio]
X_test, y_test = X[split_ratio:], y[split_ratio:]


class GaussianNaiveBayes:
    def __init__(self):
        self.prior_probs = None
        self.classes = None
        self.n_features = None
        self.mean = None
        self.var = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        # compute prior probabilities
        self.prior_probs = np.bincount(y) / len(y)

        # compute conditional probabilities
        self.mean = np.zeros((len(self.classes), self.n_features))
        self.var = np.zeros((len(self.classes), self.n_features))

        for c in self.classes:
            class_points = X[y == c]
            self.mean[c] = np.mean(class_points, axis=0)
            self.var[c] = np.var(class_points, axis=0)

    def predict(self, X):
        y_pred = []

        for x in X:
            log_probs = []

            for c in self.classes:
                log_prior = np.log(self.prior_probs[c])
                log_likelihood = -0.5 * np.sum(
                    np.log((2 * np.pi * self.var[c]))
                    + (x - self.mean[c]) ** 2 / self.var[c]
                )
                log_probs.append(log_prior + log_likelihood)

            y_pred.append(np.argmax(log_probs))

        return np.array(y_pred)


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print("Accuracy:", compute_accuracy(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
