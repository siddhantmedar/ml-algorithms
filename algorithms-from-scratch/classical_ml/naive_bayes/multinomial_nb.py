import numpy as np
from sklearn.metrics import confusion_matrix

np.random.seed(313)


n_samples, n_features = 1000, 3
X = np.random.randint(0, 100, (n_samples, n_features))
y = np.random.randint(0, 2, n_samples)

split_ratio = int(0.90 * n_samples)

X_train, y_train = X[:split_ratio], y[:split_ratio]
X_test, y_test = X[split_ratio:], y[split_ratio:]


class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        self.prior_probs = None
        self.classes = None
        self.conditional_probs = None
        self.n_features = None
        self.alpha = alpha

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.classes = np.unique(y)

        # compute prior probabilities
        self.prior_probs = np.bincount(y) / len(y)
        self.conditional_probs = np.zeros((len(self.classes), self.n_features))

        # compute conditional probabilities
        for c in self.classes:
            class_points = X[y == c]
            w_y = (class_points.sum(axis=0) + self.alpha) / (
                np.sum(class_points) + self.alpha * self.n_features
            )

            self.conditional_probs[c] = w_y

    def predict(self, X):
        y_pred = []

        for x in X:
            log_probs = []

            for c in self.classes:
                log_prior = np.log(self.prior_probs[c])
                log_likelihood = np.sum(x * np.log(self.conditional_probs[c]))
                log_probs.append(log_prior + log_likelihood)

            y_pred.append(np.argmax(log_probs))

        return np.array(y_pred)


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    nb = MultinomialNaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print("Accuracy:", compute_accuracy(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
