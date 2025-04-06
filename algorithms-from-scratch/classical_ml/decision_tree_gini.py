import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
feature_names = dict(enumerate(data["feature_names"]))

X, y = data["data"], data["target"]
indices = np.random.permutation(X.shape[0])
X, y = X[indices], y[indices]

split_ratio = int(X.shape[0] * 0.90)
X_train, y_train = X[:split_ratio], y[:split_ratio]
X_test, y_test = X[split_ratio:], y[split_ratio:]

print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")


class TreeNode:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.class_prediction = None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X_test):
        return np.array([self._traverse_tree(x, self.root) for x in X_test])

    def _gini_impurity(self, y):
        _, class_counts = np.unique(y, return_counts=True)
        probs = class_counts / y.shape[0]
        gini_score = 1 - np.sum(probs**2)
        return gini_score

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = np.inf

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                gini_score = (y_left.shape[0] / y.shape[0]) * self._gini_impurity(
                    y_left
                ) + (y_right.shape[0] / y.shape[0]) * self._gini_impurity(y_right)

                if gini_score < best_gini:
                    best_gini = gini_score
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        node = TreeNode()

        # If all labels are the same, or the stopping criteria are met, make it a leaf node
        if (
            np.unique(y).shape[0] == 1
            or y.shape[0] <= self.min_samples_split
            or (self.max_depth and depth >= self.max_depth)
        ):
            node.class_prediction = np.bincount(y).argmax()
            return node

        feature, threshold = self._best_split(X, y)

        if feature is None:
            node.class_prediction = np.bincount(y).argmax()
            return node

        node.feature_index = feature
        node.threshold = threshold

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        node.left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _traverse_tree(self, x, node):
        # base case
        if node.class_prediction is not None:
            return node.class_prediction

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left_child)
        else:
            return self._traverse_tree(x, node.right_child)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = " " * depth

        if node.class_prediction is not None:
            print(f"{indent}Predict: {node.class_prediction}")
            return

        print(
            f"{indent}Feature {feature_names[node.feature_index]} <= {node.threshold:.2f}"
        )
        self.print_tree(node.left_child, depth + 1)
        print(
            f"{indent}Feature {feature_names[node.feature_index]} > {node.threshold:.2f}"
        )
        self.print_tree(node.right_child, depth + 1)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


if __name__ == "__main__":
    model = DecisionTree()
    model.fit(X_train, y_train)  # train model

    model.print_tree()

    y_pred = model.predict(X_test)

    print(accuracy(y_test, y_pred))
