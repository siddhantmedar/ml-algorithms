import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from matplotlib import cm


def get_dataset(n_samples, noise=0.5, scale=False, shuffle=False):
    # Generate a synthetic regression dataset with 2 features
    X, y = make_regression(
        n_samples=n_samples, n_features=2, noise=noise, random_state=313
    )

    scaler_X = None
    y_scale = None
    if scale:
        # Scale X to have zero mean and unit variance
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)

        # Scale y to the range [-1, 1]
        y_min, y_max = y.min(), y.max()
        y = 2 * (y - y_min) / (y_max - y_min) - 1
        y_scale = (y_min, y_max)

    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    return X, y, scaler_X, y_scale


class SVMRegressor:
    def __init__(self, c=1.0, eps=0.0, learning_rate=0.01, epochs=1000):
        # Initialize SVR parameters
        if c <= 0:
            raise ValueError("Regularization parameter c must be positive")
        if eps < 0:
            raise ValueError("Epsilon must be non-negative")

        self.w = None
        self.b = 0.0
        self.c = c
        self.eps = eps
        self.epochs = epochs
        self.learning_rate = learning_rate

    def compute_loss(self, X, y):
        # Compute the SVR objective: J = lambda * ||w||^2 + (1/n) * sum(max(0, |y_i - (w * x_i + b)| - epsilon))
        hinge_loss = 0.0
        n_samples, n_features = X.shape
        lambda_ = 1 / self.c

        for x_i, y_i in zip(X, y):
            error_i = y_i - (x_i.dot(self.w) + self.b)
            hinge_loss += max(0, abs(error_i) - self.eps)

        hinge_loss = (1 / n_samples) * hinge_loss
        reg_loss = lambda_ * self.w.dot(self.w)

        return hinge_loss + reg_loss

    def fit(self, X, y):
        # Train the SVR using subgradient descent
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        n_samples, n_features = X.shape
        if self.w is None:
            self.w = np.random.randn(n_features)

        prev_loss = float("inf")
        tol = 1e-4

        for epoch in range(self.epochs):
            dw = np.zeros_like(self.w)
            db = 0.0

            # Compute subgradients for epsilon-insensitive loss
            for x_i, y_i in zip(X, y):
                error_i = y_i - (x_i.dot(self.w) + self.b)
                z_i = abs(error_i) - self.eps

                if z_i > 0:
                    s_i = 1 if error_i > 0 else -1
                    dw += -s_i * x_i
                    db += -s_i

            # Add regularization term to w subgradient
            dw = (2 * (1 / self.c) * self.w) + ((1 / n_samples) * dw)
            db = (1 / n_samples) * db

            # Update parameters
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # Check convergence and monitor training
            loss = self.compute_loss(X, y)
            if epoch % 100 == 0:
                y_pred_train = self.predict(X)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, w norm: {np.linalg.norm(self.w):.4f}, y_pred range: {y_pred_train.min():.2f} to {y_pred_train.max():.2f}")
            if abs(loss - prev_loss) < tol:
                print(f"Converged at epoch {epoch}, Loss: {loss:.4f}")
                break
            prev_loss = loss

    def predict(self, X):
        # Predict continuous values
        return X.dot(self.w) + self.b


def mean_squared_error(y_true, y_pred):
    # Compute Mean Squared Error
    return np.mean((y_true - y_pred) ** 2)


# Demo with synthetic regression data
X, y, scaler_X, y_scale = get_dataset(n_samples=1000, noise=10, scale=True, shuffle=True)

split_ratio = int(0.9 * X.shape[0])
X_train, y_train = X[:split_ratio], y[:split_ratio]
X_test, y_test = X[split_ratio:], y[split_ratio:]

print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
print(f"y_train (scaled) range: {y_train.min():.2f} to {y_train.max():.2f}")
print(f"y_test (scaled) range: {y_test.min():.2f} to {y_test.max():.2f}")

# Train and evaluate SVR
svr = SVMRegressor(c=50.0, eps=0.5)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

# Rescale predictions and true values back to original scale for evaluation
y_min, y_max = y_scale
y_test_rescaled = (y_test + 1) / 2 * (y_max - y_min) + y_min
y_pred_rescaled = (y_pred + 1) / 2 * (y_max - y_min) + y_min
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f"Mean Squared Error (original scale): {mse:.4f}")

# Visualize the predictions and regression surface (for 2D data)
fig = plt.figure(figsize=(12, 5))

# Scatter plot of true vs. predicted values (original scale)
ax1 = fig.add_subplot(121)
ax1.scatter(X_test[:, 0], y_test_rescaled, color='blue', label='True values', alpha=0.5)
ax1.scatter(X_test[:, 0], y_pred_rescaled, color='red', label='Predicted values', alpha=0.5)
ax1.set_xlabel('Feature 1 (scaled)')
ax1.set_ylabel('Target (original scale)')
ax1.set_title(f'SVR Predictions (MSE: {mse:.4f})')
ax1.legend()

# 3D plot of the regression surface with epsilon tube (scaled data)
ax2 = fig.add_subplot(122, projection='3d')
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the regression surface
ax2.plot_surface(xx, yy, Z, cmap=cm.viridis, alpha=0.5)

# Plot the epsilon tube (Z ± epsilon)
ax2.plot_surface(xx, yy, Z + svr.eps, color='gray', alpha=0.2)
ax2.plot_surface(xx, yy, Z - svr.eps, color='gray', alpha=0.2)

# Plot the test points
ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, color='red', label='True values', alpha=0.5)

ax2.set_xlabel('Feature 1 (scaled)')
ax2.set_ylabel('Feature 2 (scaled)')
ax2.set_zlabel('Target (scaled)')
ax2.set_title('SVR Regression Surface with Epsilon Tube')

# Adjust the viewing angle for better visibility
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()