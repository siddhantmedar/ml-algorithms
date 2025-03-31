import numpy as np 
np.random.seed(313)


n_samples = 2048
n_features = 2

X = np.random.uniform(-5,5,(n_samples,n_features))
y = 3 * X[:,0] + np.sin(X[:,1]) + np.random.randn(n_samples)

split_ratio = int(0.9*X.shape[0])

X_train, y_train = X[:split_ratio],y[:split_ratio]
X_test, y_test = X[split_ratio:],y[split_ratio:]

print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test set: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

class LinearRegression():
    def __init__(self,epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.epochs = epochs
        self.learning_rate = 0.001

    def fit(self,X,y):
        self.coef_ = np.random.randn(X.shape[1])
        self.intercept_ = np.zeros(1)

        # perform training
        for epoch in range(self.epochs):
            y_pred = X @ self.coef_ + self.intercept_
            loss = 0.5 * np.sum((y_pred-y)**2)

            # compute gradients
            dW = X.T @ (y_pred-y)
            db = np.mean(y_pred-y)

            # update parameters
            self.coef_-=self.learning_rate * dW
            self.intercept_-=self.learning_rate * db

            if epoch%10==0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}, W: {self.coef_}, b: {self.intercept_}")

    def predict(self,X_test):
        return X_test @ self.coef_ + self.intercept_

def compute_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


if __name__ == "__main__":
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)

    loss = compute_loss(y_test,y_pred)

    print(f"Mean Squared Error on Test Set: {loss}")