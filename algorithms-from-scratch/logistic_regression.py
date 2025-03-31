import numpy as np 
np.random.seed(313)


n_samples = 2048
n_features = 2

X = np.random.uniform(-5,5,(n_samples,n_features))
y = np.where(3 * X[:,0] + np.sin(X[:,1]) + np.random.randn(n_samples) >=0.65 , 1, 0)

split_ratio = int(0.9*X.shape[0])

X_train, y_train = X[:split_ratio],y[:split_ratio]
X_test, y_test = X[split_ratio:],y[split_ratio:]

print(f"Training set: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Test set: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():
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
            z = X @ self.coef_ + self.intercept_
            y_pred = sigmoid(z)
            loss = -np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15))

            # compute gradients
            dW = X.T @ (y_pred-y)
            db = np.mean(y_pred-y)

            # update parameters
            self.coef_-=self.learning_rate * dW
            self.intercept_-=self.learning_rate * db

            if epoch%10==0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}, W: {self.coef_}, b: {self.intercept_}")

    def predict(self,X_test):
        z = X_test @ self.coef_ + self.intercept_
        y_pred = sigmoid(z)
        y_pred = np.where(y_pred > 0.5, 1, 0)

        return y_pred
    
    def predict_proba(self,X_test):
        z = X_test @ self.coef_ + self.intercept_
        y_pred_proba = sigmoid(z)
        return y_pred_proba

def compute_loss(y_true,y_pred):
    return -np.mean(y_true*np.log(y_pred + 1e-15) + (1-y_true)*np.log(1-y_pred + 1e-15))


if __name__ == "__main__":
    lr = LogisticRegression()
    lr.fit(X_train,y_train)

    # y_pred = lr.predict(X_test)
    y_pred = lr.predict_proba(X_test)

    loss = compute_loss(y_test,y_pred)

    print(f"Binary Cross-Entropy Loss on Test Set: {loss}")