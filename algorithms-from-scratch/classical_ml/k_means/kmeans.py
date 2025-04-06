import numpy as np

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeans_plus_plus_init(self, X):
        centroids = []
        n_samples = X.shape[0]

        first_initial_centroid = np.random.choice(n_samples)
        centroids.append(X[first_initial_centroid])
        
        for _ in range(self.k-1):
            distances = np.array([min((np.linalg.norm(x-c)**2 
                                             for c in centroids))
                                   for x in X])
            probs = distances / distances.sum()
            index = np.random.choice(n_samples,p=probs)
            centroids.append(X[index])

        return centroids
    
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        # initial_idx = np.random.choice(n_samples, self.k, replace=False)
        # self.centroids = X[initial_idx]

        self.centroids = self._kmeans_plus_plus_init(X)

        for i in range(self.max_iter):
            # Assignment step
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=-1)
            self.labels = np.argmin(distances, axis=1)

            # Update step
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])

            # Convergence check
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.inertia_ = np.sum((X - self.centroids[self.labels]) ** 2)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=-1)
        return np.argmin(distances, axis=1)
    
    def silhouette_score(self,X):
        n_samples = X.shape[0]
        overall_score = 0
        
        for i,x in enumerate(X):
            label = self.labels[i]
            centroid = self.centroids[label]

            same_cluster = X[self.labels == label]
            same_cluster = [p for p in same_cluster if not np.array_equal(p,x)]

            centroid_dists = [(np.linalg.norm(centroid-c),i)
                                          for i,c in enumerate(self.centroids) 
                                          if not np.array_equal(c,centroid)]
            
            nearest_index = min(centroid_dists, key=lambda x: x[0])[1]
            diff_cluster = X[self.labels == nearest_index]
        
            if same_cluster:
                a = np.mean(np.linalg.norm(p-x) for p in same_cluster if not np.array_equal(p,x))
            else:
                a = 0
            
            b = np.mean(np.linalg.norm(p-x) for p in diff_cluster)
            
            if max(a,b) > 0:
                s = (b-a)/max(a,b)
            else:
                s = 0
            
            overall_score+=s

        return overall_score / n_samples

def main():
    # Generate synthetic data
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)

    # Fit KMeans
    kmeans = KMeans(k=3, max_iter=100, tol=1e-4, random_state=42)
    kmeans.fit(X)

    # Print results
    print("Centroids:")
    print(kmeans.centroids)
    print("\nInertia:")
    print(kmeans.inertia_)

    # Predict cluster labels for the same data
    labels = kmeans.predict(X)
    print("\nPredicted labels:")
    print(labels)

if __name__ == "__main__":
    main()
