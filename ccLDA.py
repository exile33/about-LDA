from sklearn.cluster import KMeans
from scipy.linalg import eigh

class ccLDA:
    def __init__(self, p=25, alpha=0.6, beta=0.6):
        self.components_ = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.n_features = None
        self.sw = None
        self.sb = None
        self.sw_c = None
        self.sb_c = None
        self.p = p
        self.alpha = alpha
        self.beta = beta

    def lda(self, X, y):
        class_labels = np.unique(y)
        n_classes = len(class_labels)
        mean_overall = np.mean(X, axis=0)
        class_samples = [X[y == c] for c in class_labels]
        mean_class = np.array([samples.mean(axis=0) for samples in class_samples])
        sw = sum([(samples - mean_class[i]).T @ (samples - mean_class[i]) for i, samples in enumerate(class_samples)])
        diff_means = mean_class - mean_overall
        sb = sum([n_c * (diff.reshape(-1, 1) @ diff.reshape(1, -1)) for n_c, diff in zip([len(s) for s in class_samples], diff_means)])
        self.sw = sw
        self.sb = sb
        return sw, sb

    def kmeans(self, X, k):
        sb_c_sum = np.zeros((self.n_features, self.n_features))
        sw_c_sum = np.zeros((self.n_features, self.n_features))
        valid_iterations = 0
        for _ in range(self.p):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) < k:
                continue
            overall_mean = np.mean(X, axis=0)
            sb_c = np.zeros((self.n_features, self.n_features))
            sw_c = np.zeros((self.n_features, self.n_features))
            for cluster in range(k):
                X_k = X[labels == cluster]
                mean_k = np.mean(X_k, axis=0)
                diff_k = (mean_k - overall_mean).reshape(-1, 1)
                sb_c += len(X_k) * np.dot(diff_k, diff_k.T)
                sw_c += np.dot((X_k - mean_k).T, (X_k - mean_k))
            sb_c_sum += sb_c
            sw_c_sum += sw_c
            valid_iterations += 1
        if valid_iterations > 0:
            self.sb_c = sb_c_sum / valid_iterations
            self.sw_c = sw_c_sum / valid_iterations
        return self.sw_c, self.sb_c

    def combine_divergence(self):
        sb_cclda = self.alpha * self.sb + (1 - self.alpha) * self.sb_c
        sw_cclda = self.beta * self.sw + (1 - self.beta) * self.sw_c
        sb_cclda = (sb_cclda + sb_cclda.T) / 2
        sw_cclda = (sw_cclda + sw_cclda.T) / 2
        return sw_cclda, sb_cclda

    def fit(self, X, y, k, n_components):
        self.n_features = X.shape[1]
        self.lda(X, y)
        self.kmeans(X, k)
        sw_cclda, sb_cclda = self.combine_divergence()
        eigenvalues, eigenvectors = eigh(sb_cclda, sw_cclda)
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx].real
        self.eigenvectors = eigenvectors[:, idx].real
        self.eigenvectors = self.eigenvectors / np.linalg.norm(self.eigenvectors, axis=0)
        n_classes = len(np.unique(y))
        self.components_ = self.eigenvectors[:, :n_components]

    def transform(self, X):
        return X @ self.components_
