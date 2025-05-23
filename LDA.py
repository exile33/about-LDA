import numpy as np
from scipy.linalg import eigh

class Original_LDA:
    def __init__(self):
        self.components_ = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X, y, alpha=1e-3):
        # 数据预处理：例如标准化
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        n_samples, n_features = X.shape
        class_labels = np.unique(y)
        n_classes = len(class_labels)

        # 计算总体均值
        mean_overall = np.mean(X, axis=0)

        # 初始化散度矩阵
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)
            # 类内散度累加
            # 可尝试 np.einsum 进一步向量化
            SW += (X_c - mean_c).T @ (X_c - mean_c)
            # 类间散度累加
            diff = (mean_c - mean_overall).reshape(-1, 1)
            SB += n_c * (diff @ diff.T)

        # 正则化 SW
        SW += alpha * np.eye(n_features)

        # 解广义特征值问题
        eigenvalues, eigenvectors = eigh(SB, SW)

        # 保证数值稳定性：归一化特征向量
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

        # 存储结果
        self.eigenvalues = eigenvalues
        self.eigenvectors = np.real(eigenvectors)

    def select_components(self, n_components):
        idx = np.argsort(self.eigenvalues)[::-1]
        self.components_ = self.eigenvectors[:, idx[:n_components]]

    def transform(self, X):
        return X @ self.components_
