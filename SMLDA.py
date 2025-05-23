import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, TransformerMixin

class SMLDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, prior_type='correlation', sigma=1.0, epsilon=1e-6):
        """
        改进后的SMLDA实现
        """
        self.n_components = n_components
        self.prior_type = prior_type
        self.sigma = sigma
        self.epsilon = epsilon
        self.W = None
        self.class_means = None
        self.global_mean = None
        self.P = None
        self.scaler = StandardScaler()  # 独立的标准化器
        self.label_correlation_ = None  # 缓存标签相关性矩阵


    def fit(self, X, y):
        # 输入验证和数据准备
        X, y = check_X_y(X, y, multi_output=True)
        self.n_samples, self.n_features = X.shape
        self.n_classes = y.shape[1]

        # 数据标准化
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y.astype(np.float32)

        # 核心计算流程
        self._compute_label_correlation_matrix()  # 预计算标签相关性
        self.P = self._compute_class_saliency_weights()
        self._compute_class_statistics()
        self._solve_projection_matrix()
        
        return self

    def _compute_class_saliency_weights(self):
        P = np.zeros((self.n_classes, self.n_samples))
        for c in range(self.n_classes):
            indices = np.where(self.y[:, c] == 1)[0]
            if len(indices) == 0:
                continue

            X_c = self.X_scaled[indices]
            A = rbf_kernel(X_c, gamma=1/(2*self.sigma**2))
            V = self._compute_prior_weights(c, indices)
            
            # 稳定性增强的矩阵构建
            D = np.diag(A.sum(axis=1)) + self.epsilon * np.eye(len(indices))
            H = 0.5*(D + D.T) - A + np.diag(V)
            
            # 带正则化的求解
            try:
                p = np.linalg.solve(H, np.ones(len(indices)))
            except np.linalg.LinAlgError:
                H_reg = H + self.epsilon * np.eye(H.shape[0])
                p = np.linalg.solve(H_reg, np.ones(H_reg.shape[0]))
            
            # 概率归一化
            p = np.clip(p, 1e-10, None)  # 防止负值
            p /= p.sum()
            P[c, indices] = p
        return P

    def _compute_prior_weights(self, c, indices):
        if self.prior_type == 'correlation':
            # 修正后的标签相关性计算
            m = self.label_correlation_[c] @ self.y[indices].T
            m = np.clip(m / (np.linalg.norm(m, ord=1) + 1e-10), 0, 1)
            return 1 - m
        elif self.prior_type == 'entropy':
            label_counts = np.sum(self.y[indices], axis=1)
            label_counts = np.clip(label_counts, 1, None)  # 防止除零
            return 1 - 1/label_counts
        else:
            raise ValueError(f"Unsupported prior type: {self.prior_type}")

    def _compute_label_correlation_matrix(self):
        # 带正则化的相关系数计算
        R = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            yi = self.y[:, i]
            norm_i = np.linalg.norm(yi) + self.epsilon
            for j in range(self.n_classes):
                yj = self.y[:, j]
                norm_j = np.linalg.norm(yj) + self.epsilon
                R[i, j] = yi.dot(yj) / (norm_i * norm_j)
        self.label_correlation_ = R

    def _compute_class_statistics(self):
        self.global_mean = np.mean(self.X_scaled, axis=0)
        self.class_means = np.zeros((self.n_classes, self.n_features))
        
        for c in range(self.n_classes):
            indices = np.where(self.y[:, c] == 1)[0]
            if len(indices) == 0:
                self.class_means[c] = self.global_mean  # 空类使用全局均值
                continue
                
            weights = self.P[c, indices]
            self.class_means[c] = np.average(self.X_scaled[indices], 
                                            axis=0, 
                                            weights=weights)

    def _solve_projection_matrix(self):
        # 散度矩阵计算
        Sw = np.zeros((self.n_features, self.n_features))
        Sb = np.zeros((self.n_features, self.n_features))
        
        for c in range(self.n_classes):
            indices = np.where(self.y[:, c] == 1)[0]
            if len(indices) == 0:
                continue
                
            # 类内散度
            X_c = self.X_scaled[indices]
            weights = self.P[c, indices]
            Sw += (X_c - self.class_means[c]).T @ np.diag(weights) @ (X_c - self.class_means[c])
            
            # 类间散度
            diff = (self.class_means[c] - self.global_mean).reshape(-1, 1)
            Sb += len(indices) * diff @ diff.T

        # 正则化处理
        Sw_reg = Sw + self.epsilon * np.eye(self.n_features)
        
        # 特征分解
        eigenvalues, eigenvectors = eigh(Sb, Sw_reg)
        
        # 降序排列
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 自动维度选择
        if self.n_components is None:
            total_variance = np.sum(eigenvalues)
            cumulative = np.cumsum(eigenvalues) / total_variance
            self.n_components = np.argmax(cumulative >= 0.999) + 1
            
        self.W = eigenvectors[:, :self.n_components]

    def transform(self, X):
        if self.W is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.scaler.transform(X) @ self.W

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
