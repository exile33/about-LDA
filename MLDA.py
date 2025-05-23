import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class MultiLabelLDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.G = None          # 投影矩阵
        self.m = None          # 全局均值
        self.C = None          # 标签相关性矩阵
        self.class_means = None # 类均值
    
    def fit(self, X, Y):
        """
        X: 输入数据矩阵, 形状为 (n_samples, n_features)
        Y: 多标签矩阵, 形状为 (n_samples, n_classes), 值为0或1
        """
        n_samples, n_features = X.shape
        n_classes = Y.shape[1]
        
        # 计算标签相关性矩阵C（余弦相似度）
        Y = Y.astype(float)
        self.C = self._compute_label_correlation(Y)
        
        # 计算归一化矩阵Z（处理过计数问题）
        Z = Y @ self.C
        Z = Z / np.linalg.norm(Z, ord=1, axis=1, keepdims=True)
        
        # 计算全局均值m（式9）
        total_weight = np.sum(Z)
        self.m = (X.T @ Z.sum(axis=0)) / total_weight
        
        # 计算类均值mk（式9）
        self.class_means = []
        weights = []
        for k in range(n_classes):
            weight_k = Z[:, k].sum()
            weights.append(weight_k)
            mk = (X.T @ Z[:, k]) / weight_k
            self.class_means.append(mk)
        self.class_means = np.array(self.class_means).T  # (n_features, n_classes)
        
        # 数据居中（式17）
        X_centered = X - self.m.reshape(1, -1)
        
        # 计算类间散度矩阵Sb（式6）
        S_b = np.zeros((n_features, n_features))
        for k in range(n_classes):
            diff = self.class_means[:, k] - self.m
            S_b += weights[k] * np.outer(diff, diff)
        
        # 计算类内散度矩阵Sw（式7）
        S_w = np.zeros((n_features, n_features))
        for k in range(n_classes):
            Xk_centered = X_centered - (self.class_means[:, k] - self.m).reshape(1, -1)
            weighted_Xk = (Z[:, k][:, np.newaxis] * Xk_centered)
            S_w += weighted_Xk.T @ weighted_Xk
        
        # 解决广义特征值问题 S_b * v = λ * S_w * v
        # 处理S_w的奇异性，使用伪逆
        S_w_pinv = np.linalg.pinv(S_w)
        eigvals, eigvecs = eigh(S_b, S_w_pinv)
        
        # 按特征值降序排列，选择前r个特征向量
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sorted_indices]
        if self.n_components is not None:
            self.G = eigvecs[:, :self.n_components]
        else:
            self.G = eigvecs
        
        return self
    
    def transform(self, X):
        """将数据投影到低维空间"""
        if self.G is None:
            raise ValueError("Model has not been fitted yet.")
        X_centered = X - self.m.reshape(1, -1)
        return X_centered @ self.G
    
    def _compute_label_correlation(self, Y):
        """计算标签相关性矩阵C（式13）"""
        # 余弦相似度
        Y_norm = Y / np.linalg.norm(Y, axis=0, keepdims=True)
        C = Y_norm.T @ Y_norm
        return C
    
    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)