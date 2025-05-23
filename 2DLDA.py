import numpy as np

class TwoLDA:
    def __init__(self):
        self.V = None             
        self.mean_total = None    
        self.class_means = {}      
        self.labels = None

    def fit(self, X_train, y_train):
        """
        拟合：只在列方向上计算 Sw (n×n) 与 Sb (n×n)，并求解广义特征值问题。
        参数：
            X_list: list of ndarray, 每个元素为 m×n 的灰度图像
            y_list: list of labels, 与 X_list 一一对应
        """
        self.labels = np.unique(y_train)
        N = len(X_train)
        m, n = X_train[0].shape

        # 计算全局均值
        self.mean_total = np.sum(X_train, axis=0) / N

        # 计算每类均值
        for lbl in self.labels:
            cls = [X for X, y in zip(X_train, y_train) if y == lbl]
            self.class_means[lbl] = np.sum(cls, axis=0) / len(cls)

        # 构造列向量散度矩阵 Sw, Sb (均为 n×n)
        Sw = np.zeros((n, n))
        Sb = np.zeros((n, n))
        for X, y in zip(X_train, y_train):
            M = self.class_means[y]
            D = X - M
            Sw += D.T @ D

        for lbl in self.labels:
            M = self.class_means[lbl]
            n_i = sum(y == lbl for y in y_train) # 实际上每个都是5
            Dm = M - self.mean_total
            Sb += n_i * (Dm.T @ Dm)

        # 求解广义特征值问题：Sb v = λ Sw v
        # 即 eig( pinv(Sw) @ Sb )
        eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(Sw) @ Sb)
        # 按降序排序
        idx = np.argsort(-eigvals.real)
        eigvecs = eigvecs.real
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
        self.V = eigvecs[:, idx]  # V 的每列是一个投影向量

    def transform(self, X_list, d):
        """
        将图像降维到 m×d。
        参数：
            X_list: list of m×n 矩阵
            d: 保留的投影方向数量
        返回：
            list of m×d 矩阵 Y = A @ V_d
        """
        V_d = self.V[:, :d]         # 取前 d 个投影向量
        return [X @ V_d for X in X_list]

    def reconstruct(self, Y_list, d):
        """
        对降维后的特征 Y (m×d) 重构回原始空间 m×n。
        参数：
            Y_list: list of m×d 矩阵
            d: 当初降维时的维度
        返回：
            list of m×n 矩阵 Ā = Y @ V_d^T
        """
        V_d = self.V[:, :d]
        return [Y @ V_d.T for Y in Y_list]

    def classify(self, X_test_list, X_train_list, y_train_list, d):
        """
        对测试集做分类。
        参数:
            X_test_list: 原始测试图像列表 (m×n)
            X_train_list: 原始训练图像列表 (m×n)
            y_train_list: 训练标签列表
            d: 降维维度
        返回:
            preds: 测试集预测标签列表
        """
        # 按当前 d 对 训练集 和 测试集 同时降维
        V_d = self.V[:, :d]
        X_train_transformed = [X @ V_d for X in X_train_list]   # m×d
        X_test_transformed  = [X @ V_d for X in X_test_list]    # m×d

        # 最近邻分类
        preds = []
        for Yt in X_test_transformed:
            best_lbl = None
            best_dist = np.inf
            for Ytr, lbl in zip(X_train_transformed, y_train_list):
                dist = sum(np.linalg.norm(Yt[:,k] - Ytr[:,k]) for k in range(d))

                if dist < best_dist:
                    best_dist = dist
                    best_lbl = lbl
            preds.append(best_lbl)
        return preds
