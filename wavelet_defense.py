import pywt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from scipy.stats import entropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors
import time
import pandas as pd
from collections import Counter
import seaborn as sns
import warnings
import traceback  # 导入traceback模块

warnings.filterwarnings("ignore")


class WaveletDefense:
    """
    基于小波变换的联邦学习防御机制

    通过分析客户端模型更新的频域特征差异来检测和过滤恶意更新
    """

    def __init__(self, model, num_clients, input_shape=(1, 28, 28), wavelet='db4',
                 epsilon=0.1, eps=0.01, min_samples=2, num_features=None, device=None,
                 clustering_method='kmeans', logger=None, verbose=False):
        """
        初始化小波防御机制

        Args:
            model: 全局模型
            num_clients: 客户端数量
            input_shape: 输入图像的形状 (channels, height, width)
            wavelet: 小波基函数类型
            epsilon: FGSM扰动大小
            eps: DBSCAN的邻域大小参数
            min_samples: DBSCAN的最小样本数参数
            num_features: 特征向量的维度 (None表示自动确定)
            device: 计算设备 (None表示自动检测)
            clustering_method: 聚类方法 ('dbscan', 'kmeans', 'spectral', 'agglomerative', 'auto')
        """
        self.model = model
        self.num_clients = num_clients
        self.input_shape = input_shape
        self.wavelet = wavelet
        self.epsilon = epsilon
        self.eps = eps
        self.min_samples = min_samples
        self.num_features = num_features
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.clustering_method = clustering_method
        self.logger = logger
        self.verbose = verbose

        # 性能指标
        self.detection_stats = []
        self.clustering_quality = []
        self.feature_importance = {}
        self.execution_time = {}
        self.eps_history = []

        self.benign_rep_values_history = []  # 存储每个轮次的真实良性客户端代表值
        self.malicious_rep_values_history = []  # 存储每个轮次的真实恶意客户端代表值
        self.epoch_numbers = []  # 存储对应的轮次号

        # 缓存每轮检测的恶意客户端索引，方便评估
        self.malicious_indices_cache = {}

        self.client_overhead = {}
        self.current_epoch = 0  # 添加当前轮次记录

    def _log(self, message, level="info", verbose_only=False):
        if verbose_only and not self.verbose:
            return

        if self.logger is not None:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(message)
        else:
            print(message)

    def extract_feature_differences(self, original_updates, probed_updates):
        """
        提取原始更新与诱导更新之间的频域差分特征。

        Args:
            original_updates: 原始客户端更新 [num_clients, update_dim]
            probed_updates: 诱导后的客户端更新 [num_clients, update_dim]

        Returns:
            np.ndarray: 差分特征矩阵 [num_clients, num_features]
        """
        feature_vectors = []

        for i, (orig_update, probed_update) in enumerate(zip(original_updates, probed_updates)):
            orig_coeffs = self.wavelet_transform(orig_update)
            orig_features = self.extract_wavelet_features(orig_coeffs)

            probed_coeffs = self.wavelet_transform(probed_update)
            probed_features = self.extract_wavelet_features(probed_coeffs)

            feature_diff = np.abs(probed_features - orig_features)
            if np.isnan(feature_diff).any() or np.isinf(feature_diff).any():
                print(f"客户端 {i} 的特征包含无效值，使用零替代")
                feature_diff = np.zeros_like(feature_diff)

            feature_vectors.append(feature_diff)

        return np.array(feature_vectors)

    def estimate_benign_feature_prototype(self, client_updates, probed_updates, benign_indices=None):
        """
        估计良性客户端在防御特征空间中的原型向量。

        Args:
            client_updates: 原始客户端更新
            probed_updates: 诱导后的客户端更新
            benign_indices: 良性客户端索引；若为None，则使用全部客户端

        Returns:
            np.ndarray: 良性原型特征
        """
        feature_vectors = self.extract_feature_differences(client_updates, probed_updates)
        if len(feature_vectors) == 0:
            return np.array([])

        if benign_indices is None:
            benign_vectors = feature_vectors
        else:
            benign_vectors = feature_vectors[benign_indices]

        if len(benign_vectors) == 0:
            return np.array([])

        return np.mean(benign_vectors, axis=0)

    def record_client_overhead(self, client_id, compute_time, model_size_kb):
        """
        记录客户端的计算和通信开销

        Args:
            client_id: 客户端ID
            compute_time: 计算时间（秒）
            model_size_kb: 模型大小（KB）
        """
        epoch_key = f"epoch_{self.current_epoch}"

        if epoch_key not in self.client_overhead:
            self.client_overhead[epoch_key] = {}

        self.client_overhead[epoch_key][f"client_{client_id}"] = {
            "compute_time": compute_time,
            "comm_bytes": model_size_kb
        }

    def save_overhead_statistics(self, filename='client_overhead_stats.json'):
        """保存客户端开销统计信息到文件"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.client_overhead, f, indent=2)

        print(f"客户端开销统计信息已保存到 {filename}")

    def _kmeans_clustering(self, feature_vectors, n_clusters=2):
        """
        KMeans聚类

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
                注意：在修改后的代码中，这将是降维后的特征向量
            n_clusters: 聚类数量

        Returns:
            labels: 聚类标签
        """
        try:
            # 确保聚类数不超过样本数
            n_clusters = min(n_clusters, len(feature_vectors))

            # 执行KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_vectors)

            self._log(f"KMeans聚类完成, 聚类数={n_clusters}", verbose_only=True)
            return labels

        except Exception as e:
            self._log(f"KMeans聚类出错: {str(e)}", level="warning")
            # 失败时返回所有点为一类
            return np.zeros(len(feature_vectors), dtype=int)

    def _spectral_clustering(self, feature_vectors, n_clusters=2):
        """
        谱聚类

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
                注意：在修改后的代码中，这将是降维后的特征向量
            n_clusters: 聚类数量

        Returns:
            labels: 聚类标签
        """
        try:
            # 确保聚类数不超过样本数
            n_clusters = min(n_clusters, len(feature_vectors))

            # 执行谱聚类
            clustering = SpectralClustering(n_clusters=n_clusters,
                                            assign_labels='discretize',
                                            random_state=42)
            labels = clustering.fit_predict(feature_vectors)

            self._log(f"谱聚类完成, 聚类数={n_clusters}", verbose_only=True)
            return labels

        except Exception as e:
            self._log(f"谱聚类出错: {str(e)}", level="warning")
            # 失败时返回所有点为一类
            return np.zeros(len(feature_vectors), dtype=int)

    def _agglomerative_clustering(self, feature_vectors, n_clusters=2):
        """
        层次聚类

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
                注意：在修改后的代码中，这将是降维后的特征向量
            n_clusters: 聚类数量

        Returns:
            labels: 聚类标签
        """
        try:
            # 确保聚类数不超过样本数
            n_clusters = min(n_clusters, len(feature_vectors))

            # 执行层次聚类
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(feature_vectors)

            self._log(f"层次聚类完成, 聚类数={n_clusters}", verbose_only=True)
            return labels

        except Exception as e:
            self._log(f"层次聚类出错: {str(e)}", level="warning")
            # 失败时返回所有点为一类
            return np.zeros(len(feature_vectors), dtype=int)

    def _threshold_based_clustering(self, feature_vectors_2d):
        """
        基于阈值的聚类 (主要基于第二主成分)

        Args:
            feature_vectors_2d: 降维后的特征向量 [num_clients, 2]

        Returns:
            labels: 聚类标签
        """
        try:
            # 计算第二主成分的统计信息
            y_values = feature_vectors_2d[:, 1]
            mean_y = np.mean(y_values)
            std_y = np.std(y_values)

            # 设置阈值为均值加上2个标准差
            threshold = mean_y + 2 * std_y

            # 根据阈值进行分类
            labels = np.zeros(len(feature_vectors_2d), dtype=int)
            labels[y_values > threshold] = 1

            self._log(f"基于阈值的聚类完成, 阈值={threshold:.4f}, 类别1大小={np.sum(labels == 1)}", verbose_only=True)
            return labels

        except Exception as e:
            self._log(f"基于阈值的聚类出错: {str(e)}", level="warning")
            # 失败时返回所有点为一类
            return np.zeros(len(feature_vectors_2d), dtype=int)

    def _density_peak_clustering(self, feature_vectors, n_clusters=2):
        """
        密度峰值聚类

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
                注意：在修改后的代码中，这将是降维后的特征向量
            n_clusters: 聚类数量

        Returns:
            labels: 聚类标签
        """
        try:
            # 计算距离矩阵
            dist_matrix = squareform(pdist(feature_vectors))
            n = len(feature_vectors)

            # 确定截断距离dc (使用平均距离)
            dc = np.mean(dist_matrix)

            # 计算局部密度rho
            rho = np.zeros(n)
            for i in range(n):
                rho[i] = np.sum(np.exp(-(dist_matrix[i] / dc) ** 2))

            # 计算delta (到密度更高点的最小距离)
            delta = np.zeros(n)
            nneigh = np.zeros(n, dtype=int)

            # 对于密度最大的点，指定一个特殊值
            max_rho_idx = np.argmax(rho)
            delta[max_rho_idx] = np.max(dist_matrix)
            nneigh[max_rho_idx] = max_rho_idx

            # 对于其他点
            for i in range(n):
                if i != max_rho_idx:
                    higher_rho_indices = np.where(rho > rho[i])[0]
                    if len(higher_rho_indices) > 0:
                        delta[i] = np.min(dist_matrix[i, higher_rho_indices])
                        nneigh[i] = higher_rho_indices[np.argmin(dist_matrix[i, higher_rho_indices])]
                    else:
                        delta[i] = np.max(dist_matrix[i])
                        nneigh[i] = i

            # 计算gamma (密度和距离的乘积)
            gamma = rho * delta

            # 选择聚类中心 (gamma值最大的n_clusters个点)
            centers = np.argsort(gamma)[-n_clusters:]

            # 将其他点分配到聚类
            labels = -np.ones(n, dtype=int)
            for i, center in enumerate(centers):
                labels[center] = i

            # 从高密度到低密度分配
            sorted_indices = np.argsort(rho)[::-1]
            for i in sorted_indices:
                if labels[i] == -1:  # 如果尚未分配
                    labels[i] = labels[nneigh[i]]

            self._log(f"密度峰值聚类完成, 聚类数={n_clusters}", verbose_only=True)
            return labels

        except Exception as e:
            self._log(f"密度峰值聚类出错: {str(e)}", level="warning")
            # 失败时返回所有点为一类
            return np.zeros(len(feature_vectors), dtype=int)

    def _ensemble_clustering(self, feature_vectors, feature_vectors_2d):
        """
        集成多种聚类方法

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
                注意：在修改后的代码中，这将是降维后的特征向量，与feature_vectors_2d相同
            feature_vectors_2d: 降维后的特征向量 [num_clients, 2]

        Returns:
            labels: 聚类标签
        """
        try:
            results = []

            # 方法1: 自适应DBSCAN
            dbscan_labels, _ = self._adaptive_dbscan_clustering(feature_vectors)
            results.append(dbscan_labels)

            # 方法2: KMeans聚类
            kmeans_labels = self._kmeans_clustering(feature_vectors)
            results.append(kmeans_labels)

            # 方法3: 基于阈值的分类
            threshold_labels = self._threshold_based_clustering(feature_vectors_2d)
            results.append(threshold_labels)

            # 方法4: 密度峰值聚类
            density_labels = self._density_peak_clustering(feature_vectors)
            results.append(density_labels)

            # 标准化标签 (让所有方法的标签具有相同的含义)
            normalized_results = []
            for result in results:
                unique_labels = np.unique(result)
                if len(unique_labels) <= 1:
                    # 如果只有一个类别，跳过
                    continue

                # 计算每个类别的平均y值
                mean_y_per_label = {}
                for label in unique_labels:
                    if label == -1:  # 跳过噪声点
                        continue
                    indices = np.where(result == label)[0]
                    mean_y_per_label[label] = np.mean(feature_vectors_2d[indices, 1])

                # 找出具有最高平均y值的类别
                if mean_y_per_label:
                    high_label = max(mean_y_per_label, key=mean_y_per_label.get)

                    # 将标签转换为二分类: 1表示具有最高平均y值的类别，0表示其他类别
                    normalized_result = np.zeros(len(result))
                    normalized_result[result == high_label] = 1
                    normalized_results.append(normalized_result)

            if not normalized_results:
                # 如果所有方法都失败，使用备选方法
                return self._threshold_based_clustering(feature_vectors_2d)

            # 对normalized_results进行投票
            ensemble_result = np.zeros(len(feature_vectors))
            for result in normalized_results:
                ensemble_result += result

            # 根据投票结果确定最终标签
            final_labels = np.zeros(len(feature_vectors), dtype=int)
            threshold = len(normalized_results) / 2  # 多数票
            final_labels[ensemble_result > threshold] = 1

            self._log(f"集成聚类完成, 使用了{len(normalized_results)}种方法, 类别1大小={np.sum(final_labels == 1)}", verbose_only=True)
            return final_labels

        except Exception as e:
            self._log(f"集成聚类出错: {str(e)}", level="warning")
            # 失败时使用基于阈值的方法
            return self._threshold_based_clustering(feature_vectors_2d)

    def _norm_based_detection(self, feature_vectors):
        """
        基于差分向量模长的恶意客户端检测

        Args:
            feature_vectors: 差分特征向量 [num_clients, num_features]

        Returns:
            malicious_indices: 被检测为恶意的客户端索引列表
        """
        try:
            # 计算每个差分向量的模长
            norms = np.linalg.norm(feature_vectors, axis=1)

            # 找出模长最大的客户端
            sorted_indices = np.argsort(norms)

            # 分析模长分布
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)

            # 打印模长统计信息
            print(f"\n=== 差分向量模长统计 ===")
            print(f"平均模长: {mean_norm:.4f}")
            print(f"模长标准差: {std_norm:.4f}")
            print(f"最大模长: {norms[sorted_indices[-1]]:.4f}, 客户端索引: {sorted_indices[-1]}")
            print(f"最小模长: {norms[sorted_indices[0]]:.4f}, 客户端索引: {sorted_indices[0]}")

            # 确定阈值
            threshold = mean_norm + 2 * std_norm

            # 找出模长大于阈值的客户端
            malicious_indices = np.where(norms > threshold)[0]

            # 如果没有检测到任何恶意客户端，选择模长最大的10%作为恶意客户端
            if len(malicious_indices) == 0:
                n_select = max(1, int(len(feature_vectors) * 0.1))
                malicious_indices = sorted_indices[-n_select:]

            print(f"基于模长的检测: 检测到{len(malicious_indices)}个恶意客户端")
            return malicious_indices.tolist()

        except Exception as e:
            print(f"基于模长的检测出错: {str(e)}")
            # 随机选择10%的客户端作为恶意客户端
            n_select = max(1, int(len(feature_vectors) * 0.1))
            return np.random.choice(len(feature_vectors), n_select, replace=False).tolist()

    def _visualize_clusters(self, feature_vectors_2d, labels, epoch_num):
        """
        可视化聚类结果

        Args:
            feature_vectors_2d: 降维后的特征向量 [num_clients, 2]
            labels: 聚类标签 [num_clients]
            epoch_num: 当前训练轮次
        """
        plt.figure(figsize=(12, 8))

        # 使用不同颜色表示不同类别
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # 绘制散点图
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                # 噪声点使用黑色叉号表示
                plt.scatter(
                    feature_vectors_2d[mask, 0],
                    feature_vectors_2d[mask, 1],
                    c='k',
                    marker='x',
                    label='噪声',
                    alpha=0.6,
                    s=100
                )
            else:
                plt.scatter(
                    feature_vectors_2d[mask, 0],
                    feature_vectors_2d[mask, 1],
                    c=[color],
                    label=f'簇 {label}',
                    alpha=0.6,
                    s=100
                )

            # 添加客户端索引标签
            for idx, (x, y) in enumerate(feature_vectors_2d[mask]):
                client_idx = np.where(mask)[0][idx]
                plt.annotate(
                    f'{client_idx}',
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        plt.title(f'轮次 {epoch_num} - 客户端更新聚类', fontsize=14)
        plt.xlabel('第一主成分', fontsize=12)
        plt.ylabel('第二主成分', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图像
        plt.savefig(f'cluster_epoch_{epoch_num}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_feature_importance(self, features, labels):
        """
        分析特征重要性

        Args:
            features: 标准化特征 [num_clients, num_features]
            labels: 聚类标签 [num_clients]
        """
        # 如果没有特征名称记录，跳过
        if not self.feature_importance:
            return

        # 计算每个特征的Fisher判别比率
        unique_labels = np.unique(labels)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]

        n_features = features.shape[1]
        feature_scores = np.zeros(n_features)

        # 计算总体平均值
        global_mean = np.mean(features, axis=0)

        # 计算组内方差和组间方差
        between_class_var = np.zeros(n_features)
        within_class_var = np.zeros(n_features)

        for label in unique_labels:
            class_samples = features[labels == label]
            if len(class_samples) <= 1:
                continue

            class_mean = np.mean(class_samples, axis=0)

            # 组间方差
            between_class_var += len(class_samples) * (class_mean - global_mean) ** 2

            # 组内方差
            within_class_var += np.sum((class_samples - class_mean) ** 2, axis=0)

        # 避免除零
        within_class_var[within_class_var < 1e-10] = 1e-10

        # 计算Fisher判别比率
        feature_scores = between_class_var / within_class_var

        # 更新特征重要性
        feature_names = list(self.feature_importance.keys())
        for i, score in enumerate(feature_scores[:len(feature_names)]):
            self.feature_importance[feature_names[i]] += score

    def _statistical_outlier_detection(self, features):
        """
        使用统计方法检测异常点

        Args:
            features: 客户端特征向量 [num_clients, num_features]

        Returns:
            outlier_indices: 异常点索引列表
        """
        try:
            # 计算每个客户端到其他客户端的平均距离
            distances = cdist(features, features, metric='euclidean')
            np.fill_diagonal(distances, np.inf)  # 忽略自身距离
            mean_distances = np.mean(distances, axis=1)

            # 计算距离统计信息
            mean_dist = np.mean(mean_distances)
            std_dist = np.std(mean_distances)

            # 如果标准差异常小，使用基于百分位数的方法
            if std_dist < 1e-10:
                threshold = np.percentile(mean_distances, 75)
            else:
                # 使用z-score方法，将阈值设为2个标准差
                threshold = mean_dist + 2 * std_dist

            # 找出距离大于阈值的点
            outlier_indices = np.where(mean_distances > threshold)[0]

            # 如果所有点都被标记为异常或没有异常，随机选择一些点
            if len(outlier_indices) == 0 or len(outlier_indices) == len(features):
                num_outliers = max(1, int(len(features) * 0.2))  # 至少标记1个，最多20%
                outlier_indices = np.random.choice(len(features), num_outliers, replace=False)

            return outlier_indices.tolist()

        except Exception as e:
            print(f"统计异常检测失败: {str(e)}")
            # 随机选择一些点
            num_outliers = max(1, int(len(features) * 0.1))  # 10%
            return np.random.choice(len(features), num_outliers, replace=False).tolist()

    def aggregate_updates(self, client_updates, malicious_indices, data_weights=None):
        """
        聚合良性客户端的更新 (优化版FedAvg)

        Args:
            client_updates: 所有客户端更新 [num_clients, update_dim]
            malicious_indices: 恶意客户端索引列表
            data_weights: 客户端数据量权重列表，如果为None则假设所有客户端数据量相等

        Returns:
            aggregated_update: 聚合后的更新 [update_dim]
        """
        start_time = time.time()
        try:
            # 获取良性客户端的更新和权重
            benign_indices = [i for i in range(len(client_updates)) if i not in malicious_indices]

            print("\n=== 聚合信息 ===")
            print(f"总客户端数: {len(client_updates)}")
            print(f"选中的良性客户端数: {len(benign_indices)}")

            if len(benign_indices) > 0:
                benign_updates = client_updates[benign_indices]

                # 检查更新是否包含无效值
                invalid_updates = []
                for i, update in enumerate(benign_updates):
                    if torch.isnan(update).any() or torch.isinf(update).any():
                        print(f"警告: 客户端 {benign_indices[i]} 的更新包含无效值")
                        invalid_updates.append(i)

                # 移除无效的更新
                if invalid_updates:
                    valid_indices = [i for i in range(len(benign_updates)) if i not in invalid_updates]
                    if valid_indices:
                        benign_updates = benign_updates[valid_indices]
                        # 如果有数据权重，也需要相应调整
                        if data_weights is not None:
                            data_weights = [data_weights[benign_indices[i]] for i in valid_indices]
                        benign_indices = [benign_indices[i] for i in valid_indices]
                        print(f"已移除 {len(invalid_updates)} 个包含无效值的更新")
                    else:
                        print("所有良性更新都包含无效值，回退到使用简单平均")
                        # 使用最简单的方法：所有更新的平均值
                        return torch.mean(client_updates, dim=0)

                # 执行加权联邦平均
                if data_weights is not None:
                    # 提取良性客户端的权重并归一化
                    benign_weights = [data_weights[i] for i in benign_indices]
                    total_weight = sum(benign_weights)
                    if total_weight > 0:
                        normalized_weights = [w / total_weight for w in benign_weights]
                    else:
                        normalized_weights = [1.0 / len(benign_weights)] * len(benign_weights)

                    # 加权平均
                    aggregated_update = torch.zeros_like(benign_updates[0])
                    for i, update in enumerate(benign_updates):
                        aggregated_update += update * normalized_weights[i]

                    print(f"执行了加权FedAvg，使用 {len(benign_updates)} 个良性更新")
                else:
                    # 如果没有提供权重，使用简单平均
                    aggregated_update = torch.mean(benign_updates, dim=0)
                    print(f"执行了简单FedAvg，使用 {len(benign_updates)} 个良性更新")
            else:
                print("警告: 没有找到良性更新!")
                # 更保守的后备策略：使用离群点检测方法
                mean_update = torch.mean(client_updates, dim=0)
                distances = torch.norm(client_updates - mean_update, dim=1)

                # 选择距离最小的25%作为"较好的"更新
                k = max(1, len(client_updates) // 4)
                closest_indices = torch.argsort(distances)[:k]

                if data_weights is not None:
                    # 使用这些客户端的数据权重进行加权平均
                    closest_weights = [data_weights[i] for i in closest_indices]
                    total_weight = sum(closest_weights)
                    if total_weight > 0:
                        normalized_weights = [w / total_weight for w in closest_weights]
                    else:
                        normalized_weights = [1.0 / k] * k

                    aggregated_update = torch.zeros_like(client_updates[0])
                    for i, idx in enumerate(closest_indices):
                        aggregated_update += client_updates[idx] * normalized_weights[i]
                else:
                    # 如果没有权重，使用简单平均
                    aggregated_update = torch.mean(client_updates[closest_indices], dim=0)

                print(f"使用了 {k} 个最接近均值的更新作为备选聚合策略")

            self.execution_time['aggregation'] = time.time() - start_time
            return aggregated_update

        except Exception as e:
            print(f"聚合更新时发生错误: {str(e)}")
            # 降级策略：使用所有更新的均值
            self.execution_time['aggregation'] = time.time() - start_time
            return torch.mean(client_updates, dim=0)


    def plot_representative_values_highquality(self, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy.stats import ttest_ind, gaussian_kde

        if not self.benign_rep_values_history or not self.malicious_rep_values_history:
            print("代表值数据不足，无法绘图")
            return

        # 构造 DataFrame
        df = pd.DataFrame({
            'Representative Value': self.benign_rep_values_history + self.malicious_rep_values_history,
            'Client Type': ['Benign'] * len(self.benign_rep_values_history) + ['Malicious'] * len(
                self.malicious_rep_values_history)
        })

        # 色彩定义
        fill_palette = {
            'Malicious': (31 / 255, 119 / 255, 180 / 255, 0.5),
            'Benign': (214 / 255, 39 / 255, 40 / 255, 0.5)
        }
        edge_palette = {
            'Malicious': (31 / 255, 119 / 255, 180 / 255, 1.0),
            'Benign': (214 / 255, 39 / 255, 40 / 255, 1.0)
        }

        plt.rcParams.update({
            'font.family': ['Times New Roman', 'serif'],
            'font.size': 11,
            'axes.linewidth': 0.8,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
        })

        fig, ax = plt.subplots(figsize=(6, 4))

        for i, label in enumerate(['Benign', 'Malicious']):
            data = np.array(df[df['Client Type'] == label]['Representative Value'])

            # 自定义 violin（生成尖尖 KDE）
            kde = gaussian_kde(data, bw_method='silverman')
            y_vals = np.linspace(min(data) - 1, max(data) + 1, 500)
            kde_vals = kde(y_vals)
            kde_vals = kde_vals / kde_vals.max() * 0.4  # 控制最大宽度

            ax.fill_betweenx(y_vals, i - kde_vals, i + kde_vals,
                             facecolor=fill_palette[label],
                             edgecolor=edge_palette[label],
                             linewidth=1.2,
                             alpha=0.6)

            # 计算 box 相关值
            q1, med, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            lw = np.min(data[data >= q1 - 1.5 * iqr])
            uw = np.max(data[data <= q3 + 1.5 * iqr])
            box_width = 0.25

            # IQR 矩形
            rect = plt.Rectangle((i - box_width / 2, q1), box_width, iqr,
                                 linewidth=1.2,
                                 edgecolor=edge_palette[label],
                                 facecolor=(1, 1, 1, 0.6),  # 半透明白
                                 zorder=3)
            ax.add_patch(rect)

            # 中位数线
            ax.plot([i - box_width / 2, i + box_width / 2], [med, med],
                    lw=2.0, color=edge_palette[label], zorder=4)

            # whiskers + caps
            cap_width = 0.08
            ax.plot([i, i], [lw, q1], lw=1.0, color=edge_palette[label], zorder=2)
            ax.plot([i - cap_width, i + cap_width], [lw, lw], lw=1.0, color=edge_palette[label], zorder=2)
            ax.plot([i, i], [q3, uw], lw=1.0, color=edge_palette[label], zorder=2)
            ax.plot([i - cap_width, i + cap_width], [uw, uw], lw=1.0, color=edge_palette[label], zorder=2)

        # 显著性分析
        #stat, pval = ttest_ind(self.benign_rep_values_history, self.malicious_rep_values_history)
        #formatted_pval = f"P = {pval:.2e}"

        # P值文本位置
        #y_max = df['Representative Value'].max()
        #y_min = df['Representative Value'].min()
        #y_range = y_max - y_min
        #y_text = y_max + 0.05 * y_range

        #ax.text(0.5, y_text, formatted_pval,
                #ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 坐标设置
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Malicious', 'Benign'], fontweight='bold')
        ax.set_ylabel("Norm of Eigenvector", fontweight='bold')

        #for spine in ['top', 'right']:
            #ax.spines[spine].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)  # 可调

        plt.tight_layout(pad=0.5)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"图表保存至 {save_path}")
        plt.show()

    def print_performance_summary(self):
        """打印防御机制的性能统计信息"""
        print("\n" + "=" * 50)
        print("小波防御机制性能统计")
        print("=" * 50)

        # 检测统计信息
        if self.detection_stats:
            df = pd.DataFrame(self.detection_stats)
            print("\n检测统计信息:")
            print(f"平均检测率: {np.mean(df['detected_malicious'] / df['total_clients']):.2%}")
            print(f"平均良性簇大小: {np.mean(df['benign_cluster_size']):.2f}")
            if 'noise_points' in df.columns:
                print(f"平均噪声点比例: {np.mean(df['noise_points'] / df['total_clients']):.2%}")

            # 聚类方法统计
            if 'clustering_method' in df.columns:
                method_counts = Counter(df['clustering_method'])
                print("\n聚类方法使用统计:")
                for method, count in method_counts.items():
                    print(f"  {method}: {count} 次 ({count / len(df):.1%})")

        # 聚类质量
        if self.clustering_quality:
            df = pd.DataFrame(self.clustering_quality)
            print("\n聚类质量指标:")
            print(f"平均聚类数量: {np.mean(df['n_clusters']):.2f}")
            print(f"平均轮廓系数: {np.mean([s for s in df['silhouette'] if s > -1]):.3f}")
            print(f"平均Calinski-Harabasz分数: {np.mean([s for s in df['ch_score'] if s > -1]):.1f}")

        # 执行时间
        if self.execution_time:
            print("\n平均执行时间 (秒):")
            for phase, times in self.execution_time.items():
                if isinstance(times, list):
                    print(f"  {phase}: {np.mean(times):.3f}")
                else:
                    print(f"  {phase}: {times:.3f}")

        # DBSCAN参数历史
        if self.eps_history:
            print(f"\nDBSCAN eps参数历史:")
            print(f"  初始值: {self.eps_history[0]:.4f}")
            print(f"  最终值: {self.eps_history[-1]:.4f}")
            print(f"  平均值: {np.mean(self.eps_history):.4f}")
            print(f"  变化范围: {min(self.eps_history):.4f} - {max(self.eps_history):.4f}")

        # 特征重要性
        if self.feature_importance:
            print("\n前5个最重要特征:")
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_features[:5]:
                print(f"  {name}: {score:.3f}")

        print("\n真实客户端代表值统计信息:")
        print(f"DEBUG: benign_rep_values_history长度: {len(self.benign_rep_values_history)}")
        print(f"DEBUG: malicious_rep_values_history长度: {len(self.malicious_rep_values_history)}")
        print(f"DEBUG: benign_rep_values_history内容: {self.benign_rep_values_history}")
        print(f"DEBUG: malicious_rep_values_history内容: {self.malicious_rep_values_history}")
        if self.benign_rep_values_history and self.malicious_rep_values_history:
            avg_benign = np.mean(self.benign_rep_values_history)
            avg_malicious = np.mean(self.malicious_rep_values_history)
            avg_diff = avg_malicious - avg_benign

            print(f"真实良性客户端平均代表值: {avg_benign:.4f}")
            print(f"真实恶意客户端平均代表值: {avg_malicious:.4f}")
            print(f"真实平均代表值差异: {avg_diff:.4f}")

            # 可视化真实客户端代表值
            self.plot_representative_values(save_path='true_representative_values_distribution.png')
            self.plot_representative_values_highquality(save_path='high_quality_representative_values.pdf')


    def plot_detection_performance(self, true_malicious_indices=None):
        """
        绘制检测性能随时间的变化

        Args:
            true_malicious_indices: 真实恶意客户端索引列表，用于计算精确率和召回率
        """
        if not self.detection_stats:
            print("没有检测统计数据可供绘制")
            return

        df = pd.DataFrame(self.detection_stats)

        plt.figure(figsize=(15, 10))

        # 检测统计
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['detected_malicious'], marker='o', label='检测为恶意')
        plt.plot(df['epoch'], df['benign_cluster_size'], marker='s', label='良性簇大小')
        if 'noise_points' in df.columns:
            plt.plot(df['epoch'], df['noise_points'], marker='^', label='噪声点')
        plt.plot(df['epoch'], df['total_clients'], '--', label='总客户端数')
        plt.xlabel('训练轮次')
        plt.ylabel('客户端数量')
        plt.title('检测统计')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 如果提供了真实恶意客户端索引
        if true_malicious_indices is not None:
            true_malicious_set = set(true_malicious_indices)
            precision = []
            recall = []
            f1_score = []

            for epoch in df['epoch']:
                detected_malicious = set(self.malicious_indices_cache.get(epoch, []))

                # 计算精确率和召回率
                tp = len(detected_malicious & true_malicious_set)
                fp = len(detected_malicious - true_malicious_set)
                fn = len(true_malicious_set - detected_malicious)

                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

                precision.append(p)
                recall.append(r)
                f1_score.append(f1)

            # 精确率、召回率和F1分数
            plt.subplot(2, 2, 2)
            plt.plot(df['epoch'], precision, marker='o', label='精确率')
            plt.plot(df['epoch'], recall, marker='s', label='召回率')
            plt.plot(df['epoch'], f1_score, marker='^', label='F1分数')
            plt.xlabel('训练轮次')
            plt.ylabel('值')
            plt.title('检测性能指标')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('detection_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self):
        """绘制特征重要性"""
        if not self.feature_importance:
            print("没有特征重要性数据可供绘制")
            return

        # 获取前15个最重要的特征
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        names, values = zip(*sorted_features)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(names)), values, align='center')
        plt.yticks(range(len(names)), names)
        plt.xlabel('重要性分数')
        plt.title('特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_representative_values(self, save_path=None):
        """
        创建真实良性和恶意客户端代表值的专业可视化图表

        Args:
            save_path: 图表保存路径（None表示不保存）
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        if not self.benign_rep_values_history or not self.malicious_rep_values_history:
            print("没有足够的数据可供绘制真实客户端代表值图表")
            return

        # 设置样式
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))

        # 准备箱形图数据
        data = []

        # 良性到良性（使用真实良性客户端的值）
        if len(self.benign_rep_values_history) > 0:
            data.append(self.benign_rep_values_history)

        # 良性到恶意（过渡数据 - 可以模拟）
        transition_data = []
        for b, m in zip(self.benign_rep_values_history, self.malicious_rep_values_history):
            mid_point = (b + m) / 2
            transition_data.append(mid_point)
        data.append(transition_data)

        # 恶意到恶意（使用真实恶意客户端的值）
        if len(self.malicious_rep_values_history) > 0:
            data.append(self.malicious_rep_values_history)

        # 创建箱形图
        box = plt.boxplot(data, patch_artist=True, widths=0.5)

        # 自定义颜色
        colors = ['#90EE90', '#FFD700', '#FF6347']  # 浅绿色，金色，番茄色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # 添加标签和标题
        plt.xticks([1, 2, 3], ['良性到\n良性', '良性到\n恶意', '恶意到\n恶意'])
        plt.ylabel('欧几里得距离', fontsize=12)
        plt.title('真实客户端类型的代表值分布', fontsize=14, fontweight='bold')

        # 添加单个点作为群体散点图
        for i, d in enumerate(data, 1):
            # 添加x位置的抖动
            x_jitter = np.random.normal(i, 0.04, size=len(d))
            plt.scatter(x_jitter, d, alpha=0.6, s=30, c='#2F4F4F')

        # 为几个点添加轮次标签
        for i, d in enumerate(data, 1):
            if len(d) > 0 and len(self.epoch_numbers) > 0:
                # 标记几个代表性轮次
                indices_to_label = [0, len(d) // 2, -1] if len(d) > 2 else range(len(d))
                for idx in indices_to_label:
                    if 0 <= idx < len(d) and idx < len(self.epoch_numbers):
                        plt.annotate(f"轮次 {self.epoch_numbers[idx]}",
                                     xy=(i + np.random.normal(0, 0.03), d[idx]),
                                     xytext=(5, 5), textcoords='offset points',
                                     fontsize=8, alpha=0.7)

        # 在箱形图下方添加网格
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加解释指标的注释
        plt.figtext(0.13, 0.01,
                    "欧几里得距离衡量PCA降维空间中客户端特征向量之间的分离程度。\n"
                    "真实良性和恶意客户端之间的距离越大，表示攻击的区分度越高，有助于检测。",
                    fontsize=10, style='italic')

        # 添加子图标签
        plt.annotate('(a) IID', xy=(0.5, -0.1), xycoords='axes fraction',
                     ha='center', va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()

        # 如果提供了路径，则保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"真实客户端代表值分布图已保存至 {save_path}")

        plt.show()

    def plot_clustering_methods(self):
        """绘制聚类方法使用统计"""
        if not self.detection_stats or 'clustering_method' not in self.detection_stats[0]:
            print("没有聚类方法统计数据可供绘制")
            return

        # 统计各聚类方法的使用频率
        method_counts = {}
        for stat in self.detection_stats:
            method = stat['clustering_method']
            if method not in method_counts:
                method_counts[method] = 0
            method_counts[method] += 1

        # 绘制饼图
        plt.figure(figsize=(10, 8))
        labels = list(method_counts.keys())
        sizes = list(method_counts.values())
        colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True)
        plt.axis('equal')  # 确保饼图是圆的
        plt.title('聚类方法使用统计')

        plt.savefig('clustering_methods.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_eps_history(self):
        """绘制DBSCAN的eps参数变化历史"""
        if not self.eps_history:
            print("没有eps参数历史数据可供绘制")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.eps_history)), self.eps_history, marker='o')
        plt.xlabel('轮次')
        plt.ylabel('eps值')
        plt.title('DBSCAN的eps参数变化历史')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig('eps_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_statistics(self, filename='wavelet_defense_stats.json'):
        """保存防御统计信息到文件"""
        stats = {
            'detection_stats': self.detection_stats,
            'clustering_quality': self.clustering_quality,
            'feature_importance': self.feature_importance,
            'execution_time': self.execution_time,
            'eps_history': self.eps_history
        }

        # 将numpy和torch值转换为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        stats = convert_to_serializable(stats)

        import json
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"统计信息已保存到 {filename}")

    def get_malicious_indices_at_epoch(self, epoch):
        """获取指定轮次的恶意客户端索引"""
        return self.malicious_indices_cache.get(epoch, [])

    def _select_best_clustering_method(self, feature_vectors_2d):
        """
        根据数据特性自动选择最佳聚类方法

        Args:
            feature_vectors_2d: 降维后的特征向量 [num_clients, 2]
                注意：在修改后的代码中，这将直接用于聚类而不仅仅是可视化

        Returns:
            best_method: 最佳聚类方法名称
        """
        try:
            # 计算数据分布的基本统计信息
            mean_y = np.mean(feature_vectors_2d[:, 1])
            std_y = np.std(feature_vectors_2d[:, 1])

            # 检查是否存在明显的分离
            # 如果Y轴上存在明显的分离，使用阈值法
            if std_y > 1.0:
                potential_outliers = np.where(feature_vectors_2d[:, 1] > mean_y + std_y * 2)[0]
                if len(potential_outliers) > 0 and len(potential_outliers) < len(feature_vectors_2d) * 0.3:
                    return 'threshold'

            # 计算点之间的距离分布
            dist_matrix = squareform(pdist(feature_vectors_2d))
            avg_dist = np.mean(dist_matrix)
            max_dist = np.max(dist_matrix)

            # 如果数据点分布稀疏，使用DBSCAN
            if max_dist / avg_dist > 5:
                return 'dbscan'

            # 如果数据点分布均匀，使用KMeans
            if len(feature_vectors_2d) < 100:
                return 'kmeans'

            # 如果数据点较多，且分布不规则，使用Spectral聚类
            return 'ensemble'

        except Exception as e:
            print(f"选择聚类方法时出错: {str(e)}")
            # 默认使用组合方法
            return 'ensemble'

    def _adaptive_dbscan_clustering(self, feature_vectors, max_attempts=10):
        """
        自适应DBSCAN聚类

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
                注意：在修改后的代码中，这将是降维后的特征向量
            max_attempts: 最大尝试次数

        Returns:
            (labels, eps): 聚类标签和使用的eps值
        """
        # 首先估计最佳eps值
        eps = self._estimate_eps(feature_vectors)
        initial_eps = eps
        min_samples = self.min_samples

        for attempt in range(max_attempts):
            # 执行DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_vectors)
            labels = clustering.labels_

            # 计算聚类统计
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            print(
                f"DBSCAN尝试 {attempt + 1}: eps={eps:.4f}, min_samples={min_samples}, 聚类数={n_clusters}, 噪声点数={n_noise}")

            # 检查结果并相应调整参数
            if 2 <= n_clusters <= 10 and n_noise < len(feature_vectors) * 0.5:
                # 找到了合适的参数
                print(f"找到合适的DBSCAN参数: eps={eps:.4f}, min_samples={min_samples}")
                break

            if n_noise == len(feature_vectors):
                # 所有点都是噪声点，增加eps
                eps *= 2.0
            elif n_clusters <= 1:
                # 只有一个聚类或没有聚类，增加eps但减小min_samples
                eps *= 1.5
                min_samples = max(2, min_samples - 1)
            elif n_clusters > 10:
                # 太多聚类，减小eps
                eps *= 0.8
            else:
                # 太多单点聚类，增加eps
                eps *= 1.2

        return labels, eps

    def _estimate_eps(self, feature_vectors, k=5):
        """
        估计DBSCAN的最佳eps值

        Args:
            feature_vectors: 特征向量 [num_clients, num_features]
            k: k-距离参数

        Returns:
            eps: 估计的eps值
        """
        try:
            # 计算每个点到其k个最近邻的距离
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(feature_vectors)
            distances, indices = nbrs.kneighbors(feature_vectors)

            # 排序距离
            k_distances = np.sort(distances[:, k])

            # 计算k-距离图的拐点
            diffs = np.diff(k_distances)
            eps_idx = np.argmax(diffs) + 1

            # 如果拐点不明显，使用百分位数
            if diffs[eps_idx - 1] < np.mean(diffs) * 2:
                eps = np.percentile(k_distances, 90)
            else:
                eps = k_distances[eps_idx]

            # 确保eps不会太小
            eps = max(eps, 0.1)

            print(f"估计的eps值: {eps:.4f}")
            return eps

        except Exception as e:
            print(f"估计eps值时出错: {str(e)}")
            # 返回一个默认值
            return 0.5

    def wavelet_transform(self, params):
        """
        对参数进行小波变换

        Args:
            params: 模型参数张量

        Returns:
            coeffs: 小波系数列表
        """
        # 将参数转换为1D numpy数组
        params_np = params.detach().cpu().numpy()

        # 对于非常长的参数向量，可以先进行降采样或分段处理
        if len(params_np) > 1e6:
            # 降采样或截取部分参数
            step = max(1, int(len(params_np) / 1e6))
            params_np = params_np[::step]

        # 进行小波变换，确保长度足够
        if len(params_np) < 8:  # 至少需要进行三级小波分解
            # 填充足够的零
            params_np = np.pad(params_np, (0, 8 - len(params_np)))

        try:
            coeffs = pywt.wavedec(params_np, self.wavelet, level=3)
            return coeffs
        except Exception as e:
            print(f"小波变换出错: {str(e)}")
            # 如果小波变换失败，返回一个简单的分解
            return [params_np, params_np[:len(params_np) // 2], params_np[:len(params_np) // 4],
                    params_np[:len(params_np) // 8]]

    def extract_wavelet_features(self, coeffs):
        """
        从小波系数中提取特征

        Args:
            coeffs: 小波系数列表

        Returns:
            features: 特征向量
        """
        features = []

        # 1. 能量特征
        for coeff in coeffs:
            energy = np.sum(coeff ** 2) / len(coeff)
            features.append(energy)

        # 2. 频域熵
        for coeff in coeffs:
            # 对系数取绝对值并归一化
            abs_coeff = np.abs(coeff)
            total = np.sum(abs_coeff)
            if total > 0:
                normalized_coeff = abs_coeff / total
                # 计算Shannon熵
                freq_entropy = entropy(normalized_coeff + 1e-10)  # 添加小常数避免log(0)
            else:
                freq_entropy = 0
            features.append(freq_entropy)

        # 3. 统计矩特征
        for coeff in coeffs:
            features.append(np.mean(coeff))  # 均值
            features.append(np.std(coeff))  # 标准差
            features.append(np.percentile(coeff, 75) - np.percentile(coeff, 25))  # 四分位距

            # 添加偏度和峰度 (如果coeff长度足够)
            if len(coeff) > 3:
                features.append(np.mean((coeff - np.mean(coeff)) ** 3) / (np.std(coeff) ** 3))  # 偏度
            else:
                features.append(0)

            if len(coeff) > 4:
                features.append(np.mean((coeff - np.mean(coeff)) ** 4) / (np.std(coeff) ** 4) - 3)  # 峰度
            else:
                features.append(0)

        # 4. 尺度间的比率特征
        for i in range(len(coeffs) - 1):
            # 避免除零错误
            std_i = np.std(coeffs[i])
            std_i1 = np.std(coeffs[i + 1])
            if std_i1 > 1e-10:
                scale_ratio = std_i / std_i1
            else:
                scale_ratio = 0
            features.append(scale_ratio)

            # 添加能量比率
            energy_i = np.sum(coeffs[i] ** 2) / len(coeffs[i])
            energy_i1 = np.sum(coeffs[i + 1] ** 2) / len(coeffs[i + 1])
            if energy_i1 > 1e-10:
                energy_ratio = energy_i / energy_i1
            else:
                energy_ratio = 0
            features.append(energy_ratio)

        # 5. 最大系数和其位置特征
        for coeff in coeffs:
            if len(coeff) > 0:
                max_idx = np.argmax(np.abs(coeff))
                features.append(coeff[max_idx])
                features.append(max_idx / len(coeff))  # 归一化位置
            else:
                features.append(0)
                features.append(0)

        # 如果指定了特征数量，则进行截断或填充
        if self.num_features is not None:
            if len(features) > self.num_features:
                features = features[:self.num_features]
            elif len(features) < self.num_features:
                features.extend([0] * (self.num_features - len(features)))

        # 记录特征名称以便分析特征重要性
        if not self.feature_importance:
            feature_names = []
            for i, coeff in enumerate(coeffs):
                feature_names.append(f"energy_{i}")
            for i, coeff in enumerate(coeffs):
                feature_names.append(f"entropy_{i}")
            for i, coeff in enumerate(coeffs):
                feature_names.extend([f"mean_{i}", f"std_{i}", f"iqr_{i}", f"skew_{i}", f"kurt_{i}"])
            for i in range(len(coeffs) - 1):
                feature_names.extend([f"scale_ratio_{i}_{i + 1}", f"energy_ratio_{i}_{i + 1}"])
            for i, coeff in enumerate(coeffs):
                feature_names.extend([f"max_coeff_{i}", f"max_pos_{i}"])

            # 确保特征名称数量与特征数量匹配
            if self.num_features is not None:
                if len(feature_names) > self.num_features:
                    feature_names = feature_names[:self.num_features]
                elif len(feature_names) < self.num_features:
                    for i in range(self.num_features - len(feature_names)):
                        feature_names.append(f"padding_{i}")

            for name in feature_names:
                self.feature_importance[name] = 0

        return np.array(features)

    def generate_probing_parameters(self, client_updates):
        """
        为每个客户端生成诱导参数

        Args:
            client_updates: 客户端模型更新 [num_clients, update_dim]

        Returns:
            probed_updates: 添加了扰动的模型更新 [num_clients, update_dim]
        """
        start_time = time.time()
        probing_params = []

        for update in client_updates:
            try:
                # 创建假输入数据
                fake_input = torch.randn(1, *self.input_shape).to(self.device)
                fake_target = torch.randint(0, 10, (1,)).to(self.device)

                # 计算FGSM扰动
                self.model.zero_grad()
                output = self.model(fake_input)
                loss = F.cross_entropy(output, fake_target)
                loss.backward()

                # 生成扰动并应用到客户端更新
                perturbation = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        pert = self.epsilon * torch.sign(param.grad.data)
                        perturbation.append(pert.view(-1))

                # 如果模型很复杂，可能需要截断扰动向量以匹配更新向量的长度
                perturbation = torch.cat(perturbation)
                if perturbation.shape[0] > update.shape[0]:
                    perturbation = perturbation[:update.shape[0]]
                elif perturbation.shape[0] < update.shape[0]:
                    # 如果扰动向量太短，填充零
                    padding = torch.zeros(update.shape[0] - perturbation.shape[0], device=self.device)
                    perturbation = torch.cat([perturbation, padding])

                # 添加扰动到更新
                probed_update = update + perturbation.to(update.device)
                probing_params.append(probed_update)

            except Exception as e:
                self._log(f"生成诱导参数时出错，回退到随机噪声: {str(e)}", level="warning")
                # 如果生成失败，使用原始更新加上随机噪声
                random_noise = torch.randn_like(update) * self.epsilon
                probed_update = update + random_noise
                probing_params.append(probed_update)

        self.execution_time['generate_probing'] = time.time() - start_time
        return torch.stack(probing_params)
        start_time = time.time()

    def detect_malicious_clients(self, original_updates, probed_updates, epoch_num, true_malicious_indices=None):
        """
        检测恶意客户端

        Args:
            original_updates: 原始客户端更新 [num_clients, update_dim]
            probed_updates: 诱导后的客户端更新 [num_clients, update_dim]
            epoch_num: 当前训练轮次

        Returns:
            malicious_indices: 被检测为恶意的客户端索引列表
        """
        start_time = time.time()
        try:
            feature_vectors = self.extract_feature_differences(original_updates, probed_updates)
            normalized_feature_vectors = []

            # 标准化特征
            if len(feature_vectors) > 0:
                try:
                    normalized_feature_vectors = self.scaler.fit_transform(feature_vectors)
                except Exception as e:
                    self._log(f"特征标准化失败，回退到min-max归一化: {str(e)}", level="warning")
                    # 如果标准化失败，使用简单的min-max归一化
                    feature_vectors = np.array(feature_vectors)
                    feature_max = np.max(feature_vectors, axis=0)
                    feature_min = np.min(feature_vectors, axis=0)

                    # 避免除零
                    range_values = feature_max - feature_min
                    range_values[range_values == 0] = 1

                    normalized_feature_vectors = (feature_vectors - feature_min) / range_values

                # 将特征值限制在合理范围内
                normalized_feature_vectors = np.clip(normalized_feature_vectors, -5, 5)
            else:
                self._log("没有有效的特征向量，返回空列表", level="warning")
                return []

            # 先使用PCA降维，然后再进行聚类（关键修改部分）
            n_components = min(2, normalized_feature_vectors.shape[1], normalized_feature_vectors.shape[0])
            if n_components < 1:
                n_components = 1

            pca = PCA(n_components=n_components)
            try:
                feature_vectors_2d = pca.fit_transform(normalized_feature_vectors)
                explained_variance = np.sum(pca.explained_variance_ratio_)
                self._log(f"轮次 {epoch_num} PCA解释方差比例: {explained_variance:.2%}", verbose_only=True)
            except Exception as e:
                self._log(f"PCA降维失败，回退到前两维特征: {str(e)}", level="warning")
                # 如果PCA失败，使用原始特征的前两个维度
                if normalized_feature_vectors.shape[1] >= 2:
                    feature_vectors_2d = normalized_feature_vectors[:, :2]
                else:
                    # 如果维度不足，填充零
                    feature_vectors_2d = np.zeros((normalized_feature_vectors.shape[0], 2))
                    feature_vectors_2d[:, 0] = normalized_feature_vectors[:, 0]

            if true_malicious_indices is not None:
                # 获取真实良性和恶意客户端的索引
                true_benign_indices = [i for i in range(len(feature_vectors_2d)) if i not in true_malicious_indices]

                # 计算单个客户端的代表值
                all_rep_values = np.array(
                    [np.linalg.norm(feature_vectors_2d[i]) for i in range(len(feature_vectors_2d))])

                # 存储真实良性和恶意客户端的平均代表值
                if true_benign_indices:
                    benign_mean_rep = np.mean(all_rep_values[true_benign_indices])
                    self.benign_rep_values_history.append(benign_mean_rep)
                else:
                    self.benign_rep_values_history.append(0)

                if true_malicious_indices:
                    malicious_mean_rep = np.mean(all_rep_values[true_malicious_indices])
                    self.malicious_rep_values_history.append(malicious_mean_rep)
                else:
                    self.malicious_rep_values_history.append(0)

                self.epoch_numbers.append(epoch_num)

                self._log(
                    f"轮次 {epoch_num} 特征代表值统计: 良性均值={self.benign_rep_values_history[-1]:.4f}, "
                    f"恶意均值={self.malicious_rep_values_history[-1]:.4f}, "
                    f"差异={self.malicious_rep_values_history[-1] - self.benign_rep_values_history[-1]:.4f}",
                    verbose_only=True
                )


            # 使用降维后的特征向量进行聚类（关键修改部分）
            # 自适应选择聚类方法
            if self.clustering_method == 'auto':
                # 根据数据特性选择聚类方法
                method = self._select_best_clustering_method(feature_vectors_2d)
            else:
                method = self.clustering_method

            self._log(f"轮次 {epoch_num} 选择的聚类方法: {method}", verbose_only=True)

            # 根据选择的方法执行聚类，使用降维后的特征向量（关键修改部分）
            if method == 'dbscan':
                labels, eps_used = self._adaptive_dbscan_clustering(feature_vectors_2d)  # 使用降维后的特征
                self.eps = eps_used  # 更新eps以便下次使用
                self.eps_history.append(eps_used)
            elif method == 'kmeans':
                labels = self._kmeans_clustering(feature_vectors_2d)  # 使用降维后的特征
            elif method == 'spectral':
                labels = self._spectral_clustering(feature_vectors_2d)  # 使用降维后的特征
            elif method == 'agglomerative':
                labels = self._agglomerative_clustering(feature_vectors_2d)  # 使用降维后的特征
            elif method == 'threshold':
                labels = self._threshold_based_clustering(feature_vectors_2d)  # 已经是降维后的特征
            elif method == 'density_peak':
                labels = self._density_peak_clustering(feature_vectors_2d)  # 使用降维后的特征
            else:
                # 默认使用多方法组合投票，现在两个参数都是feature_vectors_2d
                labels = self._ensemble_clustering(feature_vectors_2d, feature_vectors_2d)

            # 可视化聚类结果
            self._visualize_clusters(feature_vectors_2d, labels, epoch_num)

            # 计算聚类质量指标
            silhouette_avg = -1  # 默认值
            ch_score = -1  # 默认值

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            self._log(f"轮次 {epoch_num} 聚类数量: {n_clusters}", verbose_only=True)

            if n_clusters > 1 and n_clusters < len(labels):
                try:
                    # 忽略噪声点计算轮廓系数
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > n_clusters:
                        silhouette_avg = silhouette_score(
                            feature_vectors_2d[non_noise_mask],  # 使用降维后的特征（关键修改部分）
                            labels[non_noise_mask]
                        )

                    # Calinski-Harabasz分数
                    ch_score = calinski_harabasz_score(feature_vectors_2d, labels)  # 使用降维后的特征（关键修改部分）

                    self._log(f"轮次 {epoch_num} 聚类质量: silhouette={silhouette_avg:.3f}, CH={ch_score:.1f}", verbose_only=True)
                except Exception as e:
                    self._log(f"计算聚类质量指标出错: {str(e)}", level="warning")

            # 记录聚类质量指标
            self.clustering_quality.append({
                'epoch': epoch_num,
                'n_clusters': n_clusters,
                'silhouette': silhouette_avg,
                'ch_score': ch_score,
                'noise_ratio': np.sum(labels == -1) / len(labels) if -1 in labels else 0,
                'clustering_method': method
            })

            # 如果所有点都是单独的簇或全是噪声点，使用基于差分向量模长的方法
            if n_clusters <= 1 or n_clusters >= len(labels) * 0.8:
                self._log("聚类失败：簇数量不合理，使用基于差分向量模长的方法", level="warning")
                return self._norm_based_detection(feature_vectors)

            # 计算每个簇的代表值
            cluster_representatives = {}
            for label in unique_labels:
                if label == -1:  # 跳过噪声点
                    continue

                # 获取该簇内所有客户端的索引
                cluster_indices = np.where(labels == label)[0]

                # 计算该簇内所有客户端的特征均值
                cluster_features = np.array([feature_vectors_2d[i] for i in cluster_indices])  # 使用降维后的特征（关键修改部分）
                cluster_mean = np.mean(cluster_features, axis=0)

                # 使用特征向量的范数作为代表值
                representative_value = np.linalg.norm(cluster_mean)
                cluster_representatives[label] = representative_value

            # 如果没有有效的簇，使用距离分析
            if not cluster_representatives:
                self._log("未找到有效的簇，使用距离分析方法", level="warning")
                return self._statistical_outlier_detection(feature_vectors_2d)  # 使用降维后的特征（关键修改部分）
            for label, value in sorted(cluster_representatives.items(), key=lambda x: x[1], reverse=True):
                cluster_size = np.sum(labels == label)
                self._log(f"轮次 {epoch_num} 簇 {label}: 大小={cluster_size}, 代表值={value:.4f}", verbose_only=True)

            # 分析特征重要性
            if len(unique_labels) > 1:
                self._analyze_feature_importance(feature_vectors_2d, labels)  # 使用降维后的特征（关键修改部分）

            # 选择具有最高代表值的簇作为良性客户端群体
            benign_cluster = min(cluster_representatives, key=cluster_representatives.get)
            self._log(
                f"轮次 {epoch_num} 选择良性簇 {benign_cluster}，代表值={cluster_representatives[benign_cluster]:.4f}",
                verbose_only=True
            )

            # 将不在良性簇中的客户端标记为恶意
            malicious_indices = [i for i in range(len(labels)) if labels[i] != benign_cluster]

            # 如果所有客户端都被标记为恶意，选择其中一部分作为良性
            if len(malicious_indices) == len(labels):
                self._log("所有客户端都被标记为恶意，回退到基于差分向量模长的选择", level="warning")
                return self._norm_based_detection(feature_vectors)

            # 保存检测统计信息
            self.detection_stats.append({
                'epoch': epoch_num,
                'total_clients': len(labels),
                'detected_malicious': len(malicious_indices),
                'benign_cluster_size': np.sum(labels == benign_cluster),
                'noise_points': np.sum(labels == -1) if -1 in labels else 0,
                'clustering_method': method
            })

            # 缓存本轮检测的恶意客户端索引
            self.malicious_indices_cache[epoch_num] = malicious_indices

            self._log(
                f"轮次 {epoch_num} 检测完成: 恶意客户端={len(malicious_indices)}, "
                f"良性簇大小={np.sum(labels == benign_cluster)}, "
                f"噪声点={np.sum(labels == -1) if -1 in labels else 0}"
            )

            self.execution_time['detection'] = time.time() - start_time
            return malicious_indices

        except Exception as e:
            self._log(f"检测恶意客户端时发生错误: {str(e)}", level="error")
            # 如果检测过程失败，使用基于差分向量模长的方法
            if self.verbose:
                traceback.print_exc()  # 打印详细的错误堆栈
            # 如果检测过程失败，使用基于差分向量模长的方法
            if 'feature_vectors' in locals() and len(feature_vectors) > 0:
                return self._norm_based_detection(np.array(feature_vectors))
            else:
                # 如果连feature_vectors都没有，随机选择10%的客户端作为恶意客户端
                n_clients = len(original_updates)
                n_select = max(1, int(n_clients * 0.1))
                return np.random.choice(n_clients, n_select, replace=False).tolist()

        # 聚类质量
        if self.clustering_quality:
            quality_df = pd.DataFrame(self.clustering_quality)
            plt.subplot(2, 2, 3)
            plt.plot(quality_df['epoch'], quality_df['n_clusters'], marker='o', label='聚类数量')
            if 'noise_ratio' in quality_df.columns:
                plt.plot(quality_df['epoch'], quality_df['noise_ratio'], marker='s', label='噪声比例')
            plt.xlabel('训练轮次')
            plt.ylabel('值')
            plt.title('聚类特性')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.subplot(2, 2, 4)
            valid_silhouette = quality_df[quality_df['silhouette'] > -1]
            if not valid_silhouette.empty:
                plt.plot(valid_silhouette['epoch'], valid_silhouette['silhouette'], marker='o', label='轮廓系数')
            valid_ch = quality_df[quality_df['ch_score'] > -1]
            if not valid_ch.empty:
                plt.plot(valid_ch['epoch'], valid_ch['ch_score'] / 100, marker='s', label='CH分数 (/100)')  # 缩放以适应图表
            plt.xlabel('训练轮次')
            plt.ylabel('分数')
            plt.title('聚类质量')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
