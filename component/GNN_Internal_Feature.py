import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gnn_extract_excel_features(filename, graph_method='knn', **kwargs):
    """
    从 Excel 文件提取特征并生成图特征
    :param filename: 输入的 Excel 文件路径
    :param graph_method: 构图方法，'knn', 'similarity', 或 'cluster'
    :param kwargs: 构图方法的额外参数
    :return: 节点索引，三层图特征，标签
    """
    # 读取 Excel 数据
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')
    index = readbook.iloc[:, 0].to_numpy()
    labels = readbook.iloc[:, -1].to_numpy()
    features_df = readbook.iloc[:, 1:-1]
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])

    # One-hot encoding for categorical features
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)

    # Combine numeric and categorical features
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)

    # 构图方式：k-NN
    def create_graph_knn(features, k=6):
        num_nodes = features.shape[0]
        edge_index = []
        for i in range(num_nodes):
            distances = np.linalg.norm(features[i] - features, axis=1)
            nearest_neighbors = np.argsort(distances)[1:k + 1]
            for neighbor in nearest_neighbors:
                edge_index.append([i, neighbor])
                edge_index.append([neighbor, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    # 构图方式：基于特征相似性
    def create_graph_similarity(features, threshold=0.5):
        num_nodes = features.shape[0]
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 防止自环
                    similarity = np.dot(features[i], features[j]) / (
                        np.linalg.norm(features[i]) * np.linalg.norm(features[j])
                    )
                    if similarity > threshold:
                        edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    # 构图方式：基于聚类
    def create_graph_cluster(features, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
        labels = kmeans.labels_
        edge_index = []
        num_nodes = features.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if labels[i] == labels[j] and i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    # 根据选择的构图方法创建图
    if graph_method == 'knn':
        graph_data = create_graph_knn(combined_features, **kwargs)
    elif graph_method == 'similarity':
        graph_data = create_graph_similarity(combined_features, **kwargs)
    elif graph_method == 'cluster':
        graph_data = create_graph_cluster(combined_features, **kwargs)
    else:
        raise ValueError(f"Unsupported graph method: {graph_method}")

    # GCN
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.conv2 = GCNConv(128, 64)
            self.conv3 = GCNConv(64, out_channels)

        def forward(self, x, edge_index):
            layer1 = self.conv1(x, edge_index)
            layer1 = F.relu(layer1)
            layer2 = self.conv2(layer1, edge_index)
            layer2 = F.relu(layer2)
            layer3 = self.conv3(layer2, edge_index)
            return layer1, layer2, layer3
    # GAT
    # class GNN(torch.nn.Module):
    #     def __init__(self, in_channels, out_channels, heads=8):
    #         """
    #         初始化 GNN 模型，基于 GATConv。
    #         :param in_channels: 输入特征的维度
    #         :param out_channels: 输出特征的维度
    #         :param heads: GATConv 的注意力头数量（默认为 8）
    #         """
    #         super(GNN, self).__init__()
    #         # 第一层 GAT，输入特征维度 -> 128，注意力头数为 heads
    #         self.conv1 = GATConv(in_channels, 128 // heads, heads=heads, concat=True)
    #         # 第二层 GAT，128 -> 64，注意力头数为 heads
    #         self.conv2 = GATConv(128, 64 // heads, heads=heads, concat=True)
    #         # 第三层 GAT，64 -> 输出特征维度，注意力头数为 1（不拼接）
    #         self.conv3 = GATConv(64, out_channels, heads=1, concat=False)
    #
    #     def forward(self, x, edge_index):
    #         """
    #         前向传播函数
    #         :param x: 节点特征
    #         :param edge_index: 图的边索引
    #         :return: 每一层的特征表示（layer1, layer2, layer3）
    #         """
    #         # 第一层 GAT
    #         layer1 = self.conv1(x, edge_index)
    #         layer1 = F.relu(layer1)
    #
    #         # 第二层 GAT
    #         layer2 = self.conv2(layer1, edge_index)
    #         layer2 = F.relu(layer2)
    #
    #         # 第三层 GAT
    #         layer3 = self.conv3(layer2, edge_index)
    #
    #         return layer1, layer2, layer3

    # 模型初始化
    model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
    data = graph_data.to(device)

    # 模型评估并提取特征
    model.eval()
    with torch.no_grad():
        layer1, layer2, layer3 = model(data.x, data.edge_index)

    # 返回特征
    layer1_features = layer1.cpu().numpy()
    layer2_features = layer2.cpu().numpy()
    layer3_features = layer3.cpu().numpy()

    return index, layer1_features, layer2_features, layer3_features, labels

