# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def gnn_extract_excel_features(filename):
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#     index = readbook.iloc[:, 0].to_numpy()
#     labels = readbook.iloc[:, -1].to_numpy()
#     features_df = readbook.iloc[:, 1:-1]
#     numeric_features = features_df.select_dtypes(include=[np.number])
#     categorical_features = features_df.select_dtypes(exclude=[np.number])
#
#     # One-hot encoding for categorical features
#     if not categorical_features.empty:
#         categorical_features = pd.get_dummies(categorical_features)
#
#     # Combine numeric and categorical features
#     combined_features = pd.concat([numeric_features, categorical_features], axis=1)
#     combined_features = combined_features.to_numpy(dtype=np.float32)
#
#     # Create graph from features (using k nearest neighbors)
#     def create_graph_from_features(features, k=6):
#         num_nodes = features.shape[0]
#         edge_index = []
#         for i in range(num_nodes):
#             distances = np.linalg.norm(features[i] - features, axis=1)
#             nearest_neighbors = np.argsort(distances)[1:k + 1]
#             for neighbor in nearest_neighbors:
#                 edge_index.append([i, neighbor])
#                 edge_index.append([neighbor, i])
#
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         x = torch.tensor(features, dtype=torch.float)
#         return Data(x=x, edge_index=edge_index)
#
#     # Define GNN model with three GCN layers
#     class GNN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(GNN, self).__init__()
#             self.conv1 = GCNConv(in_channels, 128)
#             self.conv2 = GCNConv(128, 64)
#             self.conv3 = GCNConv(64, out_channels)
#
#         def forward(self, x, edge_index):
#             layer1 = self.conv1(x, edge_index)
#             layer1 = F.relu(layer1)
#             layer2 = self.conv2(layer1, edge_index)
#             layer2 = F.relu(layer2)
#             layer3 = self.conv3(layer2, edge_index)
#             return layer1, layer2, layer3
#
#     # Create the graph data from features
#     graph_data = create_graph_from_features(combined_features)
#     model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
#     data = graph_data.to(device)
#
#     # Model evaluation to extract features from each layer
#     model.eval()
#     with torch.no_grad():
#         layer1, layer2, layer3 = model(data.x, data.edge_index)
#
#     # Move features back to CPU and convert to numpy arrays
#     layer1_features = layer1.cpu().numpy()
#     layer2_features = layer2.cpu().numpy()
#     layer3_features = layer3.cpu().numpy()
#
#     # Return index and features for all three layers, along with labels
#     return index, layer1_features, layer2_features, layer3_features, labels
#
#
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, GATConv
# from sklearn.cluster import KMeans
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def gnn_extract_excel_features(filename, graph_method='knn', **kwargs):
#     """
#     从 Excel 文件提取特征并生成图特征
#     :param filename: 输入的 Excel 文件路径
#     :param graph_method: 构图方法，'knn', 'similarity', 或 'cluster'
#     :param kwargs: 构图方法的额外参数
#     :return: 节点索引，三层图特征，标签
#     """
#     # 读取 Excel 数据
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#     index = readbook.iloc[:, 0].to_numpy()
#     labels = readbook.iloc[:, -1].to_numpy()
#     features_df = readbook.iloc[:, 1:-1]
#     numeric_features = features_df.select_dtypes(include=[np.number])
#     categorical_features = features_df.select_dtypes(exclude=[np.number])
#
#     # One-hot encoding for categorical features
#     if not categorical_features.empty:
#         categorical_features = pd.get_dummies(categorical_features)
#
#     # Combine numeric and categorical features
#     combined_features = pd.concat([numeric_features, categorical_features], axis=1)
#     combined_features = combined_features.to_numpy(dtype=np.float32)
#
#     # 构图方式：k-NN
#     def create_graph_knn(features, k=6):
#         num_nodes = features.shape[0]
#         edge_index = []
#         for i in range(num_nodes):
#             distances = np.linalg.norm(features[i] - features, axis=1)
#             nearest_neighbors = np.argsort(distances)[1:k + 1]
#             for neighbor in nearest_neighbors:
#                 edge_index.append([i, neighbor])
#                 edge_index.append([neighbor, i])
#
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         x = torch.tensor(features, dtype=torch.float)
#         return Data(x=x, edge_index=edge_index)
#
#     # 构图方式：基于特征相似性
#     def create_graph_similarity(features, threshold=0.5):
#         num_nodes = features.shape[0]
#         edge_index = []
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if i != j:  # 防止自环
#                     similarity = np.dot(features[i], features[j]) / (
#                         np.linalg.norm(features[i]) * np.linalg.norm(features[j])
#                     )
#                     if similarity > threshold:
#                         edge_index.append([i, j])
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         x = torch.tensor(features, dtype=torch.float)
#         return Data(x=x, edge_index=edge_index)
#
#     # 构图方式：基于聚类
#     def create_graph_cluster(features, n_clusters=5):
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
#         labels = kmeans.labels_
#         edge_index = []
#         num_nodes = features.shape[0]
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if labels[i] == labels[j] and i != j:
#                     edge_index.append([i, j])
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         x = torch.tensor(features, dtype=torch.float)
#         return Data(x=x, edge_index=edge_index)
#
#     # 根据选择的构图方法创建图
#     if graph_method == 'knn':
#         graph_data = create_graph_knn(combined_features, **kwargs)
#     elif graph_method == 'similarity':
#         graph_data = create_graph_similarity(combined_features, **kwargs)
#     elif graph_method == 'cluster':
#         graph_data = create_graph_cluster(combined_features, **kwargs)
#     else:
#         raise ValueError(f"Unsupported graph method: {graph_method}")
#
#     # GCN
#     class GNN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(GNN, self).__init__()
#             self.conv1 = GCNConv(in_channels, 128)
#             self.conv2 = GCNConv(128, 64)
#             self.conv3 = GCNConv(64, out_channels)
#
#         def forward(self, x, edge_index):
#             layer1 = self.conv1(x, edge_index)
#             layer1 = F.relu(layer1)
#             layer2 = self.conv2(layer1, edge_index)
#             layer2 = F.relu(layer2)
#             layer3 = self.conv3(layer2, edge_index)
#             return layer1, layer2, layer3
#     # GAT
#     # class GNN(torch.nn.Module):
#     #     def __init__(self, in_channels, out_channels, heads=8):
#     #         """
#     #         初始化 GNN 模型，基于 GATConv。
#     #         :param in_channels: 输入特征的维度
#     #         :param out_channels: 输出特征的维度
#     #         :param heads: GATConv 的注意力头数量（默认为 8）
#     #         """
#     #         super(GNN, self).__init__()
#     #         # 第一层 GAT，输入特征维度 -> 128，注意力头数为 heads
#     #         self.conv1 = GATConv(in_channels, 128 // heads, heads=heads, concat=True)
#     #         # 第二层 GAT，128 -> 64，注意力头数为 heads
#     #         self.conv2 = GATConv(128, 64 // heads, heads=heads, concat=True)
#     #         # 第三层 GAT，64 -> 输出特征维度，注意力头数为 1（不拼接）
#     #         self.conv3 = GATConv(64, out_channels, heads=1, concat=False)
#     #
#     #     def forward(self, x, edge_index):
#     #         """
#     #         前向传播函数
#     #         :param x: 节点特征
#     #         :param edge_index: 图的边索引
#     #         :return: 每一层的特征表示（layer1, layer2, layer3）
#     #         """
#     #         # 第一层 GAT
#     #         layer1 = self.conv1(x, edge_index)
#     #         layer1 = F.relu(layer1)
#     #
#     #         # 第二层 GAT
#     #         layer2 = self.conv2(layer1, edge_index)
#     #         layer2 = F.relu(layer2)
#     #
#     #         # 第三层 GAT
#     #         layer3 = self.conv3(layer2, edge_index)
#     #
#     #         return layer1, layer2, layer3
#
#     # 模型初始化
#     model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
#     data = graph_data.to(device)
#
#     # 模型评估并提取特征
#     model.eval()
#     with torch.no_grad():
#         layer1, layer2, layer3 = model(data.x, data.edge_index)
#
#     # 返回特征
#     layer1_features = layer1.cpu().numpy()
#     layer2_features = layer2.cpu().numpy()
#     layer3_features = layer3.cpu().numpy()
#
#     return index, layer1_features, layer2_features, layer3_features, labels

# ##########节点全连接################
#
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GCNConv
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def add_self_loops(edge_index, num_nodes):
#     """
#     添加自环边（每个节点都和自己有一条边连接）
#     """
#     loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
#     edge_index = torch.cat([edge_index, loop_index], dim=1)
#     return edge_index
#
#
# def gnn_extract_excel_features(filename):
#     # 读取Excel数据
#     readbook = pd.read_excel(f'{filename}', engine='openpyxl')
#
#     # 患者 ID 和标签
#     index = readbook.iloc[:, 0].to_numpy()  # 患者ID
#     labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）
#
#     # 临床病理特征（去掉患者ID和标签）
#     features_df = readbook.iloc[:, 1:-1]
#
#     # 打印列名和特征数量
#     print(f"Feature columns: {features_df.columns}")
#     print(f"Number of features: {features_df.shape[1]}")
#
#     # 分析数值特征和类别特征
#     numeric_features = features_df.select_dtypes(include=[np.number])
#     categorical_features = features_df.select_dtypes(exclude=[np.number])
#
#     # 输出数值特征和类别特征的数量
#     print(f"Number of numeric features: {numeric_features.shape[1]}")
#     print(f"Number of categorical features: {categorical_features.shape[1]}")
#
#     # 对类别特征进行独热编码
#     if not categorical_features.empty:
#         categorical_features = pd.get_dummies(categorical_features)
#         print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")
#
#     # 合并数值特征和类别特征
#     combined_features = pd.concat([numeric_features, categorical_features], axis=1)
#     combined_features = combined_features.to_numpy(dtype=np.float32)
#     # 打印最终合并特征的数量
#     print(
#         f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")
#
#     # 构建图数据：每个患者为一个图，每个临床特征为一个节点
#     def create_graph_from_features(features):
#         num_patients = features.shape[0]  # 患者数量
#         num_features = features.shape[1]  # 每个患者的临床特征数量（每列一个特征）
#
#         edge_index_list = []  # 存储所有患者的边信息
#         node_features_list = []  # 存储所有患者的节点特征
#         batch_list = []  # 存储患者的批处理信息
#
#         for patient_idx in range(num_patients):
#             patient_features = features[patient_idx]  # 当前患者的特征
#             node_features_list.append(patient_features)
#
#             # 构建边：基于特征间的关系
#             edge_index = []
#             for i in range(num_features):
#                 for j in range(num_features):
#                     if i != j:  # 不创建自环，稍后会统一添加
#                         edge_index.append([i, j])
#
#             # 转换为 PyTorch 张量并添加自环
#             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#             edge_index = add_self_loops(edge_index, num_features)
#             edge_index_list.append(edge_index)
#
#             # 为每个患者设置一个独立的批处理 ID
#             batch_list.extend([patient_idx] * num_features)
#
#         # 合并所有患者的节点特征和边
#         all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
#         all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)
#
#         all_edge_indices = torch.cat(edge_index_list, dim=1)
#         batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息
#
#         # 使用 Batch 来将图数据合并，保持不同患者的独立性
#         graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)
#
#         return graph_data
#
#     # 创建图数据
#     graph_data = create_graph_from_features(combined_features)
#
#     # 检查 x 的维度
#     print(f"x shape: {graph_data.x.shape}")
#     print(f"edge_index shape: {graph_data.edge_index.shape}")
#
#     # 定义图神经网络模型
#     class GNN(torch.nn.Module):
#         def __init__(self, in_channels, out_channels):
#             super(GNN, self).__init__()
#             self.conv1 = GCNConv(in_channels, 128)
#             self.conv2 = GCNConv(128, 64)
#             self.conv3 = GCNConv(64, out_channels)
#
#         def forward(self, x, edge_index, batch):
#             # 第一层卷积
#             layer1 = self.conv1(x, edge_index)
#             layer1 = F.relu(layer1)
#
#             # 第二层卷积
#             layer2 = self.conv2(layer1, edge_index)
#             layer2 = F.relu(layer2)
#
#             # 第三层卷积
#             layer3 = self.conv3(layer2, edge_index)
#             return layer1, layer2, layer3  # 返回每一层的特征
#
#
#     # 模型初始化
#     model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
#     data = graph_data.to(device)
#
#     # 使用图神经网络进行节点特征聚合
#     model.eval()
#     with torch.no_grad():
#         layer1_features, layer2_features, layer3_features = model(data.x, data.edge_index, data.batch)
#
#     # 返回特征
#     layer1_features = layer1_features.cpu().numpy()
#     layer2_features = layer2_features.cpu().numpy()
#     layer3_features = layer3_features.cpu().numpy()
#
#     return index, layer1_features, layer2_features, layer3_features, labels

#############聚类构图###################
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_self_loops(edge_index, num_nodes):
    """
    添加自环边（每个节点都和自己有一条边连接）
    """
    loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index

def gnn_extract_excel_features(filename, n_clusters=5):
    # 读取Excel数据
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')

    # 患者 ID 和标签
    index = readbook.iloc[:, 0].to_numpy()  # 患者ID
    labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）

    # 临床病理特征（去掉患者ID和标签）
    features_df = readbook.iloc[:, 1:-1]

    # 打印列名和特征数量
    print(f"Feature columns: {features_df.columns}")
    print(f"Number of features: {features_df.shape[1]}")

    # 分析数值特征和类别特征
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])

    # 输出数值特征和类别特征的数量
    print(f"Number of numeric features: {numeric_features.shape[1]}")
    print(f"Number of categorical features: {categorical_features.shape[1]}")

    # 对类别特征进行独热编码
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
        print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")

    # 合并数值特征和类别特征
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)
    # 打印最终合并特征的数量
    print(
        f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")

    # 使用 K-means 聚类对每个患者的特征进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(combined_features)  # 每个患者的簇标签
    print(f"Cluster labels: {cluster_labels}")

    def create_graph_from_features(features, cluster_labels):
        num_patients = features.shape[0]  # 患者数量
        num_features = features.shape[1]  # 每个患者的临床特征数量（每列一个特征）

        edge_index_list = []  # 存储所有患者的边信息
        node_features_list = []  # 存储所有患者的节点特征
        batch_list = []  # 存储患者的批处理信息

        for patient_idx in range(num_patients):
            patient_features = features[patient_idx]  # 当前患者的特征
            node_features_list.append(patient_features)

            # 当前患者的特征节点所属的簇标签
            patient_cluster = cluster_labels[patient_idx]

            # 查找属于同一簇的所有特征节点索引
            cluster_nodes = np.where(cluster_labels == patient_cluster)[0]  # 所有属于同一簇的特征节点索引

            edge_index = []
            for i in range(num_features):
                for j in range(num_features):
                    if i != j:  # 不创建自环，稍后会统一添加
                        edge_index.append([i, j])

            # 转换为 PyTorch 张量并添加自环
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_index = add_self_loops(edge_index, num_features)
            edge_index_list.append(edge_index)

            # 为每个患者设置一个独立的批处理 ID
            batch_list.extend([patient_idx] * num_features)

        # 合并所有患者的节点特征和边
        all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
        all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)

        all_edge_indices = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息

        # 使用 Batch 来将图数据合并，保持不同患者的独立性
        graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)

        return graph_data


    # 创建图数据
    graph_data = create_graph_from_features(combined_features, cluster_labels)

    # 检查 x 的维度
    print(f"x shape: {graph_data.x.shape}")
    print(f"edge_index shape: {graph_data.edge_index.shape}")

    # 定义图神经网络模型
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.conv2 = GCNConv(128, 64)
            self.conv3 = GCNConv(64, out_channels)

        def forward(self, x, edge_index, batch):
            # 第一层卷积
            layer1 = self.conv1(x, edge_index)
            layer1 = F.relu(layer1)

            # 第二层卷积
            layer2 = self.conv2(layer1, edge_index)
            layer2 = F.relu(layer2)

            # 第三层卷积
            layer3 = self.conv3(layer2, edge_index)
            return layer1, layer2, layer3  # 返回每一层的特征

    # 模型初始化
    model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
    data = graph_data.to(device)

    # 使用图神经网络进行节点特征聚合
    model.eval()
    with torch.no_grad():
        layer1_features, layer2_features, layer3_features = model(data.x, data.edge_index, data.batch)

    # 返回特征
    layer1_features = layer1_features.cpu().numpy()
    layer2_features = layer2_features.cpu().numpy()
    layer3_features = layer3_features.cpu().numpy()

    return index, layer1_features, layer2_features, layer3_features, labels
