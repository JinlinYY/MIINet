# import numpy as np
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def combine_features(image_layer1, image_layer2, image_layer3, gnn_layer1, gnn_layer2, gnn_layer3, tabular_features):
#     """
#     将ViT的三层特征与GCN的三层特征分别融合，
#     然后将每一层融合后的特征与没有构图的临床特征融合，最后再将三层融合特征进行最终融合。
#     """

#     # Step 1: 分别融合ViT和GCN的三层特征
#     combined_layer1 = torch.cat((image_layer1, gnn_layer1), dim=1)
#     combined_layer2 = torch.cat((image_layer2, gnn_layer2), dim=1)
#     combined_layer3 = torch.cat((image_layer3, gnn_layer3), dim=1)

#     # Step 2: 分别将融合后的每一层与临床特征进行融合
#     fused_layer1 = torch.cat((combined_layer1, tabular_features), dim=1)
#     fused_layer2 = torch.cat((combined_layer2, tabular_features), dim=1)
#     fused_layer3 = torch.cat((combined_layer3, tabular_features), dim=1)

#     # Step 3: 最终将三层的融合特征进行拼接
#     final_combined_features = torch.cat((fused_layer1, fused_layer2, fused_layer3), dim=1)

#     # Step 4: 将最终的融合特征转换为numpy数组
#     final_combined_features_np = final_combined_features.detach().cpu().numpy()

#     return final_combined_features_np


# import numpy as np
# import torch
# from sklearn.decomposition import PCA

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def apply_pca(features, n_components):
#     """
#     对特征进行PCA降维
#     """
#     pca = PCA(n_components=n_components)
#     return pca.fit_transform(features)

# def combine_features(image_layer1, image_layer2, image_layer3, gnn_layer1, gnn_layer2, gnn_layer3, tabular_features):
#     """
#     将ViT的三层特征与GCN的三层特征分别融合，
#     然后将每一层融合后的特征与没有构图的临床特征融合，最后再将三层融合特征进行最终融合。
#     """

#     # Step 1: 分别融合ViT和GCN的三层特征（先降维再拼接）
#     combined_layer1 = torch.cat((torch.tensor(apply_pca(image_layer1.cpu().numpy(), n_components=30)),
#                                   torch.tensor(apply_pca(gnn_layer1.cpu().numpy(), n_components=30))), dim=1)

#     combined_layer2 = torch.cat((torch.tensor(apply_pca(image_layer2.cpu().numpy(), n_components=20)),
#                                   torch.tensor(apply_pca(gnn_layer2.cpu().numpy(), n_components=20))), dim=1)

#     combined_layer3 = torch.cat((torch.tensor(apply_pca(image_layer3.cpu().numpy(), n_components=10)),
#                                   torch.tensor(apply_pca(gnn_layer3.cpu().numpy(), n_components=10))), dim=1)

#     # Step 2: 分别将融合后的每一层与临床特征进行融合（不对临床特征降维）
#     fused_layer1 = torch.cat((combined_layer1, tabular_features), dim=1)
#     fused_layer2 = torch.cat((combined_layer2, tabular_features), dim=1)
#     fused_layer3 = torch.cat((combined_layer3, tabular_features), dim=1)

#     # Step 3: 最终将三层的融合特征进行拼接
#     final_combined_features = torch.cat((fused_layer1, fused_layer2, fused_layer3), dim=1)

#     # Step 4: 将最终的融合特征转换为numpy数组
#     final_combined_features_np = final_combined_features.detach().cpu().numpy()

#     return final_combined_features_np

################################################################################################################################
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
def apply_pca(features, n_components):
    """
    对特征进行PCA降维
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

class AttentionWeighting(torch.nn.Module):
    def __init__(self, feature_dim):
        super(AttentionWeighting, self).__init__()
        # 定义线性变换
        self.query = torch.nn.Linear(feature_dim, feature_dim)
        self.key = torch.nn.Linear(feature_dim, feature_dim)
        self.value = torch.nn.Linear(feature_dim, feature_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, *modalities):
        # 将模态特征拼接
        combined = torch.stack(modalities, dim=1)  # [batch_size, num_modalities, feature_dim]

        # 计算注意力分数
        query = self.query(combined)  # [batch_size, num_modalities, feature_dim]
        key = self.key(combined)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)  # [batch_size, num_modalities, num_modalities]

        # 加权特征
        weighted_features = torch.matmul(attention_weights, combined)
        return weighted_features.sum(dim=1), attention_weights

def combine_features_attention_weighting(image_layer1, image_layer2, image_layer3, gnn_layer1, gnn_layer2, gnn_layer3, tabular_features):
    tabular_features = tabular_features.to(device)

    # PCA降维并归一化
    def pca_and_normalize(feature, n_components):
        pca_result = apply_pca(feature.cpu().numpy(), n_components)
        return F.normalize(torch.tensor(pca_result).to(device), dim=1)

    pca_image_layer1 = pca_and_normalize(image_layer1, 30)
    pca_gnn_layer1 = pca_and_normalize(gnn_layer1, 30)

    pca_image_layer2 = pca_and_normalize(image_layer2, 20)
    pca_gnn_layer2 = pca_and_normalize(gnn_layer2, 20)

    pca_image_layer3 = pca_and_normalize(image_layer3, 8)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 8)

    # 对齐tabular_features维度
    # 统一对齐所有特征到相同维度
    align_dim = 30  # 设置统一的特征维度
    align_layer1 = torch.nn.Linear(pca_image_layer1.size(1), align_dim).to(device)
    align_layer2 = torch.nn.Linear(pca_image_layer2.size(1), align_dim).to(device)
    align_layer3 = torch.nn.Linear(pca_image_layer3.size(1), align_dim).to(device)

    # 对齐每个模态特征
    pca_image_layer1 = align_layer1(pca_image_layer1)
    pca_gnn_layer1 = align_layer1(pca_gnn_layer1)

    pca_image_layer2 = align_layer2(pca_image_layer2)
    pca_gnn_layer2 = align_layer2(pca_gnn_layer2)

    pca_image_layer3 = align_layer3(pca_image_layer3)
    pca_gnn_layer3 = align_layer3(pca_gnn_layer3)

    # 如果tabular_features维度不同，也对其对齐
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)
    tabular_features_aligned = tabular_align_layer(tabular_features)

    # 使用AttentionWeighting逐层融合
    attention_weighting_layer1 = AttentionWeighting(feature_dim=pca_image_layer1.size(1)).to(device)
    attention_weighting_layer2 = AttentionWeighting(feature_dim=pca_image_layer2.size(1)).to(device)
    attention_weighting_layer3 = AttentionWeighting(feature_dim=pca_image_layer3.size(1)).to(device)

    fused_layer1, weights1 = attention_weighting_layer1(
        pca_image_layer1, pca_gnn_layer1, tabular_features_aligned
    )
    fused_layer2, weights2 = attention_weighting_layer2(
        pca_image_layer2, pca_gnn_layer2, tabular_features_aligned
    )
    fused_layer3, weights3 = attention_weighting_layer3(
        pca_image_layer3, pca_gnn_layer3, tabular_features_aligned
    )

    # 拼接融合后的三层特征
    fused_features = torch.cat((fused_layer1, fused_layer2, fused_layer3, tabular_features), dim=1)

    return fused_features, (weights1, weights2, weights3)


def combine_features_attention_weighting_channel_shuffle(image_layer1, image_layer2, image_layer3, gnn_layer1, gnn_layer2, gnn_layer3, tabular_features):
    tabular_features = tabular_features.to(device)

    # PCA降维并归一化
    def pca_and_normalize(feature, n_components):
        pca_result = apply_pca(feature.cpu().numpy(), n_components)
        return F.normalize(torch.tensor(pca_result).to(device), dim=1)

    pca_image_layer1 = pca_and_normalize(image_layer1, 30)
    pca_gnn_layer1 = pca_and_normalize(gnn_layer1, 30)

    pca_image_layer2 = pca_and_normalize(image_layer2, 20)
    pca_gnn_layer2 = pca_and_normalize(gnn_layer2, 20)

    pca_image_layer3 = pca_and_normalize(image_layer3, 6)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 6)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_layer1 = torch.nn.Linear(pca_image_layer1.size(1), align_dim).to(device)
    align_layer2 = torch.nn.Linear(pca_image_layer2.size(1), align_dim).to(device)
    align_layer3 = torch.nn.Linear(pca_image_layer3.size(1), align_dim).to(device)

    # 对齐每个模态特征
    pca_image_layer1 = align_layer1(pca_image_layer1)
    pca_gnn_layer1 = align_layer1(pca_gnn_layer1)

    pca_image_layer2 = align_layer2(pca_image_layer2)
    pca_gnn_layer2 = align_layer2(pca_gnn_layer2)

    pca_image_layer3 = align_layer3(pca_image_layer3)
    pca_gnn_layer3 = align_layer3(pca_gnn_layer3)

    # 如果tabular_features维度不同，也对其对齐
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)
    tabular_features_aligned = tabular_align_layer(tabular_features)

    # 使用AttentionWeighting逐层融合
    attention_weighting_layer1 = AttentionWeighting(feature_dim=pca_image_layer1.size(1)).to(device)
    attention_weighting_layer2 = AttentionWeighting(feature_dim=pca_image_layer2.size(1)).to(device)
    attention_weighting_layer3 = AttentionWeighting(feature_dim=pca_image_layer3.size(1)).to(device)

    fused_layer1, weights1 = attention_weighting_layer1(
        pca_image_layer1, pca_gnn_layer1, tabular_features_aligned
    )
    fused_layer2, weights2 = attention_weighting_layer2(
        pca_image_layer2, pca_gnn_layer2, tabular_features_aligned
    )
    fused_layer3, weights3 = attention_weighting_layer3(
        pca_image_layer3, pca_gnn_layer3, tabular_features_aligned
    )

    # 拼接融合后的三层特征
    fused_features = torch.cat((fused_layer1, fused_layer2, fused_layer3, tabular_features), dim=1)

    # 通道洗牌操作
    def channel_shuffle(features, groups):
        batch_size, num_channels = features.size()
        assert num_channels % groups == 0, "The number of channels must be divisible by the number of groups."
        channels_per_group = num_channels // groups

        # Reshape into (batch_size, groups, channels_per_group)
        features = features.view(batch_size, groups, channels_per_group)

        # Transpose to (batch_size, channels_per_group, groups)
        features = features.transpose(1, 2)

        # Flatten back to (batch_size, num_channels)
        return features.contiguous().view(batch_size, -1)

    # 动态调整分组数
    shuffle_groups = 5
    while fused_features.size(1) % shuffle_groups != 0 and shuffle_groups > 1:
        shuffle_groups -= 1

    if shuffle_groups > 1:
        fused_features = channel_shuffle(fused_features, shuffle_groups)
    else:
        print("Warning: Channel shuffle skipped as no valid groups found.")

    return fused_features, (weights1, weights2, weights3)


def combine_features_direct_cat(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    tabular_features = tabular_features.to(device)

    # PCA降维并归一化
    def pca_and_normalize(feature, n_components):
        pca_result = apply_pca(feature.cpu().numpy(), n_components)
        return F.normalize(torch.tensor(pca_result).to(device), dim=1)

    pca_image_layer1 = pca_and_normalize(image_layer1, 30)
    pca_gnn_layer1 = pca_and_normalize(gnn_layer1, 30)

    pca_image_layer2 = pca_and_normalize(image_layer2, 20)
    pca_gnn_layer2 = pca_and_normalize(gnn_layer2, 20)

    pca_image_layer3 = pca_and_normalize(image_layer3, 10)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 10)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_layer1 = torch.nn.Linear(pca_image_layer1.size(1), align_dim).to(device)
    align_layer2 = torch.nn.Linear(pca_image_layer2.size(1), align_dim).to(device)
    align_layer3 = torch.nn.Linear(pca_image_layer3.size(1), align_dim).to(device)

    # 对齐每个模态特征
    pca_image_layer1 = align_layer1(pca_image_layer1)
    pca_gnn_layer1 = align_layer1(pca_gnn_layer1)

    pca_image_layer2 = align_layer2(pca_image_layer2)
    pca_gnn_layer2 = align_layer2(pca_gnn_layer2)

    pca_image_layer3 = align_layer3(pca_image_layer3)
    pca_gnn_layer3 = align_layer3(pca_gnn_layer3)

    # 如果tabular_features维度不同，也对其对齐
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)
    tabular_features_aligned = tabular_align_layer(tabular_features)

    # 融合操作使用直接cat
    fused_layer1 = torch.cat((pca_image_layer1, pca_gnn_layer1, tabular_features_aligned), dim=1)
    fused_layer2 = torch.cat((pca_image_layer2, pca_gnn_layer2, tabular_features_aligned), dim=1)
    fused_layer3 = torch.cat((pca_image_layer3, pca_gnn_layer3, tabular_features_aligned), dim=1)

    # 拼接三层的融合特征
    fused_features = torch.cat((fused_layer1, fused_layer2, fused_layer3, tabular_features), dim=1)



    # 返回直接cat后的特征
    return fused_features, None


def combine_features_direct_cat_channel_shuffle(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    tabular_features = tabular_features.to(device)

    # PCA降维并归一化
    def pca_and_normalize(feature, n_components):
        pca_result = apply_pca(feature.cpu().numpy(), n_components)
        return F.normalize(torch.tensor(pca_result).to(device), dim=1)

    pca_image_layer1 = pca_and_normalize(image_layer1, 30)
    pca_gnn_layer1 = pca_and_normalize(gnn_layer1, 30)

    pca_image_layer2 = pca_and_normalize(image_layer2, 20)
    pca_gnn_layer2 = pca_and_normalize(gnn_layer2, 20)

    pca_image_layer3 = pca_and_normalize(image_layer3, 10)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 10)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_layer1 = torch.nn.Linear(pca_image_layer1.size(1), align_dim).to(device)
    align_layer2 = torch.nn.Linear(pca_image_layer2.size(1), align_dim).to(device)
    align_layer3 = torch.nn.Linear(pca_image_layer3.size(1), align_dim).to(device)

    # 对齐每个模态特征
    pca_image_layer1 = align_layer1(pca_image_layer1)
    pca_gnn_layer1 = align_layer1(pca_gnn_layer1)

    pca_image_layer2 = align_layer2(pca_image_layer2)
    pca_gnn_layer2 = align_layer2(pca_gnn_layer2)

    pca_image_layer3 = align_layer3(pca_image_layer3)
    pca_gnn_layer3 = align_layer3(pca_gnn_layer3)

    # 如果tabular_features维度不同，也对其对齐
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)
    tabular_features_aligned = tabular_align_layer(tabular_features)

    # 融合操作使用直接cat
    fused_layer1 = torch.cat((pca_image_layer1, pca_gnn_layer1, tabular_features_aligned), dim=1)
    fused_layer2 = torch.cat((pca_image_layer2, pca_gnn_layer2, tabular_features_aligned), dim=1)
    fused_layer3 = torch.cat((pca_image_layer3, pca_gnn_layer3, tabular_features_aligned), dim=1)

    # 拼接三层的融合特征
    fused_features = torch.cat((fused_layer1, fused_layer2, fused_layer3, tabular_features), dim=1)

    # 通道洗牌操作
    def channel_shuffle(features, groups):
        batch_size, num_channels = features.size()
        assert num_channels % groups == 0, "The number of channels must be divisible by the number of groups."
        channels_per_group = num_channels // groups

        # Reshape into (batch_size, groups, channels_per_group)
        features = features.view(batch_size, groups, channels_per_group)

        # Transpose to (batch_size, channels_per_group, groups)
        features = features.transpose(1, 2)

        # Flatten back to (batch_size, num_channels)
        return features.contiguous().view(batch_size, -1)

    # 动态调整分组数
    shuffle_groups = 5
    while fused_features.size(1) % shuffle_groups != 0 and shuffle_groups > 1:
        shuffle_groups -= 1

    if shuffle_groups > 1:
        fused_features = channel_shuffle(fused_features, shuffle_groups)
    else:
        print("Warning: Channel shuffle skipped as no valid groups found.")

    # 返回直接cat后的特征
    return fused_features, None


def combine_features_1_cat_channel_shuffle(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    tabular_features = tabular_features.to(device)

    # PCA降维并归一化
    def pca_and_normalize(feature, n_components):
        pca_result = apply_pca(feature.cpu().numpy(), n_components)
        return F.normalize(torch.tensor(pca_result).to(device), dim=1)

    pca_image_layer1 = pca_and_normalize(image_layer1, 30)
    pca_gnn_layer1 = pca_and_normalize(gnn_layer1, 30)

    pca_image_layer2 = pca_and_normalize(image_layer2, 20)
    pca_gnn_layer2 = pca_and_normalize(gnn_layer2, 20)

    pca_image_layer3 = pca_and_normalize(image_layer3, 10)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 10)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_layer1 = torch.nn.Linear(pca_image_layer1.size(1), align_dim).to(device)
    align_layer2 = torch.nn.Linear(pca_image_layer2.size(1), align_dim).to(device)
    align_layer3 = torch.nn.Linear(pca_image_layer3.size(1), align_dim).to(device)

    # 对齐每个模态特征
    pca_image_layer1 = align_layer1(pca_image_layer1)
    pca_gnn_layer1 = align_layer1(pca_gnn_layer1)

    pca_image_layer2 = align_layer2(pca_image_layer2)
    pca_gnn_layer2 = align_layer2(pca_gnn_layer2)

    pca_image_layer3 = align_layer3(pca_image_layer3)
    pca_gnn_layer3 = align_layer3(pca_gnn_layer3)

    # 如果tabular_features维度不同，也对其对齐
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)
    tabular_features_aligned = tabular_align_layer(tabular_features)

    # 融合操作使用直接cat
    fused_layer1 = torch.cat((pca_image_layer1, pca_gnn_layer1, tabular_features_aligned), dim=1)
    fused_layer2 = torch.cat((pca_image_layer2, pca_gnn_layer2, tabular_features_aligned), dim=1)
    fused_layer3 = torch.cat((pca_image_layer3, pca_gnn_layer3, tabular_features_aligned), dim=1)

    # 拼接三层的融合特征
    fused_features = torch.cat((fused_layer3, tabular_features), dim=1)

    # 通道洗牌操作
    def channel_shuffle(features, groups):
        batch_size, num_channels = features.size()
        assert num_channels % groups == 0, "The number of channels must be divisible by the number of groups."
        channels_per_group = num_channels // groups

        # Reshape into (batch_size, groups, channels_per_group)
        features = features.view(batch_size, groups, channels_per_group)

        # Transpose to (batch_size, channels_per_group, groups)
        features = features.transpose(1, 2)

        # Flatten back to (batch_size, num_channels)
        return features.contiguous().view(batch_size, -1)

    # 动态调整分组数
    shuffle_groups = 5
    while fused_features.size(1) % shuffle_groups != 0 and shuffle_groups > 1:
        shuffle_groups -= 1

    if shuffle_groups > 1:
        fused_features = channel_shuffle(fused_features, shuffle_groups)
    else:
        print("Warning: Channel shuffle skipped as no valid groups found.")

    # 返回直接cat后的特征
    return fused_features, None


def combine_features_1_cat(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    tabular_features = tabular_features.to(device)

    # PCA降维并归一化
    def pca_and_normalize(feature, n_components):
        pca_result = apply_pca(feature.cpu().numpy(), n_components)
        return F.normalize(torch.tensor(pca_result).to(device), dim=1)

    pca_image_layer1 = pca_and_normalize(image_layer1, 30)
    pca_gnn_layer1 = pca_and_normalize(gnn_layer1, 30)

    pca_image_layer2 = pca_and_normalize(image_layer2, 20)
    pca_gnn_layer2 = pca_and_normalize(gnn_layer2, 20)

    pca_image_layer3 = pca_and_normalize(image_layer3, 10)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 10)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_layer1 = torch.nn.Linear(pca_image_layer1.size(1), align_dim).to(device)
    align_layer2 = torch.nn.Linear(pca_image_layer2.size(1), align_dim).to(device)
    align_layer3 = torch.nn.Linear(pca_image_layer3.size(1), align_dim).to(device)

    # 对齐每个模态特征
    pca_image_layer1 = align_layer1(pca_image_layer1)
    pca_gnn_layer1 = align_layer1(pca_gnn_layer1)

    pca_image_layer2 = align_layer2(pca_image_layer2)
    pca_gnn_layer2 = align_layer2(pca_gnn_layer2)

    pca_image_layer3 = align_layer3(pca_image_layer3)
    pca_gnn_layer3 = align_layer3(pca_gnn_layer3)

    # 如果tabular_features维度不同，也对其对齐
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)
    tabular_features_aligned = tabular_align_layer(tabular_features)

    # 融合操作使用直接cat
    fused_layer1 = torch.cat((pca_image_layer1, pca_gnn_layer1, tabular_features_aligned), dim=1)
    fused_layer2 = torch.cat((pca_image_layer2, pca_gnn_layer2, tabular_features_aligned), dim=1)
    fused_layer3 = torch.cat((pca_image_layer3, pca_gnn_layer3, tabular_features_aligned), dim=1)

    # 拼接三层的融合特征
    fused_features = torch.cat((fused_layer3, tabular_features), dim=1)

    # 返回直接cat后的特征
    return fused_features, None

def combine_features_1_cat_without_graph_pca(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    # 确保所有输入都在同一设备上
    image_layer3 = image_layer3.to(device)
    tabular_features = tabular_features.to(device)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_image_layer3 = torch.nn.Linear(image_layer3.size(1), align_dim).to(device)
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)

    # 对齐每个模态特征
    aligned_image_layer3 = align_image_layer3(image_layer3)
    aligned_tabular_features = tabular_align_layer(tabular_features)

    # 仅将最后一层图像特征和表格特征直接cat
    fused_features = torch.cat((aligned_image_layer3, aligned_tabular_features), dim=1)

    # 返回融合后的特征
    return fused_features, None


def combine_features_1_kronecker_product_without_graph_pca(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    # 确保所有输入都在同一设备上
    image_layer3 = image_layer3.to(device)
    tabular_features = tabular_features.to(device)

    # 对齐 tabular_features 维度
    align_dim = 30  # 设置统一的特征维度
    align_image_layer3 = torch.nn.Linear(image_layer3.size(1), align_dim).to(device)
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)

    # 对齐每个模态特征
    aligned_image_layer3 = align_image_layer3(image_layer3)
    aligned_tabular_features = tabular_align_layer(tabular_features)

    # 确保批量维度一致
    if aligned_image_layer3.size(0) != aligned_tabular_features.size(0):
        raise ValueError("Batch sizes of aligned_image_layer3 and aligned_tabular_features must match.")

    # Kronecker product 融合特征
    kronecker_features = torch.kron(aligned_image_layer3, aligned_tabular_features)

    # 降维操作，确保输出与 cat 操作的维度一致
    output_dim = aligned_image_layer3.size(1) + aligned_tabular_features.size(1)
    reduce_dim_layer = torch.nn.Linear(kronecker_features.size(1), output_dim).to(device)
    fused_features = reduce_dim_layer(kronecker_features)

    # 确保样本数匹配
    fused_features = fused_features[:tabular_features.size(0)]

    # 返回融合后的特征
    return fused_features, None

def combine_features_1_outer_product_without_graph_pca(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    # 确保所有输入都在同一设备上
    image_layer3 = image_layer3.to(device)
    tabular_features = tabular_features.to(device)

    # 对齐 tabular_features 维度
    align_dim = 30  # 设置统一的特征维度
    align_image_layer3 = torch.nn.Linear(image_layer3.size(1), align_dim).to(device)
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)

    # 对齐每个模态特征
    aligned_image_layer3 = align_image_layer3(image_layer3)
    aligned_tabular_features = tabular_align_layer(tabular_features)

    # 外积融合特征
    outer_product_features = torch.bmm(
        aligned_image_layer3.unsqueeze(2),  # 扩展维度以计算外积
        aligned_tabular_features.unsqueeze(1)  # 扩展维度以计算外积
    )  # 输出维度为 [batch_size, align_dim, align_dim]

    # 降维操作，确保输出维度与 cat 操作的结果一致
    output_dim = aligned_image_layer3.size(1) + aligned_tabular_features.size(1)
    reduce_dim_layer = torch.nn.Linear(outer_product_features.view(outer_product_features.size(0), -1).size(1), output_dim).to(device)

    # 降维至与 cat 融合方式一致的维度
    fused_features = reduce_dim_layer(outer_product_features.view(outer_product_features.size(0), -1))

    # 返回融合后的特征
    return fused_features, None



import torch
import torch.nn as nn

class AttentionBottleneck(nn.Module):
    def __init__(self, input_dim1, input_dim2, bottleneck_dim):
        super(AttentionBottleneck, self).__init__()
        self.fc1 = nn.Linear(input_dim1, bottleneck_dim)
        self.fc2 = nn.Linear(input_dim2, bottleneck_dim)
        self.attention = nn.Linear(bottleneck_dim * 2, bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, input_dim1 + input_dim2)  # 输出维度与输入一致

    def forward(self, x1, x2):
        # 投影到瓶颈空间
        h1 = torch.relu(self.fc1(x1))
        h2 = torch.relu(self.fc2(x2))
        # 计算注意力权重
        combined = torch.cat((h1, h2), dim=1)
        attention_weights = torch.sigmoid(self.attention(combined))
        # 加权融合
        fused = attention_weights * h1 + (1 - attention_weights) * h2
        # 恢复到原始维度
        return self.output(fused)

def combine_features_attention_bottleneck(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    # 确保所有输入都在同一设备上
    image_layer3 = image_layer3.to(device)
    tabular_features = tabular_features.to(device)

    # 对齐tabular_features维度
    align_dim = 30  # 设置统一的特征维度
    align_image_layer3 = torch.nn.Linear(image_layer3.size(1), align_dim).to(device)
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)

    # 对齐每个模态特征
    aligned_image_layer3 = align_image_layer3(image_layer3)
    aligned_tabular_features = tabular_align_layer(tabular_features)

    # 注意力瓶颈融合
    bottleneck_dim = 15  # 设置瓶颈维度
    attention_bottleneck = AttentionBottleneck(align_dim, align_dim, bottleneck_dim).to(device)
    fused_features = attention_bottleneck(aligned_image_layer3, aligned_tabular_features)

    # 返回融合后的特征
    return fused_features, None



class TransformerFusion(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(TransformerFusion, self).__init__()
        # 投影到嵌入维度
        self.projection_image = nn.Linear(input_dim, embed_dim)
        self.projection_tabular = nn.Linear(input_dim, embed_dim)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出投影回原始维度
        self.output_projection = nn.Linear(embed_dim, input_dim)

    def forward(self, image_features, tabular_features):
        # 投影到统一嵌入空间
        image_embed = self.projection_image(image_features).unsqueeze(1)  # [batch_size, 1, embed_dim]
        tabular_embed = self.projection_tabular(tabular_features).unsqueeze(1)  # [batch_size, 1, embed_dim]
        # 拼接两个模态的嵌入
        combined_embed = torch.cat((image_embed, tabular_embed), dim=1)  # [batch_size, 2, embed_dim]
        # 通过 Transformer 编码器
        fused_embed = self.transformer(combined_embed)  # [batch_size, 2, embed_dim]
        # 对两个模态分别投影回原始维度
        fused_image = self.output_projection(fused_embed[:, 0, :])  # [batch_size, input_dim]
        fused_tabular = self.output_projection(fused_embed[:, 1, :])  # [batch_size, input_dim]
        # 将结果拼接
        return torch.cat((fused_image, fused_tabular), dim=1)

def combine_features_transformer_fusion_without_graph_pca(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    # 确保所有输入都在同一设备上
    image_layer3 = image_layer3.to(device)
    tabular_features = tabular_features.to(device)

    # 对齐 tabular_features 维度
    align_dim = 30  # 设置统一的特征维度
    align_image_layer3 = torch.nn.Linear(image_layer3.size(1), align_dim).to(device)
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)

    # 对齐每个模态特征
    aligned_image_layer3 = align_image_layer3(image_layer3)
    aligned_tabular_features = tabular_align_layer(tabular_features)

    # Transformer-based 融合
    embed_dim = align_dim  # 嵌入维度
    num_heads = 2  # 多头注意力头数
    num_layers = 2  # Transformer 层数
    transformer_fusion = TransformerFusion(align_dim, embed_dim, num_heads, num_layers).to(device)
    fused_features = transformer_fusion(aligned_image_layer3, aligned_tabular_features)

    # 返回融合后的特征
    return fused_features, None


class CoAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(CoAttentionFusion, self).__init__()
        self.query_image = nn.Linear(feature_dim, feature_dim)
        self.key_tabular = nn.Linear(feature_dim, feature_dim)
        self.value_tabular = nn.Linear(feature_dim, feature_dim)

        self.query_tabular = nn.Linear(feature_dim, feature_dim)
        self.key_image = nn.Linear(feature_dim, feature_dim)
        self.value_image = nn.Linear(feature_dim, feature_dim)

        self.output_projection = nn.Linear(feature_dim * 2, feature_dim * 2)  # 融合输出映射到最终维度

    def forward(self, image_features, tabular_features):
        # 图像特征对表格特征的注意力
        query_img = self.query_image(image_features)
        key_tab = self.key_tabular(tabular_features)
        value_tab = self.value_tabular(tabular_features)

        attention_weights_image = F.softmax(torch.matmul(query_img, key_tab.T), dim=-1)  # 注意力权重
        attended_tabular = torch.matmul(attention_weights_image, value_tab)  # 计算注意力加权表格特征

        # 表格特征对图像特征的注意力
        query_tab = self.query_tabular(tabular_features)
        key_img = self.key_image(image_features)
        value_img = self.value_image(image_features)

        attention_weights_tabular = F.softmax(torch.matmul(query_tab, key_img.T), dim=-1)  # 注意力权重
        attended_image = torch.matmul(attention_weights_tabular, value_img)  # 计算注意力加权图像特征

        # 将两个模态的注意力特征拼接并映射到最终维度
        fused_features = torch.cat((attended_image, attended_tabular), dim=1)
        return self.output_projection(fused_features)


def combine_features_coattention_fusion_without_graph_pca(
    image_layer1, image_layer2, image_layer3,
    gnn_layer1, gnn_layer2, gnn_layer3,
    tabular_features
):
    # 确保所有输入都在同一设备上
    image_layer3 = image_layer3.to(device)
    tabular_features = tabular_features.to(device)

    # 对齐 tabular_features 维度
    align_dim = 30  # 设置统一的特征维度
    align_image_layer3 = torch.nn.Linear(image_layer3.size(1), align_dim).to(device)
    tabular_align_layer = torch.nn.Linear(tabular_features.size(1), align_dim).to(device)

    # 对齐每个模态特征
    aligned_image_layer3 = align_image_layer3(image_layer3)
    aligned_tabular_features = tabular_align_layer(tabular_features)

    # 使用协同注意力进行特征融合
    coattention_fusion = CoAttentionFusion(align_dim).to(device)
    fused_features = coattention_fusion(aligned_image_layer3, aligned_tabular_features)

    # 返回融合后的特征
    return fused_features, None
