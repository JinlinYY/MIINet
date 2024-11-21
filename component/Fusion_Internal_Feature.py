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

    pca_image_layer3 = pca_and_normalize(image_layer3, 10)
    pca_gnn_layer3 = pca_and_normalize(gnn_layer3, 10)

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
