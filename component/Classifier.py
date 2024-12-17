import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classifier(nn.Module):
    def __init__(self, feature_dim, output_size):
        super().__init__()
        self.linear1 = nn.Linear(feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层，防止过拟合

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)  # 在第一层后添加Dropout
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)  # 在第二层后添加Dropout
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)  # 在第三层后添加Dropout
        x = F.relu(self.bn4(self.linear4(x)))
        x = self.dropout(x)  # 在第四层后添加Dropout
        out = self.classifier(x)
        return out

class SNN(nn.Module):
    def __init__(self, feature_dim, output_size):
        super(SNN, self).__init__()
        # 定义网络层
        self.linear1 = nn.Linear(feature_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)  # Dropout 防止过拟合

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # 使用 LeCun 正态初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=np.sqrt(1 / module.weight.size(1)))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 前向传播，使用 SELU 激活函数
        x = F.selu(self.linear1(x))
        x = self.dropout(x)
        x = F.selu(self.linear2(x))
        x = self.dropout(x)
        x = F.selu(self.linear3(x))
        x = self.dropout(x)
        x = F.selu(self.linear4(x))
        x = self.dropout(x)
        out = self.classifier(x)  # 最后一层为线性层
        return out

