import torch
import torch.nn as nn
import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)                          # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         # logpt = F.log_softmax(input)
#         logpt = F.log_softmax(input, dim=1)
#
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = logpt.data.exp()
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * at
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 对输入做 softmax 操作，得到每个类别的概率分布
        inputs = F.softmax(inputs, dim=1)

        # 选择正确类别的概率
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.view(-1, 1), 1)
        prob = (inputs * targets_one_hot).sum(dim=1)

        # 计算 Focal Loss 的核心部分
        loss = -self.alpha * (1 - prob) ** self.gamma * prob.log()

        # 根据指定的 reduction 方法计算损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss