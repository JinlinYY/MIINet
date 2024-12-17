# import numpy as np
# import torch
#
# def inputtotensor(inputtensor, labeltensor):
#     inputtensor = np.array(inputtensor)
#     inputtensor = torch.FloatTensor(inputtensor)
#
#     labeltensor = np.array(labeltensor)
#     labeltensor = labeltensor.astype(float)
#     labeltensor = torch.LongTensor(labeltensor)
#
#     return inputtensor, labeltensor
import numpy as np
import torch


def inputtotensor(inputtensor, label):
    # 如果 inputtensor 是 GPU 张量，需要将其移动到 CPU
    if inputtensor.is_cuda:
        inputtensor = inputtensor.cpu()

    # 如果 inputtensor 需要梯度，则先调用 .detach()
    if inputtensor.requires_grad:
        inputtensor = inputtensor.detach()

    # 将张量转换为 NumPy 数组
    inputtensor = np.array(inputtensor)
    labeltensor = torch.tensor(label, dtype=torch.long)  # 转换为 PyTorch 张量

    return torch.tensor(inputtensor, dtype=torch.float32), labeltensor
