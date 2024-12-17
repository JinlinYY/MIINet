# import numpy as np
# import torch
# from torchvision import models, transforms
# from PIL import Image
# from torch.nn import Identity
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def extract_image_features(image_filenames):
#     # 加载ViT模型，并删除最后的分类层
#     model = models.vit_l_32(weights='DEFAULT')
#     model.heads = Identity()  # 去掉分类头部
#
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     model.eval()
#     model.to(device)
#
#     # 定义一个钩子函数来捕获中间层输出
#     intermediate_outputs = []
#
#     def hook(module, input, output):
#         intermediate_outputs.append(output)
#
#     # 在模型中注册钩子，捕获中间层输出
#     hooks = []
#     for i in [6, 9, 11]:  # 根据需要选择ViT模型中的层索引
#         hooks.append(model.encoder.layers[i].register_forward_hook(hook))
#
#     layer6_features = []
#     layer9_features = []
#     final_features = []
#
#     with torch.no_grad():
#         for filename in image_filenames:
#             image = Image.open(filename).convert('RGB')
#             input_tensor = preprocess(image).unsqueeze(0).to(device)
#
#             # 执行前向传播，钩子会捕获中间层输出
#             model(input_tensor)
#
#             # 获取中间层输出
#             layer6 = intermediate_outputs[0].mean(dim=1).cpu().numpy()
#             layer9 = intermediate_outputs[1].mean(dim=1).cpu().numpy()
#             final_output = intermediate_outputs[2].mean(dim=1).cpu().numpy()
#
#             layer6_features.append(layer6)
#             layer9_features.append(layer9)
#             final_features.append(final_output)
#
#             # 清除中间输出以备下次使用
#             intermediate_outputs.clear()
#
#     # 移除钩子
#     for hook in hooks:
#         hook.remove()
#
#     return (
#         np.vstack(layer6_features),
#         np.vstack(layer9_features),
#         np.vstack(final_features)
#     )



# def extract_image_features(image_filenames):
#     # 加载ResNet50模型，并删除最后的分类层
#     model = models.resnet50(weights='DEFAULT')
#     model.fc = Identity()  # 去掉分类头部
#
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     model.eval()
#     model.to(device)
#
#     # 定义一个钩子函数来捕获中间层输出
#     intermediate_outputs = []
#
#     def hook(module, input, output):
#         intermediate_outputs.append(output)
#
#     # 在模型中注册钩子，捕获中间层输出
#     hooks = []
#     for name, module in model.named_children():
#         if name in ['layer2', 'layer3', 'layer4']:  # 选择ResNet50的中间层
#             hooks.append(module.register_forward_hook(hook))
#
#     layer2_features = []
#     layer3_features = []
#     layer4_features = []
#
#     with torch.no_grad():
#         for filename in image_filenames:
#             image = Image.open(filename).convert('RGB')
#             input_tensor = preprocess(image).unsqueeze(0).to(device)
#
#             # 执行前向传播，钩子会捕获中间层输出
#             model(input_tensor)
#
#             # 获取中间层输出
#             layer2 = intermediate_outputs[0].mean(dim=[2, 3]).cpu().numpy()  # 全局平均池化
#             layer3 = intermediate_outputs[1].mean(dim=[2, 3]).cpu().numpy()
#             layer4 = intermediate_outputs[2].mean(dim=[2, 3]).cpu().numpy()
#
#             layer2_features.append(layer2)
#             layer3_features.append(layer3)
#             layer4_features.append(layer4)
#
#             # 清除中间输出以备下次使用
#             intermediate_outputs.clear()
#
#     # 移除钩子
#     for hook in hooks:
#         hook.remove()
#
#     return (
#         np.vstack(layer2_features),
#         np.vstack(layer3_features),
#         np.vstack(layer4_features)
#     )


#################ISIC的超声图像特征提取#####################
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn import Identity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, image_filenames, transform=None):
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def extract_image_features(image_filenames, batch_size=16, num_workers=4):
    # Load the ViT model and remove the classification head
    model = models.vit_l_32(weights='DEFAULT')
    model.heads = Identity()  # Remove the classification head

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval()
    model.to(device)

    # Create a dataset and DataLoader for batching and multi-threading
    dataset = ImageDataset(image_filenames, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    intermediate_outputs = []

    def hook(module, input, output):
        intermediate_outputs.append(output)

    # Register hooks for the intermediate layers
    hooks = []
    for i in [6, 9, 11]:  # Select layers of ViT model
        hooks.append(model.encoder.layers[i].register_forward_hook(hook))

    layer6_features = []
    layer9_features = []
    final_features = []

    with torch.no_grad():  # Disable gradients for inference
        for inputs in dataloader:
            inputs = inputs.to(device)

            # Perform forward pass and capture intermediate outputs
            model(inputs)

            # Process the intermediate layer outputs
            layer6 = intermediate_outputs[0].mean(dim=1).cpu().numpy()
            layer9 = intermediate_outputs[1].mean(dim=1).cpu().numpy()
            final_output = intermediate_outputs[2].mean(dim=1).cpu().numpy()

            layer6_features.append(layer6)
            layer9_features.append(layer9)
            final_features.append(final_output)

            # Clear intermediate outputs for the next batch
            intermediate_outputs.clear()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return (
        np.vstack(layer6_features),
        np.vstack(layer9_features),
        np.vstack(final_features)
    )
