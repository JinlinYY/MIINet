import numpy as np
import torch
from sklearn.decomposition import PCA, MiniBatchSparsePCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from component.Cli_Encoder import extract_excel_features
from component.USI_Internal_Feature import extract_image_features
from component.GNN_Internal_Feature import gnn_extract_excel_features
from component.Fusion_Internal_Feature import combine_features_attention_weighting_channel_shuffle, \
    combine_features_attention_weighting
from metrics.plot_roc_curve import plot_roc_curve
from metrics.plot_dac_curve import plot_dac_curve
from module.inputtotensor import inputtotensor
from component.Classifier import Classifier, SNN
from metrics.print_metrics import print_average_metrics, print_average_metrics_2
from module.set_seed import set_seed
from module.train_test import train_test
from module.my_loss import FocalLoss
from metrics.compute_specificity_fnr import compute_specificity_fnr, compute_specificity_fnr_2
from sklearn.metrics import  roc_curve, auc
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():
    # 提取原始表格特征
    index, excel_feature, label = extract_excel_features(
        '/tmp/pycharm_project_403/public_dataset/BCa/metadata-pRC.xlsx')
    excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # 提取超声图像特征（输出三层特征）
    image_filenames = ['/tmp/pycharm_project_403/public_dataset/BCa/CDIs_images_visualized/{}.png'.format(idx) for idx in
                       index.astype(int)]
    image_layer1, image_layer2, image_layer3 = extract_image_features(image_filenames)  # 修改为输出三层
    image_layer1_tensor = torch.tensor(image_layer1, dtype=torch.float32)
    image_layer2_tensor = torch.tensor(image_layer2, dtype=torch.float32)
    image_layer3_tensor = torch.tensor(image_layer3, dtype=torch.float32)

    # 提取GNN的表格特征（输出三层特征）
    _, gnn_layer1, gnn_layer2, gnn_layer3, _ = gnn_extract_excel_features(
        '/tmp/pycharm_project_403/public_dataset/BCa/metadata-pRC.xlsx', n_clusters=10
    )

    gnn_layer1_tensor = torch.tensor(gnn_layer1, dtype=torch.float32)
    gnn_layer2_tensor = torch.tensor(gnn_layer2, dtype=torch.float32)
    gnn_layer3_tensor = torch.tensor(gnn_layer3, dtype=torch.float32)

    # 特征融合
    combined_features, attention_weights = combine_features_attention_weighting_channel_shuffle(
        image_layer1_tensor, image_layer2_tensor, image_layer3_tensor,  # 超声图像特征
        gnn_layer1_tensor, gnn_layer2_tensor, gnn_layer3_tensor,  # 临床图特征
        excel_feature_tensor  # 临床特征
    )

    # combined_features = combine_features(excel_feature_pca_tensor, image_features_pca_tensor)  # 两模态
    combined_features_tensor, label_tensor = inputtotensor(combined_features, label)

    # K-fold cross-validation
    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores, specificity_scores, FNR_scores = [], [], [], [], [], [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features, label):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor[train_index], combined_features_tensor[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]
        print(f"combined_features.shape[1]:{combined_features.shape[1]}")

        test_patient_indices = index[test_index]  # 获取测试集患者索引号

        net = Classifier(feature_dim=combined_features.shape[1], output_size=len(set(label))).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = FocalLoss(gamma=2.0, alpha=1.2)

        batch_size = 256
        model_path = f'./pth/best_model_fold{fold}.pth'

        cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred, train_losses, val_losses = train_test(
            x_train, y_train, x_test, y_test,
            x_test, y_test,
            net, optimizer, loss_func, batch_size, model_path
        )

        # 打印测试集预测结果
        print(f"Test patient indices (fold {fold}): {test_patient_indices}")
        print(f"Fold {fold} - Test set true labels: {y_test.numpy()}")
        print(f"Fold {fold} - Test set predicted labels: {y_test_pred}")
        print(f"Fold {fold} - Test set predicted probabilities: {test_probs}")

        accuracy_scores.append(accuracy_score(y_test, y_test_pred))
        precision_scores.append(precision_score(y_test, y_test_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_test_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_test_pred, average='weighted'))
        fpr, tpr, _ = roc_curve(y_test, y_test_pred)
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)
        specificity, fnr = compute_specificity_fnr_2(y_test, y_test_pred)
        specificity_scores.append(specificity)
        FNR_scores.append(fnr)

    # 打印平均指标
    print_average_metrics_2(accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores,
                              specificity_scores, FNR_scores)


if __name__ == "__main__":
    SEED = 45
    set_seed(SEED)
    main()
