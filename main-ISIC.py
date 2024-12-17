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
    # 设定特征文件路径
    excel_feature_path = '/tmp/pycharm_project_403/public_dataset/ISIC/train-del.xlsx'
    image_feature_path = '/tmp/pycharm_project_403/public_dataset/ISIC/train_128'
    gnn_feature_path = '/tmp/pycharm_project_403/public_dataset/ISIC/train-del.xlsx'

    # 提取原始表格特征
    feature_file = '/tmp/pycharm_project_403/public_dataset/ISIC/features/excel_features.pth'
    if os.path.exists(feature_file):
        print("Loading stored Excel features...")
        excel_feature_tensor = torch.load(feature_file)
        label = torch.load('label.pth')  # 确保加载标签
        print("Excel features and labels loaded.")
    else:
        print("Extracting Excel features...")
        index, excel_feature, label = extract_excel_features(excel_feature_path)
        excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)
        torch.save(excel_feature_tensor, feature_file)
        torch.save(label, 'label.pth')  # 保存标签
        print("Excel features and labels saved.")

    # 提取超声图像特征
    image_feature_file = '/tmp/pycharm_project_403/public_dataset/ISIC/features/image_features.pth'
    if os.path.exists(image_feature_file):
        print("Loading stored image features...")
        image_layer1_tensor, image_layer2_tensor, image_layer3_tensor = torch.load(image_feature_file)
        print("Image features loaded.")
    else:
        print("Extracting image features...")
        image_filenames = [f'{image_feature_path}/{idx}.jpg' for idx in index.astype(int)]
        image_layer1, image_layer2, image_layer3 = extract_image_features(image_filenames)
        image_layer1_tensor = torch.tensor(image_layer1, dtype=torch.float32)
        image_layer2_tensor = torch.tensor(image_layer2, dtype=torch.float32)
        image_layer3_tensor = torch.tensor(image_layer3, dtype=torch.float32)
        torch.save((image_layer1_tensor, image_layer2_tensor, image_layer3_tensor), image_feature_file)
        print("Image features saved.")

    # 提取GNN的表格特征
    gnn_feature_file = '/tmp/pycharm_project_403/public_dataset/ISIC/features/gnn_features.pth'
    if os.path.exists(gnn_feature_file):
        print("Loading stored GNN features...")
        gnn_layer1_tensor, gnn_layer2_tensor, gnn_layer3_tensor = torch.load(gnn_feature_file)
        print("GNN features loaded.")
    else:
        print("Extracting GNN features...")
        _, gnn_layer1, gnn_layer2, gnn_layer3, _ = gnn_extract_excel_features(gnn_feature_path, n_clusters=3)
        gnn_layer1_tensor = torch.tensor(gnn_layer1, dtype=torch.float32)
        gnn_layer2_tensor = torch.tensor(gnn_layer2, dtype=torch.float32)
        gnn_layer3_tensor = torch.tensor(gnn_layer3, dtype=torch.float32)
        torch.save((gnn_layer1_tensor, gnn_layer2_tensor, gnn_layer3_tensor), gnn_feature_file)
        print("GNN features saved.")

    # 特征融合
    combined_features_file = '/tmp/pycharm_project_403/public_dataset/ISIC/features/combined_features.pth'
    if os.path.exists(combined_features_file):
        print("Loading stored combined features...")
        combined_features_tensor, label_tensor = torch.load(combined_features_file)
        print("Combined features loaded.")
    else:
        print("Combining features...")
        combined_features, attention_weights = combine_features_attention_weighting_channel_shuffle(
            image_layer1_tensor, image_layer2_tensor, image_layer3_tensor,  # 超声图像特征
            gnn_layer1_tensor, gnn_layer2_tensor, gnn_layer3_tensor,  # 临床图特征
            excel_feature_tensor  # 临床特征
        )

        combined_features_tensor, label_tensor = inputtotensor(combined_features, label)
        torch.save((combined_features_tensor, label_tensor), combined_features_file)


    # K-fold cross-validation
    k_folds = 20
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores, specificity_scores, FNR_scores = [], [], [], [], [], [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features_tensor, label):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor[train_index], combined_features_tensor[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]
        print(f"combined_features.shape[1]:{combined_features_tensor.shape[1]}")


        net = Classifier(feature_dim=combined_features_tensor.shape[1], output_size=len(set(label))).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = FocalLoss(gamma=2.0, alpha=0.5)

        batch_size = 5096
        model_path = f'./pth/best_model_fold{fold}.pth'

        cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred, train_losses, val_losses = train_test(
            x_train, y_train, x_test, y_test,
            x_test, y_test,
            net, optimizer, loss_func, batch_size, model_path
        )


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
    SEED = 20
    set_seed(SEED)
    main()
