import numpy as np
import torch
from sklearn.decomposition import PCA, MiniBatchSparsePCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from component.Cli_Encoder import extract_excel_features
from component.USI_Internal_Feature import extract_image_features
from component.GNN_Internal_Feature import gnn_extract_excel_features
from component.Fusion_Internal_Feature import combine_features_attention_weighting_channel_shuffle, combine_features_attention_weighting
from metrics.plot_roc_curve import plot_roc_curve
from module.inputtotensor import inputtotensor
from component.Classifier import Classifier, SNN
from metrics.print_metrics import print_average_metrics
from module.set_seed import set_seed
from module.train_test import train_test
from module.my_loss import FocalLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # 提取原始表格特征
    index, excel_feature, label = extract_excel_features('/tmp/pycharm_project_403/excel_data/datasets-internal-del-name-3.xlsx')
    excel_feature_tensor = torch.tensor(excel_feature, dtype=torch.float32)

    # 提取超声图像特征（输出三层特征）
    image_filenames = ['/tmp/pycharm_project_403/image_data_internal_external/{}.bmp'.format(idx) for idx in index.astype(int)]
    image_layer1, image_layer2, image_layer3 = extract_image_features(image_filenames)  # 修改为输出三层
    image_layer1_tensor = torch.tensor(image_layer1, dtype=torch.float32)
    image_layer2_tensor = torch.tensor(image_layer2, dtype=torch.float32)
    image_layer3_tensor = torch.tensor(image_layer3, dtype=torch.float32)

    _, gnn_layer1, gnn_layer2, gnn_layer3, _ = gnn_extract_excel_features(
        '/tmp/pycharm_project_403/excel_data/datasets-internal-del-name-3.xlsx', graph_method='cluster', n_clusters=5
    )

    gnn_layer1_tensor = torch.tensor(gnn_layer1, dtype=torch.float32)
    gnn_layer2_tensor = torch.tensor(gnn_layer2, dtype=torch.float32)
    gnn_layer3_tensor = torch.tensor(gnn_layer3, dtype=torch.float32)

    # 特征融合
    final_combined_features, attention_weights  = combine_features_attention_weighting_channel_shuffle(
        image_layer1_tensor, image_layer2_tensor, image_layer3_tensor,   # 超声图像特征
        gnn_layer1_tensor, gnn_layer2_tensor, gnn_layer3_tensor,  # 临床图特征
        excel_feature_tensor  # 临床特征
    )

    # 转换为张量并获取标签
    combined_features_tensor, label_tensor = inputtotensor(final_combined_features, label)

    # K-fold cross-validation
    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro = [], [], [], [], [], []
    all_y_true, all_y_probs = [], []

    fold = 0
    for train_index, test_index in skf.split(combined_features_tensor, label_tensor):
        fold += 1
        print(f'Processing fold {fold}/{k_folds}...')
        x_train, x_test = combined_features_tensor[train_index], combined_features_tensor[test_index]
        y_train, y_test = label_tensor[train_index], label_tensor[test_index]

        net = Classifier(feature_dim=final_combined_features.shape[1], output_size=len(set(label))).to(device)
        # net = SNN(feature_dim=final_combined_features.shape[1], output_size=len(set(label))).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = FocalLoss(gamma=2)
        batch_size = 2048
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

        # ROC curve and AUC for the current fold
        all_y_true.extend(y_test)
        all_y_probs.extend(test_probs)
        roc_auc_fold = plot_roc_curve(y_test, test_probs, dataset_type=f"Fold {fold} Test", save_path=f"ROC_fig/roc_curve_fold{fold}.png")
        AUC_score_macro.append(roc_auc_fold['macro'])
        AUC_score_micro.append(roc_auc_fold['micro'])

    # Print average metrics
    print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro)

    # ROC and AUC for overall performance
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)
    overall_roc_auc = plot_roc_curve(all_y_true, all_y_probs, dataset_type="Overall", save_path="ROC_fig/roc_curve.png")
    print(overall_roc_auc)


if __name__ == "__main__":
    SEED = 45
    set_seed(SEED)
    main()
