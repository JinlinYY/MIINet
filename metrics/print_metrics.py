import numpy as np


def print_average_metrics(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC_score_macro, AUC_score_micro, specificitiy_scores, FNR_scores):
    print(f'\nAverage metrics across {len(accuracy_scores)} folds:')
    print(f'Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}')
    print(f'Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}')
    print(f'Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}')
    print(f'F1-score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
    print(f'AUC_macro: {np.mean(AUC_score_macro):.4f} ± {np.std(AUC_score_macro):.4f}')
    print(f'AUC_micro: {np.mean(AUC_score_micro):.4f} ± {np.std(AUC_score_micro):.4f}')
    print(f'Average Specificity: {np.mean(specificitiy_scores):.4f} ± {np.std(specificitiy_scores):.4f}')
    print(f'Average FNR: {np.mean(FNR_scores):.4f} ± {np.std(FNR_scores):.4f}')

def print_average_metrics_2(accuracy_scores, precision_scores, recall_scores, f1_scores, AUC,specificitiy_scores, FNR_scores):
    print(f'\nAverage metrics across {len(accuracy_scores)} folds:')
    print(f'Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}')
    print(f'Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}')
    print(f'Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}')
    print(f'F1-score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
    print(f'AUC: {np.mean(AUC):.4f} ± {np.std(AUC):.4f}')
    print(f'Average Specificity: {np.mean(specificitiy_scores):.4f} ± {np.std(specificitiy_scores):.4f}')
    print(f'Average FNR: {np.mean(FNR_scores):.4f} ± {np.std(FNR_scores):.4f}')

def print_mean_std_metrics(all_metrics):
    for dataset_type, metrics_list in all_metrics.items():
        mean_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0]}
        std_metrics = {metric: np.std([m[metric] for m in metrics_list]) for metric in metrics_list[0]}
        print(f'\nMean and Standard Deviation of {dataset_type} Metrics:')
        for metric, mean_value in mean_metrics.items():
            std_value = std_metrics[metric]
            print(f'{metric}: {mean_value:.4f} ± {std_value:.4f}')