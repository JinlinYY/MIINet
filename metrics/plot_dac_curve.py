from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dac_curve(y_true, y_probs, dataset_type="Test", save_path=None):
    thresholds = np.linspace(0, 1, 101)

    # 将 y_true 转换为二进制标签矩阵
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]

    # Prepare dictionaries to store metrics for each class
    specificities = {i: [] for i in range(n_classes)}
    fnrs = {i: [] for i in range(n_classes)}  # Changed npvs to fnrs
    precisions = {i: [] for i in range(n_classes)}
    accuracies = {i: [] for i in range(n_classes)}
    recalls = {i: [] for i in range(n_classes)}
    f1_scores = {i: [] for i in range(n_classes)}

    # Iterate over thresholds
    for threshold in thresholds:
        for i in range(n_classes):  # For each class
            # Generate predicted classes based on the current threshold
            y_pred = (y_probs[:, i] >= threshold).astype(int)

            # Compute confusion matrix for each class (treat it as positive)
            cm = confusion_matrix(y_true_binarized[:, i], y_pred)  # Binarized y_true vs. predicted class

            # For each class, compute the metrics based on confusion matrix
            if cm.shape == (2, 2):  # Ensure that it's a 2x2 confusion matrix for binary classification
                tn, fp, fn, tp = cm.ravel()

                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificity
                fnr = fn / (fn + tp) if (fn + tp) != 0 else 0  # False Negative Rate (FNR)
                precision = precision_score(y_true_binarized[:, i], y_pred)  # Precision for the class
                accuracy = accuracy_score(y_true_binarized[:, i], y_pred)  # Accuracy
                recall = recall_score(y_true_binarized[:, i], y_pred)  # Recall for the class
                f1 = f1_score(y_true_binarized[:, i], y_pred)  # F1-Score for the class

                # Store metrics for this class
                specificities[i].append(specificity)
                fnrs[i].append(fnr)  # Store FNR instead of NPV
                precisions[i].append(precision)
                accuracies[i].append(accuracy)
                recalls[i].append(recall)
                f1_scores[i].append(f1)

    # 计算平均的DAC曲线（跨类别）
    avg_specificity = np.mean(list(specificities.values()), axis=0)
    avg_fnr = np.mean(list(fnrs.values()), axis=0)  # Average FNR
    avg_precision = np.mean(list(precisions.values()), axis=0)
    avg_accuracy = np.mean(list(accuracies.values()), axis=0)
    avg_recall = np.mean(list(recalls.values()), axis=0)
    avg_f1 = np.mean(list(f1_scores.values()), axis=0)

    # Create subplots for each metric (4x2 grid for 7 metrics now)
    fig, axs = plt.subplots(3, 2, figsize=(20, 16))  # 4x2 grid for 7 metrics (no Sensitivity)
    plt.subplots_adjust(hspace=0.4, wspace=0.8)

    # Define x and y axis limits
    axis_limits = [-0.05, 1.05]

    # Plot Specificity
    for i, color in zip(range(n_classes), ['#1f77b4', '#ff7f0e', '#2ca02c']):
        axs[0, 0].plot(thresholds, specificities[i], label=f'Class {i} Specificity', linestyle='-', color=color)
    axs[0, 0].plot(thresholds, avg_specificity, label='Average Specificity', linestyle='-', color='black', lw=3)
    axs[0, 0].set_title('Specificity')
    axs[0, 0].set_xlabel('Threshold')
    axs[0, 0].set_ylabel('Specificity')
    axs[0, 0].set_xlim(axis_limits)  # Set x-axis limits
    axs[0, 0].set_ylim(axis_limits)  # Set y-axis limits
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0, 0].grid(True)

    # Plot FNR
    for i, color in zip(range(n_classes), ['#1f77b4', '#ff7f0e', '#2ca02c']):
        axs[1, 0].plot(thresholds, fnrs[i], label=f'Class {i} FNR', linestyle='-', color=color)  # Plot FNR
    axs[1, 0].plot(thresholds, avg_fnr, label='Average FNR', linestyle='-', color='black', lw=3)  # Plot average FNR
    axs[1, 0].set_title('False Negative Rate (FNR)')
    axs[1, 0].set_xlabel('Threshold')
    axs[1, 0].set_ylabel('FNR')
    axs[1, 0].set_xlim(axis_limits)  # Set x-axis limits
    axs[1, 0].set_ylim(axis_limits)  # Set y-axis limits
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1, 0].grid(True)

    # Plot Precision
    for i, color in zip(range(n_classes), ['#1f77b4', '#ff7f0e', '#2ca02c']):
        axs[2, 0].plot(thresholds, precisions[i], label=f'Class {i} Precision', linestyle='-', color=color)
    axs[2, 0].plot(thresholds, avg_precision, label='Average Precision', linestyle='-', color='black', lw=3)
    axs[2, 0].set_title('Precision')
    axs[2, 0].set_xlabel('Threshold')
    axs[2, 0].set_ylabel('Precision')
    axs[2, 0].set_xlim(axis_limits)  # Set x-axis limits
    axs[2, 0].set_ylim(axis_limits)  # Set y-axis limits
    axs[2, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[2, 0].grid(True)

    # Plot Accuracy
    for i, color in zip(range(n_classes), ['#1f77b4', '#ff7f0e', '#2ca02c']):
        axs[0, 1].plot(thresholds, accuracies[i], label=f'Class {i} Accuracy', linestyle='-', color=color)
    axs[0, 1].plot(thresholds, avg_accuracy, label='Average Accuracy', linestyle='-', color='black', lw=3)
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Threshold')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_xlim(axis_limits)  # Set x-axis limits
    axs[0, 1].set_ylim(axis_limits)  # Set y-axis limits
    axs[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0, 1].grid(True)

    # Plot Recall
    for i, color in zip(range(n_classes), ['#1f77b4', '#ff7f0e', '#2ca02c']):
        axs[1, 1].plot(thresholds, recalls[i], label=f'Class {i} Recall', linestyle='-', color=color)
    axs[1, 1].plot(thresholds, avg_recall, label='Average Recall', linestyle='-', color='black', lw=3)
    axs[1, 1].set_title('Recall (Sensitivity)')
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].set_xlim(axis_limits)  # Set x-axis limits
    axs[1, 1].set_ylim(axis_limits)  # Set y-axis limits
    axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1, 1].grid(True)

    # Plot F1-Score
    for i, color in zip(range(n_classes), ['#1f77b4', '#ff7f0e', '#2ca02c']):
        axs[2, 1].plot(thresholds, f1_scores[i], label=f'Class {i} F1-Score', linestyle='-', color=color)
    axs[2, 1].plot(thresholds, avg_f1, label='Average F1-Score', linestyle='-', color='black', lw=3)
    axs[2, 1].set_title('F1-Score')
    axs[2, 1].set_xlabel('Threshold')
    axs[2, 1].set_ylabel('F1-Score')
    axs[2, 1].set_xlim(axis_limits)  # Set x-axis limits
    axs[2, 1].set_ylim(axis_limits)  # Set y-axis limits
    axs[2, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[2, 1].grid(True)

    # Adjust layout and save or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"DAC曲线已保存到: {save_path}")
    else:
        plt.show()

    # Return calculated metrics
    return {
        "thresholds": thresholds,
        "specificity": specificities,
        "fnr": fnrs,  # Changed npv to fnr
        "precision": precisions,
        "accuracy": accuracies,
        "recall": recalls,
        "f1_score": f1_scores,
        "avg_specificity": avg_specificity,
        "avg_fnr": avg_fnr,  # Average FNR
        "avg_precision": avg_precision,
        "avg_accuracy": avg_accuracy,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1
    }
