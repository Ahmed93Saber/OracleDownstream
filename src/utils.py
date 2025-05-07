import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import itertools


def load_and_preprocess_data(geo_csv_path, curated_csv_path, label_col):
    """
    Loads GEO and curated datasets, merges them on patient identifiers, filters out rows with missing labels,
    and computes the number of weeks from a reference date.
    Args:
        geo_csv_path (str): Path to the GEO dataset CSV file.
        curated_csv_path (str): Path to the curated dataset CSV file.
        label_col (str): The column name for the labels in the curated dataset.
    Returns:
        pd.DataFrame: Preprocessed GEO dataset with labels and computed weeks.
        list: List of columns excluded from features.
    """
    geo_df = pd.read_csv(geo_csv_path)
    curated_df = pd.read_csv(curated_csv_path)
    geo_df = geo_df.merge(curated_df[['Patient ID', 'id', label_col]], on=['Patient ID', 'id'], how='left')
    geo_df = geo_df[geo_df[label_col].notna()]
    geo_df = add_num_weeks_column(geo_df, 'CROSSING_TIME_POINT')
    return geo_df

def split_and_scale_data(df, label_col, feature_cols, test_size=0.2, random_state=0):
    """
    Splits the dataframe into train and test sets and scales feature columns.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col])
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df

def add_num_weeks_column(df, date_col, reference_date=None):
    """
    Adds a 'num_weeks' column to the dataframe, representing the number of weeks from a reference date.
    """
    if reference_date is None:
        reference_date = pd.to_datetime('1900-01-01')
    else:
        reference_date = pd.to_datetime(reference_date)
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y-%m-%d')
    df["num_weeks"] = ((df[date_col] - reference_date).dt.days // 7).astype(int)
    return df


def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def log_optuna_metrics(trial, metrics, is_test=False):
    """
    Logs the metrics to Optuna.
    """
    for metric, value in metrics.items():
        if is_test:
            trial.set_user_attr(f"test_{metric}", value)
        else:
            trial.set_user_attr(metric, value)


def save_models(models_stat_dicts, trial_number, mean_metrics, saving_path="weights"):


    if len(models_stat_dicts) == 5 and mean_metrics['accuracy'] > 0.7 and mean_metrics['auc'] > 0.7:

        for fold, state_dict in enumerate(models_stat_dicts):
            torch.save(state_dict, f"{saving_path}/model_fold_{fold}_trial_{trial_number}.pth")


def set_random(seed=1, deterministic=True):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_auc(predictions, ground_truths, num, mode, save=False):
    """
    Plot the ROC curve and display in the upper third of an A4 paper.

    :param num: trial number
    :param predictions: list of the list the predictions for each fold
    :param ground_truths: list of the list of the ground truth for each fold
    :param mode: mode of the model
    :param save: save the plot
    :return: plot the ROC curve
    """
    # Set font sizes
    axis_label_fontsize = 32
    legend_fontsize = 32
    tick_label_fontsize = 32
    line_width = 4

    # Prepare figure with A4 upper third dimensions
    fig = plt.figure(figsize=(15, 15))  # A4 width, one-third A4 height

    colors = itertools.cycle(['blue', 'darkorange', 'purple', 'green', 'black', 'red'])
    fprs = []
    tprs = []
    aucs = []

    # Compute ROC curves for each fold and store them
    for fold_pred, fold_gt in zip(predictions, ground_truths):
        fpr, tpr, _ = roc_curve(fold_gt, fold_pred)
        auc_value = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc_value)

    # Define a common set of points for averaging across all folds
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    # Plot individual ROC curves in faded lines
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        plt.plot(fpr, tpr, color=next(colors), lw=line_width, alpha=0.2,  # Faded line for individual folds
                 label=f'ROC curve fold {i + 1} (AUC = {aucs[i]:0.2f})')
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    # Calculate mean TPR and standard deviation
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot mean ROC curve in bold solid line
    plt.plot(mean_fpr, mean_tpr, color='black', lw=line_width + 2, linestyle='-',
             label=f'Mean ROC (AUC = {str(mean_auc)[:4]})')

    # Plot diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Labels, title, and grid
    plt.xlabel('False Positive Rate', fontsize=axis_label_fontsize)
    plt.ylabel('True Positive Rate', fontsize=axis_label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    plt.legend(loc='lower right', fontsize=legend_fontsize)
    plt.grid()

    # Adjust layout to fit in upper third of A4
    plt.subplots_adjust(top=0.85, bottom=0.15)  # Adjust padding to fit text

    # Save or show plot
    if save:
        os.makedirs('./plots/production/', exist_ok=True)
        plt.savefig(f'./plots/production/roc_{num}_{mode}.pdf', bbox_inches='tight')
    plt.show()
