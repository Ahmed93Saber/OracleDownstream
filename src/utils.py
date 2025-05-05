import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import random
import numpy as np


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

def split_and_scale_data(df, label_col, feature_cols, test_size=0.2, random_state=101):
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
