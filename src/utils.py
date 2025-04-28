import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def load_and_preprocess_data(geo_csv_path, curated_csv_path, label_col, exclude_columns):
    """
    Loads GEO and curated datasets, merges them on patient identifiers, filters out rows with missing labels,
    and computes the number of weeks from a reference date.
    Args:
        geo_csv_path (str): Path to the GEO dataset CSV file.
        curated_csv_path (str): Path to the curated dataset CSV file.
        label_col (str): The column name for the labels in the curated dataset.
        exclude_columns (list): List of columns to exclude from features.
    Returns:
        pd.DataFrame: Preprocessed GEO dataset with labels and computed weeks.
        list: List of columns excluded from features.
    """
    geo_df = pd.read_csv(geo_csv_path)
    curated_df = pd.read_csv(curated_csv_path)
    geo_df = geo_df.merge(curated_df[['Patient ID', 'id', label_col]], on=['Patient ID', 'id'], how='left')
    geo_df = geo_df[geo_df[label_col].notna()]
    geo_df = add_num_weeks_column(geo_df, 'CROSSING_TIME_POINT')
    return geo_df, exclude_columns

def split_and_scale_data(df, label_col, feature_cols, test_size=0.2, random_state=42):
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
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
    df["num_weeks"] = ((df[date_col] - reference_date).dt.days // 7).astype(int)
    return df


def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

