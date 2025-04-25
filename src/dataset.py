import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

class ClinicalDataset(Dataset):
    """
    PyTorch Dataset for clinical data.
    """
    def __init__(self, dataframe: pd.DataFrame, columns_to_drop: list = None):
        self.dataframe = dataframe
        self.columns_to_drop = columns_to_drop

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        label = row['label-1RN-0Normal']
        features = row.drop(self.columns_to_drop).values.astype('float32')
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return features_tensor, label

def create_dataloaders(train_df: pd.DataFrame, label_column: str, exclude_columns: list, batch_size: int, n_splits: int = 5):
    """
    Creates stratified K-fold dataloaders for training and validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_columns = [col for col in train_df.columns if col not in exclude_columns]
    dataloaders = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[label_column])):
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]
        train_loader = DataLoader(
            ClinicalDataset(train_data, columns_to_drop=exclude_columns),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            ClinicalDataset(val_data, columns_to_drop=exclude_columns),
            batch_size=batch_size, shuffle=False
        )
        dataloaders[fold] = {'train': train_loader, 'val': val_loader}
    return dataloaders, feature_columns
