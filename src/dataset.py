import torch
import pandas as pd
import numpy as np
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
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return features_tensor, label_tensor


class ImagingDataset(Dataset):
    """
    PyTorch Dataset for Imaging data (Representations).
    """
    def __init__(self, dataframe: pd.DataFrame, data_dir : str, is_gap : bool = False):
        self.dataframe = dataframe

        if is_gap:
            self.embedd_type = 'embedding_attn_pool'  # 2D
        else:
            self.embedd_type = "embedding_norm"  # 3D

        self.img_seq = np.load(data_dir, allow_pickle=True).item()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        label = row['label-1RN-0Normal']
        patient_id = row["Patient ID"]
        met_id = row["id"]
        scan_date = row["scan_date"].split()[0]
        dict_key = f"{patient_id}_{scan_date}_{met_id}"

        try:
            features_tensor = self.img_seq[dict_key][self.embedd_type].clone().detach().float().squeeze()
        except KeyError:
            print(f" KeyError: {dict_key} not found in img_seq")
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features_tensor, label_tensor

def create_dataloaders(
    train_df: pd.DataFrame,
    label_column: str,
    exclude_columns: list,
    batch_size: int,
    n_splits: int = 5,
    dataset_cls=ClinicalDataset,
    dataset_kwargs: dict = None
):
    """
    Creates stratified K-fold dataloaders for training and validation using a generic dataset class.
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_columns = [col for col in train_df.columns if col not in exclude_columns]
    dataloaders = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[label_column])):
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]
        train_loader = DataLoader(
            dataset_cls(train_data, **dataset_kwargs),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset_cls(val_data, **dataset_kwargs),
            batch_size=batch_size, shuffle=False
        )
        dataloaders[fold] = {'train': train_loader, 'val': val_loader}
    return dataloaders, feature_columns
