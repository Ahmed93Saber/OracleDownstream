import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Optional



class BasePatientDataset(Dataset):
    """
    Shared utilities for clinical and imaging datasets.
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def get_label_tensor(self, row) -> torch.Tensor:
        return torch.tensor(row['label-1RN-0Normal'], dtype=torch.float32)

    def get_dict_key(self, row) -> str:
        patient_id = row["Patient ID"]
        met_id = row["id"]
        scan_date = row["scan_date"].split()[0]
        return f"{patient_id}_{scan_date}_{met_id}"


class ClinicalDataset(BasePatientDataset):
    def __init__(self, dataframe: pd.DataFrame, columns_to_drop: List[str]):
        super().__init__(dataframe)
        self.columns_to_drop = columns_to_drop

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        features = row.drop(self.columns_to_drop).values.astype('float32')
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = self.get_label_tensor(row)
        return features_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.dataframe)


class ImagingDataset(BasePatientDataset):
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, is_gap: bool = False, is_img: bool = False):
        super().__init__(dataframe)
        self.embedd_type = 'embedding_attn_pool' if is_gap else 'embedding_norm'
        self.img_seq = np.load(data_dir, allow_pickle=True).item()
        self.is_gap = is_gap
        self.is_img = is_img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        dict_key = self.get_dict_key(row)

        if self.is_img:
            img_tensor = self.img_seq[dict_key][0, ...].clone().detach().float()
        else:
            img_tensor = self.img_seq[dict_key][self.embedd_type].clone().detach().float().squeeze()

        label_tensor = self.get_label_tensor(row)
        return img_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.dataframe)


class MultimodalDataset(BasePatientDataset):
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, columns_to_drop: List[str], is_gap: bool = False):
        super().__init__(dataframe)
        self.columns_to_drop = columns_to_drop
        self.embedd_type = 'embedding_attn_pool' if is_gap else 'embedding_norm'
        self.img_seq = np.load(data_dir, allow_pickle=True).item()
        self.is_gap = is_gap

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        row = self.dataframe.iloc[index]
        dict_key = self.get_dict_key(row)

        # Clinical features
        clinical_data = row.drop(self.columns_to_drop).values.astype('float32')
        clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32)

        # Imaging features
        if self.is_gap:
            img_tensor = self.img_seq[dict_key][0, ...].clone().detach().float()
        else:
            img_tensor = self.img_seq[dict_key][self.embedd_type].clone().detach().float().squeeze()

        label_tensor = self.get_label_tensor(row)
        return (clinical_tensor, img_tensor), label_tensor

    def __len__(self) -> int:
        return len(self.dataframe)



def create_dataloaders(
    train_df: pd.DataFrame,
    label_column: str,
    exclude_columns: list,
    batch_size: int,
    n_splits: int = 5,
    dataset_cls=ClinicalDataset,
    dataset_kwargs: dict = None,
    random_state: int = 42
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
