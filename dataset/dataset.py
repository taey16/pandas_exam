from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torchvision.datasets.vision import VisionDataset


class NumpyDataset(VisionDataset):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(root=None)

        self.data = torch.from_numpy(data[:, :-1])
        self.target = torch.from_numpy(data[:, -1])
        # NOTE: target values seems not in a set of [0, 1].
        self.target = torch.where(
            self.target > 0.5,
            torch.ones_like(self.target),
            torch.zeros_like(self.target)
        )

        assert len(self.data) == len(self.target), \
            "len(self.data) and  len(self.target) is mismatched"

        print(f"NumpyDataset: len(self.data): {len(self.data)}")
        print(f"NumpyDataset: len(self.target): {len(self.target)}")

    def __len__(self) -> int:
        return len(self.data) 

    def __getitem__(self, idx: int) -> Tuple:
        features = self.data[idx].float()
        features = (features - 0.5) * 2.0
        targets = self.target[idx]
        one_hot = torch.nn.functional.one_hot(
            targets.long(), num_classes=2
        ).float()
        return features, one_hot


def get_loader(
    dataset: VisionDataset,
    batch_size: int,
    mode: str,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        shuffle=True if mode == "train" else False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    assert loader is not None, "Check get_loader()"

    return loader


def build_dataloader(
    data: np.ndarray,
    batch_size: int,
    mode: str,
) -> torch.utils.data.DataLoader: 
    dataset = NumpyDataset(data)
    loader = get_loader(dataset, batch_size=batch_size, mode=mode)
    return loader

def train_val_split(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    num_train_samples = len(dataframe)
    train_val_split_idx = int(num_train_samples * 0.9)

    val_data = dataframe.iloc[train_val_split_idx:]
    train_data = dataframe.iloc[0:train_val_split_idx]

    return train_data, val_data


def read_csv(csv_filename: str) -> pd.DataFrame:
    return pd.read_csv(csv_filename)


def print_dtype_min_max(data: pd.DataFrame) -> None:
    for col_name in data.keys():
        col_value = data[col_name]
        print_str = \
            f"{col_name}: "\
            f"dtype: {col_value.dtype}, "\
            f"min: {col_value.min()}, "\
            f"max: {col_value.max()}"
        print(print_str)
    return None


def normalize_data(data: np.ndarray, method="minmax") -> np.ndarray:
    if method == "minmax":
        min_value = data.min(axis=0)
        max_value = data.max(axis=0)
        data = (data - min_value) / (max_value - min_value + 1e-12)
    else:
        assert 0, f"check method: {method}"
    return data


def check_data(data: np.ndarray) -> None:
    assert not np.all(np.isnan(data)), \
        f"np.all(np.isnan(data)): {np.all(np.isnan(data))}"
    assert not np.all(np.isinf(data)), \
        f"np.all(np.isinf(data)): {np.all(np.isnan(data))}"
    return


