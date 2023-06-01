from typing import Tuple, List, Dict

import os
import copy
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

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


def collate_fn(batch):

    features = [d[0][None] for d in batch]
    labels = [d[1][None] for d in batch]

    try:
        features = torch.cat(features)
        labels = torch.cat(labels)
    except:
        import pdb; pdb.set_trace()
        dmy = 1

    return features, labels


class DictDataset(VisionDataset):
    def __init__(self, data: Dict, mode: str) -> None:
        super().__init__(root=None)

        self.data = dict()
        for k in data.keys():
            _data = data[k]
            # In the provided dataset, there are samples whose seq_length are not 10, so
            # we ignore them. In sum, we consider a equal size of seq_length only for convenience.
            if len(_data) == 10:
                self.data[k] = _data
            
        # Sampling is conducted by the key values
        self.keys = list(self.data.keys())

        # Store meta. for using in later (See. main.run_train())
        if mode == "test":
            # Test dataset does not contain the ground-truth at the end of the feature.
            self.feature_dim = len(self.data[self.keys[0]][0])
        else:
            self.feature_dim = len(self.data[self.keys[0]][0]) - 1 
        self.seq_length = len(self.data[self.keys[0]])

        print(f"DictDataset: len(self.keys): {len(self.keys)}")
        print(f"DictDataset: feature dim: {self.feature_dim}")
        print(f"DictDataset: seq_length: {self.seq_length}")

    def __len__(self) -> int:
        return len(self.keys) 

    def __getitem__(self, key_idx: int) -> Tuple:
        data = torch.from_numpy(
            np.array(self.data[self.keys[key_idx]])
        ).float()
        target = data[:,-1]
        features = data[:,0:-1]
        features = (features - 0.5) * 2.0
        target = torch.where(target > 0.5, torch.ones_like(target), torch.zeros_like(target))
        one_hot = torch.nn.functional.one_hot(
            target.long(), num_classes=2
        ).float()
        return features, one_hot


def get_loader(
    dataset: VisionDataset,
    batch_size: int,
    mode: str,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=None,
        shuffle=True if mode == "train" else False,
        pin_memory=True,
        drop_last=True if mode == "train" else False,
        num_workers=num_workers,
    )

    assert loader is not None, "Check get_loader()"

    return loader


def build_dataloader(
    data: np.ndarray,
    batch_size: int,
    mode: str,
) -> torch.utils.data.DataLoader: 
    dataset = DictDataset(data, mode=mode)
    loader = get_loader(dataset, batch_size=batch_size, mode=mode)
    return loader


def train_val_split(
    dataframe: pd.DataFrame,
    split_ratio: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    num_train_samples = len(dataframe)
    train_val_split_idx = int(num_train_samples * split_ratio)

    val_data = dataframe.iloc[train_val_split_idx:]
    train_data = dataframe.iloc[0:train_val_split_idx]

    return train_data, val_data


def train_val_split_by_key(train_data: Dict, split_ratio: float = 0.1) -> Tuple[Dict, Dict]:

    dump_filename = f"dump_train.pkl"
    if os.path.exists(dump_filename):
        print(f"Loading dump_filename: {dump_filename}")
        with open(dump_filename, "rb") as fp:
            train_data = pickle.load(fp)
        dump_filename = f"dump_val.pkl"
        with open(dump_filename, "rb") as fp:
            val_data = pickle.load(fp)
        return train_data, val_data

    total_keys = len(train_data.keys())
    train_val_split_idx = int(total_keys * split_ratio)
    val_data = dict()
    _train_data = copy.deepcopy(train_data)
    for idx, key in enumerate(
        tqdm(train_data.keys(), total=len(train_data.keys()))
    ):
        if idx < train_val_split_idx:
            val_data[key] = _train_data.pop(key)

    dump_filename = f"dump_train.pkl"
    print(f"Dumping dump_filename: {dump_filename}")
    with open(dump_filename, "wb") as fp:
        pickle.dump(train_data, fp)
    dump_filename = f"dump_val.pkl"
    print(f"Dumping dump_filename: {dump_filename}")
    with open(dump_filename, "wb") as fp:
        pickle.dump(val_data, fp)

    return train_data, val_data


def read_csv(csv_filename: str) -> pd.DataFrame:
    return pd.read_csv(csv_filename)


def print_data_info(data: pd.DataFrame) -> Dict:
    min_max_dict = dict()
    drop_column_name = []
    target_value = data["blocked"]
    for col_name in data.keys():
        col_value = data[col_name]

        col_value_min = col_value.min()
        col_value_max = col_value.max()
        col_value_uniq = np.unique(col_value)
        col_value_uniq_len = len(np.unique(col_value))
        if col_value.dtype != object and col_value_uniq_len > 1:
            corr = np.corrcoef(col_value, target_value)[0,1]
        else: corr = 0.0

        if col_value_uniq_len == 1 or np.abs(corr) < 0.3:
            drop_column_name.append(col_name)

        print_str = \
            f"{col_name}: "\
            f"dtype: {col_value.dtype}, "\
            f"min: {col_value_min}, "\
            f"max: {col_value_max} "\
            f"len(uniq): {col_value_uniq_len} "\
            f"corr: {corr:.6f} "
            #f"uniq: {unique_value} "
        print(print_str)

        if col_value.dtype != object:
            min_max_dict[col_name] = [
                col_value_min,
                col_value_max,
                col_value_max - col_value_min,
                corr,
            ]

    print(f"drop_column_name: {drop_column_name}")

    return min_max_dict, drop_column_name


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


def group_by_newID(
    data: pd.DataFrame,
    min_max_dict: Dict,
    drop_label_name=["logging_timestamp", "newID"],
    mode: str = "train",
) -> Dict[str, List]:

    dump_filename = f"dump_{mode}_normalized.pkl"
    if os.path.exists(dump_filename):
        print(f"Loading dump_filename: {dump_filename}")
        with open(dump_filename, "rb") as fp:
            samples_per_id = pickle.load(fp)
        return samples_per_id

    samples_per_id = dict()
    for row_data in tqdm(data.iloc, total=data.shape[0]):
        newID = row_data.newID
        row_data = row_data.drop(labels=drop_label_name)
        for col_name in row_data.keys():
            if col_name == "blocked": continue
            row_data[col_name] = \
                (row_data[col_name] - min_max_dict[col_name][0]) / (min_max_dict[col_name][2] + 1e-10)
        row_data = row_data.to_list()
        if newID in samples_per_id.keys():
            samples_per_id[newID].append(row_data)
        else:
            samples_per_id[newID] = []
            samples_per_id[newID].append(row_data)

    with open(dump_filename, "wb") as fp:
        pickle.dump(samples_per_id, fp)
    print(f"Complete Dumping in {dump_filename}")

    return samples_per_id


def shuffle_by_key(data: Dict):
    data = {k:data[k] for k in random.sample(list(data.keys()), len(data))}
    return data
