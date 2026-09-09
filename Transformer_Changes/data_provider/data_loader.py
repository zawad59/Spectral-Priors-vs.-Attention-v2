import copy
import os
import numpy as np
import pandas as pd
import glob
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from data_provider.uea import (normalize_batch_ts,bandpass_filter_func)
import warnings
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from natsort import natsorted

warnings.filterwarnings("ignore")

def normalize_sid(x):
    """
    Normalize subject IDs across formats.

    Examples:
        sub-001       -> 1
        sub_001       -> 1
        subject-001   -> 1
        feature_01    -> 1
        feature_01.npy -> 1
        01            -> 1
        1.0           -> 1
        v107          -> v107
    """
    s = str(x).strip()

    # remove extension if accidentally included
    s = os.path.splitext(s)[0]

    lower = s.lower()

    # remove common prefixes
    for prefix in ["sub-", "sub_", "subject-", "subject_", "feature_", "features_"]:
        if lower.startswith(prefix):
            s = s[len(prefix):]
            break

    # convert numeric strings with leading zeros to integer string
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def parse_sid_from_feature_filename(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return normalize_sid(stem)


def read_subject_ids_from_csv(csv_path, subject_col="subj_ID"):
    """
    Reads subject IDs from train_subject_ids.csv / val_subject_ids.csv / test_subject_ids.csv.
    """
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    if subject_col not in df.columns:
        raise ValueError(
            f"Subject column '{subject_col}' not found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return set(df[subject_col].map(normalize_sid).tolist())


def get_split_subjects(args, root_path, flag):
    """
    Reads the correct subject IDs for the requested split.

    Expected folder structure:

        root_path/
            stratified_hold_outs/
                split_1/
                    train_subject_ids.csv
                    val_subject_ids.csv
                    test_subject_ids.csv
                split_2/
                ...

    You can override folder name using:
        args.stratified_splits_dir

    You can choose split using:
        args.split_id
    """
    split_id = getattr(args, "split_id", 1)
    subject_col = getattr(args, "split_subject_col", "subj_ID")

    splits_dir_name = getattr(args, "stratified_splits_dir", "stratified_hold_outs")

    splits_root = os.path.join(root_path, splits_dir_name)
    split_dir = os.path.join(splits_root, f"split_{split_id}")

    flag_lower = flag.lower()

    if flag_lower == "train":
        csv_name = "train_subject_ids.csv"
    elif flag_lower == "val":
        csv_name = "val_subject_ids.csv"
    elif flag_lower == "test":
        csv_name = "test_subject_ids.csv"
    else:
        raise ValueError(f"Unknown flag: {flag}. Expected TRAIN, VAL, or TEST.")

    csv_path = os.path.join(split_dir, csv_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find split subject file:\n{csv_path}"
        )

    return read_subject_ids_from_csv(csv_path, subject_col=subject_col)


def load_eeg_from_subject_split(
    data_path,
    label_path,
    target_subjects,
    use_filename_subject_id=True,
):
    """
    Generic EEG loader using externally provided subject IDs.

    FIXED: label lookup is now by subject ID (via a sid -> label dict),
    not by row position in label.npy. The previous version assumed
    label.npy's row order exactly matched natsorted(Feature/)'s file
    order, which does not hold here (label.npy is lexicographically
    sorted, not natsorted) -- that mismatch silently assigned ~90% of
    subjects someone else's label.
    """

    feature_list = []
    label_list = []
    subj_list = []

    subject_label = np.load(label_path, allow_pickle=True)

    # Build an explicit sid -> label lookup, order-independent.
    # Assumes label.npy columns are [label, subj_ID] (col0=label, col1=subj_ID).
    sid_to_label = {
        normalize_sid(row[1]): int(row[0]) for row in subject_label
    }

    filenames = natsorted([
        f for f in os.listdir(data_path)
        if f.endswith(".npy")
    ])

    target_subjects = set(map(normalize_sid, target_subjects))

    for filename in filenames:
        path = os.path.join(data_path, filename)

        if use_filename_subject_id:
            sid = parse_sid_from_feature_filename(filename)
        else:
            # Fallback path unchanged from before -- only used for datasets
            # that don't carry subject ID in the filename.
            sid = None  # resolved per-row below if needed

        if sid is None or sid not in sid_to_label:
            # Can't resolve a label for this file at all -- skip rather than
            # silently mis-assign (this replaces the old fallback that used
            # positional index j+1).
            continue

        if sid not in target_subjects:
            continue

        label = sid_to_label[sid]
        subject_feature = np.load(path, allow_pickle=True)

        for trial_feature in subject_feature:
            feature_list.append(trial_feature)
            label_list.append(label)
            subj_list.append(sid)

    X = np.array(feature_list)
    y = np.array(label_list, dtype=np.int64)
    subj = np.array(subj_list)

    if len(y) == 0:
        raise RuntimeError(
            "No samples were loaded for this split. "
            "This usually means subject IDs in stratified_hold_outs do not match "
            "the IDs in label.npy or the feature filenames."
        )

    X, y, subj = shuffle(X, y, subj, random_state=42)

    return X, y, subj


class APAVALoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature_Random_Bands/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[APAVALoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)

class ADHDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[ADHDLoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)

class TDBRAINLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[TDBRAINLoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)
    
    
class TDBRAIN_ADHDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature_Random_Bands/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[TDBRAINLoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)
    


class TDBRAIN_ParkinsonLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[TDBRAINLoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)
    
    
class AMPSCZLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature_Bands/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[AMPSCZLoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)

class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature_Random_Bands/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        if flag is None:
            raise ValueError("flag must be TRAIN, VAL, or TEST")

        target_subjects = get_split_subjects(
            args=args,
            root_path=root_path,
            flag=flag,
        )

        self.X, self.y, self.subj_id = load_eeg_from_subject_split(
            data_path=self.data_path,
            label_path=self.label_path,
            target_subjects=target_subjects,
            use_filename_subject_id=True,
        )

        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

        unique_subjects = sorted(set(map(str, self.subj_id.tolist())))
        self.sid2int = {s: i for i, s in enumerate(unique_subjects)}
        self.subj_id_int = np.array(
            [self.sid2int[str(s)] for s in self.subj_id],
            dtype=np.int64,
        )

        print(
            f"[ADFTDLoader] flag={flag}, "
            f"samples={len(self.y)}, "
            f"subjects={len(unique_subjects)}, "
            f"class_counts={dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(int(self.y[index]), dtype=torch.long)

        padding_mask = torch.ones(self.max_seq_len, dtype=torch.float32)

        sid = torch.tensor(int(self.subj_id_int[index]), dtype=torch.long)

        return x, y, padding_mask, sid

    def __len__(self):
        return len(self.y)

class PTBLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.55, 0.7

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptb(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids

    def load_ptb(self, data_path, label_path, flag=None):
        """
        Loads ptb data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")

        a, b = 0.6, 0.8

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_ptbxl(self.data_path, self.label_path, flag=flag)

        # pre_process
        self.X = normalize_batch_ts(self.X)
        # self.X = bandpass_filter_func(self.X, fs=250, lowcut=0.5, highcut=45)

        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(label_path)
        no_list = list(
            data_list[np.where(data_list[:, 0] == 0)][:, 1]
        )  # Normal ECG IDs
        mi_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial Infarction IDs
        sttc_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # ST/T Change IDs
        cd_list = list(
            data_list[np.where(data_list[:, 0] == 3)][:, 1]
        )  # Conduction Disturbance IDs
        hyp_list = list(
            data_list[np.where(data_list[:, 0] == 4)][:, 1]
        )  # Hypertrophy IDs

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_ptbxl(self, data_path, label_path, flag=None):
        """
        Loads ptb-xl data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        feature_list = []
        label_list = []
        filenames = []
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames = natsorted(filenames)
        if flag == "TRAIN":
            ids = self.train_ids
            # print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            # print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            # print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
            # print("all ids:", ids)

        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            for trial_feature in subject_feature:
                # load data by ids
                if j + 1 in ids:  # id starts from 1, not 0.
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        # reshape and shuffle
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(
            np.asarray(self.y[index])
        )

    def __len__(self):
        return len(self.y)



class FLAAPLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_flaap_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_flaap_dependent(self, data_path, label_path, flag=None):
        '''
        Loads fl-aap data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        # 60 : 20 : 20
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class UCIHARLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/feature.npy')
        self.label_path = os.path.join(root_path, 'Label/label.npy')

        self.X, self.y = self.load_har_dependent(self.data_path, self.label_path, flag=flag)

        # pre_process
        # self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    def load_har_dependent(self, data_path, label_path, flag=None):
        '''
        Loads fl-aap data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        '''
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        # print(X_train.shape, y_train.shape)

        X_test = X_train[-2947:]
        y_test = y_train[-2947:]

        X_train, X_val, y_train, y_val = train_test_split(X_train[:-2947], y_train[:-2947], test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        if flag == 'TRAIN':
            return X_train, y_train
        elif flag == 'VAL':
            return X_val, y_val
        elif flag == 'TEST':
            return X_test, y_test
        else:
            raise Exception('flag must be TRAIN, VAL, or TEST')

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), \
            torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)
