import os
import numpy as np
import pandas as pd
import torch
from itertools import repeat
from scipy.signal import butter, lfilter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def collate_fn(data, max_len=None):
    """
    Minimal TeCh-compatible collate_fn.

    Supports:
      (x, y)
      (x, y, subject_id)
      (x, y, padding_mask, subject_id)

    Returns:
      X, targets, padding_masks
    or:
      X, targets, padding_masks, subject_ids
    """

    batch_size = len(data)

    if batch_size == 0:
        raise ValueError("Empty batch received in collate_fn.")

    n_items = len(data[0])

    if n_items == 2:
        features, labels = zip(*data)
        subject_ids = None

    elif n_items == 3:
        features, labels, subject_ids = zip(*data)

    elif n_items == 4:
        features, labels, _, subject_ids = zip(*data)

    else:
        raise ValueError(f"Unexpected dataset item length: {n_items}")

    # Convert features to tensors
    features = [
        x if torch.is_tensor(x) else torch.as_tensor(x)
        for x in features
    ]
    features = [x.float() for x in features]

    # Keep TeCh logic: assume input already shaped as (T, C), e.g. (256, 19)
    features = [
        x.unsqueeze(-1) if x.dim() == 1 else x
        for x in features
    ]

    lengths = [X.shape[0] for X in features]

    if max_len is None:
        max_len = max(lengths)

    feat_dim = features[0].shape[-1]

    X = torch.zeros(batch_size, max_len, feat_dim)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack([
        y if torch.is_tensor(y) else torch.as_tensor(y)
        for y in labels
    ], dim=0).long().reshape(-1)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16),
        max_len=max_len
    )

    if subject_ids is not None:
        subject_ids = torch.stack([
            sid if torch.is_tensor(sid) else torch.as_tensor(sid)
            for sid in subject_ids
        ], dim=0).long().reshape(-1)

        return X, targets, padding_masks, subject_ids

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)  # convert to same type as lengths tensor
        .repeat(batch_size, 1)  # (batch_size, max_len)
        .lt(lengths.unsqueeze(1))
    )


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(
        self,
        norm_type="standardization",
        mean=None,
        std=None,
        min_val=None,
        max_val=None,
    ):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


def bandpass_filter_func(signal, fs, lowcut, highcut):
    # length of signal
    fft_len = signal.shape[1]
    # FFT
    fft_spectrum = np.fft.rfft(signal, n=fft_len, axis=1)
    # get frequency bins
    freqs = np.fft.rfftfreq(fft_len, d=1/fs)
    # create mask for freqs
    mask = (freqs >= lowcut) & (freqs <= highcut)
    # expand mask to match fft_spectrum dimensions
    mask = mask[:, np.newaxis]  # Adjust mask shape if necessary
    # apply mask
    fft_spectrum = fft_spectrum * mask
    # IFFT
    filtered_signal = np.fft.irfft(fft_spectrum, n=fft_len, axis=1)

    return filtered_signal


def normalize_batch_ts(batch):
    """Normalize a batch of time-series data.

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (N, T, C).

    Returns:
        numpy.ndarray: A batch of processed time-series, normalized for each channel of each sample.
    """
    # Calculate mean and std for each sample's each channel
    mean_values = batch.mean(axis=1, keepdims=True)  # Shape: (N, 1, C)
    std_values = batch.std(axis=1, keepdims=True)  # Shape: (N, 1, C)

    # Avoid division by zero by setting small std to 1
    std_values[std_values == 0] = 1.0

    # Perform standard normalization
    normalized_batch = (batch - mean_values) / std_values

    return normalized_batch