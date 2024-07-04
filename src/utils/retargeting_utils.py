
"""
Functions to compute distance ratios between specific pairs of facial landmarks
"""

import numpy as np


def calculate_distance_ratio(lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6) -> np.ndarray:
    """
    Calculate the ratio of the distance between two pairs of landmarks.

    Parameters:
    lmk (np.ndarray): Landmarks array of shape (B, N, 2).
    idx1, idx2, idx3, idx4 (int): Indices of the landmarks.
    eps (float): Small value to avoid division by zero.

    Returns:
    np.ndarray: Calculated distance ratio.
    """
    return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
            (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))


def calc_eye_close_ratio(lmk: np.ndarray, target_eye_ratio: np.ndarray = None) -> np.ndarray:
    """
    Calculate the eye-close ratio for left and right eyes.

    Parameters:
    lmk (np.ndarray): Landmarks array of shape (B, N, 2).
    target_eye_ratio (np.ndarray, optional): Additional target eye ratio array to include.

    Returns:
    np.ndarray: Concatenated eye-close ratios.
    """
    lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12)
    righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36)
    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)


def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
    """
    Calculate the lip-close ratio.

    Parameters:
    lmk (np.ndarray): Landmarks array of shape (B, N, 2).

    Returns:
    np.ndarray: Calculated lip-close ratio.
    """
    return calculate_distance_ratio(lmk, 90, 102, 48, 66)
