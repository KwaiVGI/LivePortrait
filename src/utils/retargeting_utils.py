
"""
Functions to compute distance ratios between specific pairs of facial landmarks
"""

import numpy as np
import torch


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


def compute_eye_delta(frame_idx, input_eye_ratios, source_landmarks, portrait_wrapper, kp_source):
    input_eye_ratio = input_eye_ratios[frame_idx][0][0]
    eye_close_ratio = calc_eye_close_ratio(source_landmarks[None])
    eye_close_ratio_tensor = torch.from_numpy(eye_close_ratio).float().cuda(portrait_wrapper.device_id)
    input_eye_ratio_tensor = torch.Tensor([input_eye_ratio]).reshape(1, 1).cuda(portrait_wrapper.device_id)
    combined_eye_ratio_tensor = torch.cat([eye_close_ratio_tensor, input_eye_ratio_tensor], dim=1)
    # print(combined_eye_ratio_tensor.mean())
    eye_delta = portrait_wrapper.retarget_eye(kp_source, combined_eye_ratio_tensor)
    return eye_delta


def compute_lip_delta(frame_idx, input_lip_ratios, source_landmarks, portrait_wrapper, kp_source):
    input_lip_ratio = input_lip_ratios[frame_idx][0]
    lip_close_ratio = calc_lip_close_ratio(source_landmarks[None])
    lip_close_ratio_tensor = torch.from_numpy(lip_close_ratio).float().cuda(portrait_wrapper.device_id)
    input_lip_ratio_tensor = torch.Tensor([input_lip_ratio]).cuda(portrait_wrapper.device_id)
    combined_lip_ratio_tensor = torch.cat([lip_close_ratio_tensor, input_lip_ratio_tensor], dim=1)
    lip_delta = portrait_wrapper.retarget_lip(kp_source, combined_lip_ratio_tensor)
    return lip_delta
