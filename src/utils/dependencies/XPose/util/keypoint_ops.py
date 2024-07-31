import torch, os

def keypoint_xyxyzz_to_xyzxyz(keypoints: torch.Tensor):
    """_summary_

    Args:
        keypoints (torch.Tensor): ..., 51
    """
    res = torch.zeros_like(keypoints)
    num_points = keypoints.shape[-1] // 3
    Z = keypoints[..., :2*num_points]
    V = keypoints[..., 2*num_points:]
    res[...,0::3] = Z[..., 0::2]
    res[...,1::3] = Z[..., 1::2]
    res[...,2::3] = V[...]
    return res

def keypoint_xyzxyz_to_xyxyzz(keypoints: torch.Tensor):
    """_summary_

    Args:
        keypoints (torch.Tensor): ..., 51
    """
    res = torch.zeros_like(keypoints)
    num_points = keypoints.shape[-1] // 3
    res[...,0:2*num_points:2] = keypoints[..., 0::3]
    res[...,1:2*num_points:2] = keypoints[..., 1::3]
    res[...,2*num_points:] = keypoints[..., 2::3]
    return res