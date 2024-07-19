# coding: utf-8

import torch
import numpy as np
from pykalman import KalmanFilter


def smooth(x_d_lst, shape, device, observation_variance=3e-6, process_variance=1e-5):
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst]
    x_d_stacked = np.vstack(x_d_lst_reshape)
    kf = KalmanFilter(
        initial_state_mean=x_d_stacked[0],
        n_dim_obs=x_d_stacked.shape[1],
        transition_covariance=process_variance * np.eye(x_d_stacked.shape[1]),
        observation_covariance=observation_variance * np.eye(x_d_stacked.shape[1])
    )
    smoothed_state_means, _ = kf.smooth(x_d_stacked)
    x_d_lst_smooth = [torch.tensor(state_mean.reshape(shape[-2:]), dtype=torch.float32, device=device) for state_mean in smoothed_state_means]
    return x_d_lst_smooth
