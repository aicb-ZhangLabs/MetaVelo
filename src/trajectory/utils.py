import numpy as np


def robust_log1p(x: np.array):
    invalid_x_mask = x <= -1
    x[invalid_x_mask] = 0
    log1p_x = np.where(invalid_x_mask, 0, np.log1p(x))
    return log1p_x
