import numpy as np
import torch

def dropout_mask(score, d_rate, inv):
    _size = score.shape[0]

    if d_rate == 1.0:
        result_mask = torch.zeros(_size)
    elif d_rate == 0.0:
        result_mask = torch.ones(_size)
    else:
        _n_zero_units = int(_size * d_rate)  # the number of units to be zeroed
        _sort_mi_index = np.argsort(score)
        if inv:
            _drop_index = _sort_mi_index[-_n_zero_units:]
        else:
            _drop_index = _sort_mi_index[0:_n_zero_units]
        result_mask = torch.ones(_size)
        result_mask[_drop_index] = 0.0

    return result_mask



