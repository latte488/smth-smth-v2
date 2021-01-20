import numpy as np
import torch

def dropout_mask(score, d_rate, mode):
    _size = score.shape[0]

    if d_rate == 1.0:
        result_mask = torch.zeros(_size)
    elif d_rate == 0.0:
        result_mask = torch.ones(_size)
    else:
        _n_zero_units = int(_size * d_rate)  # the number of units to be zeroed
        _sort_mi_index = np.argsort(score)

        if mode == 'low':
            _drop_index = _sort_mi_index[-_n_zero_units:]
        elif mode == 'middle':
            front = _n_zero_units // 2
            back = front + _n_zero_units % 2
            _drop_index = np.hstack([_sort_mi_index[0:front], _sort_mi_index[-back:-1]])
        elif mode == 'high':
            _drop_index = _sort_mi_index[:_n_zero_units]
        elif mode == 'sparse':
            low_i = 0
            middle_i = _size // 3
            high_i = middle_i * 2
            _drop_index = []
            if _size % 3 == 2:
                middle_i += 1
                high_i += 2
            elif _size % 3 == 1:
                middle_i += 1
                high_i += 1
            for i in range(_n_zero_units):
                if i % 3 == 0:
                    _drop_index.append(_sort_mi_index[low_i])
                    low_i += 1
                elif i % 3 == 1:
                    _drop_index.append(_sort_mi_index[middle_i])
                    middle_i += 1
                else:
                    _drop_index.append(_sort_mi_index[high_i])
                    high_i += 1
        elif mode[0:6] == 'random':
            random_score = np.random.permutation(np.arange(_size))
            _sort_mi_index = np.argsort(random_score)
            _drop_index = _sort_mi_index[0:_n_zero_units]
        elif mode[:3] == 'low':
            number = int(mode[3:])
            _drop_index = _sort_mi_index[-number:]
        elif mode[:4] == 'high':
            number = int(mode[4:])
            _drop_index = _sort_mi_index[:number]
        elif mode[:6] == 'number':
            _drop_index = [int(mode[6:])]
        else:
            raise Exception('This mode is unknown. Must be selected from [low, middle, high].')

        print(f'len(_drop_index) : {len(_drop_index)}')
        result_mask = torch.ones(_size)
        result_mask[_drop_index] = 0.0

    return result_mask



