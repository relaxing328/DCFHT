import numpy as np
import pandas as pd
import math
import os

def mask_Trap(X, num_sample):
    dimen = np.arange(0.1, 1.1, 0.1)
    step = max(1, int(1 / len(dimen) * num_sample))

    X_masked = np.copy(X).astype(float)
    for t in range(num_sample):
        j = max(math.floor((t - 1) / step) + 1, 1) - 1
        if j > len(dimen) - 1:
            j = len(dimen) - 1
        dimen_t = max(1, math.floor(dimen[j] * X.shape[1]))
        X_masked[t, dimen_t:] = np.nan

    return X_masked