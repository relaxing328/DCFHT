import numpy as np
import pandas as pd
import math
import os

def mask_FESL(X, num_sample, num_fea):
    T1 = np.floor(num_sample/5).astype(int)
    T2 = num_sample - T1
    fea_T1 = np.round(num_fea*2/3).astype(int)
    # fea_T2 = num_fea - fea_T1
    B = 30

    X_masked = np.copy(X).astype(float)
    for t in range(T1-B):
        X_masked[t, fea_T1:] = np.nan
    for t in range(T2, num_sample):
        X_masked[t, :fea_T1] = np.nan

    return X_masked