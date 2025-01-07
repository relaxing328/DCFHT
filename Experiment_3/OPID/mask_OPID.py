import numpy as np
import pandas as pd
import math
import os

def mask_OPID(X, num_sample, num_C, num_vanish, num_survive):
    X_masked = np.copy(X).astype(float)

    for t in range(num_C):
        X_masked[t, (num_vanish + num_survive):] = np.nan

    for t in range(num_C, num_sample):
        X_masked[t, :num_vanish] = np.nan

    return X_masked