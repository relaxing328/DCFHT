import numpy as np
import pandas as pd
import openpyxl
import os
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection.adwin import ADWIN
from helpers import mask_types_old
from random_perm_n import random_perm_n_old
from DCFHT import OCFHT_drift_new

dataset = ['diabetes']

for i in range(len(dataset)):
    X = pd.read_csv("data/dataset_UCI/MaskData/" + dataset[i] + "/X_process.txt", sep=" ", header=None)
    Y_label = pd.read_csv("data/dataset_UCI/Datalabel/" + dataset[i] + "/Y_label.txt", sep=' ', header=None)
    MASK_NUM = 1
    X_masked = mask_types_old(X, MASK_NUM, seed=1)  # arbitrary setting Nan；seed=1
    # X = X.values
    Y_label = Y_label.values

    n = X_masked.shape[0]
    feat = X_masked.shape[1]
    Y_label = Y_label.flatten()

    is_drift = True

    acc = np.zeros((1, 10))
    runtime = np.zeros((1, 10))
   
    permutations = random_perm_n_old(n)
    for j in range(10):
        perm = permutations[j]
        Y = Y_label[perm]
        X = X_masked[perm]
        Y = Y.astype(float)

        model = OCFHT_drift_new(HoeffdingTreeClassifier(), 0, ADWIN(0.01), ADWIN(0.01), False, is_drift)
        classifier1, err_count1, correct_cnt1, runtime1, acc_all = model.OCFHT_drift_OVFM(X, Y)
        acc[0, j] = correct_cnt1 / n
        runtime[0, j] = runtime1