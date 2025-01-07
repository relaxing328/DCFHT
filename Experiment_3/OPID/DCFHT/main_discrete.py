import numpy as np
import pandas as pd
import scipy.io
import openpyxl
import os
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.page_hinkley import PageHinkley
from DCFHT_OPID_discrete import OCFHT_drift_new
from mask_OPID import mask_OPID

dataset = ['kr']

for i in range(len(dataset)):
    file = '/Experiment_3/OPID/data_UCI/' + dataset[i] + '.mat'
    mat_data = scipy.io.loadmat(file)
    data = mat_data['data']
    ID_list = mat_data['ID_ALL']
    num_C = mat_data['num_C']
    num_E = data.shape[0] - num_C
    num_vanish = mat_data['num_vanish']
    num_survive = mat_data['num_survive']
    e_test_num = mat_data['e_test_num']

    iter = ID_list.shape[0]
    X = data[:, 1:data.shape[1]]
    Y = data[:, 0]
    num = data.shape[0]

    is_drift = True

    acc = np.zeros((1, iter))
    acc_test = np.zeros((1, iter))
    runtime = np.zeros((1, iter))

    for j in range(iter):
        ID = ID_list[[j]]-1
        ID = ID.flatten()
        X = X[ID]
        Y = Y[ID]
        X_masked = mask_OPID(X, num, num_C[0,0].astype(int), num_vanish[0,0].astype(int), num_survive[0,0].astype(int))

        model = OCFHT_drift_new(HoeffdingTreeClassifier(), 0, PageHinkley(delta=0.01), PageHinkley(delta=0.005), False, is_drift,
                                num_C[0,0].astype(int), num_E[0,0].astype(int))
        classifier1, err_count1, correct_cnt1, correct_cnt_E1, runtime1, acc_all = model.OCFHT_drift_OVFM(X_masked, Y)
        acc[0, j] = correct_cnt1 / num
        acc_test[0, j] = correct_cnt_E1 / e_test_num
        runtime[0, j] = runtime1