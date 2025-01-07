import numpy as np
import scipy.io
import os
import openpyxl
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection.page_hinkley import PageHinkley
from helpers import mask_types_old
from random_perm_n import random_perm_n_old
from DCFHT_FESL_discrete import OCFHT_drift_new
from mask_FESL import mask_FESL

dataset = ['kr']

for i in range(len(dataset)):
    data_file = 'data/Dataset_UCI/' + dataset[i] + '.mat'
    mat_data = scipy.io.loadmat(data_file)
    data = mat_data['data']

    ID_route = '/Experiment_3/FESL/data_UCI/' + dataset[i] + '.mat'
    ID_mat = scipy.io.loadmat(ID_route)
    ID_list = ID_mat['index_new']
    ID_list = ID_list.flatten()

    num_sample = data.shape[0]
    iter = ID_list.shape[0]
    X = data[:, 1:data.shape[1]]
    Y = data[:, 0]
    num_fea = X.shape[1]

    is_drift = True

    acc = np.zeros((1, iter)) 
    runtime = np.zeros((1, iter))

    for j in range(iter):
        ID = ID_list[j]-1
        ID = ID.flatten()
        X = X[ID]
        Y = Y[ID]
        X_masked = mask_FESL(X, num_sample, num_fea)

        model = OCFHT_drift_new(HoeffdingTreeClassifier(), 0, PageHinkley(delta=0.01), PageHinkley(delta=0.005), False, is_drift)
        classifier1, err_count1, correct_cnt1, runtime1, acc_all = model.OCFHT_drift_OVFM(X_masked, Y)
        acc[0, j] = correct_cnt1 / num_sample
        runtime[0, j] = runtime1