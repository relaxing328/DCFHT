import numpy as np
import scipy.io
import openpyxl
import os
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.page_hinkley import PageHinkley
from DCFHT_Trap_discrete import OCFHT_drift_new
from mask_Trap import mask_Trap

dataset = ['kr']

for i in range(len(dataset)):
    file = 'data/Dataset_UCI/' + dataset[i] + '.mat'
    mat_data = scipy.io.loadmat(file)
    data = mat_data['data']
    ID_list = mat_data['ID_ALL']

    num = data.shape[0]
    iter = ID_list.shape[0]
    X = data[:, 1:data.shape[1]] 
    Y = data[:, 0]
    
    is_drift = True

    acc = np.zeros((1, iter))
    runtime = np.zeros((1, iter)) 

    for j in range(iter):
        ID = ID_list[[j]]-1
        ID = ID.flatten()
        X = X[ID]
        Y = Y[ID]
        X_masked = mask_Trap(X, num)

        model = OCFHT_drift_new(HoeffdingTreeClassifier(), 0, PageHinkley(delta=0.01), PageHinkley(delta=0.005), False, is_drift)
        classifier1, err_count1, correct_cnt1, runtime1, acc_all = model.OCFHT_drift_OVFM(X_masked, Y)
        acc[0, j] = correct_cnt1 / num
        runtime[0, j] = runtime1