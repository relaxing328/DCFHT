## DCFHT: online learning from Drifting Capricious data streams with Flexible Hoeffding Tree

## Overview
This repository contains datasets and implementation codes of the paper, titled "Online Learning from Drifting Capricious Data Streams with Flexible Hoeffding Tree".

## Requirements:
  numpy
  pandas
  openpyxl
  os
  skmultiflow (We have modified the original codes provided in https://scikit-multiflow.github.io/. And the modified library is provided.)
  python 3.7

## Parameters
drift threshold b: it can be adjusted in Line 256 in DCFHT.py.
disappearance threshold s_{max}: it can be adjusted in Line 80 in findAttr.py.

## Explanation of Folders
Experiment_1 corresponds to Section 5.2;
Experiment_2 corresponds to Section 5.3;
Experiment_3 corresponds to Section 5.4, including OLSF, FESL, OPID and the corresponding experimental data;
data stores all the experimental datasets.

## Run Example
In each code folder, we provide three files for different running examples:
  1."main.py" is used to run datasets without discrete features;
  2."main_OVFM.py" is used to run datasets handled by OVFM, which are pro-vided in the folders "data/Dataset_UCI/DataLabel" and "da-ta/Dataset_UCI/MaskData".
  3."main_discrete.py" is used to run datasets with discrete features.
  The tested dataset can be changed at the beginning of the files. They can be run without any processing.
