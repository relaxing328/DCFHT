## DCFHT: online learning from Drifting Capricious data streams with Flexible Hoeffding Tree

## Overview
This repository contains datasets and implementation codes of the paper, titled "Online Learning from Drifting Capricious Data Streams with Flexible Hoeffding Tree".
Experiment_1 corresponds to Section 5.2; Experiment_2 corresponds to Section 5.3; Experiment_3 corresponds to Section 5.4.

## Requirements:
  numpy
  pandas
  openpyxl
  os
  skmultiflow (We have modified the original codes provided in https://scikit-multiflow.github.io/. And the modified library is provided.)
  python 3.7

## Run Example
In each code folder, we provide three files for different running examples:
  1."main.py" is used to run datasets without discrete features;
  2."main_OVFM.py" is used to run datasets handled by OVFM, which are provided in the folders "data/Dataset_UCI/DataLabel" and "data/Dataset_UCI/MaskData".
  3."main_discrete.py" is used to run datasets with discrete features.
  The tested dataset can be changed at the beginning of the files. They can be run without any processing.
