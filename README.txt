# DCFHT: online learning from Drifting Capricious data streams with Flexible Hoeffding Tree

## Overview
This repository contains datasets and implementation codes of the paper, titled "Online Learning from Drifting Capricious Data Streams with Flexible Hoeffding Tree".
Experiment_1 corresponds to Section 5.2; Experiment_2 corresponds to Section 5.3; Experiment_3 corresponds to Section 5.4.

## Requirements:
  numpy
  pandas
  openpyxl
  os
  skmultiflow (We have modified the original codes in https://scikit-multiflow.github.io/. And the modified library is provided as a rar file.)
  python 3.7

## Run Example
In each code folder, we provide three files for different running examples:
  "main.py" is used to run datasets without discrete features;
  "main_OVFM.py" is used to run datasets handled by OVFM algorithm
