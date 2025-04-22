# Dataset and Label
Datasets include: SelfDeploy24 and SelfDeploy25. Corresponding labels are stored in the "label" folder.


# Scannergrouper running guide

python environment:python3.9,torch2.4.1,cuda11.8 ("py39.yml" for details)

## Scannergrouper-i running step:

### 1. Scanner level identification Evaluation 

scannergrouper/scannergrouper-i/ensemble_learning.ipynb

### 2.Identification Result Analysis

scannergrouper/scannergrouper-i/cluster/

#### divide probes according to service type

python divide_csv_proto.py

#### visualize indentified probe features

cluster_visualization.ipynb

#### generate report for still-unknown probes

stat_cluster_new.ipynb



### (Optional) Generating intermediate results: packet-level identification with IOMATCH

#### IOMATCH TRAINNING:
scannergrouper/scannergrouper-i/train_selfdeploy24.py
scannergrouper/scannergrouper-i/train_selfdeploy25.py
#### Generate IOMATCH identification result:

**generating packet-level identification result for training set and testing test:**

scannergrouper/scannergrouper-i/IOMatch-main_per_service_total_sample_incre/evaluate_selfdeploy_train.ipynb
scannergrouper/scannergrouper-i/IOMatch-main_per_service_total_sample_incre/evaluate_selfdeploy_test.ipynb

**generating 128 dimension feature for each probe:**

scannergrouper/scannergrouper-i/IOMatch-main_per_service_total_sample_incre/evaluate_selfdeploy_test_proj.ipynb



## Scannergrouper-f running step:

### 1.Result Aggregation Module

scannergrouper/scannergrouper-f/ensemble_learning.ipynb

### 2.Identification Result Analysis

scannergrouper/scannergrouper-f/cluster/
The rest running steps of files in 'cluster/' is the same as scannergrouper-i.



### (Optional) Generating intermediate results: packet-level identification with IOMATCH

#### IOMATCH TRAINNING:

scannergrouper/scannergrouper-f/IOMatch-main_per_service_total_sample/run.sh

#### Generate IOMATCH identification result:

**generating packet-level identification result for training set and testing test:**

scannergrouper/scannergrouper-f/IOMatch-main_per_service_total_sample/evaluate_selfdeploy_train.ipynb
scannergrouper/scannergrouper-f/IOMatch-main_per_service_total_sample/evaluate_selfdeploy_test.ipynb

**generating 128 dimension feature for each probe:**

scannergrouper/scannergrouper-f/IOMatch-main_per_service_total_sample/evaluate_selfdeploy_test_proj.ipynb