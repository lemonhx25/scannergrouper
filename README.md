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


## Scannergrouper-f running step:

### 1.Scanner level identification Evaluation
scannergrouper/scannergrouper-f/ensemble_learning.ipynb

### 2.Identification Result Analysis
scannergrouper/scannergrouper-f/cluster/
The rest running steps of files in 'cluster/' is the same as scannergrouper-i.



# Reimplementation of Kallitsis’s framework

**This code is reimplementation accroding to the following paper:**
Michalis Kallitsis,Rupesh Prajapati,Vasant Honavar,Dinghao Wu,and John Yen. Detecting and Interpreting Changes in Scanning Behavior in Large Network Telescopes. IEEE TIFS, 17:3611–3625, 2022.

## Evaluate scanner level identification
python Optuna.py --dataset_name SelfDeploy24;


python Optuna.py --dataset_name SelfDeploy25

