import sys
import os
sys.path.insert(1, "../")  

import urllib
import numpy as np
import pandas as pd
from aif360.datasets import GermanDataset, CompasDataset, BankDataset
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas, load_preproc_data_adult, load_preproc_data_german
from Control.control_runner import control_runner
from FairnessPipeline import fairness_executor
np.random.seed(0)

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

dataset_orig = load_preproc_data_compas()

fairness_executor(dataset_orig, unprivileged_groups, privileged_groups, "compas")
control_runner(dataset_orig, unprivileged_groups, privileged_groups, "compas", 'race')

dataset_orig = load_preproc_data_adult()

fairness_executor(dataset_orig, unprivileged_groups, privileged_groups, "adult")
control_runner(dataset_orig, unprivileged_groups, privileged_groups, "adult", 'race')
