import sys
sys.path.insert(1, "../")  

import urllib
import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(0)

from aif360.datasets import GermanDataset, CompasDataset
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas, load_preproc_data_adult
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from Pipeline.Preprocessing import preprocessing
from Pipeline.Inprocessing import inprocessing
from Pipeline.Postprocessing import postprocessing
from Utilities.generate_results import record_results
from Utilities.print_metrics import print_binary_label_metrics, print_classification_metrics

from sklearn.metrics import accuracy_score
import tensorflow as tf

def fairness_executor(dataset_orig, unprivileged_groups, privileged_groups, dataset_type):
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    dataset_transf_train, dataset_transf_test = preprocessing(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups)

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    undebiased_train_pred, undebiased_test_pred = inprocessing(dataset_transf_train, dataset_transf_test, unprivileged_groups, privileged_groups, 'plain_classifier', False)

    print_binary_label_metrics(undebiased_train_pred, undebiased_test_pred, unprivileged_groups, privileged_groups)

    print_classification_metrics(dataset_transf_train, undebiased_train_pred, unprivileged_groups, privileged_groups)
    record_results("Undebiased Training " + dataset_type, dataset_transf_train, undebiased_train_pred, unprivileged_groups, privileged_groups, "FAIR", dataset_type)

    print_classification_metrics(dataset_transf_test, undebiased_test_pred, unprivileged_groups, privileged_groups)
    record_results("Undebiased Test " + dataset_type, dataset_transf_test, undebiased_test_pred, unprivileged_groups, privileged_groups, "FAIR", dataset_type)

    debiased_train_pred, debiased_test_pred = inprocessing(dataset_transf_train, dataset_transf_test, unprivileged_groups, privileged_groups, 'debiased_classifier', True)

    print_binary_label_metrics(debiased_train_pred, debiased_test_pred, unprivileged_groups, privileged_groups)

    print_classification_metrics(dataset_transf_train, debiased_train_pred, unprivileged_groups, privileged_groups)
    record_results("Debiased Training " + dataset_type, dataset_transf_train, debiased_train_pred, unprivileged_groups, privileged_groups, "FAIR", dataset_type)

    print_classification_metrics(dataset_transf_test, debiased_test_pred, unprivileged_groups, privileged_groups)
    record_results("Debiased Test " + dataset_type, dataset_transf_test, debiased_test_pred, unprivileged_groups, privileged_groups, "FAIR", dataset_type)

    roc_train_pred, roc_test_pred = postprocessing(dataset_transf_train, debiased_train_pred, debiased_test_pred, unprivileged_groups, privileged_groups)

    print_classification_metrics(dataset_transf_train, roc_train_pred, unprivileged_groups, privileged_groups)
    record_results("ROC Training " + dataset_type, dataset_transf_train, roc_train_pred, unprivileged_groups, privileged_groups, "FAIR", dataset_type)

    print_classification_metrics(dataset_transf_test, roc_test_pred, unprivileged_groups, privileged_groups)
    record_results("ROC Test " + dataset_type, dataset_transf_test, roc_test_pred, unprivileged_groups, privileged_groups, "FAIR", dataset_type)
