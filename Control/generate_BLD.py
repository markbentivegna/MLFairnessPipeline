import numpy as np
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from Utilities.print_metrics import print_classification_metrics
from Utilities.generate_results import record_results

def generateBLD(X, y, pred, copy, label, unprivileged_groups, privileged_groups, protected_attribute):
    df = X
    df[label] = np.array(y)
    df[protected_attribute] = copy[protected_attribute]
    binary_dataset = BinaryLabelDataset(0.0, 1.0,df=df, label_names=[label], protected_attribute_names=['race'])

    pred_df = X
    pred_df[label] = pred
    pred_df[protected_attribute] = copy[protected_attribute]
    pred_binary_dataset = BinaryLabelDataset(0.0, 1.0,df=pred_df, label_names=[label], protected_attribute_names=['race'])

    return binary_dataset, pred_binary_dataset