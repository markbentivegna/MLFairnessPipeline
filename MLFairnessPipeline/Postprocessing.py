from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from MLFairnessPipeline.Utilities import generate_binary_label_dataset

def Postprocessing(reweighted_data, pred, label, unprivileged_groups, privileged_groups, protected_attribute, favorable_label, unfavorable_label, threshold=0.01):
    reweighted_binary_dataset = generate_binary_label_dataset(reweighted_data, label, protected_attribute, favorable_label, unfavorable_label)
    prediction_binary_dataset = generate_binary_label_dataset(pred, label, protected_attribute, favorable_label, unfavorable_label)
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, low_class_thresh=threshold)
    ROC.fit(reweighted_binary_dataset, prediction_binary_dataset)

    return ROC.predict(prediction_binary_dataset).convert_to_dataframe()[0]
