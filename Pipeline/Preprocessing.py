from aif360.algorithms.preprocessing import Reweighing
from Utilities.generate_binary_dataset import generate_binary_label_dataset

def Preprocessing(dataset, label, unprivileged_groups, privileged_groups, protected_attribute, favorable_label, unfavorable_label):
    binary_dataset = generate_binary_label_dataset(dataset, label, protected_attribute, favorable_label, unfavorable_label)
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transformed = RW.fit_transform(binary_dataset)
    return dataset_transformed.convert_to_dataframe()[0]
