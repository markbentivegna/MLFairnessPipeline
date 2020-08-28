from aif360.datasets.binary_label_dataset import BinaryLabelDataset

def generate_binary_label_dataset(df, label, protected_attribute, favorable_label, unfavorable_label):
    return BinaryLabelDataset(unfavorable_label, favorable_label, df=df, label_names=[label], protected_attribute_names=[protected_attribute])
