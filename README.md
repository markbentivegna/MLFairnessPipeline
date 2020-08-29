# MLFairnessPipeline

To combat the implicit bias in Machine Learning, MLFairnessPipeline allows data scientists to gain insight into how their models perform from an equality perspective. It provides fairness metrics and provides added context against control groups without any fairness smoothing.

### To install using Pip:

pip install MLFairnessPipeline

### Example code snippet:
```
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
protected_attribute = 'race'
favorable_label = 1.0
unfavorable_label = 0.0
dataset_original = shuffle(load_preproc_data_compas().convert_to_dataframe()[0]).reset_index()
label = 'two_year_recid'

reweighted_df = MLFairnessPipeline.Preprocessing(dataset_original, label, unprivileged_groups, privileged_groups, protected_attribute, favorable_label, unfavorable_label)
row_count = reweighted_df.shape[0]
train_test_split_index = math.floor(0.7*row_count)

train_reweighted_df = reweighted_df[:train_test_split_index]
test_reweighted_df = reweighted_df[train_test_split_index:]

classifier = MLFairnessPipeline.Fair_Model(unprivileged_groups, privileged_groups, label, protected_attribute, favorable_label, unfavorable_label)
classifier.fit(train_reweighted_df)
train_pred = classifier.predict(train_reweighted_df) 
test_pred = classifier.predict(test_reweighted_df)

train_fair_pred = MLFairnessPipeline.Postprocessing(train_reweighted_df, train_pred, label, unprivileged_groups, privileged_groups, protected_attribute, favorable_label, unfavorable_label)
test_fair_pred = MLFairnessPipeline.Postprocessing(test_reweighted_df, test_pred, label, unprivileged_groups, privileged_groups, protected_attribute, favorable_label, unfavorable_label)
```

### Contributors:

##### Mark Bentivegna
##### Srishti Karakoti
##### Aman Kumar Sinha
##### Pantelis Monogioudis