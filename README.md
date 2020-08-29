# MLFairnessPipeline

To combat the implicit bias in Machine Learning, MLFairnessPipeline allows data scientists to gain insight into how their models perform from an equality perspective. It provides fairness metrics and provides added context against control groups without any fairness smoothing.

 MLFairnessPipeline is an end-to-end machine learning pipeline with the three following stages:
	
1.	Pre-processing – Factor re-weighting
2.	In-processing – Adversarial debiasing neural network
3.	Post-processing – Reject Option Based Classification


### Pre-processing
•	Weights examples differently to provide a boost before classification (ensures mean of favorable outcomes between privileged and unprivileged groups are similar)
•	Happens before classification, levels the playing field before predictions

### In-processing
•	Adversarial debiasing maximizes accuracy while simultaneously reducing ability to predict protected attribute based on prediction
•	Breaks link between protected attribute and outcome, ensures the relationship is a one way street

### Post-processing
•	Reject Option Based Classification (ROC) has threshold within decision boundary and swaps predictions after classification
•	Provides favorable outcomes to unprivileged groups and unfavorable outcomes to privileged groups
•	A slight boost for unprivileged group to ensure equality



### To install using Pip:

```
pip install MLFairnessPipeline
```

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