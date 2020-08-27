from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def print_binary_label_metrics(training_data, test_data, unprivileged_groups, privileged_groups):
    metric_train = BinaryLabelDatasetMetric(training_data, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Training set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_train.mean_difference())

    metric_test = BinaryLabelDatasetMetric(test_data, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_test.mean_difference())



def print_classification_metrics(features, labels, unprivileged_groups, privileged_groups):
    classified_metric = ClassificationMetric(features, labels, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Classification accuracy = %f" % classified_metric.accuracy())
    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    bal_acc_debiasing_test = 0.5*(TPR+TNR)
    print("Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    print("Statistical parity difference = %f" % classified_metric.statistical_parity_difference())
    print("Disparate impact = %f" % classified_metric.disparate_impact())
    print("Equal opportunity difference = %f" % classified_metric.equal_opportunity_difference())
    print("Average odds difference = %f" % classified_metric.average_odds_difference())
    print("Theil_index = %f" % classified_metric.theil_index())