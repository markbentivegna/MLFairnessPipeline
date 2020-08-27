from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

def postprocessing(dataset_transf_train, debiased_train_pred, debiased_test_pred, unprivileged_groups, privileged_groups):
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    ROC = ROC.fit(dataset_transf_train, debiased_train_pred)

    print("Dataset Train: Optimal classification threshold = %.4f" % ROC.classification_threshold)

    roc_train_pred = ROC.predict(debiased_train_pred)
    roc_test_pred = ROC.predict(debiased_test_pred)

    return roc_train_pred, roc_test_pred