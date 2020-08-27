from Control.generate_BLD import generateBLD
from Utilities.print_metrics import print_classification_metrics
from Utilities.generate_results import record_results

def execute_classifier(clf, X_train, y_train, X_test, y_test, copy, label, comment, unprivileged_groups, privileged_groups, protected_attribute, dataset_type):
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_binary_dataset, train_pred_binary_dataset = generateBLD(X_train, y_train, train_pred, copy, label, unprivileged_groups, privileged_groups, protected_attribute)
    print_classification_metrics(train_binary_dataset, train_pred_binary_dataset, unprivileged_groups, privileged_groups)
    record_results(comment + ' train', train_binary_dataset, train_pred_binary_dataset, unprivileged_groups, privileged_groups, "CONTROL", dataset_type)
    
    test_binary_dataset, test_pred_binary_dataset = generateBLD(X_test, y_test, test_pred, copy, label, unprivileged_groups, privileged_groups, protected_attribute)
    print_classification_metrics(test_binary_dataset, test_pred_binary_dataset, unprivileged_groups, privileged_groups)
    record_results(comment + ' test', test_binary_dataset, test_pred_binary_dataset, unprivileged_groups, privileged_groups, "CONTROL", dataset_type)