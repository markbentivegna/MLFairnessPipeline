from aif360.metrics import ClassificationMetric
import pandas as pd
import datetime
import os.path

def get_file_name(simulation_type, dataset_type):
    current_time = datetime.datetime.now()
    return "{dataset_type}_results_{simulation_type}_{year}_{month}_{day}.csv".format(dataset_type=dataset_type, simulation_type=simulation_type, year=current_time.year, month=current_time.month, day=current_time.day)
    

def generate_results_df(comment, classified_metric, balanced_accuracy):
    return pd.DataFrame({
        'Comment': [comment],
        'Classification Accuracy': [classified_metric.accuracy()],
        'Balanced Classification Accuracy': [balanced_accuracy],
        'Statistical Parity Difference': [classified_metric.statistical_parity_difference()],
        'Disparate Impact': [classified_metric.disparate_impact()],
        'Equal Opportunity Difference': [classified_metric.equal_opportunity_difference()],
        'Average Odds Difference': [classified_metric.average_odds_difference()],
        'Theil Index': [classified_metric.theil_index()]
    })


def record_results(comment, features, labels, unprivileged_groups, privileged_groups, simulation_type, dataset_type):
    classified_metric = ClassificationMetric(features, labels, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    bal_acc_debiasing_test = 0.5*(classified_metric.true_positive_rate()+classified_metric.true_negative_rate())
    results_df = generate_results_df(comment, classified_metric, bal_acc_debiasing_test)
    results_file = get_file_name(simulation_type, dataset_type)
    if os.path.exists(results_file):
        append_csv_file(results_df, results_file)
    else:
        generate_csv_file(results_df, results_file)


def generate_csv_file(df, filename):
    df.to_csv(filename, index=False)

def append_csv_file(df, filename):
    df.to_csv(filename, mode='a', header=False, index=False)

