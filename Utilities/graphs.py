import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
fig = plt.figure()

classifiers = ['XGBClassifier', 'KNeighborsClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier']

classifier_abbr = {
    'XGBClassifier': 'XGB', 
    'KNeighborsClassifier': 'KN', 
    'RandomForestClassifier': 'RF', 
    'AdaBoostClassifier': 'ADB', 
    'GradientBoostingClassifier': 'GB'
}

def fetch_classifier(x):
    for classifier in classifiers:
        if classifier in x:
            return classifier_abbr[classifier]


compas_control_file = "compas_results_CONTROL_2020_8_26.csv"
compas_control_df = pd.read_csv(compas_control_file, encoding="ISO-8859-1")
compas_control_df = compas_control_df[~compas_control_df['Comment'].str.contains('PROTECTED ATTRIBUTE INCLUDED')]
compas_control_df = compas_control_df[~compas_control_df['Comment'].str.contains('train')]
compas_control_df = compas_control_df[~compas_control_df['Comment'].str.contains('SVC')]
compas_control_df = compas_control_df[~compas_control_df['Comment'].str.contains('LinearSVC')]
compas_control_df = compas_control_df[~compas_control_df['Comment'].str.contains('DecisionTreeClassifier')]
compas_control_df['Classifier'] = compas_control_df['Comment'].apply(lambda x: fetch_classifier(x))
compas_control_df.drop('Comment', axis=1, inplace=True)

compas_fair_file = "compas_results_FAIR_2020_8_26.csv"
compas_fair_df = pd.read_csv(compas_fair_file, encoding="ISO-8859-1")
compas_fair_df = compas_fair_df[~compas_fair_df['Comment'].str.contains('ebiased')]
compas_fair_df = compas_fair_df[~compas_fair_df['Comment'].str.contains('Training')]
compas_fair_df['Classifier'] = 'Fair Classifier'

all_classifiers = compas_control_df['Classifier'].tolist()
all_classifiers.append(compas_fair_df['Classifier'].tolist()[0])
graph_boundaries = {
    'Classification Accuracy': [0, 1.0],
    'Balanced Classification Accuracy': [0, 1.0],
    'Statistical Parity Difference': [-0.5, 0.5],
    'Disparate Impact': [0, 1],
    'Equal Opportunity Difference': [-0.5, 0.5],
    'Average Odds Difference': [-0.5, 0.5],
    'Theil Index': [0, 0.5]
}

def y_position_offset(x):
    if x < 0:
        return -0.04
    else:
        return 0

metrics = ['Classification Accuracy','Balanced Classification Accuracy','Statistical Parity Difference','Disparate Impact','Equal Opportunity Difference','Average Odds Difference','Theil Index']
for metric in metrics:
    all_metrics = compas_control_df[metric].tolist()
    all_metrics.append(compas_fair_df[metric].tolist()[0])
    
    plt.bar(all_classifiers, all_metrics, color ='blue', width = 0.4) 
    

    plt.xlabel("Classifiers") 
    plt.ylabel(metric) 
    plt.title(metric + " Across Classifiers") 
    plt.axhline(0, color='black')
    plt.ylim([graph_boundaries[metric][0], graph_boundaries[metric][1]])

    for index, data in enumerate(all_metrics):
        data_rounded = round(data, 2)
        plt.text(x=index - 0.2, y=data_rounded + y_position_offset(data_rounded), s=f"{data_rounded}", fontdict=dict(fontsize=10))
    
    plt.tight_layout()

    plt.show() 
