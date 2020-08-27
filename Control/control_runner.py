from sklearn.model_selection import train_test_split
from Control.classifiers import execute_classifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pandas as pd

def control_runner(dataset_orig, unprivileged_groups, privileged_groups, dataset_type, protected_attribute):
    df = pd.DataFrame(dataset_orig.features, columns=dataset_orig.feature_names)
    label = dataset_orig.label_names[0]
    df[label] = dataset_orig.labels

    copy_df = df.copy()
    df.columns=df.columns.str.replace('.','')
    df.columns=df.columns.str.replace(',','')
    df.columns=df.columns.str.replace('>','+')
    df.columns=df.columns.str.replace('<','-')

    X_train, X_test, y_train, y_test = train_test_split(df.drop(label, axis=1), df[label], test_size=0.33, random_state=42)
    
    execute_classifier(XGBClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " XGBClassifier - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(KNeighborsClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " KNeighborsClassifier - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(SVC(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " SVC - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(LinearSVC(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " LinearSVC - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    # execute_classifier(NuSVC(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " NuSVC - RACE", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " DecisionTreeClassifier - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(RandomForestClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " RandomForestClassifier - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(AdaBoostClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " AdaBoostClassifier - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(GradientBoostingClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " GradientBoostingClassifier - PROTECTED ATTRIBUTE INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)

    df.drop([protected_attribute], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(label, axis=1), df[label], test_size=0.33, random_state=42)
    
    execute_classifier(XGBClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " XGBClassifier - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(KNeighborsClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " KNeighborsClassifier - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(SVC(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " SVC - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(LinearSVC(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " LinearSVC - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    # execute_classifier(NuSVC(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " NuSVC - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " DecisionTreeClassifier - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(RandomForestClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " RandomForestClassifier - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(AdaBoostClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " AdaBoostClassifier - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)
    execute_classifier(GradientBoostingClassifier(), X_train, y_train, X_test, y_test, copy_df, label, dataset_type + " GradientBoostingClassifier - PROTECTED ATTRIBUTE NOT INCLUDED", unprivileged_groups, privileged_groups, protected_attribute, dataset_type)