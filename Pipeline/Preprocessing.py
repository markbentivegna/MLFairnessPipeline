from aif360.algorithms.preprocessing import Reweighing

def preprocessing(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups):
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    dataset_transf_test = RW.fit_transform(dataset_orig_test)

    return dataset_orig_train, dataset_orig_test