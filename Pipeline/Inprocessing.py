from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow as tf

def inprocessing(dataset_transf_train, dataset_transf_test, unprivileged_groups, privileged_groups, scope, debiased):
    sess = tf.Session()

    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name=scope, debias=debiased, sess=sess)
    debiased_model.fit(dataset_transf_train)

    debiased_train_pred = debiased_model.predict(dataset_transf_train) 
    debiased_test_pred = debiased_model.predict(dataset_transf_test)

    sess.close()
    tf.reset_default_graph()

    return debiased_train_pred, debiased_test_pred
