from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from MLFairnessPipeline.Utilities import generate_binary_label_dataset
import tensorflow as tf

class Fair_Model():
    def __init__(self, unprivileged_groups, privileged_groups, label, protected_attribute, favorable_label, unfavorable_label):
        self.sess = tf.Session()
        self.model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name='debiased_classifier', debias=True, sess=self.sess)
        self.protected_attribute = protected_attribute
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        self.label = label
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        
    
    def fit(self, training_data):
        training_binary_dataset = generate_binary_label_dataset(training_data, self.label, self.protected_attribute, self.favorable_label, self.unfavorable_label)
        self.model.fit(training_binary_dataset)
        

    def predict(self, test_data):
        test_binary_dataset = generate_binary_label_dataset(test_data, self.label, self.protected_attribute, self.favorable_label, self.unfavorable_label)
        return self.model.predict(test_binary_dataset).convert_to_dataframe()[0]

    def destroy(self):
        self.sess.close()
        tf.reset_default_graph()
