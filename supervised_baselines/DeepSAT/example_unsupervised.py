"""
==============================================================
Deep Belief Network features for Satellite Image classification which replicated DeepSAT paper
==============================================================

Adapted from http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py

This example shows how to build a classification pipeline with a UnsupervisedDBN
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the UnsupervisedDBN help improve the
classification accuracy.
"""

from __future__ import print_function

print(__doc__)

import numpy as np
import time
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dbn.models import UnsupervisedDBN # use "from dbn.tensorflow import SupervisedDBNClassification" for computations on TensorFlow
import pandas as pd

###############################################################################
# Setting up


def prepare_data():
    ground_truth = pd.read_csv("../data2/ground_truth/ground_truth_2000.csv")
    def map_id_to_origin(id):
        return "image"+str(id)+".tif"

    def convert_class(label):
        if label == -1:
            return 1
        else:
            return label

    def process_class(x):
        if x == -2 or x == 2:
            return -1
        else:
            return x

    train_df = pd.read_csv("../data2/supervised_learner_data/train_200000.csv")
    test_df = pd.read_csv("../data2/supervised_learner_data/test_200000.csv")

    feat_df = pd.read_csv("../data2/unsupervised_learner_data/train_2000.csv")
    ground_truth['id'] = ground_truth['ID'].map(map_id_to_origin)
    ground_truth['Class'] = ground_truth['Class'].apply(process_class)
    ground_truth['Class'] = ground_truth['Class'].map(convert_class)
    ground_truth = ground_truth[['id', 'Class']]


    train_images_labeled  = ground_truth.merge(train_df, on="id", how="inner")
    test_images_labeled = ground_truth.merge(test_df, on="id", how="inner")

    df_train = train_images_labeled.merge(feat_df, on="id", how="inner")
    df_test = test_images_labeled.merge(feat_df, on="id", how="inner")
    cols = ["glcm_contrast_Scaled", "glcm_dissimilarity_Scaled", "glcm_homogeneity_Scaled", "glcm_energy_Scaled","glcm_correlation_Scaled", "glcm_ASM_Scaled"]
    train_X = df_train[cols]
    test_X = df_test[cols]

    train_Y = df_train['Class']
    test_Y = df_test['Class']

    train_feat_new = np.repeat(np.array(train_X), 1, axis=1)
    test_feat_new = np.repeat(np.array(test_X), 1, axis=1)

    

    return train_feat_new, test_feat_new, np.array(train_Y), np.array(test_Y)


X_train, X_test, Y_train, Y_test = prepare_data()
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)  # 0-1 scaling

print(X_train.shape)
print(X_test.shape)

#Models we will use
# logistic = linear_model.LogisticRegression(max_iter=4000)
neural_net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 5, 3, 2), random_state=1, max_iter=1000000)
dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=10,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=20,
                      activation_function='sigmoid')

classifier = Pipeline(steps=[('dbn', dbn),
                             ('neural_net', neural_net)])

###############################################################################
# Training
# logistic.C = 6000.0

# Training RBM-Logistic Pipeline
t1 = time.time()
classifier.fit(X_train, Y_train)
t2 = time.time()

print(t2 - t1)
# Training Logistic regression
# logistic_classifier = linear_model.LogisticRegression(C=100.0)
# logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print(classifier.predict(X_test))
print("Neural Net using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

'''
print("Neural net using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))
'''
##############################################################################
