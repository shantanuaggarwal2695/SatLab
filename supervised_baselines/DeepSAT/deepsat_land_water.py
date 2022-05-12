from __future__ import print_function

print(__doc__)

import numpy as np
import time
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dbn.tensorflow import UnsupervisedDBN # use "from dbn.tensorflow import SupervisedDBNClassification" for computations on TensorFlow
import pandas as pd
import numpy as np
import time
np.random.seed(1234)
feats = ["ndwi","mdwi", "ndmi", "ndvi", "awei", "bi", "rvi" , "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "glcm_ASM"]


df = pd.read_csv("../data2/euro_sat/textural_features/textural.csv")
sample_per_fold = df.shape[0]/5
labeled_train = df.iloc[np.random.permutation(len(df))]

def generate_samples(df, n=5):
    result = []
    for i in range(n):
        sample = df.sample(int(sample_per_fold))
        df.drop(index=sample.index, inplace=True)
        result.append(sample)

    return result

k_fold_sets = generate_samples(labeled_train)


for i in range(5):
    found = False
    for j in range(5):
        if i == j:
            df_test = k_fold_sets[i]
        else:
            if not found:
                df_train = k_fold_sets[j]
                found = True
            else:
                df_train = df_train.append(k_fold_sets[j])
    X_train, Y_train, X_test, Y_test = np.array(df_train[feats]),  np.array(df_train['class']), np.array(df_test[feats]), np.array(df_test['class'])

#    ground_truth = pd.read_csv("../data2/ground_truth/ground_truth_2000.csv")
    X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)  # 0-1 scaling
    X_test = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) + 0.0001)  # 0-1 scaling
    neural_net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 5, 3, 2), random_state=1, max_iter=1000000)
    dbn = UnsupervisedDBN(hidden_layers_structure=[256, 512],
                      batch_size=32,
                      learning_rate_rbm=0.06,
                      n_epochs_rbm=5,
                      activation_function='sigmoid')

    classifier = Pipeline(steps=[('dbn', dbn),('neural_net', neural_net)])
    t1 = time.time()
    classifier.fit(X_train, Y_train)
    t2 = time.time()

    print(t2-t1)
    #t2 = time.time()
    #print(classifier.predict(X_test))
    print("Neural Net using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

    print("__________________________________________________________________")

