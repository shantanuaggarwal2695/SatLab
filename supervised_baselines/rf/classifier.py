import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import time 

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
        
    train_x, train_y, test_x, test_y = df_train[feats], df_train['class'], df_test[feats], df_test['class']
    clf = RandomForestClassifier(max_depth=5, random_state=0, class_weight="balanced_subsample")
    t1 = time.time()
    model = clf.fit(np.array(train_x), np.array(train_y).ravel())
    t2 = time.time()
    prediction = model.predict(np.array(test_x))
    actual = test_y.to_list()
    predict = prediction.tolist()

    precision = true_positive(actual, predict)/(true_positive(actual, predict)+false_positive(actual, predict))
    recall = true_positive(actual, predict)/(true_positive(actual, predict) + false_negative(actual, predict))

    print(precision, recall)

    print(2*(precision*recall)/(precision+recall))
    
    print(t2-t1)
    
    print("____________________________________________________")


    
