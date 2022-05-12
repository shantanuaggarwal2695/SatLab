import pandas as pd
import rasterio
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time

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
    train_x, train_y, test_x, test_y = df_train[['origin']], df_train['class'], df_test[['origin']], df_test['class']
    def map_origin(x):
        return x.split("/")[-1].strip()

    train_x['origin'] = train_x['origin'].apply(map_origin)
    test_x['origin'] = test_x['origin'].apply(map_origin)
    def getBands(rasterImage):
        with rasterio.open(rasterImage) as dataset:
            band0 = dataset.read(2)
            band1 = dataset.read(3)
            band2 = dataset.read(4)
            max_0 = np.amax(band0)
            band0 = band0/(max_0/255.0)
            max_1 = np.amax(band1)
            band1 = band1/(max_1/255.0)
            max_2 = np.amax(band2)
            band2 = band2/(max_2/255.0)
            return np.dstack((band0, band1, band2))
#         return np.ravel(dataset.read(band)).tolist()

    def true_positive(actual, predicted):
        count = 0
        for (i, label) in enumerate(actual):
            if actual[i]==0 and predicted[i]==0:
                count+=1
        return count

    def true_negative(actual, predicted):
        count = 0
        for (i, label) in enumerate(actual):
            if actual[i]==1 and predicted[i]==1:
                count+=1
        return count

    def false_positive(actual, predicted):
        count = 0
        for (i, label) in enumerate(actual):
            if actual[i]==1 and predicted[i]==0:
                count+=1
        return count

    def false_negative(actual, predicted):
        count = 0
        for (i, label) in enumerate(actual):
            if actual[i]==0 and predicted[i]==1:
                count+=1
        return count


    train = train_x['origin'].to_list()
    test = test_x['origin'].to_list()
    train_label = train_y.to_list()
    test_label = test_y.to_list()

    train_dir = "../data2/euro_sat/load_data/"
    train_feat = []
    
    print("starting to load data ")
    print("_______________________________________")
    load_t1 = time.time()
    for (i,images) in enumerate(train):
        train_feat.append(getBands(train_dir+images))
    load_t2 = time.time()
    load_time = load_t1 - load_t2

    train_feat = np.array(train_feat)


    test_feat = []
    for (i,images) in enumerate(test):
        test_feat.append(getBands(train_dir+images))

    test_feat = np.array(test_feat)

    nClasses = 2
    train_X = train_feat.reshape(-1, 64,64, 3)
    test_X = test_feat.reshape(-1, 64,64, 3)
    # train_X.shape, test_X.shape
    train_X = train_X / 255.
    test_X = test_X / 255.

    train_Y_one_hot = to_categorical(train_label)
    test_Y_one_hot = to_categorical(test_label)

    batch_size = 16
    epochs = 5
    num_classes = 2

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(64,64,3),padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(Dense(num_classes, activation='softmax'))


    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    train_t1 = time.time()
    weight_0 = train_label.count(1)/train_label.count(0)
    fashion_train = fashion_model.fit(train_X, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(test_X, test_Y_one_hot), callbacks=[es], class_weight={0:weight_0, 1:1})

    train_t2 = time.time()
    train_time = train_t2-train_t1

    predictions = np.argmax(fashion_model.predict(test_X), axis=1)
    actual = test_label
    predict = predictions.tolist()
    print(actual, predict)
    precision = true_positive(actual, predict)/(true_positive(actual, predict)+false_positive(actual, predict))
    recall = true_positive(actual, predict)/(true_positive(actual, predict) + false_negative(actual, predict))
    f1_score = 2*(precision*recall)/(precision+recall)

    result = {"f1":f1_score, "prec":precision, "recall":recall, "load_time":load_time, "train_time":train_time}
    #print("Precision and Recall are:", precision, recall)
    #print(2*(precision*recall)/(precision+recall))
    print(result)
    print("-----------------------------------------------------------------------------")

