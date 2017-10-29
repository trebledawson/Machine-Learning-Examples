import numpy as np
import pandas as pd
import string
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.initializers import lecun_normal
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt


##########
#  Main  #
##########
def main():
    UCI_data = 1

    if UCI_data == 1:
        letters()
    elif UCI_data == 2:
        images()
    elif UCI_data == 3:
        drive()
    elif UCI_data == 4:
        crime()
    elif UCI_data == 5:
        slice()

##############################
#  CLASSIFICATION  EXAMPLES  #
##############################

def letters():
    """
    UCI Letter Recognition Dataset
    ------------------------------
    Number of instances: 20,000
    Number of attributes: 16
    Number of classes: 26
    Training size: 14,000
    Validation size: 3,000
    Test size: 3,000
    Best test performance: 98.2666666667 percent accuracy
    """

    # Load data
    file = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI Datasets\Letter " \
           "Recognition Data\letter-recognition.data"

    names = ['letter', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    data = pd.read_csv(file, names=names)
    labels = np.zeros((data.shape[0], 1))

    for letter, index in zip(list(string.ascii_uppercase), list(range(26))):
        labels[data.index[data['letter'] == letter]] = index

    num_classes = len(np.unique(labels))
    labels = to_categorical(labels, num_classes)
    data = np.array(data.drop('letter', 1)).astype('float32')
    data_train, data_test, labels_train, labels_test = tts(data, labels,
                                                           test_size=0.15)
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Generate model
    seed = 5
    model = Sequential()
    model.add(Dense(800, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed), input_dim=16))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(400, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed)))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(200, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed)))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    train_start = time.time()
    early = EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    model.fit(data_train, labels_train, callbacks=[early], epochs=1000,
              batch_size=500, validation_split=0.17647, shuffle=True,
              verbose=2)
    print('Training time:', time.time() - train_start, 'seconds.')

    # Evaluate model
    score = model.evaluate(data_test, labels_test, verbose=0)
    print('Accuracy is', 100 * score[1], 'percent.')

def images():
    """
    UCI Image Segmentation Dataset
    ------------------------------
    Number of instances: 2,310
    Number of attributes: 19
    Number of classes: 7
    Training size: 1,660
    Validation size: 300
    Test size: 350
    Best test performance: 97.4285714286 percent accuracy
    """

    # Load data
    file_train = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI " \
                 "Datasets\Image Segmentation Data\segmentation.data"

    file_test = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI " \
                 "Datasets\Image Segmentation Data\segmentation.test"

    names = list(range(20))
    data = pd.read_csv(file_train, names=names)
    test = pd.read_csv(file_test, names=names)

    labels = np.zeros((data.shape[0], 1))
    test_labels = np.zeros((test.shape[0], 1))
    classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH',
               'GRASS']
    for class_, index in zip(classes, list(range(7))):
        labels[data.index[data[0] == class_]] = index
        test_labels[test.index[test[0] == class_]] = index

    data = np.array(data.drop(0, 1)).astype('float32')
    test = np.array(test.drop(0, 1)).astype('float32')
    data = np.vstack((data, test))

    num_feat = data.shape[1]

    num_classes = len(np.unique(labels))
    labels = to_categorical(labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    labels = np.vstack((labels, test_labels))

    data_train, data_test, labels_train, labels_test = tts(data, labels,
                                                           test_size=350)

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Generate model
    seed = 5
    model = Sequential()
    model.add(Dense(1000, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed), input_dim=num_feat))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(500, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed)))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(100, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed)))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    train_start = time.time()
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    model.fit(data_train, labels_train, callbacks=[early], epochs=1000,
              batch_size=50, validation_split=0.17644, shuffle=True, verbose=2)
    print('Training time:', time.time() - train_start, 'seconds.')

    # Evaluate model
    score = model.evaluate(data_test, labels_test, verbose=0)
    print('Accuracy is', 100 * score[1], 'percent.')

def drive():
    """
    UCI Sensorless Drive Diagnosis Dataset
    --------------------------------------
    Number of instances: 58,509
    Number of attributes: 48
    Number of classes: 11
    Training size: 40,959
    Validation size: 8,775
    Test size: 8,775
    Best test performance: 97.6529565979 percent accuracy
    """

    # Load data
    file = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI " \
           "Datasets\Sensorless Drive Diagnosis\Sensorless_drive_diagnosis.txt"

    names = list(range(49))

    data = pd.read_csv(file, delim_whitespace=True, names=names)
    labels = np.array(data[48]).reshape(-1, 1)
    labels -= 1

    num_classes = len(np.unique(labels))
    labels = to_categorical(labels, num_classes)
    data = np.array(data.drop(48, 1)).astype('float32')
    num_feat = data.shape[1]

    data_train, data_test, labels_train, labels_test = tts(data, labels,
                                                           test_size=0.15)

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Generate model
    seed = 5
    np.random.seed = seed

    model = Sequential()
    model.add(Dense(2048, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed), input_dim=num_feat))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(512, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed)))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(128, activation='relu',
                    kernel_initializer=lecun_normal(seed),
                    bias_initializer=lecun_normal(seed)))
    model.add(Dropout(0.5, seed=seed))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    train_start = time.time()
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    model.fit(data_train, labels_train, callbacks=[early], epochs=1000,
              batch_size=50, validation_split=0.17647, shuffle=True,
              verbose=2)
    print('Training time:', time.time() - train_start, 'seconds.')

    # Evaluate model
    score = model.evaluate(data_test, labels_test, verbose=0)
    print('Accuracy is', 100 * score[1], 'percent.')

#########################
#  REGRESSION EXAMPLES  #
#########################
def crime():
    """
    UCI Communities and Crime Dataset
    --------------------------------------
    Number of instances: 1994
    Number of attributes: 122
    Training size: 1390
    Validation size: 300
    Test size: 300
    Best test performance:
    R2: 0.634906146265 | MSE: 0.14279287312947544 | RMSE: 0.37787944258648876
    """

    # Load data
    file = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI " \
           "Datasets\Communities and Crime Data\communities.data"
    names = list(range(128))
    data = pd.read_csv(file, names=names)

    # Drop non-predictive data columns
    data = data.drop((range(0, 5)), 1)

    # Drop missing data columns
    data = data.drop(list(range(101, 118)), 1)
    data = data.drop([30, 121, 122, 123, 124, 126], 1)
    data = np.array(data).astype('float32')

    labels = data[:, -1]
    data = data[:, :-1]


    num_feat = data.shape[1]

    dtrain, dtest, ltrain, ltest = tts(data, labels, test_size=300)

    scaler = StandardScaler()
    dtrain = scaler.fit_transform(dtrain)
    dtest = scaler.transform(dtest)

    # Model generation
    seed = 5
    np.random.seed(seed)

    model = Sequential()
    model.add(Dense(70, activation='sigmoid', input_dim=num_feat))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='nadam', loss='mean_squared_error',
                  metrics=['mse'])

    # Model training
    print('Training model...')
    train_start = time.time()
    early = EarlyStopping(monitor='val_mean_squared_error', patience=20,
                          verbose=0)
    model.fit(dtrain, ltrain, callbacks=[early], epochs=1000,
              batch_size=100, validation_split=0.177, shuffle=True,
              verbose=2)

    print('Training time:', time.time() - train_start, 'seconds.')

    # Evaluate model
    pred = model.predict(dtest, verbose=0)

    r2 = r2_score(ltest, pred)
    mse = sqrt(mean_squared_error(ltest, pred))
    print('R2:', r2, '| MSE:', mse, '| RMSE:', sqrt(mse))

def slice():
    """
    UCI Slice Localization Dataset
    ------------------------------
    Number of instances: 53,500
    Number of attributes: 385
    Training size: 37,500
    Validation size: 8,000
    Test size: 8,000
    Best test performance:
    R2: 0.999084790614 | MSE: 0.6829713658422585 | RMSE: 0.8264208164381258
    """

    # Load data
    file = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI Datasets\Slice " \
           "Localization Data\slice_localization_data.csv"

    names = list(range(386))
    data = np.array(pd.read_csv(file, names=names))[1:, :].astype('float32')

    labels = data[:, -1]
    data = data[:, :-1]

    num_feat = data.shape[1]

    dtrain, dtest, ltrain, ltest = tts(data, labels, test_size=8000)

    scaler = StandardScaler()
    dtrain = scaler.fit_transform(dtrain)
    dtest = scaler.transform(dtest)

    # Model generation
    seed = 5
    np.random.seed(seed)

    model = Sequential()
    model.add(Dense(2000, activation='sigmoid', input_dim=num_feat))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='nadam', loss='mean_squared_error',
                  metrics=['mse'])

    # Model training
    print('Training model...')
    train_start = time.time()
    early = EarlyStopping(monitor='val_mean_squared_error', patience=20,
                          verbose=0)
    model.fit(dtrain, ltrain, callbacks=[early], epochs=1000,
              batch_size=1000, validation_split=0.17582, shuffle=True,
              verbose=2)

    print('Training time:', time.time() - train_start, 'seconds.')

    # Evaluate model
    pred = model.predict(dtest, verbose=0)

    r2 = r2_score(ltest, pred)
    mse = sqrt(mean_squared_error(ltest, pred))
    print('R2:', r2, '| MSE:', mse, '| RMSE:', sqrt(mse))

if __name__ == '__main__':
    main()