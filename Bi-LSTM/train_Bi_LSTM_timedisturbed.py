from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.models import load_model
from keras.layers import TimeDistributed
import numpy as np


def validate_dataset(name, model, n_timesteps, n_classes):

    # test prediction of the model with a different test subject
    dataset = read_csv(prepared_dataset_path + 'prepared_' + name + '.csv', header=0, index_col=0)

    # split the dataset into input and output data and reshape input for LSTM [samples, timesteps, features]
    values_X, values_y, values_y_dec = split_dataset_into_input_and_output(dataset, n_timesteps, n_classes)
    y_pred = model.predict(values_X)

    # make final prediction => transform float prediction values to int
    # (Softmax => max value in a row = highest likelihood)
    y_pred_final = (y_pred == y_pred.max(axis=2, keepdims=True)).astype(int)

    # reshape because axis for timesteps is no longer needed
    y_pred_final = y_pred_final.reshape(y_pred_final.shape[0] * y_pred_final.shape[1], y_pred_final.shape[2])
    values_y = values_y.reshape(values_y.shape[0] * values_y.shape[1], values_y.shape[2])
    values_y_dec = values_y_dec.reshape(values_y_dec.shape[0] * values_y_dec.shape[1])

    # transform the predicted binary matrix in a decimal vector
    y_pred_dec = np.empty(y_pred_final.shape[0], dtype=np.int)
    iterator = np.nditer(y_pred_final, flags=['multi_index'])
    while not iterator.finished:
        if (iterator[0] == 1):
            mi = iterator.multi_index
            y_pred_dec[mi[0]] = mi[1]
        iterator.iternext()

    # F1-Score
    f1_score_val = f1_score(values_y, y_pred_final, average='micro')
    print('\nF1-Score for ' + name + ': %f' % (f1_score_val))

    # Cohen's Kappa Score (compare the decimal vectors)
    kappa_score = cohen_kappa_score(values_y_dec, y_pred_dec)
    print('Cohens-Kappa-Score for ' + name + ': %f' % (kappa_score))

    # Accuracy Score
    #scores = model.evaluate(values_X, values_y, verbose=0)
    #print('Test accuracy for ' + name + ': ', scores[1])



def split_dataset_into_input_and_output(dataset, timesteps, n_classes ):

    # drop the timestamp
    dataset = dataset.drop('Sec', 0)

    # show the distribution of the activities in the dataset
    #value_count = dataset['Activity'].value_counts(normalize=True)
    #print(value_count)

    # dataset.dropna(inplace=True)
    val = dataset.values

    # verify if all data is float
    val = val.astype('float32')

    # normalize only the input features
    scaler = MinMaxScaler(feature_range=(0, 1))
    val[:, :-1] = scaler.fit_transform(val[:, :-1])

    # calculate the remainder and drop the remaining data
    # this step is needed before reshaping the dataset
    val_remainder = val.shape[0] % timesteps

    if val_remainder > 0:
        val = val[:-val_remainder, :]

    # reshape data for LSTM [samples, timesteps, features] (round because: float => int)
    val = val.reshape((round(val.shape[0] / timesteps), timesteps, val.shape[1]))

    # TODO: delete 0 activitys => imbalanced dataset
    # TODO: Aktivität 0 ist zu 71% im Datensatz vorhanden. Das zweit häufigste zu 2%

    # shuffle the time sequences (shuffles the array along the first axis of a multi-dimensional array)
    np.random.shuffle(val)

    # split into input and output data
    # keep only the target y from the end of the sequence
    # (1 sequence of x-timesteps = 1 target y)
    val_X, val_y_dec = val[:, :, :-1], val[:, :, -1]

    val_y_dec = val_y_dec.reshape(val_X.shape[0], timesteps, 1)

    # create a binary output vector (e.g.: Activity 4 => [0,0,0,0,1,0,0,...,0]
    val_y_bin = np.zeros(n_classes * val_y_dec.shape[0] * timesteps, dtype=np.int).reshape(val_y_dec.shape[0], timesteps, n_classes)
    val_y_dec = val_y_dec.astype(int)

    iterator = np.nditer(val_y_dec, flags=['multi_index'])
    while not iterator.finished:
        mi = iterator.multi_index
        val_y_bin[mi[0], mi[1], iterator[0]] = 1
        iterator.iternext()

    #print(val_X.shape, val_y.shape, val_y_bin.shape)

    return val_X, val_y_bin, val_y_dec

# init
n_classes = 34
n_timesteps = 20
load_existing_model = False
save_model_to_disk = True
prepared_dataset_path = '../Dataset/'

# load dataset
dataset = read_csv(prepared_dataset_path + 'prepared_combination_subject5_ideal_et_al.csv', header=0, index_col=0)


# split the dataset into input and output data and reshape input for LSTM [samples, timesteps, features]
values_X, values_y, _ = split_dataset_into_input_and_output(dataset, n_timesteps, n_classes)

# split into train and test sets
train_to_test_ratio = 0.7
spl = round(train_to_test_ratio * values_X.shape[0])
train_X, test_X = values_X[:spl, :, :], values_X[spl:, :, :]
train_y, test_y = values_y[:spl, :],    values_y[spl:, :]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



# load a existing model or create a new model

if load_existing_model:
    model = load_model('my_model.h5')
else:
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(train_y.shape[2], activation='softmax')))
    #model.add(Dense(train_y.shape[1], activation='softmax'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #cus_met.fbeta_score,

# start Training
history = model.fit(train_X, train_y, batch_size=30, epochs=10, validation_data=[test_X, test_y],verbose=2)

# save model to disk
if save_model_to_disk:
    model.save('my_model.h5')


# compare the model with different datasets
validate_dataset('subject5_ideal', model, n_timesteps, n_classes)
validate_dataset('subject5_self', model, n_timesteps, n_classes)
validate_dataset('subject3_ideal', model, n_timesteps, n_classes)
validate_dataset('subject3_self', model, n_timesteps, n_classes)
validate_dataset('subject7_ideal', model, n_timesteps, n_classes)


# plot the history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


print('Ende')



