from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dropout
import numpy as np
import custom_metrics as cus_met

def validate_dataset(name, model, n_timesteps, n_classes):

    # test prediction of the model with a different test subject
    dataset = read_csv('prepared_' + name + '.csv', header=0, index_col=0)

    # split the dataset into input and output data and reshape input for LSTM [samples, timesteps, features]
    values_X, values_y = split_dataset_into_input_and_output(dataset, n_timesteps, n_classes)
    y_pred = model.predict(values_X)

    # make final prediction => transform float prediction values to int
    # (Softmax => max value in a row = highest likelihood)
    y_pred_final = (y_pred == y_pred.max(axis=1, keepdims=True)).astype(int)
    f1_score_val = f1_score(values_y, y_pred_final, average='macro')

    print('F1-Score for ' + name + ': %f' % (f1_score_val))

def split_dataset_into_input_and_output(dataset, timesteps, n_classes ):

    # drop the timestamp
    dataset = dataset.drop('Sec', 1)
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
    val= val[:-val_remainder, :]

    # reshape data for LSTM [samples, timesteps, features] (round because: float => int)
    val = val.reshape((round(val.shape[0] / timesteps), timesteps, val.shape[1]))

    # shuffle the time sequences (shuffles the array along the first axis of a multi-dimensional array)
    np.random.shuffle(val)

    # split into input and output data
    # keep only the target y from the end of the sequence
    # (1 sequence of x-timesteps = 1 target y)
    # TODO: select the the target y which is dominated in the last column
    val_X, val_y = val[:, :, :-1], val[:, -1, -1]


    # create a binary output vector (e.g.: Activity 4 => [0,0,0,0,1,0,0,...,0]
    val_y_bin = np.zeros(n_classes * val_y.shape[0], dtype=np.int).reshape(val_y.shape[0], n_classes)
    val_y = val_y.astype(int)

    for counter, dec_val in enumerate(val_y):
        val_y_bin[counter, dec_val] = 1


    #print(val_X.shape, val_y.shape, val_y_bin.shape)

    return val_X, val_y_bin


n_classes = 34
n_timesteps = 5

# load dataset
dataset = read_csv('prepared_subject5_ideal.csv', header=0, index_col=0)


# split the dataset into input and output data and reshape input for LSTM [samples, timesteps, features]
values_X, values_y = split_dataset_into_input_and_output(dataset, n_timesteps, n_classes)

# split into train and test sets
train_to_test_ratio = 0.7
spl = round(train_to_test_ratio * values_X.shape[0])
train_X, test_X = values_X[:spl, :, :], values_X[spl:, :, :]
train_y, test_y = values_y[:spl, :],    values_y[spl:, :]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# network architecture
# TODO: use a sequence as return (maybe more target y are needed)
model = Sequential()
model.add(Bidirectional(LSTM(100), input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1], activation='softmax'))
model.load_weights("model.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', cus_met.fbeta_score, ])

# start Training
history = model.fit(train_X, train_y, batch_size=30, epochs=2, validation_data=[test_X, test_y],verbose=2)

# save model to disk
model.save_weights("model.h5")

# plot the history
pyplot.plot(history.history['fbeta_score'], label='train')
pyplot.plot(history.history['val_fbeta_score'], label='test')
pyplot.legend()
#pyplot.show()


validate_dataset('subject5_ideal', model, n_timesteps, n_classes)

validate_dataset('subject5_self', model, n_timesteps, n_classes)

validate_dataset('subject3_ideal', model, n_timesteps, n_classes)

validate_dataset('subject3_self', model, n_timesteps, n_classes)


print('Ende')



