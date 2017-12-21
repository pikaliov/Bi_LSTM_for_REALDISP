from deepautoencoder import StackedAutoEncoder
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib



# init Model
dataset_name = 'prepared_subject3_ideal.csv'
activation_function = 'softmax'
layer_epoch = 10000
sda_dims = [100, 70, 50, 30]
print_step = round(layer_epoch/2)

# init same settings for each layer
epochs = []
acti_func_arr = []
for x in range(len(sda_dims)):
    acti_func_arr.append(activation_function)
    epochs.append(layer_epoch)




# load dataset
dataset = read_csv(dataset_name, header=0, index_col=0)

# drop the timestamp
dataset = dataset.drop('Sec', 0)

val = dataset.values

# verify if all data is float
val = val.astype('float32')

# normalize only the input features
scaler = MinMaxScaler(feature_range=(0, 1))
val[:, :-1] = scaler.fit_transform(val[:, :-1])

# split into train and test data
train_to_test_ratio = 0.7
spl = np.random.rand(val.shape[0]) < train_to_test_ratio
train_X, train_y = val[spl, :-1], val[spl, -1]
test_X, test_y = val[~spl, :-1], val[~spl, -1]

model = StackedAutoEncoder(dims=sda_dims, activations=acti_func_arr, epoch=epochs, loss= 'rmse', lr=0.007,
                           batch_size=100, print_step=print_step, noise='gaussian')
model.fit(train_X, test_X)

test_X_latent = model.transform(test_X)

joblib.dump(model, 'SDAmodel.sav')

print('End')

