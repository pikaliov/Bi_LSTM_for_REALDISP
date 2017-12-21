from deepautoencoder import StackedAutoEncoder
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib



dataset_name = 'prepared_subject3_ideal.csv'

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


loaded_model = joblib.load('SDAmodel.sav')

test_X_latent = loaded_model.transform(test_X)

print('End')