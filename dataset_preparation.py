from pandas import read_csv

# read the dataset from log file with Tab as Delimiter
name_logfile = 'subject3_ideal'
logfile_path = '../Bi_LSTM_for_REALDISP/Dataset/realistic_sensor_displacement/'
dataset = read_csv(logfile_path + name_logfile + '.log', sep='\t', lineterminator='\n')

# generate labels for the dataset
columns_names = ['Sec', 'MicSec']
for x in range(117):
    columns_names.append('Sensor%d' % (x))
columns_names.append('Activity')
dataset.columns = columns_names

# Summarize both time columns to one column in Seconds
dataset['Sec'] = dataset['Sec'] + dataset['MicSec']*float(0.0000001)
dataset = dataset.drop(columns=['MicSec'])
dataset.index.name = 'index'

print(dataset.head())

# save prepared dataset as csv file
dataset.to_csv('prepared_' + name_logfile + '.csv')

