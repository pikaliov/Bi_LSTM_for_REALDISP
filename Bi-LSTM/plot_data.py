from pandas import read_csv
from matplotlib import pyplot


#load dataset
prepared_dataset_path = '../Dataset/'
dataset = read_csv(prepared_dataset_path + 'prepared_subject5_ideal.csv', header=0, index_col=0)
values = dataset.values

#specifiy columns to plot
groups = [1,2,3,4,5,6,7,8,9,10,11,12,13]
i = 1

#plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1

pyplot.show()