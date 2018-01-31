from pandas import read_csv

subject_number = [1,2,3,4,5,7,8,9,10]
logfile_names = []
for subject in subject_number:
    logfile_names.append('subject{}_ideal'.format(subject))
    logfile_names.append('subject{}_self'.format(subject))

#logfile_names = ['subject5_ideal', 'subject3_ideal', 'subject5_self', 'subject3_self', 'subject4_ideal', 'subject4_self']
logfile_path = '../Dataset/realistic_sensor_displacement/'
prepared_dataset_output_path = '../Dataset/'

save_as_HDF5_File = True

for counter, name in enumerate(logfile_names):

    # read the dataset from log file with Tab as Delimiter
    dataset = read_csv(logfile_path + name + '.log', sep='\t', lineterminator='\n', header=None)

    # generate labels for the dataset
    columns_names = ['Sec', 'MicSec']
    for x in range(117):
        columns_names.append('Sensor%d' % (x))
    columns_names.append('Activity')
    dataset.columns = columns_names

    # Summarize both time columns to one column in Seconds
    dataset['Sec'] = dataset['Sec'] + dataset['MicSec'] * float(0.0000001)
    dataset = dataset.drop(columns=['MicSec'])
    dataset.index.name = 'index'

    #print(dataset.head())

    if len(logfile_names) > 1:
        if counter == 0:
            # Overwrite a existing file at the first loop run
            print('Combining {} logfiles to one File'.format(len(logfile_names)))
            if save_as_HDF5_File:
                dataset.to_hdf(prepared_dataset_output_path + 'prepared_' + 'combination_' + logfile_names[0] + '_et_al' + '.hdf', 'key', mode='w',
                               append=True)
            else:
                dataset.to_csv(prepared_dataset_output_path + 'prepared_' + 'combination_' + logfile_names[0] + '_et_al' + '.csv', mode='w',
                               index=False)
        else:
            # Append the following logfiles to the existing csv-file
            if save_as_HDF5_File:
                dataset.to_hdf(prepared_dataset_output_path + 'prepared_' + 'combination_' + logfile_names[0] + '_et_al' + '.hdf', 'key', mode='a',
                               append=True)
            else:
                dataset.to_csv(prepared_dataset_output_path + 'prepared_' + 'combination_' + logfile_names[0] + '_et_al' + '.csv', mode='a',
                               header=False, index=False)


    else:
        if save_as_HDF5_File:
            dataset.to_hdf(prepared_dataset_output_path + 'prepared_' + logfile_names[0] + '.hdf', 'key')
        else:
            dataset.to_csv(prepared_dataset_output_path + 'prepared_' + logfile_names[0] + '.csv', index=False)

    print('Finished preparing {} logfile '.format(name))

print('Finished logfile queue!')


