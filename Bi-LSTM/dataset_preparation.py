from pandas import read_csv


logfile_names = ['subject5_ideal', 'subject3_ideal', 'subject5_self', 'subject3_self']
logfile_path = '../Dataset/realistic_sensor_displacement/'
prepared_dataset_output_path = '../Dataset/'


for counter, name in enumerate(logfile_names):

    # read the dataset from log file with Tab as Delimiter
    dataset = read_csv(logfile_path + name + '.log', sep='\t', lineterminator='\n')

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
            print('Combining %d logfiles to one CSV-File' % len(logfile_names))
            dataset.to_csv('prepared_' + 'combination_' + logfile_names[0] + '_et_al' + '.csv', mode='w', index=False)
        else:
            # Append the following logfiles to the existing csv-file
            dataset.to_csv('prepared_' + 'combination_' + logfile_names[0] + '_et_al' + '.csv', mode='a', header=False, index=False)


    else:
        dataset.to_csv(prepared_dataset_output_path + 'prepared_' + logfile_names[0] + '.csv', index=False)

    print('Finished preparing %d logfile ' % (counter + 1))

print('Finished logfile queue!')


