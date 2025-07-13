import numpy as np


def reshape_data(input_file):
    data = np.loadtxt(input_file)
    reshaped_data = data.reshape(500, 2048)
    return reshaped_data

def convert_data(input_data_files, output_data='ver_clean_merged.1.npy'):
    data_list = [reshape_data(file) for file in input_data_files]
    merged_data = np.vstack(data_list)
    np.save(output_data, merged_data)