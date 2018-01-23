import numpy as np

def save_data_to_file(out_file, conv1_w, conv2_w, conv2_in_arr, conv2_out_arr):

    np.savez_compressed(OUTPUT_FILE, conv1_weights = conv1_w,
                                     conv2_weights = conv2_w,
                                     conv2_input = conv2_in_arr,
                                     conv2_output = conv2_out_arr)

def load_data_from_file(in_file):

    data_dict = {}

    with np.load(in_file) as data:

        data_dict['conv1_weights'] = data['conv1_weights']
        data_dict['conv2_weights'] = data['conv2_weights']

        data_dict['conv2_input']   = data['conv2_input']
        data_dict['conv2_output']  = data['conv2_output']

    print('Loaded %d samples' % (data_dict['conv2_input'].shape[0]))

    return data_dict
 
