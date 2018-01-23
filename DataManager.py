import np

def save_data_to_file(out_file, conv1_w, conv2_w, conv2_in_arr, conv2_out_arr):

    np.savez_compressed(OUTPUT_FILE, conv1_weights = conv1_w,
                                     conv2_weights = conv2_w,
                                     conv2_input = conv2_in_arr,
                                     conv2_output = conv2_out_arr)

def load_data_from_file(in_file):

    with np.load(in_file) as data:

        conv1_weights = data['conv1_weights']
        conv2_weights = data['conv2_weights']

        conv2_input   = data['conv2_input']
        conv2_output  = data['conv2_output']

    return conv1_weights, conv2_weights, conv2_input, conv2_output
 
