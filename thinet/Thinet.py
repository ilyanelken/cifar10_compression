# built-in

# 3rd party
import numpy as np

# misc
import config

class Thinet:

    def __init__(self, input_map, weights, output_map, rate):
        """
        :param input_map:   input activation map     [batch_size x W x H x C1]
        :param weights:     convolutional filters    [K x K x C1 x C2]
        :param output_map:  output activation map    [batch_size x W x H x C2]
        :param rate:        desired compression rate [0 <= rate <= 1.0]
        """
        # Input data
        self.input      = input_map
        self.weights    = weights
        self.output     = output_map
        self.rate       = rate

        # Comression results placeholders
        self.keep_filters = None
        self.remove_filters = None
        self.weights_wo_reconst = None
        self.weights_reconst = None

    @staticmethod
    def get_random_coords(width, height, channels, n):
        """ Generate random coordinates in the following shape [n x 3]"""
        coords = np.zeros((n, 3), dtype=np.int16)
    
        coords[:, 0] = np.random.randint(0, width, size=n)
        coords[:, 1] = np.random.randint(0, height, size=n)
        coords[:, 2] = np.random.randint(0, channels, size=n)
    
        return coords  

    def __compute_eq6(self, x_idx, channels):

        weights = self.weights
        input_map = self.input

        assert (weights.shape[0] == weights.shape[1]), "invalid kernel size"

        res = 0
        offset = weights.shape[0] // 2

        for map_in in input_map:
            for p in x_idx:
                x, y, c = p
                T_sum = 0
                for j in channels:
                    kernel = np.squeeze(weights[:, :, j, c])
                    roi = map_in[y - offset:y + offset + 1, x - offset:x + offset + 1, j]
 
                    T_sum += np.sum(roi * kernel)
                res += np.power(T_sum, 2)

        return res

    def __compute_eq6_fast(self, x_idx, channels):

        weights = self.weights
        input_map = self.input

        assert (weights.shape[0] == weights.shape[1]), "invalid kernel size"

        res = 0
        offset = weights.shape[0] // 2

        for p in x_idx:
            x, y, c = p
            T_sum = np.zeros((input_map.shape[0],))
            for j in channels:
                kernel = np.expand_dims(np.squeeze(weights[:, :, j, c]), axis=0)
                roi = input_map[:, y - offset:y + offset + 1, x - offset:x + offset + 1, j]
                T_sum += np.sum(roi * kernel, axis=(1, 2))
            res += np.sum(np.power(T_sum, 2))

        return res

    def __best_filters_scales(self, x_idx, y_idx, filters):
        """
        Minimize the reconstruction error by weighting valid channels,
        based on: Thinet: equation 7
        """
        batch_size1 = self.input.shape[0]
        batch_size2 = self.output.shape[0]

        assert (batch_size1 == batch_size2), "batch size is not consistent"

        batch_size  = batch_size1
        input_map   = self.input
        output_map  = self.output

        channels    = len(filters)
        sample_pts  = y_idx.shape[0]
        offset      = self.weights.shape[0] // 2

        X = np.zeros((sample_pts * batch_size, channels), dtype=np.float32)
        Y = np.zeros((sample_pts * batch_size,), dtype=np.float32)

        # Compute Y vector [samples*batch_size x 1]
        for i, p in enumerate(y_idx):
            x, y, c = p
            Y[i*batch_size : (i+1)*batch_size] = output_map[:, y, x, c]

        # Compute X matrix [samples*batch_size x channels]
        for i, p in enumerate(x_idx):
            x, y, c = p
            for j, f in enumerate(filters):
                kernel = np.expand_dims(np.squeeze(self.weights[:, :, f, c]), axis=0)
                roi    = input_map[:, y - offset:y + offset + 1, x - offset:x + offset + 1, f]
                X[i*batch_size : (i+1)*batch_size, j] = np.sum(kernel * roi, axis=(1, 2))

        #
        # Eq. 7:
        #
        #      w_hat = (X^T * X)^(-1) * X^T * Y
        #
        var_1 = np.linalg.inv(np.dot(X.T, X))
        var_2 = np.dot(X.T, Y)

        # Channel filters weights [channels x 1]
        w_hat = np.dot(var_1, var_2)

        return np.squeeze(w_hat.T)


    def __prune_weights(self, filters, scales):

        weights = self.weights
        K1      = weights.shape[0]
        K2      = weights.shape[1]
        C1      = len(filters)
        C2      = weights.shape[3]

        assert (K1 == K2), "invalid kernel size"

        weights_wo_reconst = np.zeros((K1, K2, C1, C2), dtype=weights.dtype)
        weights_reconst = np.zeros((K1, K2, C1, C2), dtype=weights.dtype)
        for i, f in enumerate(filters):
            weights_wo_reconst[:, :, i, :] = weights[:, :, f, :]
            weights_reconst[:, :, i, :] = weights[:, :, f, :] * scales[i]

        return weights_wo_reconst, weights_reconst


    def compress(self, sample_points=10):
        """
        Compute the most important filters according to compression rate,
        based on: ThiNet: Algorithm 1 - a greedy algorithm for minimizing Eq.6

        :param sample_points:  number of points to use in each activation map
        :return I:             filters to be preserved in layer i
        :return w_reconst:     layer i+1 reconstructed filters               
        """

        H2 = self.output.shape[1]
        W2 = self.output.shape[2]
        C1 = self.weights.shape[2]
        C2 = self.weights.shape[3]

        y_idx = Thinet.get_random_coords(W2, H2, C2, sample_points)
        x_idx = np.copy(y_idx)
        x_idx[:, 0:2] += 1  # corresponding input (x, y) coordinates (assumed kernel size: 3x3)

        T = []                # list of filters to prune
        I = list(range(C1))   # list of remaining filters
        while len(T) < round(C1 * (1 - self.rate)):
            min_value = np.inf
            for i in I:
                tmpT = list(T)
                tmpT.append(i)

                value = self.__compute_eq6_fast(x_idx, tmpT)

                #value_fast = self.__compute_eq6_fast(x_idx, tmpT)
                #if value != value_fast:
                #    raise ValueError("%f != %f" % (value, value_fast))

                if value < min_value:
                    min_value = value
                    min_i = i
            I.remove(min_i)
            T.append(min_i)

        # Minimize reconstruction error
        W_hat = self.__best_filters_scales(x_idx, y_idx, I)
        W_wo_reconst, W_reconst = self.__prune_weights(I, W_hat)

        # Update compression results
        self.keep_filters = I
        self.remove_filters = T
        self.weights_wo_reconst = W_wo_reconst
        self.weights_reconst = W_reconst

        print("Filters to preserve: ", I)

    def compress_callback(self, data_dict, params, with_reconstruction):
        """
        :param data_dict:  all network variables
        :param params:     network variables to remove in the following format:

               params = { 'remove' : [('conv1',  'weights', 3),
                                      ('conv1',  'biases',  0),
                                      ('PReLU1', 'alpha',   0)],
                           'update' : ('conv2', 'weights') }
        """

        print('\nRemoved parameters:\n')
        for rec in params['remove']:
            layer, key, axis = rec
            print("\t[%s][%s] %s --> " % (
                  layer, key, str(data_dict[layer][key].shape)), end="")
            data_dict[layer][key] = np.delete(data_dict[layer][key], self.remove_filters, axis=axis)
            print("%s" % str(data_dict[layer][key].shape))

        print('\nUpdated parameters:\n')
        layer, key = params['update']
        print("\t[%s][%s] %s --> " % (
              layer, key, str(data_dict[layer][key].shape)), end="")
        if with_reconstruction:
            data_dict[layer][key] = self.weights_reconst
        else:
            data_dict[layer][key] = self.weights_wo_reconst
        print(data_dict[layer][key].shape)

        print('')


    def update_model(self, pnet_orig_path, pnet_compressed_path, layer, with_reconstruction=True):

        if self.keep_filters is None:
            print("Please run compress() method first")
            return

        if layer not in config.PNET_LAYERS_PRUNE:
            print("Invalid layer name: %s")
            return

        data_dict = np.load(pnet_orig_path, encoding='latin1').item()

        # data dict is updated inplace
        self.compress_callback(data_dict,
                               config.PNET_LAYERS_PRUNE[layer],
                               with_reconstruction)

        np.save(pnet_compressed_path, data_dict)

