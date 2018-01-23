
PNET_LAYERS_PRUNE = {
                      'conv1': {
                                 'remove' : [('conv1',  'weights', 3),
                                             ('conv1',  'biases',  0),
                                             ('PReLU1', 'alpha',   0)],
                                 'update' : ('conv2', 'weights')
                               },

                      'conv2': {
                                 'remove' : [('conv2',  'weights', 3),
                                             ('conv2',  'biases',  0),
                                             ('PReLU2', 'alpha',   0)],
                                 'update' : ('conv3', 'weights')
                               }
                    }

