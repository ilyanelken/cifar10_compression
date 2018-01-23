# built-in
import os
import random

# 3rd party
from scipy import misc
import numpy as np
from tqdm import tqdm

# misc
from FaceDetectorMTCNN import FaceDetectorMTCNN


class DataManager():

    def __init__(self, mtcnn_model_path, pnet_channels=(10, 16)):

        self.weights_conv1 = None
        self.weights_conv2 = None
        self.weights_conv3 = None

        self.scale_1_conv2_in = None
        self.scale_1_conv2_out = None
        self.scale_1_conv3_in = None
        self.scale_1_conv3_out = None

        self.scale_2_conv2_in = None
        self.scale_2_conv2_out = None
        self.scale_2_conv3_in = None
        self.scale_2_conv3_out = None

        self.scale_3_conv2_in = None
        self.scale_3_conv2_out = None
        self.scale_3_conv3_in = None
        self.scale_3_conv3_out = None

        self.scale_4_conv2_in = None
        self.scale_4_conv2_out = None
        self.scale_4_conv3_in = None
        self.scale_4_conv3_out = None

        self.scale_5_conv2_in = None
        self.scale_5_conv2_out = None
        self.scale_5_conv3_in = None
        self.scale_5_conv3_out = None

        self.scale_6_conv2_in = None
        self.scale_6_conv2_out = None
        self.scale_6_conv3_in = None
        self.scale_6_conv3_out = None

        self.mtcnn = FaceDetectorMTCNN(mtcnn_model_path,
                                       pnet_channels=pnet_channels,
                                       gpu_memory_fraction=0.2)

        # Load PNet model weights 
        pnet_model_path = os.path.join(mtcnn_model_path, 'det1.npy')
        if not os.path.isfile(pnet_model_path):
            raise ValueError('PNet load model: invalid file specified (%s)' % load_file)

        data_dict = np.load(pnet_model_path, encoding='latin1').item()

        self.weights_conv1 = data_dict['conv1']['weights']
        self.weights_conv2 = data_dict['conv2']['weights']
        self.weights_conv3 = data_dict['conv3']['weights'] 

        #for i, w in enumerate(self.weights):
        #    print("conv%d " % i, w.shape) 


    def load_data(self, img_list_file, num_images_max=None, crop_size=(480, 640)):

        num_images = 0

        self.scale_1_conv2_in = []
        self.scale_1_conv2_out = []
        self.scale_1_conv3_in = []
        self.scale_1_conv3_out = []

        self.scale_2_conv2_in = []
        self.scale_2_conv2_out = []
        self.scale_2_conv3_in = []
        self.scale_2_conv3_out = []

        self.scale_3_conv2_in = []
        self.scale_3_conv2_out = []
        self.scale_3_conv3_in = []
        self.scale_3_conv3_out = []

        self.scale_4_conv2_in = []
        self.scale_4_conv2_out = []
        self.scale_4_conv3_in = []
        self.scale_4_conv3_out = []

        self.scale_5_conv2_in = []
        self.scale_5_conv2_out = []
        self.scale_5_conv3_in = []
        self.scale_5_conv3_out = []

        self.scale_6_conv2_in = []
        self.scale_6_conv2_out = []
        self.scale_6_conv3_in = []
        self.scale_6_conv3_out = []
       
        with open(img_list_file, 'r') as f:
            img_list = f.readlines()

        random.shuffle(img_list)

        for i, img_file in tqdm(enumerate(img_list)):

            img_file = img_file.strip()
            
            if not os.path.isfile(img_file):
                print("Image %s doesn't exist" % img_file)
                continue
            
            img = misc.imread(img_file, mode='RGB')
            
            h, w = img.shape[0:2]
        
            if w < crop_size[1] or h < crop_size[0]:
                continue
            
            # Crop center
            xs = (w - crop_size[1]) // 2
            ys = (h - crop_size[0]) // 2
            
            img_crop = img[ys:ys + crop_size[0], xs:xs + crop_size[1], :]
           
            bbox, pnet_run_time, heatmaps = self.mtcnn.apply_mtcnn(img_crop)

            if bbox.shape[0] == 0:
                #print("No face found in %s" % img_file)
                continue
            
            num_images += 1

            self.scale_1_conv2_in.append(heatmaps[1][0])
            self.scale_1_conv2_out.append(heatmaps[2][0])
            self.scale_1_conv3_in.append(heatmaps[3][0])
            self.scale_1_conv3_out.append(heatmaps[4][0])

            self.scale_2_conv2_in.append(heatmaps[1][1])
            self.scale_2_conv2_out.append(heatmaps[2][1])
            self.scale_2_conv3_in.append(heatmaps[3][1])
            self.scale_2_conv3_out.append(heatmaps[4][1])

            self.scale_3_conv2_in.append(heatmaps[1][2])
            self.scale_3_conv2_out.append(heatmaps[2][2])
            self.scale_3_conv3_in.append(heatmaps[3][2])
            self.scale_3_conv3_out.append(heatmaps[4][2])

            self.scale_4_conv2_in.append(heatmaps[1][3])
            self.scale_4_conv2_out.append(heatmaps[2][3])
            self.scale_4_conv3_in.append(heatmaps[3][3])
            self.scale_4_conv3_out.append(heatmaps[4][3])

            self.scale_5_conv2_in.append(heatmaps[1][4])
            self.scale_5_conv2_out.append(heatmaps[2][4])
            self.scale_5_conv3_in.append(heatmaps[3][4])
            self.scale_5_conv3_out.append(heatmaps[4][4])

            self.scale_6_conv2_in.append(heatmaps[1][5])
            self.scale_6_conv2_out.append(heatmaps[2][5])
            self.scale_6_conv3_in.append(heatmaps[3][5])
            self.scale_6_conv3_out.append(heatmaps[4][5])

            if num_images_max is not None:
                if num_images >= num_images_max:
                    break

            #if plot_output:
            #    FaceDetectorMTCNN.draw_bbox(img, bbox)
            #    plt.figure(figsize=(8, 8));
            #    plt.imshow(img);
            #    plt.title('%s' % (img_file))
            #    plt.show()

        self.scale_1_conv2_in  = np.array(self.scale_1_conv2_in)
        self.scale_1_conv2_out = np.array(self.scale_1_conv2_out)
        self.scale_1_conv3_in  = np.array(self.scale_1_conv3_in)
        self.scale_1_conv3_out = np.array(self.scale_1_conv3_out)
        
        self.scale_2_conv2_in  = np.array(self.scale_2_conv2_in)
        self.scale_2_conv2_out = np.array(self.scale_2_conv2_out)
        self.scale_2_conv3_in  = np.array(self.scale_2_conv3_in)
        self.scale_2_conv3_out = np.array(self.scale_2_conv3_out)

        self.scale_3_conv2_in  = np.array(self.scale_3_conv2_in)
        self.scale_3_conv2_out = np.array(self.scale_3_conv2_out)
        self.scale_3_conv3_in  = np.array(self.scale_3_conv3_in)
        self.scale_3_conv3_out = np.array(self.scale_3_conv3_out)

        self.scale_4_conv2_in  = np.array(self.scale_4_conv2_in)
        self.scale_4_conv2_out = np.array(self.scale_4_conv2_out)
        self.scale_4_conv3_in  = np.array(self.scale_4_conv3_in)
        self.scale_4_conv3_out = np.array(self.scale_4_conv3_out)

        self.scale_5_conv2_in  = np.array(self.scale_5_conv2_in)
        self.scale_5_conv2_out = np.array(self.scale_5_conv2_out)
        self.scale_5_conv3_in  = np.array(self.scale_5_conv3_in)
        self.scale_5_conv3_out = np.array(self.scale_5_conv3_out)

        self.scale_6_conv2_in  = np.array(self.scale_6_conv2_in)
        self.scale_6_conv2_out = np.array(self.scale_6_conv2_out)
        self.scale_6_conv3_in  = np.array(self.scale_6_conv3_in)
        self.scale_6_conv3_out = np.array(self.scale_6_conv3_out)

        print("Loaded %d out of %d available images" % (num_images, len(img_list)))


    def save_data_to_file(self, out_file):

        np.savez_compressed(out_file,

                            weights_conv1 = self.weights_conv1,
                            weights_conv2 = self.weights_conv2,
                            weights_conv3 = self.weights_conv3,

                            scale_1_conv2_in  = self.scale_1_conv2_in,
                            scale_1_conv2_out = self.scale_1_conv2_out,
                            scale_1_conv3_in  = self.scale_1_conv3_in,
                            scale_1_conv3_out = self.scale_1_conv3_out,

                            scale_2_conv2_in  = self.scale_2_conv2_in,
                            scale_2_conv2_out = self.scale_2_conv2_out,
                            scale_2_conv3_in  = self.scale_2_conv3_in,
                            scale_2_conv3_out = self.scale_2_conv3_out,

                            scale_3_conv2_in  = self.scale_3_conv2_in,
                            scale_3_conv2_out = self.scale_3_conv2_out,
                            scale_3_conv3_in  = self.scale_3_conv3_in,
                            scale_3_conv3_out = self.scale_3_conv3_out,

                            scale_4_conv2_in  = self.scale_4_conv2_in,
                            scale_4_conv2_out = self.scale_4_conv2_out,
                            scale_4_conv3_in  = self.scale_4_conv3_in,
                            scale_4_conv3_out = self.scale_4_conv3_out,

                            scale_5_conv2_in  = self.scale_5_conv2_in,
                            scale_5_conv2_out = self.scale_5_conv2_out,
                            scale_5_conv3_in  = self.scale_5_conv3_in,
                            scale_5_conv3_out = self.scale_5_conv3_out,

                            scale_6_conv2_in  = self.scale_5_conv2_in,
                            scale_6_conv2_out = self.scale_5_conv2_out,
                            scale_6_conv3_in  = self.scale_5_conv3_in,
                            scale_6_conv3_out = self.scale_5_conv3_out)


    def load_data_from_file(self, in_file):

        with np.load(in_file) as data:

            self.weights_conv1 = data['weights_conv1']
            self.weights_conv2 = data['weights_conv2']
            self.weights_conv3 = data['weights_conv3']

            self.scale_1_conv2_in  = data['scale_1_conv2_in']
            self.scale_1_conv2_out = data['scale_1_conv2_out']
            self.scale_1_conv3_in  = data['scale_1_conv3_in']
            self.scale_1_conv3_out = data['scale_1_conv3_out']

            self.scale_2_conv2_in  = data['scale_2_conv2_in']
            self.scale_2_conv2_out = data['scale_2_conv2_out']
            self.scale_2_conv3_in  = data['scale_2_conv3_in']
            self.scale_2_conv3_out = data['scale_2_conv3_out']

            self.scale_3_conv2_in  = data['scale_3_conv2_in']
            self.scale_3_conv2_out = data['scale_3_conv2_out']
            self.scale_3_conv3_in  = data['scale_3_conv3_in']
            self.scale_3_conv3_out = data['scale_3_conv3_out']

            self.scale_4_conv2_in  = data['scale_4_conv2_in']
            self.scale_4_conv2_out = data['scale_4_conv2_out']
            self.scale_4_conv3_in  = data['scale_4_conv3_in']
            self.scale_4_conv3_out = data['scale_4_conv3_out']

            self.scale_5_conv2_in  = data['scale_5_conv2_in']
            self.scale_5_conv2_out = data['scale_5_conv2_out']
            self.scale_5_conv3_in  = data['scale_5_conv3_in']
            self.scale_5_conv3_out = data['scale_5_conv3_out']

            self.scale_6_conv2_in  = data['scale_6_conv2_in']
            self.scale_6_conv2_out = data['scale_6_conv2_out']
            self.scale_6_conv3_in  = data['scale_6_conv3_in']
            self.scale_6_conv3_out = data['scale_6_conv3_out']

        print('Loaded data for {} examples'.format(self.scale_1_conv2_in.shape[0]))

