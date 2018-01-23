import os
import cv2
import numpy as np
from scipy import misc
import tensorflow as tf

import detect_face

class FaceDetectorMTCNN:

    def __init__(self, model_path, pnet_channels=(10, 16), scale=0.3, gpu_memory_fraction=0.2):
        """
        Initialize the MTCNN face detector.

        :param pnet_channels:       PNet convolution layers channels
        :param model_path:          Directory containing the MTCNN model files.
        :param gpu_memory_fraction: Maximum GPU memory fraction to use.
        """

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1}, gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, model_path, pnet_channels)

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # pyramid scale factor

        self.scale = scale  # input scale factor


    def apply_mtcnn(self, img):
        """
        Apply the MTCNN face detector to the specified image.

        :param img: Input image (grayscale or color).
        :return: Tuple (bounding_boxes, pnet_run_time, layers), where:
          bounding_boxes  : Nx5 numpy array, containing (left, top, right, bottom, confidence) for each face
          pnet_run_time   : total pnet run time (sum for all input scales)
          layers          : a dictionary with all layers activation maps and weights
        """
        img_resized = misc.imresize(img, self.scale)

        if img.ndim == 2:
            img = FaceDetectorMTCNN.to_rgb(img)

        img = img[:, :, 0:3]
        bounding_boxes, pnet_run_time, heatmaps = detect_face.detect_face(
            img_resized, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor
        )

        bounding_boxes /= self.scale

        return bounding_boxes, pnet_run_time, heatmaps

    @staticmethod
    def draw_bbox(img, bboxes, color = (255, 0, 0), thickness = 2):
    
        for face_idx in range(bboxes.shape[0]):
            x1 = int(bboxes[face_idx, 0])
            y1 = int(bboxes[face_idx, 1])
            x2 = int(bboxes[face_idx, 2])
            y2 = int(bboxes[face_idx, 3])
    
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def to_rgb(img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
