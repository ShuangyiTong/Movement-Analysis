import os
import time
import numpy as np
import tensorflow as tf

import pose_playground.pose_models.VNect.utils as utils
from pose_playground.pose_models.VNect.net import VNectNet
from pose_playground.utils.one_euro_filter import OneEuroFilter

class VNectEstimator:
    # the side length of the CNN input box
    box_size = 368
    # the input box size is 8 times the side length of the output heatmaps
    hm_factor = 8
    # sum of the joints to be detected
    joints_sum = 21
    # parent joint indexes of each joint (for plotting the skeletal lines)
    joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]
    # joint names by index
    joint_names = ['head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                   'left_wrist', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                   'pelvis', 'spine', 'head', 'right_hand', 'left_hand', 'right_toe', 'left_toe']

    def __init__(self, existing_sess=None):
        print('Initializing VNect Estimator...')
        # the scale factors to zoom down the input image crops
        # put different scales to get better average performance
        # for faster loops, use less scales e.g. [1], [1, 0.7]
        self.scales = [1, 0.85, 0.7]
        # initializing one euro filters for all the joints
        filter_config_2d = {
            'freq': 30,        # system frequency about 30 Hz
            'mincutoff': 1.7,  # value refer to the paper
            'beta': 0.3,       # value refer to the paper
            'dcutoff': 0.1     # not mentioned, empirically set
        }
        filter_config_3d = {
            'freq': 30,        # system frequency about 30 Hz
            'mincutoff': 0.8,  # value refer to the paper
            'beta': 0.4,       # value refer to the paper
            'dcutoff': 0.1     # not mentioned, empirically set
        }
        self.filter_2d = [(OneEuroFilter(**filter_config_2d),
                           OneEuroFilter(**filter_config_2d))
                          for _ in range(self.joints_sum)]
        self.filter_3d = [(OneEuroFilter(**filter_config_3d),
                           OneEuroFilter(**filter_config_3d),
                           OneEuroFilter(**filter_config_3d))
                          for _ in range(self.joints_sum)]
        self.filter_time = None


        if existing_sess:
            self.sess = existing_sess
        else:
            self.sess = tf.Session()
            self.net = VNectNet()
            # load pretrained VNect model
            this_file_path = os.path.dirname(os.path.abspath(__file__))
            weight_file = os.path.join(this_file_path, '../../model_data/vnect_weights.pkl')
            self.net.load_weights(self.sess, weight_file)

        graph = tf.get_default_graph()
        self.input_crops = graph.get_tensor_by_name('Placeholder:0')
        self.heatmap = graph.get_tensor_by_name('split_2:0')
        self.x_heatmap = graph.get_tensor_by_name('split_2:1')
        self.y_heatmap = graph.get_tensor_by_name('split_2:2')
        self.z_heatmap = graph.get_tensor_by_name('split_2:3')

        print('VNect Estimator initialization complete.')

    @staticmethod
    def gen_input_batch(img_input, box_size, scales):
        # input image --> sqrared image acceptable for the model
        img_square, scaler, [offset_x, offset_y] = utils.img_scale_squarify(img_input, box_size)
        # generate multi-scale image batch
        input_batch = []
        for scale in scales:
            img = utils.img_scale_padding(img_square, scale, box_size) if scale < 1 else img_square
            input_batch.append(img)
        # image value range: [0, 255) --> [-0.4, 0.6)
        input_batch = np.asarray(input_batch, dtype=np.float32) / 255 - 0.4
        return input_batch, scaler, [offset_x, offset_y]

    def joint_filter(self, joints, dim, time_delta=None):
        if time_delta and self.filter_time:
            self.filter_time += time_delta
        else:
            self.filter_time = time.time()

        if dim == 2:
            for i in range(self.joints_sum):
                joints[i, 0] = self.filter_2d[i][0](joints[i, 0], self.filter_time)
                joints[i, 1] = self.filter_2d[i][1](joints[i, 1], self.filter_time)
        elif dim == 3:
            for i in range(self.joints_sum):
                joints[i, 0] = self.filter_3d[i][0](joints[i, 0], self.filter_time)
                joints[i, 1] = self.filter_3d[i][1](joints[i, 1], self.filter_time)
                joints[i, 2] = self.filter_3d[i][2](joints[i, 2], self.filter_time)
        else:
            raise NotImplementedError('Filter not implemented for dim ' + str(dim))

        return joints

    def __call__(self, img_input, time_delta=None):
        t0 = time.time()
        img_batch, scaler, [offset_x, offset_y] = self.gen_input_batch(img_input, self.box_size, self.scales)
        hm, xm, ym, zm = self.sess.run([self.heatmap,
                                        self.x_heatmap,
                                        self.y_heatmap,
                                        self.z_heatmap],
                                       {self.input_crops: img_batch})
        # averaging the outputs with different scales
        hm_size = self.box_size // self.hm_factor
        hm_avg = np.zeros((hm_size, hm_size, self.joints_sum))
        xm_avg = np.zeros((hm_size, hm_size, self.joints_sum))
        ym_avg = np.zeros((hm_size, hm_size, self.joints_sum))
        zm_avg = np.zeros((hm_size, hm_size, self.joints_sum))
        for i in range(len(self.scales)):
            rescale = 1.0 / self.scales[i]
            scaled_hm = utils.img_scale(hm[i, :, :, :], rescale)
            scaled_x_hm = utils.img_scale(xm[i, :, :, :], rescale)
            scaled_y_hm = utils.img_scale(ym[i, :, :, :], rescale)
            scaled_z_hm = utils.img_scale(zm[i, :, :, :], rescale)
            mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
            hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            xm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            ym_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            zm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
        hm_avg /= len(self.scales)
        xm_avg /= len(self.scales)
        ym_avg /= len(self.scales)
        zm_avg /= len(self.scales)

        # joints_2d are in box size scale
        joints_2d = utils.extract_2d_joints(hm_avg, self.box_size, self.hm_factor)
        joints_2d = self.joint_filter(joints_2d, 2, time_delta)
        joints_3d = utils.extract_3d_joints(joints_2d, xm_avg, ym_avg, zm_avg, self.hm_factor)
        joints_3d = self.joint_filter(joints_3d, 3, time_delta)

        # rescale joints_2d to input image scale
        joints_2d[:, 0] = (joints_2d[:, 0] - offset_y) / scaler
        joints_2d[:, 1] = (joints_2d[:, 1] - offset_x) / scaler

        print('FPS: {:>2.2f}'.format(1 / (time.time() - t0)), end='\r')
        return joints_2d, joints_3d