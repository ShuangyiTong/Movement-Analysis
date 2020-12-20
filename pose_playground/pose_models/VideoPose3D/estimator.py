import os
import torch
import numpy as np

from pose_playground.pose_models.VideoPose3D.net_coco2human36m import TemporalModel
from pose_playground.pose_models.OpenPose.estimator import OpenPose2DEstimator
from pose_playground.pose_models.VideoPose3D.utils import normalizeScreenCoordinates

class VideoPose3DEstimator:
    def __init__(self, causal):
        self.coco_2d_estimator = OpenPose2DEstimator()
        self.num_joints = self.coco_2d_estimator.num_joints
        self.causal = causal

        self.net = TemporalModel(17, # pretrained model is trained on 2D Coco data input with 17 joints
                                 2,  # input is 2D joint coordinates
                                 17, # pretrained model outputs Human3.6 joints
                                 filter_widths=(3, 3, 3, 3, 3), # use recommended values
                                 causal=self.causal, # False for non-real time inference
                                 dropout=0.25, channels=1024, dense=False)
    
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        checkpoint = torch.load(os.path.join(this_file_path, '../../model_data/pretrained_h36m_detectron_coco.bin'),
                                map_location=lambda storage, loc: storage)

        self.net.load_state_dict(checkpoint['model_pos'])
        self.net.eval()
        torch.no_grad()

        self.has_cuda = self.coco_2d_estimator.has_cuda
        if self.has_cuda:
            self.net.cuda()

        self.receptive_field = self.net.receptive_field()
        self.pad = (self.receptive_field - 1) // 2
        if self.causal:
            self.causal_shift = self.pad
        else:
            self.causal_shift = 0

        # Coco is symmetric on all these joints except the head 0,
        # use this to augment input space and averaging to get better result
        self.input_left_joints = [1, 3, 5, 7, 9, 11, 13, 15]
        self.input_right_joints = [2, 4, 6, 8, 10, 12, 14, 16]

        self.output_left_joints = [4, 5, 6, 11, 12, 13]
        self.output_right_joints = [1, 2, 3, 14, 15, 16]

        self.joints_pool_array = None
    
    def __call__(self, img_input, time_delta=None):
        joints_2d = self.coco_2d_estimator(img_input, time_delta=time_delta)
        joints_2d = np.expand_dims(joints_2d, axis=0)
        joints_2d[..., :2] = normalizeScreenCoordinates(joints_2d[..., :2], img_input.shape[1], img_input.shape[0])
        if self.joints_pool_array != None:
            if joints_pool_array.shape[0] >= self.receptive_field:
                self.joints_pool_array = np.append(self.joints_pool_array[1:], joints_2d, axis=0)
        else:
            self.joints_pool_array = joints_2d
        
        input_2d = np.expand_dims(np.pad(self.joints_pool_array,
                                        ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                                        'edge'), axis=0)
        
        input_2d = np.concatenate((input_2d, input_2d), axis=0)
        input_2d[1, :, :, 0] *= -1
        input_2d[1, :, self.input_left_joints + self.input_right_joints] = input_2d[1, :, self.input_right_joints + self.input_left_joints]

        input_2d = torch.from_numpy(input_2d)
        if self.has_cuda:
            input_2d.cuda()

        joints_3ds = model_pos(input_2d)
        joints_3ds[1, :, :, 0] *= -1
        joints_3ds[1, :, self.output_left_joints + self.output_right_joints] = joints_3ds[1, :, self.output_right_joints + self.output_left_joints]
        joints_3ds = torch.mean(joints_3ds, dim=0, keepdim=True)

        joints_3d = (joints_3ds.squeeze(0).cpu().detach().numpy())[-1]

        del input_2d
        del joints_3ds

        return joints_2d, joints_3d