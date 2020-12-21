import os
import torch
import numpy as np

from pose_playground.pose_models.VideoPose3D.net_coco2human36m import TemporalModel
from pose_playground.pose_models.OpenPose.estimator import OpenPose2DEstimator
from pose_playground.pose_models.VideoPose3D.utils import normalizeScreenCoordinates

class VideoPose3DEstimator:
    def __init__(self, causal):
        self.coco_2d_estimator = OpenPose2DEstimator(data_format='coco')
        self.causal = causal

        # human3.6m joints config
        self.num_output_joints = 17
        self.joint_parents = [7, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.joint_names = ['pelvis', 'left_hip', 'left_knee', 'left_foot', 'right_hip', 'right_knee', 'right_food',
                            'spine', 'upper_torso', 'neck_base', 'head', 'right_shoulder', 'right_elbow', 'right_hand',
                            'left_shoulder', 'left_elbow', 'left_hand']

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

        # Human3.6M symmetric joints
        self.output_left_joints = [4, 5, 6, 11, 12, 13]
        self.output_right_joints = [1, 2, 3, 14, 15, 16]

        self.joints_pool_array = None

    def Coco18to17(self, joints_2d):
        assert joints_2d.shape[0] == 18

        COCO17MAPTO18 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
        relocated_joints = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            relocated_joints[i] = joints_2d[COCO17MAPTO18[i]]

        return relocated_joints

    def getIntermediateInputTemplate(self):
        return np.zeros((0, 17, 2), dtype=np.float32)

    def getIntermediateInput(self, img_input, time_delta=None):
        joints_2d = self.coco_2d_estimator(img_input, time_delta=time_delta)
        joints_2d = self.Coco18to17(joints_2d)
        joints_2d = np.expand_dims(joints_2d, axis=0)
        return joints_2d
    
    def prepInputs(self, joints_2d_input):
        input_2d = np.expand_dims(np.pad(joints_2d_input,
                                        ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                                        'edge'), axis=0)
        
        input_2d = np.concatenate((input_2d, input_2d), axis=0)
        input_2d[1, :, :, 0] *= -1
        input_2d[1, :, self.input_left_joints + self.input_right_joints] = input_2d[1, :, self.input_right_joints + self.input_left_joints]

        input_2d = torch.from_numpy(input_2d)
        if self.has_cuda:
            input_2d = input_2d.cuda()

        return input_2d

    def processOutputs(self, joints_3d_output):
        joints_3d_output[1, :, :, 0] *= -1
        joints_3d_output[1, :, self.output_left_joints + self.output_right_joints] = joints_3d_output[1, :, self.output_right_joints + self.output_left_joints]
        joints_3d_output = torch.mean(joints_3d_output, dim=0, keepdim=True)
        joints_3d_output = joints_3d_output.squeeze(0).cpu().detach().numpy()
        joints_3d_output *= 1000 # scale to mm to be compatible with 3d ploting functions
        return joints_3d_output
    
    def __call__(self, img_input, time_delta=None):
        joints_2d = self.coco_2d_estimator(img_input, time_delta=time_delta)
        joints_2d = self.Coco18to17(joints_2d)
        output_joints_2d = joints_2d.astype(int)
        joints_2d = np.expand_dims(joints_2d, axis=0)
        joints_2d[..., :2] = normalizeScreenCoordinates(joints_2d[..., :2], img_input.shape[1], img_input.shape[0])
        if self.joints_pool_array is not None:
            if self.joints_pool_array.shape[0] >= self.receptive_field:
                self.joints_pool_array = np.append(self.joints_pool_array[1:], joints_2d, axis=0)
            else:
                self.joints_pool_array = np.append(self.joints_pool_array, joints_2d, axis=0)
        else:
            self.joints_pool_array = joints_2d

        input_2d = self.prepInputs(self.joints_pool_array)
        joints_3ds = self.net(input_2d)
        joints_3ds = self.processOutputs(joints_3ds)

        joints_3d = joints_3ds[-1]

        return output_joints_2d, joints_3d

    def batchInput(self, joints_2ds, w=None, h=None):
        assert w
        assert h
        output_joints_2ds = joints_2ds.copy()
        joints_2ds[..., :2] = normalizeScreenCoordinates(joints_2ds[..., :2], w, h)
        input_2d = self.prepInputs(joints_2ds)
        joints_3ds = self.net(input_2d)
        joints_3ds = self.processOutputs(joints_3ds)
        
        return output_joints_2ds, joints_3ds