import os
import torch
import torchvision
import cv2
import numpy as np

from pose_playground.pose_models.OpenPose.net_coco import OpenPoseCoco

class OpenPose2DEstimator:
    def __init__(self, data_format='coco'):
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        self.net = OpenPoseCoco(os.path.join(this_file_path, '../../model_data/openpose_coco_weights.pkl'))
        
        if data_format == 'coco':
            self.num_output_joints = 18
        else:
            raise ValueError('Unrecognised data format: ' + data_format)

        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.net.cuda()

        self.net.eval()
        torch.no_grad()

    def __call__(self, img_input, time_delta=None):
        img_height, img_width = img_input.shape[:2]
        img_tensor = torchvision.transforms.ToTensor()((img_input))
        if self.has_cuda:
            img_tensor = img_tensor.cuda()
        img_tensor = img_tensor.unsqueeze(0)
        out_tensor = self.net(img_tensor)
        H = out_tensor.shape[2]
        W = out_tensor.shape[3]
        np_res = out_tensor.cpu().detach().numpy()
        kps = np.zeros((self.num_output_joints, 2), dtype=np.float32)
        for i in range(self.num_output_joints):
            img_prob_map = np_res[0, i, :, :]
            _, _, _, point = cv2.minMaxLoc(img_prob_map)

            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            kps[i] = [x, y]
        
        return kps