import time
import numpy as np

from pose_playground.utils.one_euro_filter import OneEuroFilter

class Joints2Angles:
    def __init__(self, angle_joint, sideA_joint, sideB_joint, filtering=True):
        self.angle_joint = angle_joint
        self.sideA_joint = sideA_joint
        self.sideB_joint = sideB_joint

        if filtering:
            # filter configuration
            config_filter = {
                'freq': 30,
                'mincutoff': 1.0,
                'beta': 0,
                'dcutoff': 0.1
            }
            self.filter = OneEuroFilter(**config_filter)
            self.filter_time = None
        else:
            self.filter = None

    def calculateAngle(self, joints_3d):
        is_normal = False

        # check if the input is a vector
        if hasattr(self.sideA_joint, '__len__'):
            sideA = self.sideA_joint
            # check if the input is a plane
            if len(self.sideA_joint) == 2 and hasattr(self.sideA_joint[0], '__len__'):
                sideA = np.cross(self.sideA_joint[0], self.sideA_joint[1])
                is_normal = True
        else:
            sideA = joints_3d[self.sideA_joint] - joints_3d[self.angle_joint]
        
        # check if the input is a vector
        if hasattr(self.sideB_joint, '__len__'):
            # check if the input is a plane
            if len(self.sideB_joint) == 2 and hasattr(self.sideB_joint[0], '__len__'):
                sideB = np.cross(self.sideB_joint[0], self.sideB_joint[1])
                is_normal = True
            sideB = self.sideB_joint
        else:
            sideB = joints_3d[self.sideB_joint] - joints_3d[self.angle_joint]
            
        cos_angle = np.dot(sideA, sideB) / (np.linalg.norm(sideA) * np.linalg.norm(sideB))

        angle = np.arccos(cos_angle)
        if is_normal:
            angle -= np.pi / 2
            if angle < 0:
                angle = -angle
        return angle

    def __call__(self, joints_3d, time_delta=None):
        angle = self.calculateAngle(joints_3d)

        if self.filter:
            if time_delta and self.filter_time:
                self.filter_time += time_delta
            else:
                self.filter_time = time.time()
            angle = self.filter(angle, self.filter_time)
        
        return angle