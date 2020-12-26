import sys
import os
import cv2
import numpy as np

from pose_playground.utils.hog_human import HOGHuman

class ModelManaged:
    '''Unified model interface

    You can get the following member variables:
    self.num_joints: total number of joints
    self.joint_parents: connected parents of the joint
    self.joint_names: named joints ordered by index
    self.use_sess: existing session
    '''
    def __init__(self, model_type, **kwargs):
        '''Args:
        model_type (string): one of {'VNect'}

        kwargs:
            with_hog (bool): use histogram of oriented gradients to refine human body region
            use_sess: force using existing session, this bypasses restrictions like tensorflow scope
        '''
        self.model_type = model_type
        self.hog = None
        if model_type == 'VNect':
            from pose_playground.pose_models.VNect.estimator import VNectEstimator

            if 'use_sess' in kwargs.keys():
                self.model_instance = VNectEstimator(existing_sess=kwargs['use_sess'])
            else:
                self.model_instance = VNectEstimator()
            self.sess = self.model_instance.sess

            hog_box = False
            if 'with_hog' in kwargs.keys():
                hog_box = kwargs['with_hog']
        
            if hog_box:
                self.hog = HOGHuman()
            
            self.causal = True
        elif model_type == 'VideoPose3D':
            from pose_playground.pose_models.VideoPose3D.estimator import VideoPose3DEstimator
            
            self.causal = kwargs['causal']
            if 'backend' in kwargs.keys():
                self.model_instance = VideoPose3DEstimator(self.causal, backend=kwargs['backend'])
            else:
                self.model_instance = VideoPose3DEstimator(self.causal)
        else:
            raise NotImplementedError(model_type + ' unrecognised')

        self.num_joints = self.model_instance.num_output_joints
        self.joint_parents = self.model_instance.joint_parents
        self.joint_names = self.model_instance.joint_names

    def estimateFromImage(self, image_file, time_delta=None):
        '''Args:
        image_file: image file path to be processed

        Returns:
            (2D joint coordinates, 3D joint coordinates)
                with shape [self.num_joints, 2] and [self.num_joints, 3] 
        '''
        img = cv2.imread(image_file)
        return self.estimateFromCV2Image(img, time_delta)

    def estimateFromCV2Image(self, img, time_delta=None):
        if self.hog:
            x, y, w, h = self.hog(img)
        else:
            x, y = 0, 0
            h, w = img.shape[:2]
        self.last_estimate_rectangle = (x, y, w, h)
        
        img_cropped = img[y:y + h, x:x + w, :]
        joints_2d, joints_3d = self.model_instance(img_cropped, time_delta)
        return joints_2d, joints_3d

    def estimateFromVideo(self, video_name, interval=None, resize=1):
        video = cv2.VideoCapture(video_name)
        succeed, frame = video.read()
        frame_time = 1 / video.get(cv2.CAP_PROP_FPS)

        if self.causal:
            joints_2ds = []
            joints_3ds = []
        else:
            intermediate_inputs = []
        bounding_boxes = []
        frames = []
        while succeed:
            if interval:
                # filter time range of the video
                if video.get(cv2.CAP_PROP_POS_MSEC) < interval[0]:
                    succeed, frame = video.read()
                    continue
                elif video.get(cv2.CAP_PROP_POS_MSEC) > interval[1]:
                    break

            if resize != 1:
                frame = cv2.resize(frame, (int(frame.shape[1] * resize), int(frame.shape[0] * resize)))

            if self.hog:
                x, y, w, h = self.hog(frame)
            else:
                x, y = 0, 0
                h, w = frame.shape[:2]

            frames.append(frame)
            bounding_boxes.append((x, y, w, h))
            print('Processing frame:', len(frames), 'timestamp (ms):', video.get(cv2.CAP_PROP_POS_MSEC))
            frame_cropped = frame[y:y + h, x:x + w, :]

            if self.causal:
                joints_2d, joints_3d = self.model_instance(frame_cropped, frame_time)
                joints_2ds.append(joints_2d)
                joints_3ds.append(joints_3d)
            else:
                intermediate_input = self.model_instance.getIntermediateInput(frame_cropped, frame_time)
                intermediate_inputs.append(intermediate_input)
            succeed, frame = video.read()

        if not self.causal:
            joints_2ds, joints_3ds = self.model_instance.batchInput(np.array(intermediate_inputs),
                                                                    w=w, h=h)
        else:
            joints_2ds = np.array(joints_2ds).astype(int)
            joints_3ds = np.array(joints_3ds)

        return joints_2ds, joints_3ds, frames, bounding_boxes

    def getJointIndexByName(self, joint_name):
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            raise ValueError(joint_name + ' not listed in this model: ' + self.joint_names)

