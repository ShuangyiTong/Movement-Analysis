import cv2

from pose_playground.pose_models.VNect.estimator import VNectEstimator
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
            if 'use_sess' in kwargs.keys():
                self.model_instance = VNectEstimator(existing_sess=kwargs['use_sess'])
            else:
                self.model_instance = VNectEstimator()
            self.num_joints = VNectEstimator.joints_sum
            self.joint_parents = VNectEstimator.joint_parents
            self.joint_names = VNectEstimator.joint_names
            self.sess = self.model_instance.sess

            hog_box = False
            if 'with_hog' in kwargs.keys():
                hog_box = kwargs['with_hog']
        
            if hog_box:
                self.hog = HOGHuman()
        else:
            raise NotImplementedError(model_type + ' unrecognised')

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

    def getJointIndexByName(self, joint_name):
        try:
            return self.joint_names.index(joint_name)
        except ValueError:
            raise ValueError(joint_name + ' not listed in this model: ' + self.joint_names)

