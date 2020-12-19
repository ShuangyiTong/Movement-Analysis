import numpy as np

from pose_playground.pose_models.model_managed import ModelManaged

# You create a model by calling ModelManaged. Here we are telling ModelManaged to select 
# VNect model and apply histogram of oriented gradient method to refine body region
pose_model = ModelManaged('VNect', with_hog=True)
# Call estimateFromImage to get both 2d and 3d coordinates
joints_2d, joints_3d = pose_model.estimateFromImage('test_pic.jpg')

# Print these coordinates
print('2D projection coordinates on the image: ', joints_2d)
print('3D Cartesian coordinates: ', joints_3d)
# Calculate left arm angle
from pose_playground.joint_angle import Joints2Angles
j2angle = Joints2Angles(pose_model.getJointIndexByName('left_shoulder'),
                        pose_model.getJointIndexByName('neck'),
                        pose_model.getJointIndexByName('left_elbow'))
print('left neck-shoulder-elbow angle: ', str(j2angle(joints_3d) / np.pi) + 'π')

# Import visualise functions
from pose_playground.utils.visualise import drawLimbs3d, drawLimbs2d, drawLimbs2dOnCV2Image, drawRectangle
# Draw a 3D skeleton with matplotlib
drawLimbs3d(joints_3d, pose_model.joint_parents)
# Draw a 2D projection over the original image
x, y, _, _ = pose_model.last_estimate_rectangle
drawLimbs2d('test_pic.jpg', joints_2d, pose_model.joint_parents, 'result.jpg', x=x, y=y)

# For videos
pose_model = ModelManaged('VNect', with_hog=True, use_sess=pose_model.sess) # initialise a new one
import cv2
video = cv2.VideoCapture('Diabetic_Neuropathy.mp4')
import os
try:
    os.mkdir('video_results/')
except FileExistsError:
    pass
succeed, frame = video.read()
frame_count = 1
angle_list = []
j2angle = Joints2Angles(pose_model.getJointIndexByName('spine'),
                        pose_model.getJointIndexByName('neck'),
                        pose_model.getJointIndexByName('pelvis'))
while succeed:
    # filter time range of the video
    if video.get(cv2.CAP_PROP_POS_MSEC) < 22000:
        succeed, frame = video.read()
        continue
    elif video.get(cv2.CAP_PROP_POS_MSEC) > 30000:
        break
    joints_2d, joints_3d = pose_model.estimateFromCV2Image(frame, time_delta=1/30)
    angle_list.append(j2angle(joints_3d, time_delta=1/30) / np.pi)
    x, y, w, h = pose_model.last_estimate_rectangle
    drawRectangle(frame, (x, y, w, h))
    drawLimbs2dOnCV2Image(frame, joints_2d, pose_model.joint_parents, 'video_results/' + str(frame_count) + '.jpg', x=x, y=y)
    succeed, frame = video.read()
    frame_count += 1

import matplotlib.pyplot as plt
plt.xlabel('frame')
plt.ylabel('Spine angle (rad)')
plt.plot(angle_list)
plt.show()