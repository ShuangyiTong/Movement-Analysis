import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pose_playground.pose_models.model_managed import ModelManaged
from pose_playground.joint_angle import Joints2Angles
from pose_playground.utils.visualise import drawLimbs2dOnCV2Image, drawRectangle

try:
    os.mkdir('video_results/')
except FileExistsError:
    pass
PD_LBP_patient = 'PD_LBP_Trim.mp4'
# Post Guillain-Barré syndrome
GBS_patient = 'GBS_Trim.mp4'
healthy_patient = 'healthy_Trim.mp4'

PD_LBP_video = cv2.VideoCapture(PD_LBP_patient)
GBS_video = cv2.VideoCapture(GBS_patient)
healthy_video = cv2.VideoCapture(healthy_patient)

P_succeed, P_frame = PD_LBP_video.read()
G_succeed, G_frame = GBS_video.read()
h_succeed, h_frame = healthy_video.read()

P_pose_model = ModelManaged('VNect', with_hog=True)
P_angle_list = []
P_j2angle = Joints2Angles(P_pose_model.getJointIndexByName('spine'),
                          P_pose_model.getJointIndexByName('neck'),
                          P_pose_model.getJointIndexByName('pelvis'))

G_pose_model = ModelManaged('VNect', with_hog=True, use_sess=P_pose_model.sess)
G_angle_list = []
G_j2angle = Joints2Angles(G_pose_model.getJointIndexByName('spine'),
                          G_pose_model.getJointIndexByName('neck'),
                          G_pose_model.getJointIndexByName('pelvis'))

h_pose_model = ModelManaged('VNect', with_hog=True, use_sess=P_pose_model.sess)
h_angle_list = []
h_j2angle = Joints2Angles(h_pose_model.getJointIndexByName('spine'),
                          h_pose_model.getJointIndexByName('neck'),
                          h_pose_model.getJointIndexByName('pelvis'))

frame_count = 1
while P_succeed and G_succeed and h_succeed:
    P_joints_2d, P_joints_3d = P_pose_model.estimateFromCV2Image(P_frame, time_delta=1/30)
    G_joints_2d, G_joints_3d = G_pose_model.estimateFromCV2Image(G_frame, time_delta=1/30)
    h_joints_2d, h_joints_3d = h_pose_model.estimateFromCV2Image(h_frame, time_delta=1/30)
    
    P_angle_list.append(P_j2angle(P_joints_3d, time_delta=1/30) / np.pi)
    G_angle_list.append(G_j2angle(G_joints_3d, time_delta=1/30) / np.pi)
    h_angle_list.append(h_j2angle(h_joints_3d, time_delta=1/30) / np.pi)

    
    x, y, w, h = P_pose_model.last_estimate_rectangle
    drawRectangle(P_frame, (x, y, w, h))
    drawLimbs2dOnCV2Image(P_frame, P_joints_2d, P_pose_model.joint_parents, 'video_results/P_' + str(frame_count) + '.jpg', x=x, y=y)

    x, y, w, h = G_pose_model.last_estimate_rectangle
    drawRectangle(G_frame, (x, y, w, h))
    drawLimbs2dOnCV2Image(G_frame, G_joints_2d, G_pose_model.joint_parents, 'video_results/G_' + str(frame_count) + '.jpg', x=x, y=y)

    x, y, w, h = h_pose_model.last_estimate_rectangle
    drawRectangle(h_frame, (x, y, w, h))
    drawLimbs2dOnCV2Image(h_frame, h_joints_2d, h_pose_model.joint_parents, 'video_results/h_' + str(frame_count) + '.jpg', x=x, y=y)
    
    frame_count += 1
    if frame_count > 160:
        break

    P_succeed, P_frame = PD_LBP_video.read()
    G_succeed, G_frame = GBS_video.read()
    h_succeed, h_frame = healthy_video.read()

plt.xlabel('frame')
plt.ylabel('Spine angle (rad)')
plt.plot(P_angle_list, label='PD and LPD patient')
plt.plot(G_angle_list, label='GBS patient')
plt.plot(h_angle_list, label='healthy')
plt.legend()
plt.show()