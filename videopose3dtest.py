import os
import numpy as np

from pose_playground.utils.visualise import drawLimbs3d, drawLimbs2dOnCV2Image
from pose_playground.pose_models.model_managed import ModelManaged

vp3d_model = ModelManaged('VideoPose3D', causal=True, backend='detectron')

joints_2ds, joints_3ds, frames, bboxes = vp3d_model.estimateFromVideo('Healthy_Trim.mp4', resize=1)
print(joints_2ds.shape)
print(joints_3ds.shape)
np.save('joints_2d', joints_2ds)
np.save('joints_3d', joints_3ds)
for i, frame in enumerate(frames):
    try:
        os.mkdir('Healthy_Trim_results/')
    except FileExistsError:
        pass
    drawLimbs2dOnCV2Image(frame, joints_2ds[i], None, save_figure='Healthy_Trim_results/2d_' + str(i).zfill(5) + '.jpg')
    drawLimbs3d(joints_3ds[i], vp3d_model.joint_parents, save_figure='Healthy_Trim_results/3d_' + str(i).zfill(5) + '.jpg')