from pose_playground.pose_models.model_managed import ModelManaged

vp3d_model = ModelManaged('VideoPose3D', causal=False)

joints_2d, joints3d = vp3d_model.estimateFromImage('test_pic.jpg')