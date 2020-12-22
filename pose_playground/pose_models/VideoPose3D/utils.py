import numpy as np

def normalizeScreenCoordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

def getDetectron2Predictor(model_cfg='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'):
    import detectron2
    from detectron2.utils.logger import setup_logger
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg)
    predictor = DefaultPredictor(cfg)
    return predictor

def detectronOutput2Keypoints(outputs):
    has_bbox = False
    if outputs.has('pred_boxes'):
        bbox_tensor = outputs.pred_boxes.tensor.numpy()
        if len(bbox_tensor) > 0:
            has_bbox = True
            scores = outputs.scores.numpy()[:, None]
            bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
    if has_bbox:
        kps = outputs.pred_keypoints.numpy()
        kps_xy = kps[:, :, :2]
        kps_prob = kps[:, :, 2:3]
        kps_logit = np.zeros_like(kps_prob) # Dummy
        kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
        kps = kps.transpose(0, 2, 1)
    else:
        kps = []
        bbox_tensor = []

    if len(kps) == 0:
        return np.full((17, 2), np.nan, dtype=np.float32)
    else:
        best_match = np.argmax(bbox_tensor[:, 4])
        best_bb = bbox_tensor[best_match, :4]
        best_kp = kps[best_match].T.copy()
        best_kp = best_kp[:, :2] # Extract (x, y)
        return best_kp