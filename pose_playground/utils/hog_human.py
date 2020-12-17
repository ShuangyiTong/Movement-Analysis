import cv2
import numpy as np


class HOGHuman:
    """
    a simple HOG-method-based human tracking box
    """

    def __init__(self):
        print('Initializing HOGBox...')
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print('HOGBox initialized.')

    def __call__(self, img):
        H, W = img.shape[:2]
        found, w = self.hog.detectMultiScale(img)
        rect = self.cal_rect(found[np.argmax([found[i, 2] * found[i, 3] for i in range(len(found))])], H, W) \
            if len(found) else [0, 0, W, H]  # biggest area
        
        return rect

    @staticmethod
    def cal_rect(rect, H, W):
        """
        calculate the box size and position
        """
        x, y, w, h = rect
        offset_w = int(0.4 / 2 * W)
        offset_h = int(0.2 / 2 * H)
        return [np.max([x - offset_w, 0]),  # x
                np.max([y - offset_h, 0]),  # y
                np.min([x + w + offset_w, W]) - np.max([x - offset_w, 0]),  # w
                np.min([y + h + offset_h, H]) - np.max([y - offset_h, 0])]  # h
