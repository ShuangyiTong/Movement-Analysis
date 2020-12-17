import cv2
import numpy as np

def img_scale(img, scale):
    """
    Resize a image by s scaler in both x and y directions.

    :param img: input image
    :param scale: scale  factor, new image side length / raw image side length
    :return: the scaled image
    """
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
def hm_pt_interp_bilinear(src, scale, point):
    """
    Determine the value of one desired point by bilinear interpolation.

    :param src: input heatmap
    :param scale: scale factor, input box side length / heatmap side length
    :param point: position of the desired point in input box, [row, column]
    :return: the value of the desired point
    """
    src_h, src_w = src.shape[:]
    dst_y, dst_x = point
    src_x = (dst_x + 0.5) / scale - 0.5
    src_y = (dst_y + 0.5) / scale - 0.5
    src_x_0 = int(src_x)
    src_y_0 = int(src_y)
    src_x_1 = min(src_x_0 + 1, src_w - 1)
    src_y_1 = min(src_y_0 + 1, src_h - 1)

    value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0] + (src_x - src_x_0) * src[src_y_0, src_x_1]
    value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0] + (src_x - src_x_0) * src[src_y_1, src_x_1]
    dst_val = (src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1
    return dst_val


def img_padding(img, box_size, color='black'):
    """
    Given the input image and side length of the box, put the image into the center of the box.

    :param img: the input color image, whose longer side is equal to box size
    :param box_size: the side length of the square box
    :param color: indicating the padding area color
    :return: the padded image
    """
    h, w = img.shape[:2]
    offset_x, offset_y = 0, 0
    if color == 'black':
        pad_color = [0, 0, 0]
    elif color == 'grey':
        pad_color = [128, 128, 128]
    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    if h > w:
        offset_x = box_size // 2 - w // 2
        img_padded[:, offset_x: box_size // 2 + int(np.ceil(w / 2)), :] = img
    else:  # h <= w
        offset_y = box_size // 2 - h // 2
        img_padded[offset_y: box_size // 2 + int(np.ceil(h / 2)), :, :] = img
    return img_padded, [offset_x, offset_y]


def img_scale_squarify(img, box_size):
    """
    To scale and squarify the input image into a square box with fixed size.

    :param img: the input color image
    :param box_size: the length of the square box
    :return: box image, scaler and offsets
    """
    h, w = img.shape[:2]
    scaler = box_size / max(h, w)
    img_scaled = img_scale(img, scaler)
    img_padded, [offset_x, offset_y] = img_padding(img_scaled, box_size)
    assert img_padded.shape == (box_size, box_size, 3), 'padded image shape invalid'
    return img_padded, scaler, [offset_x, offset_y]


def img_scale_padding(img, scaler, box_size, color='black'):
    """
    For a box image, scale down it and then pad the former area.

    :param img: the input box image
    :param scaler: scale factor, new image side length / raw image side length, < 1
    :param box_size: side length of the square box
    :param color: the padding area color
    """
    img_scaled = img_scale(img, scaler)
    if color == 'black':
        pad_color = (0, 0, 0)
    elif color == 'grey':
        pad_color = (128, 128, 128)
    pad_h = (box_size - img_scaled.shape[0]) // 2
    pad_w = (box_size - img_scaled.shape[1]) // 2
    pad_h_offset = (box_size - img_scaled.shape[0]) % 2
    pad_w_offset = (box_size - img_scaled.shape[1]) % 2
    img_scale_padded = np.pad(img_scaled,
                              ((pad_w, pad_w + pad_w_offset),
                               (pad_h, pad_h + pad_h_offset),
                               (0, 0)),
                              mode='constant',
                              constant_values=(
                                  (pad_color[0], pad_color[0]),
                                  (pad_color[1], pad_color[1]),
                                  (pad_color[2], pad_color[2])))
    return img_scale_padded


def extract_2d_joints(heatmaps, box_size, hm_factor):
    """
    Rescale the heatmap to input box size, then extract the coordinates for every joint.

    :param heatmaps: the input heatmaps
    :param box_size: the side length of the input box
    :param hm_factor: heatmap factor, indicating box size / heatmap size
    :return: a 2D array with [joints_num, 2], each row of which means [row, column] coordinates of corresponding joint
    """
    joints_2d = np.zeros((heatmaps.shape[2], 2))
    for joint_num in range(heatmaps.shape[2]):
        heatmap_scaled = cv2.resize(heatmaps[:, :, joint_num], (0, 0), 
                                    fx=hm_factor, fy=hm_factor, 
                                    interpolation=cv2.INTER_LINEAR)
        joint_coord = np.unravel_index(np.argmax(heatmap_scaled), 
                                       (box_size, box_size))
        joints_2d[joint_num, :] = joint_coord
    return joints_2d


def extract_3d_joints(joints_2d, x_hm, y_hm, z_hm, hm_factor):
    """
    Extract 3D coordinates of each joint according to its 2D coordinates.

    :param joints_2d: 2D array with [joints_num, 2], containing 2D coordinates the joints
    :param x_hm: x coordinate heatmaps
    :param y_hm: y coordinate heatmaps
    :param z_hm: z coordinate heatmaps
    :param hm_factor: heatmap factor, indicating box size / heatmap size
    :return: a 3D array with [joints_num, 3], each row of which contains [x, y, z] coordinates of corresponding joint

    Notation:
    x direction: left --> right
    y direction: up --> down
    z direction: nearer --> farther
    """
    scaler = 100  # scaler=100 -> mm unit; scaler=10 -> cm unit
    joints_3d = np.zeros((x_hm.shape[2], 3), dtype=np.float32)
    for joint_num in range(x_hm.shape[2]):
        y_2d, x_2d = joints_2d[joint_num][:]
        joint_x = hm_pt_interp_bilinear(x_hm[:, :, joint_num], 
                                        hm_factor,
                                        (y_2d, x_2d)) * scaler
        joint_y = hm_pt_interp_bilinear(y_hm[:, :, joint_num], 
                                        hm_factor,
                                        (y_2d, x_2d)) * scaler
        joint_z = hm_pt_interp_bilinear(z_hm[:, :, joint_num], 
                                        hm_factor,
                                        (y_2d, x_2d)) * scaler
        joints_3d[joint_num, :] = [joint_x, joint_y, joint_z]
    # Subtract the root location to normalize the data
    joints_3d -= joints_3d[14, :]
    return joints_3d