import math
import cv2
import matplotlib.pyplot as plt

def drawLimbs3d(joints_3d, joint_parents, save_figure=None):
    fig = plt.figure()
    ax_3d = plt.axes(projection='3d')
    ax_3d.clear()
    ax_3d.view_init(-90, -90)
    ax_3d.set_xlim(-500, 500)
    ax_3d.set_ylim(-500, 500)
    ax_3d.set_zlim(-500, 500)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])
    white = (1.0, 1.0, 1.0, 0.0)
    ax_3d.w_xaxis.set_pane_color(white)
    ax_3d.w_yaxis.set_pane_color(white)
    ax_3d.w_xaxis.line.set_color(white)
    ax_3d.w_yaxis.line.set_color(white)
    ax_3d.w_zaxis.line.set_color(white)
    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[joint_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[joint_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[joint_parents[i], 2]]
        ax_3d.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
    if save_figure:
        plt.savefig(save_figure)
        plt.close(fig)
    else:
        plt.show()

def rgb2bgr(rgb):
    r, g, b = rgb
    return b, g, r

def drawLimbs2dCV2(img_to_draw, joints_2d_copy, limb_parents, save_figure, x=0, y=0):
    img = img_to_draw.copy()
    joints_2d = joints_2d_copy.copy()
    joints_2d[:, 0] += y
    joints_2d[:, 1] += x
    colour_map = [(239,65,54), (247,148,29), (251,176,64), (249,237,50), (0,174,239)]
    for joint_id, joint in enumerate(joints_2d):
        cv2.circle(img, (joint[0], joint[1]), 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        if limb_parents:
            cv2.line(img, (joint[0], joint[1]), 
                    (joints_2d[limb_parents[joint_id]][0], joints_2d[limb_parents[joint_id]][1]),
                    colour_map[joint_id % len(colour_map)], 2)

    return img
    
def drawLimbs2dOnCV2Image(img, joints_2d_copy, limb_parents, save_figure, x=0, y=0):
    img = drawLimbs2dCV2(img, joints_2d_copy, limb_parents, save_figure, x, y)
    cv2.imwrite(save_figure, img)

def drawLimbs2d(image_file, joints_2d, limb_parents, save_figure, x=0, y=0):
    img = cv2.imread(image_file)
    drawLimbs2dOnCV2Image(img, joints_2d, limb_parents, save_figure, x, y)

def drawRectangle(img, rect):
    '''Draw a rectangle on the image

    rect (4-tuple): x, y, widthm height
    '''
    x, y, w, h = rect
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (60, 66, 207), 4)