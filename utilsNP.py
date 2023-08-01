import random

import numpy as np
import cv2

random.seed(2023)

def resize_image(image, size):

    image = np.array(image)

    # 获得现在的shape
    shape = np.shape(image)[:2]
    # 获得输出的shape
    if isinstance(size, int):
        size = (size, size)

    # 计算缩放的比例
    r = min(size[0] / shape[0], size[1] / shape[1])

    # 计算缩放后图片的高宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]

    # 除以2以padding到两边
    dw /= 2
    dh /= 2

    # 对图像进行resize
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=(128, 128, 128))  # add border
    return new_image


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object (numpy)
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img,label,(c1[0], c1[1] - 2),0,tl / 3,[225, 255, 255],thickness=tf,lineType=cv2.LINE_AA,)

