import onnxruntime
import json
import sys
import numpy as np
import cv2
import torch
from PIL import Image

from utils.utils import get_classes
from decodeNP import DecodeBoxNP
from utilsNP import resize_image, plot_one_box

import warnings
warnings.filterwarnings('ignore')

onnx_model_path = './model_data/models.onnx'
classes_path = './model_data/coco_classes.txt'
input_shape = [640, 640]
class_names, num_classes = get_classes(classes_path)
decode = DecodeBoxNP(num_classes=num_classes, input_shape=input_shape)


def init():
    session = onnxruntime.InferenceSession(onnx_model_path, providers='CUDAExecutionProvider')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print('input_name :',input_name)
    print('output_name :',output_name)
    print(session.get_inputs()[0].shape)
    print(session.get_outputs()[0].shape)
    return session


def process_image(handle=None, input_image=None, args=None, **kwargs):
    vis = True
    input_name = 'images'
    fake_result = {"model_data": {"objects": []}}
    image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    image_shape = np.array(np.shape(image)[0:2])
    image_data = resize_image(image, (input_shape[1], input_shape[0]))
    image_data = np.array(image_data, dtype='float32')
    image_data /= 255.0
    image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
    outputs = handle.run([], {input_name: image_data})
    outputs = np.squeeze(outputs, axis=0)
    outputs = np.array(outputs)
    # print(outputs.shape)  # (1, 8400, 84)
    results = decode.non_max_suppression(outputs, num_classes, input_shape,
                                         image_shape, True, conf_thres=0.4,
                                         nms_thres=0.3)
    if results[0] is None:
        return json.dumps(fake_result, indent=4)

    top_label = np.array(results[0][:, 5], dtype='int32')
    top_conf = results[0][:, 4]
    top_boxes = results[0][:, :4]
    if vis:
        new_image = input_image
        for j in range(len(top_boxes)):
            box = top_boxes[j]
            top, left, bottom, right = box
            right_box = [left,top,right,bottom]
            plot_one_box(right_box, new_image, label="{}:{:.2f}".format(class_names[int(top_label[j])], top_conf[j]))
        cv2.imwrite('1.jpg', new_image)
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i].astype('float')

        top, left, bottom, right = box

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom).astype('int32'))
        right = min(image.shape[1], np.floor(right).astype('int32'))
        # print(predicted_class, top, left, bottom, right, score)
        fake_result['model_data']['objects'].append({
            "xmin": int(left),
            "ymin": int(top),
            "xmax": int(right),
            "ymax": int(bottom),
            "confidence": score,
            "name": predicted_class
        })
    return json.dumps(fake_result, indent=4)


if __name__ == '__main__':
    img = cv2.imread('images/bus.jpg')
    session = init()
    import time

    s = time.time()
    fake_result = process_image(handle=session, input_image=img)
    e = time.time()
    print(fake_result)
    print((e - s))