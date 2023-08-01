import json
import torch
import sys
import numpy as np
import cv2
from utils.utils import cvtColor, resize_image, preprocess_input, get_classes
from utils.utils_bbox import DecodeBox
from nets.yolo import YoloBody

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0")
model_path='./model_data/yolov8_s.pth'
classes_path='./model_data/coco_classes.txt'
input_shape = [640, 640]
class_names, num_classes  = get_classes(classes_path)

decode = DecodeBox(num_classes=num_classes, input_shape=input_shape)

def init():
    net = YoloBody(input_shape=input_shape, num_classes=num_classes, phi='s')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.fuse().eval()
    net = net.cuda()
    return net

def process_image(handle=None, input_image=None, args=None, **kwargs):
    fake_result = {"model_data": {"objects": []}}
    if input_image is None:
        return json.dumps(fake_result, indent=4)
    image = Image.fromarray(cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB))
    # image = input_image
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_data = resize_image(image, (input_shape[1], input_shape[0]), True)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        outputs = handle(images)
        outputs = decode.decode_box(outputs)
        # print(np.array(outputs.cpu().numpy()).shape)
        # ---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        # ---------------------------------------------------------#
        results = decode.non_max_suppression(outputs, num_classes, input_shape,
                                                     image_shape, True, conf_thres=0.4,
                                                     nms_thres=0.3)
        if results[0] is None:
            return json.dumps(fake_result, indent=4)

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box             = top_boxes[i]
        score           = top_conf[i].astype('float')

        top, left, bottom, right = box
        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))
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
    # Test API
    from PIL import Image
    img =cv2.imread('images/bus.jpg')
    predictor = init()
    import time
    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print((e-s))
