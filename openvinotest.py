import glob
import json
import os

import cv2
import numpy as np

from openvino.inference_engine import IECore
from utilsNP import resize_image, plot_one_box
from utils.utils import get_classes
from decodeNP import DecodeBoxNP

classes_path = './model_data/coco_classes.txt'
class_names, num_classes = get_classes(classes_path)
input_shape = [640, 640]
decode = DecodeBoxNP(num_classes=num_classes, input_shape=input_shape)

if __name__ == "__main__":
    vis = True
    fake_result = {"model_data": {"objects": []}}
    # load model
    xml_dir = './model_data/models.xml'
    bin_dir = './model_data/models.bin'
    # inference engine
    ie = IECore()
    # read IR
    model = ie.read_network(model=xml_dir, weights=bin_dir)
    # load model
    compiled_model = ie.load_network(model, "CPU")
    input_layer_ir = next(iter(model.input_info))
    output_layer_ir = next(iter(model.outputs))

    print("- Input layer name: {}".format(input_layer_ir))
    print("- Output layer name: {}".format(output_layer_ir))

    # load images
    images_list = glob.glob('./images/*.jpg')
    images_name = [i.split('\\')[-1][:-4] for i in images_list]
    # print(images_name)
    model_shape = model.input_info[input_layer_ir].input_data
    _, _, input_h, input_w = model_shape.shape
    for i, input_image in enumerate(images_list):
        input_image = cv2.imread(input_image)
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image_shape = np.array(np.shape(image)[0:2])
        image_data = resize_image(image, (input_h,input_w))
        image_data = np.array(image_data, dtype='float32')
        image_data /= 255.0
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
        # print(image_data.shape)  # 1,3,640,640
        # inference
        result = compiled_model.infer(inputs={input_layer_ir: image_data})[output_layer_ir]
        # print(result.shape)  # (1, 8400, 84)
        results = decode.non_max_suppression(result, num_classes, input_shape,
                                             image_shape, True, conf_thres=0.4,
                                             nms_thres=0.5)
        if results[0] is None:
            json.dumps(fake_result, indent=4)

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]
        if vis:
            new_image = input_image
            save_name = ['output/' + 'ir_' + str(images_name[i]) + '.jpg' for i in range(len(images_name))]
            for j in range(len(top_boxes)):
                box = top_boxes[j]
                top, left, bottom, right = box
                right_box = [left, top, right, bottom]
                plot_one_box(right_box, new_image,
                             label="{}:{:.2f}".format(class_names[int(top_label[j])], top_conf[j]))
            cv2.imwrite(save_name[i], new_image)
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
    json.dumps(fake_result, indent=4)
    print(fake_result)