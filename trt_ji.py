import os
import json
import cv2
import numpy as np
from totrt import ONNX_build_engine
from trttest import YoLov8TRT, warmUpThread, plot_one_box
from utils.utils import get_classes


classes_path = './model_data/coco_classes.txt'
input_shape = [640, 640]
class_names, num_classes = get_classes(classes_path)

def init():
    """Initialize model
    Returns: model
    """
    # onnxpath = "yolov5s.onnx"
    # trtpath  = "yolov5s.trt"
    # ONNX_build_engine(onnxpath, trtpath, write_engine=True, batch_size=1, imgsz=640,inputname="images")
    engine_file_path = "model_data/models.trt"
    model = YoLov8TRT(engine_file_path)
    try:
        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(model)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        model.destroy()
    return model

def process_image(handle=None, input_image=None, args=None, **kwargs):
    vis = True
    batch_size = 1
    fake_result = {}
    fake_result["model_data"] = {"objects": []}
    image = np.expand_dims(input_image, axis=0)
    # print(input_image.shape)
    batch_image_raw, infer_time, results_batch = handle.infer(image, vis=False)
    for i in range(batch_size):
        results = results_batch[i]

        if results[0] is None:
            fake_result["model_data"]["objects"].append([])

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        if vis:
            save_name = ['out' + str(n) + '.jpg' for n in range(1, batch_size + 1)]
            new_image = input_image
            for j in range(len(top_boxes)):
                box = top_boxes[j]
                top, left, bottom, right = box
                right_box = [left, top, right, bottom]
                plot_one_box(right_box, new_image,
                             label="{}:{:.2f}".format(class_names[int(top_label[j])], top_conf[j]))
            cv2.imwrite(save_name[i], batch_image_raw[i])

        for j, c in list(enumerate(top_label)):
            box = top_boxes[j]
            left, top, right, bottom = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(input_image.shape[1], np.floor(bottom).astype('int32'))
            right = min(input_image.shape[2], np.floor(right).astype('int32'))
            # print(left, top, right, bottom)
            fake_result['model_data']['objects'].append({
                "xmin": int(left),
                "ymin": int(top),
                "xmax": int(right),
                "ymax": int(bottom),
                "confidence": top_conf[j],
                "name": class_names[int(c)]
            })

    return json.dumps(fake_result, indent=4)


if __name__ == '__main__':
    # Test API
    img = cv2.imread('images/bus.jpg')
    # print(img.shape)
    predictor = init()
    res = process_image(predictor, img)
    print(res)
