# yolov8_demo
YOLOv8的多框架后处理，使用onnxruntime、tensorrt、openvino进行yolov8的模型推理，不依赖torch，借助b导的yolov8代码完成。


1.基于b导的yolov8代码，进行不同后端框架的推理

2.模型结果解码可以加在nets/yolo.py的输出中，如dist2bbox，nms等，这样在转换为其他框架推理时就不需要进行该操作

3.使用numpy写了yolov8的后处理部分，不依赖torch，不需要编译，可在win上运行

4.model_data文件中存有基于YOLOv8s的基于coco的不同框架的推理模型，可以借鉴使用

5.部分代码为极市平台模型测试代码，可忽略


