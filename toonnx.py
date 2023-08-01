from hrnet import HRnet_Segmentation
simplify=True

onnx_save_path = "/project/train/models/seg_model.onnx"
hrnet = HRnet_Segmentation(model_path ="/project/train/models/best_epoch_weights.pth",num_classes = 2,backbone = "hrnetv2_w32",input_shape=[512,512] )
hrnet.convert_to_onnx(simplify, onnx_save_path)