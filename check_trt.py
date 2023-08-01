#!/usr/bin/env python
# -- coding: utf-8 --

import argparse
import numpy as np

def check_trt(model_path, image_size):
    """
    检查TRT模型
    """
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    # 必须导入包，import pycuda.autoinit，否则报错

    print('[Info] model_path: {}'.format(model_path))
    img_shape = (1, 3, image_size, image_size)
    print('[Info] img_shape: {}'.format(img_shape))

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_path = model_path  # TRT模型路径
    with open(trt_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print("[Info] binding: {}, binding_idx: {}, size: {}, dtype: {}"
                  .format(binding, binding_idx, size, dtype))

    input_image = np.random.randn(*img_shape).astype(np.float32)  # 图像尺寸
    input_image = np.ascontiguousarray(input_image)
    print('[Info] input_image: {}'.format(input_image.shape))

    with engine.create_execution_context() as context:
        stream = cuda.Stream()
        bindings = [0] * len(engine)

        for binding in engine:
            idx = engine.get_binding_index(binding)

            if engine.binding_is_input(idx):
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings[idx] = int(input_memory)
                cuda.memcpy_htod_async(input_memory, input_image, stream)
            else:
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                shape = context.get_binding_shape(idx)

                output_buffer = np.empty(shape, dtype=dtype)
                output_buffer = np.ascontiguousarray(output_buffer)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings[idx] = int(output_memory)

        context.execute_async_v2(bindings, stream.handle)
        stream.synchronize()

        cuda.memcpy_dtoh(output_buffer, output_memory)
    print("[Info] output_buffer: {}".format(output_buffer))


def parse_args():
    """
    处理脚本参数
    """
    parser = argparse.ArgumentParser(description='检查TRT模型')
    parser.add_argument('--model_path', default='./model_data/models.trt', help='TRT模型路径', type=str)
    parser.add_argument('--image_size', default=640, help='图像尺寸，如336', type=int)

    args = parser.parse_args()

    arg_model_path = args.model_path
    print("[Info] 模型路径: {}".format(arg_model_path))

    arg_image_size = args.image_size
    print("[Info] image_size: {}".format(arg_image_size))

    return arg_model_path, arg_image_size


if __name__ == '__main__':
    arg_model_path, arg_image_size = parse_args()
    check_trt(arg_model_path, arg_image_size)  # 检查TRT模型
