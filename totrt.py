import tensorrt as trt


def GiB(val):
    return val * 1 << 30


def ONNX_build_engine(onnx_file_path,out_file_path, write_engine = True,batch_size = 5,imgsz=512,inputname = "images"):
    '''
    通过加载onnx文件，构建engine
    :param onnx_file_path: onnx文件路径
    :return: engine
    '''
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 1、动态输入第一点必须要写的
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    batch_size = batch_size # trt推理时最大支持的batchsize
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(2)
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        # 重点 
        profile = builder.create_optimization_profile() # 动态输入时候需要 分别为最小输入、常规输入、最大输入
        # 有几个输入就要写几个profile.set_shape 名字和转onnx的时候要对应
        profile.set_shape(inputname, (1,3,imgsz,imgsz), (1,3,imgsz,imgsz), (batch_size,3,imgsz,imgsz))
        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        print("Completed creating Engine")
        # 保存engine文件
        if write_engine:
            engine_file_path = out_file_path
            with open(engine_file_path, "wb") as f:
                f.write(engine)
        return engine

if __name__ == "__main__":
    onnx_path = './model_data/models.onnx'
    trt_path  = './model_data/models.trt'
    batch_size = 1
    image_size = 640
    input_name = "images"
    ONNX_build_engine(onnx_path, trt_path, write_engine=True, batch_size=batch_size, imgsz=image_size, inputname=input_name)
