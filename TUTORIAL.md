# YOLOv8-TensorRT 完整详细教程

本教程分为8个章节，详细讲解 YOLOv8-TensorRT 项目的原理与实践。

## 目录
- [第一章：项目概述与环境准备](#第一章)
- [第二章：YOLOv8 模型导出](#第二章)
- [第三章：TensorRT 引擎构建](#第三章)
- [第四章：Python 推理实现](#第四章)
- [第五章：C++ 推理实现](#第五章)
- [第六章：高级特性](#第六章)
- [第七章：性能优化与调试](#第七章)
- [第八章：实战案例与最佳实践](#第八章)

---

# 第一章：项目概述与环境准备 {#第一章}

## 1.1 项目背景

YOLOv8-TensorRT 是一个将 Ultralytics YOLOv8 模型加速的推理框架。

### 工作流程
\`\`\`
PyTorch (.pt) → ONNX (.onnx) → TensorRT (.engine) → 推理
\`\`\`

### 核心优势
- 速度提升 2-10倍
- 支持 FP16/INT8 量化
- Python 和 C++ 接口
- 多任务：检测/分割/姿态/分类/OBB

## 1.2 环境安装

### 1.2.1 CUDA 安装
\`\`\`bash
# 下载并安装 CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 配置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
nvidia-smi
\`\`\`

### 1.2.2 TensorRT 安装
\`\`\`bash
# 下载 TensorRT tar包
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
cd TensorRT-8.6.1.6

# 配置环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib

# 安装 Python wheel
cd python
pip install tensorrt-*.whl

# 验证
python -c "import tensorrt; print(tensorrt.__version__)"
\`\`\`

### 1.2.3 Python 依赖
\`\`\`bash
git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git
cd YOLOv8-TensorRT

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
pip install ultralytics
\`\`\`

## 1.3 快速开始

\`\`\`bash
# 1. 下载模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# 2. 导出 ONNX
python3 export-det.py --weights yolov8s.pt --sim

# 3. 构建引擎
python3 build.py --weights yolov8s.onnx --fp16

# 4. 推理测试
python3 infer-det.py --engine yolov8s.engine --imgs data/ --show
\`\`\`

---

# 第二章：YOLOv8 模型导出 {#第二章}

## 2.1 ONNX 格式

ONNX (Open Neural Network Exchange) 是神经网络交换格式。

### 为什么需要 ONNX
- **跨框架**：训练和部署可用不同框架
- **优化友好**：便于图优化
- **标准化**：统一的算子定义

### YOLOv8 导出挑战
标准导出输出原始特征图，需要复杂后处理。本项目将后处理集成到 ONNX 中。

## 2.2 检测模型导出

### 代码实现 (export-det.py)

\`\`\`python
from ultralytics import YOLO
from models.common import PostDetect, optim

# 加载模型
model = YOLO('yolov8s.pt').model.fuse().eval()

# 优化层（替换 Detect 为 PostDetect）
for m in model.modules():
    optim(m)

# 导出 ONNX
torch.onnx.export(
    model,
    torch.randn(1, 3, 640, 640),
    'yolov8s.onnx',
    input_names=['images'],
    output_names=['num_dets', 'bboxes', 'scores', 'labels']
)
\`\`\`

### PostDetect 类

\`\`\`python
class PostDetect(nn.Module):
    """带 NMS 后处理的检测头"""
    conf_thres = 0.25
    iou_thres = 0.65
    topk = 100
    
    def forward(self, x):
        # 1. 提取 boxes 和 scores
        boxes, scores = self.extract(x)
        
        # 2. 应用 NMS
        return ops.v8_detect_nms(
            boxes, scores,
            self.conf_thres,
            self.iou_thres,
            self.topk
        )
\`\`\`

### 实战操作

\`\`\`bash
python3 export-det.py \
  --weights yolov8s.pt \
  --iou-thres 0.65 \
  --conf-thres 0.25 \
  --topk 100 \
  --opset 11 \
  --sim \
  --input-shape 1 3 640 640 \
  --device cuda:0
\`\`\`

## 2.3 分割模型导出

\`\`\`bash
python3 export-seg.py \
  --weights yolov8s-seg.pt \
  --opset 11 \
  --sim \
  --input-shape 1 3 640 640
\`\`\`

输出包含分割掩码：
- num_dets: (1, 1)
- bboxes: (1, 100, 4)
- scores: (1, 100)
- labels: (1, 100)
- masks: (1, 100, 160, 160)

---

# 第三章：TensorRT 引擎构建 {#第三章}

## 3.1 TensorRT 优化原理

### 层融合
\`\`\`
优化前: Conv → BatchNorm → ReLU (3层)
优化后: ConvBnReLU (1层)
\`\`\`

### 精度校准
- **FP32**: 全精度，慢但准确
- **FP16**: 半精度，速度提升 2-3倍
- **INT8**: 整数量化，速度提升 4-5倍

### 内核自动调优
TensorRT 自动选择最优 GPU kernel。

## 3.2 从 ONNX 构建引擎

### build.py 核心代码

\`\`\`python
from models import EngineBuilder

# 创建构建器
builder = EngineBuilder('yolov8s.onnx', device='cuda:0')

# 构建引擎
builder.build(
    fp16=True,                    # FP16 加速
    input_shape=(1, 3, 640, 640),
    iou_thres=0.65,
    conf_thres=0.25,
    topk=100
)
# 生成 yolov8s.engine
\`\`\`

### EngineBuilder 类实现

\`\`\`python
class EngineBuilder:
    def __build_engine(self, fp16, ...):
        # 1. 创建 builder 和 network
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(EXPLICIT_BATCH)
        
        # 2. 解析 ONNX
        parser = trt.OnnxParser(network, logger)
        parser.parse(onnx_model.SerializeToString())
        
        # 3. 配置优化
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2GB)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # 4. 构建引擎
        engine = builder.build_serialized_network(network, config)
        
        # 5. 保存
        with open('yolov8s.engine', 'wb') as f:
            f.write(engine)
\`\`\`

## 3.3 从 API 构建引擎

### 生成 pickle 权重

\`\`\`bash
python3 gen_pkl.py -w yolov8s.pt -o yolov8s.pkl
\`\`\`

### API 构建

\`\`\`bash
python3 build.py \
  --weights yolov8s.pkl \
  --fp16 \
  --input-shape 1 3 640 640 \
  --iou-thres 0.65 \
  --conf-thres 0.25 \
  --topk 100
\`\`\`

### build_from_api 实现

\`\`\`python
def build_from_api(self, fp16, input_shape, ...):
    # 1. 加载权重
    with open('yolov8s.pkl', 'rb') as f:
        state_dict = pickle.load(f)
    
    # 2. 构建网络层
    # 输入层
    images = network.add_input('images', trt.float32, input_shape)
    
    # Backbone
    x = Conv(network, state_dict, images, 64, 3, 2, 1, 'Conv.0')
    x = C2f(network, state_dict, x, 128, 3, True, 1, 0.5, 'C2f.2')
    ...
    
    # Neck (FPN)
    ...
    
    # Head (Detection + NMS)
    batched_nms = Detect(network, state_dict, [p3, p4, p5],
                        strides, 'Detect.22', reg_max, fp16,
                        iou_thres, conf_thres, topk)
    
    # 标记输出
    for i in range(batched_nms.num_outputs):
        network.mark_output(batched_nms.get_output(i))
\`\`\`

## 3.4 使用 trtexec

\`\`\`bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8s.onnx \
  --saveEngine=yolov8s.engine \
  --fp16 \
  --workspace=2048
\`\`\`

优点：
- 无需 Python 环境
- 命令行操作简单
- 支持 profiling

---

# 第四章：Python 推理实现 {#第四章}

## 4.1 TRTModule 推理引擎

### 核心类定义

\`\`\`python
class TRTModule(nn.Module):
    def __init__(self, weight, device):
        # 1. 加载引擎
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(
                Path(weight).read_bytes()
            )
        
        # 2. 创建执行上下文
        context = engine.create_execution_context()
        
        # 3. 创建 CUDA stream
        stream = torch.cuda.Stream(device)
        
        self.engine = engine
        self.context = context
        self.stream = stream
    
    def forward(self, *inputs):
        # 1. 设置输入
        for i, inp in enumerate(inputs):
            self.context.set_tensor_address(
                self.input_names[i],
                inp.data_ptr()
            )
        
        # 2. 分配输出
        outputs = []
        for i in range(self.num_outputs):
            output = torch.empty(
                self.out_info[i].shape,
                dtype=self.out_info[i].dtype,
                device=self.device
            )
            self.context.set_tensor_address(
                self.output_names[i],
                output.data_ptr()
            )
            outputs.append(output)
        
        # 3. 执行推理
        self.context.execute_async_v3(self.stream.cuda_stream)
        
        # 4. 同步等待
        self.stream.synchronize()
        
        return tuple(outputs)
\`\`\`

## 4.2 完整推理流程

### infer-det.py 实现

\`\`\`python
from models import TRTModule
from models.utils import letterbox, blob
from models.torch_utils import det_postprocess

# 1. 加载引擎
device = torch.device('cuda:0')
engine = TRTModule('yolov8s.engine', device)
H, W = 640, 640

# 2. 读取图像
bgr = cv2.imread('bus.jpg')

# 3. 预处理
bgr, ratio, dwdh = letterbox(bgr, (W, H))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
tensor = blob(rgb, return_seg=False)
tensor = torch.asarray(tensor, device=device)
dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)

# 4. 推理
engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
data = engine(tensor)

# 5. 后处理
bboxes, scores, labels = det_postprocess(data)
bboxes -= dwdh
bboxes /= ratio

# 6. 可视化
for bbox, score, label in zip(bboxes, scores, labels):
    x1, y1, x2, y2 = bbox.round().int().tolist()
    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
    cv2.putText(draw, f'{cls}:{score:.3f}', (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
\`\`\`

## 4.3 预处理详解

### letterbox 函数

\`\`\`python
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """调整图像大小并填充"""
    shape = im.shape[:2]  # 当前 [height, width]
    
    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 新尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    # 分配填充
    dw /= 2
    dh /= 2
    
    # 调整大小
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)
\`\`\`

### blob 函数

\`\`\`python
def blob(im, return_seg=False):
    """转换图像为网络输入"""
    # 归一化
    im = im.astype(np.float32) / 255.0
    
    if not return_seg:
        # HWC → CHW
        im = im.transpose([2, 0, 1])
    
    # 添加 batch 维度
    im = np.ascontiguousarray(im)
    im = im[None, ...]  # (1, 3, H, W)
    
    return im
\`\`\`

## 4.4 后处理详解

### det_postprocess 函数

\`\`\`python
def det_postprocess(data):
    """检测后处理"""
    num_dets, bboxes, scores, labels = data
    
    # 获取有效检测数量
    num = int(num_dets[0, 0])
    
    # 截取有效部分
    bboxes = bboxes[0, :num]
    scores = scores[0, :num]
    labels = labels[0, :num]
    
    return bboxes, scores, labels
\`\`\`

## 4.5 批量推理

\`\`\`python
# 批处理示例
images = path_to_list('data/')  # 获取所有图像

for image_path in images:
    # 读取
    bgr = cv2.imread(str(image_path))
    
    # 预处理
    bgr, ratio, dwdh = letterbox(bgr, (640, 640))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    tensor = torch.asarray(tensor, device=device)
    
    # 推理
    data = engine(tensor)
    
    # 后处理
    bboxes, scores, labels = det_postprocess(data)
    
    # ... 可视化和保存
\`\`\`

---

# 第五章：C++ 推理实现 {#第五章}

## 5.1 C++ YOLOv8 类

### 头文件 (yolov8.hpp)

\`\`\`cpp
class YOLOv8 {
public:
    YOLOv8(const std::string& engine_path);
    ~YOLOv8();
    
    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void infer();
    void postprocess(std::vector<Object>& objs);
    void draw_objects(const cv::Mat& image, cv::Mat& res,
                     const std::vector<Object>& objs);

private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    
    float* device_ptrs[5];  // input, num_dets, bboxes, scores, labels
    int num_bindings;
};

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};
\`\`\`

### 实现文件

\`\`\`cpp
YOLOv8::YOLOv8(const std::string& engine_path) {
    // 1. 加载引擎
    std::ifstream file(engine_path, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buffer = new char[size];
    file.read(buffer, size);
    file.close();
    
    // 2. 反序列化
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(buffer, size);
    delete[] buffer;
    
    // 3. 创建执行上下文
    context = engine->createExecutionContext();
    
    // 4. 创建 CUDA stream
    cudaStreamCreate(&stream);
    
    // 5. 分配设备内存
    num_bindings = engine->getNbBindings();
    for (int i = 0; i < num_bindings; i++) {
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; j++)
            vol *= dims.d[j];
        
        cudaMalloc(&device_ptrs[i], vol * sizeof(float));
    }
}

void YOLOv8::infer() {
    // 执行推理
    context->enqueueV2((void**)device_ptrs, stream, nullptr);
    cudaStreamSynchronize(stream);
}
\`\`\`

## 5.2 编译与构建

### CMakeLists.txt

\`\`\`cmake
cmake_minimum_required(VERSION 3.10)
project(yolov8_tensorrt)

set(CMAKE_CXX_STANDARD 14)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT
set(TENSORRT_DIR /path/to/TensorRT)
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 可执行文件
add_executable(yolov8 main.cpp)

target_link_libraries(yolov8
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)
\`\`\`

### 编译步骤

\`\`\`bash
cd csrc/detect/end2end
mkdir build && cd build
cmake ..
make -j8
\`\`\`

## 5.3 使用示例

\`\`\`cpp
int main(int argc, char** argv) {
    // 1. 创建 YOLOv8 对象
    YOLOv8* yolov8 = new YOLOv8("yolov8s.engine");
    yolov8->make_pipe(true);  // 预热
    
    // 2. 读取图像
    cv::Mat image = cv::imread("bus.jpg");
    cv::Size size(640, 640);
    
    // 3. 预处理
    yolov8->copy_from_Mat(image, size);
    
    // 4. 推理
    auto start = std::chrono::system_clock::now();
    yolov8->infer();
    auto end = std::chrono::system_clock::now();
    
    // 5. 后处理
    std::vector<Object> objs;
    yolov8->postprocess(objs);
    
    // 6. 可视化
    cv::Mat result;
    yolov8->draw_objects(image, result, objs, CLASS_NAMES, COLORS);
    
    // 7. 显示
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference time: " << time << " ms" << std::endl;
    cv::imshow("result", result);
    cv::waitKey(0);
    
    delete yolov8;
    return 0;
}
\`\`\`

---

# 第六章：高级特性 {#第六章}

## 6.1 分割模型推理

### Python 实现

\`\`\`python
from models.torch_utils import seg_postprocess

# 推理
data = engine(tensor)

# 后处理（包含分割掩码）
bboxes, scores, labels, masks = seg_postprocess(
    data, conf_thres=0.25
)

# 绘制分割结果
for bbox, mask in zip(bboxes, masks):
    # 调整掩码大小
    mask = cv2.resize(mask, (W, H))
    
    # 应用掩码
    colored_mask = np.zeros_like(bgr)
    colored_mask[mask > 0.5] = color
    
    # 融合
    result = cv2.addWeighted(bgr, 0.5, colored_mask, 0.5, 0)
\`\`\`

## 6.2 姿态估计

\`\`\`python
from models.torch_utils import pose_postprocess

# 推理
data = engine(tensor)

# 后处理
bboxes, scores, kpts = pose_postprocess(data)
# kpts: (N, 17, 3) - 17个关键点，每个(x, y, conf)

# 绘制骨架
SKELETON = [[16,14], [14,12], [17,15], ...]
for kpt in kpts:
    # 绘制关键点
    for i, (x, y, conf) in enumerate(kpt):
        if conf > 0.5:
            cv2.circle(image, (int(x), int(y)), 5, KPS_COLORS[i], -1)
    
    # 绘制骨架
    for (i, j) in SKELETON:
        if kpt[i-1][2] > 0.5 and kpt[j-1][2] > 0.5:
            cv2.line(image,
                    (int(kpt[i-1][0]), int(kpt[i-1][1])),
                    (int(kpt[j-1][0]), int(kpt[j-1][1])),
                    LIMB_COLORS[idx], 2)
\`\`\`

## 6.3 分类模型

\`\`\`python
# 推理
output = engine(tensor)  # (1, 1000)

# 获取 top-5 类别
top5 = torch.topk(output, 5)
scores, indices = top5.values[0], top5.indices[0]

for score, idx in zip(scores, indices):
    print(f"{CLASSES_CLS[idx]}: {score:.3f}")
\`\`\`

## 6.4 OBB (旋转检测)

\`\`\`python
# OBB 输出
bboxes, scores, labels = obb_postprocess(data)
# bboxes: (N, 5) - [x, y, w, h, angle]

# 绘制旋转框
for bbox in bboxes:
    x, y, w, h, angle = bbox
    rect = ((x, y), (w, h), angle * 180 / np.pi)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, 2)
\`\`\`

## 6.5 DeepStream 集成

\`\`\`bash
cd csrc/deepstream

# 编译自定义解析器
make

# 运行 DeepStream
deepstream-app -c deepstream_app_config.txt
\`\`\`

## 6.6 Jetson 部署

\`\`\`bash
# Jetson 专用编译
cd csrc/jetson/detect
mkdir build && cd build
cmake ..
make

# 运行
./yolov8_jetson yolov8s.engine image.jpg
\`\`\`

---

# 第七章：性能优化与调试 {#第七章}

## 7.1 性能分析

### Python Profiling

\`\`\`python
from models.engine import TRTProfilerV1

# 创建 profiler
profiler = TRTProfilerV1()
engine.set_profiler(profiler)

# 推理
for _ in range(100):
    engine(tensor)

# 报告
profiler.report()
\`\`\`

### 使用 trt-profile.py

\`\`\`bash
python3 trt-profile.py --engine yolov8s.engine --device cuda:0
\`\`\`

输出示例：
\`\`\`
Layer Name                              Time (us)
Conv_0                                  120.5
C2f_2                                   350.2
...
Total: 8520.3 us (117 FPS)
\`\`\`

## 7.2 精度对比

### FP32 vs FP16 vs INT8

\`\`\`python
import time
import numpy as np

def benchmark(engine_path, iterations=100):
    engine = TRTModule(engine_path, device)
    times = []
    
    for _ in range(iterations):
        start = time.time()
        engine(tensor)
        times.append(time.time() - start)
    
    return np.mean(times) * 1000  # ms

# 测试
fp32_time = benchmark('yolov8s_fp32.engine')
fp16_time = benchmark('yolov8s_fp16.engine')

print(f"FP32: {fp32_time:.2f} ms")
print(f"FP16: {fp16_time:.2f} ms")
print(f"Speedup: {fp32_time / fp16_time:.2f}x")
\`\`\`

## 7.3 内存优化

### Workspace 设置

\`\`\`python
# build.py 中调整
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    4 * (1 << 30)  # 4GB
)
\`\`\`

### 批处理优化

\`\`\`python
# 使用批处理提高吞吐量
batch_size = 4
tensor = torch.randn(batch_size, 3, 640, 640).cuda()

# 推理
data = engine(tensor)
# 输出: (batch_size, topk, ...)
\`\`\`

## 7.4 常见问题

### 问题 1：引擎不兼容

**错误**：
\`\`\`
[TRT] engine.cpp:1025: [EnforceCheck] Error Code 1: Cuda Runtime (invalid argument)
\`\`\`

**原因**：引擎在不同 GPU 架构间不兼容

**解决**：在目标设备上重新构建引擎

### 问题 2：内存不足

**错误**：
\`\`\`
[TRT] Out of memory
\`\`\`

**解决**：
- 减小 batch size
- 降低输入分辨率
- 使用 FP16 或 INT8

### 问题 3：推理速度慢

**排查**：
1. 检查是否使用 FP16
2. 确认 CUDA/cuDNN 版本最新
3. 使用 profiler 定位瓶颈层
4. 考虑动态 shape 开销

---

# 第八章：实战案例与最佳实践 {#第八章}

## 8.1 视频流处理

\`\`\`python
import cv2
from models import TRTModule

engine = TRTModule('yolov8s.engine', device='cuda:0')

cap = cv2.VideoCapture('video.mp4')
# 或摄像头: cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 预处理
    bgr, ratio, dwdh = letterbox(frame, (640, 640))
    tensor = blob(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tensor = torch.asarray(tensor, device='cuda:0')
    
    # 推理
    data = engine(tensor)
    bboxes, scores, labels = det_postprocess(data)
    
    # 绘制
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = (bbox / ratio).round().int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
\`\`\`

## 8.2 多模型集成

\`\`\`python
# 检测 + 分割
det_engine = TRTModule('yolov8s.engine', device)
seg_engine = TRTModule('yolov8s-seg.engine', device)

# 先检测
det_data = det_engine(tensor)
bboxes, scores, labels = det_postprocess(det_data)

# 对感兴趣区域分割
for bbox in bboxes:
    x1, y1, x2, y2 = bbox.int().tolist()
    roi = tensor[:, :, y1:y2, x1:x2]
    seg_data = seg_engine(roi)
    # ... 处理分割结果
\`\`\`

## 8.3 生产环境部署

### Flask API 服务

\`\`\`python
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)
engine = TRTModule('yolov8s.engine', device='cuda:0')

@app.route('/detect', methods=['POST'])
def detect():
    # 接收 base64 图像
    img_data = base64.b64decode(request.json['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 预处理
    bgr, ratio, dwdh = letterbox(img, (640, 640))
    tensor = blob(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tensor = torch.asarray(tensor, device='cuda:0')
    
    # 推理
    data = engine(tensor)
    bboxes, scores, labels = det_postprocess(data)
    
    # 格式化结果
    results = []
    for bbox, score, label in zip(bboxes, scores, labels):
        results.append({
            'bbox': (bbox / ratio).tolist(),
            'score': float(score),
            'class': int(label)
        })
    
    return jsonify({'detections': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
\`\`\`

## 8.4 最佳实践总结

### 模型选择
- **速度优先**：yolov8n, yolov8s
- **精度优先**：yolov8m, yolov8l, yolov8x
- **平衡选择**：yolov8s (推荐)

### 分辨率选择
- **实时应用**：640x640
- **高精度**：1280x1280
- **低延迟**：320x320 或 416x416

### 精度配置
- **开发阶段**：FP32 (验证精度)
- **生产部署**：FP16 (速度优先)
- **边缘设备**：INT8 (Jetson)

### 批处理
- **单帧推理**：batch_size=1
- **离线处理**：batch_size=4/8/16
- **显存限制**：动态调整 batch_size

### 错误处理
\`\`\`python
try:
    data = engine(tensor)
    bboxes, scores, labels = det_postprocess(data)
except RuntimeError as e:
    print(f"Inference failed: {e}")
    # 降级处理或跳过
\`\`\`

### 日志记录
\`\`\`python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Engine loaded: {engine_path}")
logger.info(f"Input shape: {engine.inp_info[0].shape}")
logger.info(f"Inference time: {time_ms:.2f} ms")
\`\`\`

## 8.5 总结

本教程涵盖了 YOLOv8-TensorRT 的完整使用流程：

✅ **第1章**：环境准备和快速开始  
✅ **第2章**：ONNX 模型导出原理与实践  
✅ **第3章**：TensorRT 引擎构建方法  
✅ **第4章**：Python 推理实现  
✅ **第5章**：C++ 高性能推理  
✅ **第6章**：多任务和多平台支持  
✅ **第7章**：性能优化与调试技巧  
✅ **第8章**：生产环境部署实践  

### 进一步学习

- TensorRT 官方文档：https://docs.nvidia.com/deeplearning/tensorrt/
- YOLOv8 文档：https://docs.ultralytics.com/
- 项目仓库：https://github.com/triple-Mu/YOLOv8-TensorRT

---

**教程版本**: 1.0  
**最后更新**: 2025-12-29  
**作者**: YOLOv8-TensorRT 项目团队
