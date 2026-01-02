# YOLOv8-TensorRT 系统实现架构文档

## 目录
- [1. 系统概述](#1-系统概述)
- [2. 整体架构](#2-整体架构)
- [3. 核心模块架构](#3-核心模块架构)
- [4. 数据流与处理流程](#4-数据流与处理流程)
- [5. 技术栈与依赖](#5-技术栈与依赖)
- [6. 目录结构](#6-目录结构)
- [7. 接口设计](#7-接口设计)

---

## 1. 系统概述

### 1.1 项目简介
YOLOv8-TensorRT 是一个基于 NVIDIA TensorRT 的 YOLOv8 深度学习模型加速推理框架。该项目通过将 YOLOv8 模型转换为 TensorRT 引擎，实现了高性能的目标检测、实例分割、姿态估计、图像分类和旋转目标检测（OBB）等任务。

### 1.2 核心功能
- **模型转换**: PyTorch → ONNX → TensorRT Engine
- **多任务支持**: 检测、分割、姿态估计、分类、OBB
- **多端部署**: Python API、C++ API、DeepStream、Jetson
- **性能优化**: FP16 精度、动态/静态 shape、NMS 插件集成
- **灵活推理**: 支持 PyTorch、CUDA-Python、PyCUDA 后端

### 1.3 设计目标
1. **高性能**: 利用 TensorRT 优化，最大化推理速度
2. **易用性**: 提供简洁的 Python 和 C++ 接口
3. **灵活性**: 支持多种部署方式和精度配置
4. **可扩展**: 模块化设计，便于功能扩展

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户应用层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Python 应用   │  │  C++ 应用    │  │ DeepStream   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        推理接口层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │TRTModule(Py) │  │YOLOv8(C++)   │  │DeepStream API│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        引擎执行层                                 │
│                    ┌──────────────────┐                         │
│                    │ TensorRT Runtime │                         │
│                    │  - Context       │                         │
│                    │  - Engine        │                         │
│                    │  - Stream        │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        引擎构建层                                 │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │ ONNX Parser  │  │  TensorRT    │                            │
│  │              │  │  API Builder │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        模型导出层                                 │
│                    ┌──────────────────┐                         │
│                    │   Ultralytics    │                         │
│                    │   YOLOv8 Export  │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       底层硬件层                                  │
│            ┌────────────┐        ┌────────────┐                │
│            │  CUDA      │        │  cuDNN     │                │
│            └────────────┘        └────────────┘                │
│                        NVIDIA GPU                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 架构层次说明

#### 2.1.1 用户应用层
最上层，用户可以通过 Python、C++ 或 DeepStream 应用程序调用推理功能。

#### 2.1.2 推理接口层
封装了推理逻辑，提供统一的接口供应用层调用：
- **TRTModule**: Python 推理模块，基于 PyTorch 实现
- **YOLOv8**: C++ 推理类，直接调用 TensorRT API
- **DeepStream API**: NVIDIA DeepStream 集成接口

#### 2.1.3 引擎执行层
TensorRT Runtime 负责执行推理任务：
- **Context**: 执行上下文，管理推理状态
- **Engine**: 优化后的模型引擎
- **Stream**: CUDA 流，管理异步执行

#### 2.1.4 引擎构建层
将 ONNX 模型或通过 API 构建 TensorRT 引擎：
- **ONNX Parser**: 解析 ONNX 模型
- **TensorRT API Builder**: 使用 API 直接构建网络

#### 2.1.5 模型导出层
使用 Ultralytics 将 PyTorch 模型导出为 ONNX 格式

#### 2.1.6 底层硬件层
CUDA、cuDNN 等 NVIDIA 库和 GPU 硬件

---

## 3. 核心模块架构

### 3.1 模型导出模块

#### 3.1.1 组件关系图
```
┌─────────────────────────────────────────┐
│          export-det.py                  │
│  ┌───────────────────────────────────┐  │
│  │ 1. 加载 PyTorch 模型 (YOLO)       │  │
│  │ 2. 添加后处理层 (PostDetect)      │  │
│  │ 3. 导出为 ONNX                    │  │
│  │ 4. 简化 ONNX (可选)               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↓                    ↓
┌──────────────┐    ┌─────────────────────┐
│ .pt 模型文件  │    │  models/common.py   │
└──────────────┘    │  - PostDetect       │
                    │  - PostSeg          │
                    │  - optim            │
                    └─────────────────────┘
```

#### 3.1.2 核心类和函数
- **PostDetect**: 添加 NMS 后处理到检测模型
- **PostSeg**: 添加后处理到分割模型
- **optim**: 优化模型层，替换不支持的操作

### 3.2 引擎构建模块

#### 3.2.1 组件关系图
```
┌────────────────────────────────────────────────────┐
│                  build.py                          │
│  ┌──────────────────────────────────────────────┐  │
│  │         EngineBuilder                        │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │  - __build_engine()                    │  │  │
│  │  │  - build_from_onnx()                   │  │  │
│  │  │  - build_from_api()                    │  │  │
│  │  └────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
                       ↓
        ┌──────────────────────────┐
        │   TensorRT Builder       │
        │  - Network Definition    │
        │  - Builder Config        │
        │  - Plugin Registry       │
        └──────────────────────────┘
                       ↓
        ┌──────────────────────────┐
        │     .engine 文件          │
        └──────────────────────────┘
```

#### 3.2.2 构建路径
1. **ONNX 路径**: ONNX → TensorRT Parser → Engine
2. **API 路径**: Pickle 权重 → TensorRT API → Engine

#### 3.2.3 关键配置
- **FP16**: 半精度加速
- **Workspace**: 内存池大小
- **输入 Shape**: 动态/静态
- **NMS 参数**: IOU、Conf 阈值

### 3.3 推理执行模块

#### 3.3.1 Python 推理架构
```
┌─────────────────────────────────────────────────┐
│              infer-det.py                       │
│  ┌───────────────────────────────────────────┐  │
│  │  1. 加载 Engine (TRTModule)              │  │
│  │  2. 预处理图像 (letterbox, blob)         │  │
│  │  3. 推理 (Engine.forward)                │  │
│  │  4. 后处理 (det_postprocess)             │  │
│  │  5. 可视化 (draw_bboxes)                 │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            models/engine.py                     │
│  ┌───────────────────────────────────────────┐  │
│  │          TRTModule                        │  │
│  │  - __init_engine()                        │  │
│  │  - __init_bindings()                      │  │
│  │  - forward()                              │  │
│  │  - set_profiler()                         │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### 3.3.2 C++ 推理架构
```
┌─────────────────────────────────────────────────┐
│         csrc/detect/end2end/main.cpp            │
│  ┌───────────────────────────────────────────┐  │
│  │              main()                       │  │
│  │  1. 解析命令行参数                         │  │
│  │  2. 创建 YOLOv8 对象                      │  │
│  │  3. make_pipe() - 初始化推理管道           │  │
│  │  4. 循环读取输入                          │  │
│  │  5. copy_from_Mat()                       │  │
│  │  6. infer()                               │  │
│  │  7. postprocess()                         │  │
│  │  8. draw_objects()                        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              YOLOv8 类                          │
│  - 成员变量：                                    │
│    * nvinfer1::ICudaEngine* engine             │
│    * nvinfer1::IExecutionContext* context      │
│    * cudaStream_t stream                       │
│  - 核心方法：                                    │
│    * make_pipe()                               │
│    * copy_from_Mat()                           │
│    * infer()                                   │
│    * postprocess()                             │
│    * draw_objects()                            │
└─────────────────────────────────────────────────┘
```

### 3.4 工具模块

#### 3.4.1 辅助工具
```
models/
├── utils.py          # 通用工具函数
│   ├── letterbox()   # 图像预处理
│   ├── blob()        # 转换为网络输入
│   ├── path_to_list()# 路径处理
│
├── torch_utils.py    # PyTorch 相关工具
│   ├── det_postprocess()  # 检测后处理
│   ├── seg_postprocess()  # 分割后处理
│   ├── pose_postprocess() # 姿态后处理
│
├── cudart_api.py     # CUDA Runtime API
│   └── TRTEngine (无 PyTorch)
│
└── pycuda_api.py     # PyCUDA API
    └── TRTEngine (无 PyTorch)
```

---

## 4. 数据流与处理流程

### 4.1 完整工作流程

```
┌──────────────┐
│ 训练好的     │
│ .pt 模型     │
└──────┬───────┘
       │
       ↓ export-det.py / export-seg.py
┌──────────────┐
│ .onnx 模型   │
└──────┬───────┘
       │
       ↓ build.py / trtexec
┌──────────────┐
│ .engine 文件 │
└──────┬───────┘
       │
       ↓ infer-det.py / C++ 应用
┌──────────────┐
│ 推理结果     │
│ (检测框/分割) │
└──────────────┘
```

### 4.2 推理数据流详解

#### 4.2.1 输入数据处理
```
原始图像 (H, W, 3)
    ↓
letterbox 填充 → (640, 640, 3)
    ↓
BGR → RGB 转换
    ↓
归一化 (0-255 → 0-1)
    ↓
HWC → CHW 转换
    ↓
Batch 维度添加 → (1, 3, 640, 640)
    ↓
转为 Tensor/GPU 内存
```

#### 4.2.2 引擎推理流程
```
┌─────────────────────────────────────┐
│         1. 设置输入绑定              │
│  context.set_tensor_address()       │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│         2. 执行推理                  │
│  context.execute_async_v3()         │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│         3. 同步等待                  │
│  stream.synchronize()               │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│         4. 读取输出                  │
│  - num_dets: 检测数量               │
│  - bboxes: 边界框坐标               │
│  - scores: 置信度                   │
│  - labels: 类别标签                 │
└─────────────────────────────────────┘
```

#### 4.2.3 输出后处理
```
推理输出
    ↓
过滤低置信度检测 (scores > threshold)
    ↓
坐标转换 (去除 padding)
    ↓
坐标缩放 (恢复原始尺寸)
    ↓
绘制结果 / 保存输出
```

### 4.3 不同任务的数据流

#### 4.3.1 目标检测
```
Input: (1, 3, 640, 640)
    ↓
Engine Forward
    ↓
Output:
  - num_dets: (1, 1)
  - bboxes: (1, 100, 4)
  - scores: (1, 100)
  - labels: (1, 100)
```

#### 4.3.2 实例分割
```
Input: (1, 3, 640, 640)
    ↓
Engine Forward
    ↓
Output:
  - num_dets: (1, 1)
  - bboxes: (1, 100, 4)
  - scores: (1, 100)
  - labels: (1, 100)
  - masks: (1, 100, 160, 160)  # 或原图尺寸
```

#### 4.3.3 姿态估计
```
Input: (1, 3, 640, 640)
    ↓
Engine Forward
    ↓
Output:
  - num_dets: (1, 1)
  - bboxes: (1, 100, 4)
  - scores: (1, 100)
  - labels: (1, 100)
  - keypoints: (1, 100, 51)  # 17个关键点 * 3(x,y,conf)
```

---

## 5. 技术栈与依赖

### 5.1 核心技术栈

#### 5.1.1 Python 栈
```yaml
运行时:
  - Python: 3.8-3.10
  - CUDA: >= 11.4
  - cuDNN: 与 CUDA 匹配
  - TensorRT: >= 8.4

深度学习框架:
  - PyTorch: >= 1.10
  - Ultralytics: 最新版

模型处理:
  - ONNX: >= 1.12
  - onnxsim: 最新版 (可选)

推理后端:
  - cuda-python: 最新版 (可选)
  - pycuda: 最新版 (可选)

图像处理:
  - OpenCV (cv2): >= 4.0
  - NumPy: >= 1.19
```

#### 5.1.2 C++ 栈
```yaml
编译器:
  - C++ 标准: C++11/C++14
  - GCC/G++: >= 7.0
  - CMake: >= 3.10

依赖库:
  - CUDA Toolkit: >= 11.4
  - TensorRT: >= 8.4
  - OpenCV: >= 4.0

Jetson 平台:
  - JetPack: 对应版本
  - L4T: 对应版本
```

### 5.2 依赖关系图

```
┌──────────────────────────────────────────┐
│            应用程序                       │
└────────────┬─────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼─────┐
│PyTorch │      │ OpenCV   │
└───┬────┘      └──────────┘
    │
┌───▼────────────────┐
│  TensorRT          │
│  - nvinfer         │
│  - nvonnxparser    │
│  - nvparsers       │
└───┬────────────────┘
    │
┌───▼────────────────┐
│  CUDA/cuDNN        │
└───┬────────────────┘
    │
┌───▼────────────────┐
│  NVIDIA GPU Driver │
└────────────────────┘
```

### 5.3 版本兼容性矩阵

| 组件 | 推荐版本 | 最低版本 | 说明 |
|------|---------|---------|------|
| CUDA | 11.8 | 11.4 | 更高版本性能更好 |
| TensorRT | 8.6 | 8.4 | 支持更多优化 |
| Python | 3.10 | 3.8 | 3.11+ 可能有兼容性问题 |
| PyTorch | 2.0 | 1.10 | 与 CUDA 版本匹配 |
| Ultralytics | 8.0+ | 8.0 | YOLOv8 支持 |
| OpenCV | 4.7 | 4.0 | Python 和 C++ 都需要 |

---

## 6. 目录结构

### 6.1 项目文件组织

```
YOLOv8-TensorRT/
│
├── README.md                 # 项目说明
├── LICENSE                   # 许可证
├── requirements.txt          # Python 依赖
├── config.py                 # 全局配置 (类别、颜色)
│
├── models/                   # Python 模块
│   ├── __init__.py
│   ├── engine.py            # TensorRT 引擎封装
│   ├── api.py               # TensorRT API 构建
│   ├── common.py            # 通用层定义
│   ├── utils.py             # 工具函数
│   ├── torch_utils.py       # PyTorch 工具
│   ├── cudart_api.py        # CUDA Runtime API
│   └── pycuda_api.py        # PyCUDA API
│
├── export-det.py            # 检测模型导出
├── export-seg.py            # 分割模型导出
├── build.py                 # 引擎构建
├── gen_pkl.py               # 生成 pickle 权重
│
├── infer-det.py             # 检测推理 (PyTorch)
├── infer-seg.py             # 分割推理
├── infer-pose.py            # 姿态推理
├── infer-cls.py             # 分类推理
├── infer-obb.py             # OBB 推理
│
├── infer-det-without-torch.py   # 无 PyTorch 推理
├── infer-seg-without-torch.py
├── infer-pose-without-torch.py
├── infer-cls-without-torch.py
├── infer-obb-without-torch.py
│
├── trt-profile.py           # 性能分析
│
├── csrc/                    # C++ 源代码
│   ├── detect/
│   │   ├── end2end/         # 端到端检测
│   │   │   ├── main.cpp
│   │   │   ├── yolov8.hpp
│   │   │   └── CMakeLists.txt
│   │   └── normal/          # 常规检测
│   │
│   ├── segment/
│   │   ├── normal/
│   │   └── simple/
│   │
│   ├── pose/
│   │   └── normal/
│   │
│   ├── cls/
│   │   └── normal/
│   │
│   ├── obb/
│   │   └── normal/
│   │
│   ├── jetson/              # Jetson 平台
│   │   ├── detect/
│   │   ├── pose/
│   │   └── segment/
│   │
│   └── deepstream/          # DeepStream 集成
│       ├── README.md
│       ├── CMakeLists.txt
│       └── custom_bbox_parser/
│
├── docs/                    # 文档
│   ├── API-Build.md
│   ├── Normal.md
│   ├── Segment.md
│   ├── Pose.md
│   ├── Cls.md
│   ├── Obb.md
│   ├── Jetson.md
│   └── star.md
│
└── data/                    # 测试数据 (示例图像/视频)
```

### 6.2 模块职责划分

#### 6.2.1 Python 模块
- **models/engine.py**: 核心推理引擎，封装 TensorRT Runtime
- **models/api.py**: TensorRT API 构建网络层
- **models/common.py**: 后处理层和优化函数
- **models/utils.py**: 图像预处理、路径处理等工具
- **models/torch_utils.py**: 后处理函数（检测、分割、姿态）
- **models/cudart_api.py**: 基于 CUDA Runtime 的推理（无 PyTorch）
- **models/pycuda_api.py**: 基于 PyCUDA 的推理（无 PyTorch）

#### 6.2.2 导出脚本
- **export-det.py**: 导出检测模型为 ONNX + NMS
- **export-seg.py**: 导出分割模型为 ONNX
- **gen_pkl.py**: 提取模型权重为 pickle 格式

#### 6.2.3 构建脚本
- **build.py**: 从 ONNX 或 pickle 构建 TensorRT 引擎

#### 6.2.4 推理脚本
- **infer-*.py**: 各任务 Python 推理脚本
- **infer-*-without-torch.py**: 无 PyTorch 依赖的推理脚本

#### 6.2.5 C++ 应用
- **csrc/detect/end2end**: 端到端检测应用
- **csrc/segment**: 分割应用
- **csrc/pose**: 姿态估计应用
- **csrc/cls**: 分类应用
- **csrc/obb**: OBB 应用
- **csrc/jetson**: Jetson 平台优化版本
- **csrc/deepstream**: DeepStream 集成

---

## 7. 接口设计

### 7.1 Python API

#### 7.1.1 TRTModule 类
```python
from models import TRTModule

# 初始化
engine = TRTModule(
    weight='yolov8s.engine',  # 引擎文件路径
    device=torch.device('cuda:0')  # 推理设备
)

# 查看输入输出信息
print(engine.inp_info)  # 输入张量信息
print(engine.out_info)  # 输出张量信息

# 设置期望的输出顺序
engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

# 推理
outputs = engine(input_tensor)  # input: torch.Tensor
# outputs: Tuple[torch.Tensor, ...] 或 torch.Tensor

# 性能分析
from models.engine import TRTProfilerV1
profiler = TRTProfilerV1()
engine.set_profiler(profiler)
```

#### 7.1.2 EngineBuilder 类
```python
from models import EngineBuilder

# 从 ONNX 构建
builder = EngineBuilder(
    checkpoint='yolov8s.onnx',
    device='cuda:0'
)
builder.build(
    fp16=True,                    # 使用 FP16
    input_shape=(1, 3, 640, 640), # 输入形状
    iou_thres=0.65,               # NMS IOU 阈值
    conf_thres=0.25,              # 置信度阈值
    topk=100                      # 最大检测数
)

# 从 API 构建 (需要 .pkl 文件)
builder = EngineBuilder(
    checkpoint='yolov8s.pkl',
    device='cuda:0'
)
builder.build(
    fp16=True,
    input_shape=(1, 3, 640, 640),
    iou_thres=0.65,
    conf_thres=0.25,
    topk=100
)
```

#### 7.1.3 推理工具函数
```python
from models.utils import letterbox, blob, path_to_list
from models.torch_utils import det_postprocess

# 图像预处理
bgr = cv2.imread('image.jpg')
bgr, ratio, dwdh = letterbox(bgr, (640, 640))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
tensor = blob(rgb, return_seg=False)  # (1, 3, 640, 640)

# 后处理
bboxes, scores, labels = det_postprocess(data)
# data: Tuple(num_dets, bboxes, scores, labels)
# 返回: (N, 4), (N,), (N,) 只包含有效检测

# 路径处理
images = path_to_list('data/')  # 支持文件/目录
```

### 7.2 C++ API

#### 7.2.1 YOLOv8 类接口
```cpp
#include "yolov8.hpp"

// 构造
YOLOv8* yolov8 = new YOLOv8(engine_file_path);

// 初始化推理管道
yolov8->make_pipe(warmup=true);

// 图像输入
cv::Mat image = cv::imread("image.jpg");
cv::Size size(640, 640);
yolov8->copy_from_Mat(image, size);

// 推理
yolov8->infer();

// 后处理
std::vector<Object> objs;
yolov8->postprocess(objs);

// 可视化
cv::Mat result;
yolov8->draw_objects(image, result, objs, CLASS_NAMES, COLORS);

// 清理
delete yolov8;
```

#### 7.2.2 Object 结构体
```cpp
struct Object {
    cv::Rect_<float> rect;  // 边界框
    int              label; // 类别 ID
    float            prob;  // 置信度
    cv::Mat          boxMask; // 分割掩码 (仅分割任务)
    std::vector<cv::Point2f> kps; // 关键点 (仅姿态任务)
};
```

### 7.3 命令行接口

#### 7.3.1 模型导出
```bash
# 检测模型
python3 export-det.py \
  --weights yolov8s.pt \
  --iou-thres 0.65 \
  --conf-thres 0.25 \
  --topk 100 \
  --opset 11 \
  --sim \
  --input-shape 1 3 640 640 \
  --device cuda:0

# 分割模型
python3 export-seg.py \
  --weights yolov8s-seg.pt \
  --opset 11 \
  --sim \
  --input-shape 1 3 640 640 \
  --device cuda:0
```

#### 7.3.2 引擎构建
```bash
# 从 ONNX
python3 build.py \
  --weights yolov8s.onnx \
  --iou-thres 0.65 \
  --conf-thres 0.25 \
  --topk 100 \
  --fp16 \
  --device cuda:0

# 从 API (需要 .pkl)
python3 gen_pkl.py -w yolov8s.pt -o yolov8s.pkl
python3 build.py \
  --weights yolov8s.pkl \
  --iou-thres 0.65 \
  --conf-thres 0.25 \
  --topk 100 \
  --fp16 \
  --input-shape 1 3 640 640 \
  --device cuda:0
```

#### 7.3.3 Python 推理
```bash
python3 infer-det.py \
  --engine yolov8s.engine \
  --imgs data/ \
  --show \
  --out-dir outputs \
  --device cuda:0
```

#### 7.3.4 C++ 推理
```bash
# 编译
cd csrc/detect/end2end
mkdir build && cd build
cmake ..
make

# 推理
./yolov8 yolov8s.engine image.jpg     # 单张图像
./yolov8 yolov8s.engine data/         # 图像目录
./yolov8 yolov8s.engine video.mp4     # 视频
```

### 7.4 配置接口

#### 7.4.1 config.py
```python
# 类别名称
CLASSES_DET = ('person', 'bicycle', 'car', ...)
CLASSES_SEG = CLASSES_DET
CLASSES_OBB = ('plane', 'ship', ...)
CLASSES_CLS = ([...])  # ImageNet 1000 类

# 颜色映射
COLORS = {cls: [r, g, b] for cls in CLASSES_DET}
COLORS_OBB = {cls: [r, g, b] for cls in CLASSES_OBB}

# 分割掩码颜色
MASK_COLORS = np.array([...], dtype=np.float32) / 255.

# 姿态估计配置
KPS_COLORS = [[0, 255, 0], ...]  # 关键点颜色
SKELETON = [[16, 14], [14, 12], ...]  # 骨架连接
LIMB_COLORS = [[51, 153, 255], ...]  # 肢体颜色

# 透明度
ALPHA = 0.5
```

---

## 附录

### A. 关键术语表

| 术语 | 说明 |
|------|------|
| TensorRT | NVIDIA 的高性能深度学习推理优化器和运行时库 |
| ONNX | Open Neural Network Exchange，神经网络交换格式 |
| NMS | Non-Maximum Suppression，非极大值抑制 |
| FP16 | 16位浮点数，半精度 |
| Engine | TensorRT 优化后的可执行模型 |
| Context | TensorRT 执行上下文，管理推理状态 |
| Binding | TensorRT 输入/输出张量绑定 |
| Stream | CUDA 流，用于异步执行 |
| Letterbox | 保持宽高比的图像填充方法 |
| Blob | 神经网络输入张量 |

### B. 性能优化建议

1. **使用 FP16**: 在支持的 GPU 上启用半精度加速
2. **静态 Shape**: 尽量使用静态输入形状，避免动态 shape 开销
3. **批处理**: 使用批量推理提高吞吐量
4. **CUDA Stream**: 利用异步执行隐藏数据传输延迟
5. **Workspace**: 适当增加 workspace 大小
6. **版本选择**: 使用最新的 CUDA 和 TensorRT 版本
7. **预分配内存**: 复用输出缓冲区，减少内存分配

### C. 常见问题

**Q: 为什么 API 构建不支持分割模型？**
A: 分割模型的后处理较复杂，当前版本通过 ONNX 路径实现更简单。

**Q: 如何选择 PyTorch 后端 vs 无 PyTorch 后端？**
A: PyTorch 后端性能更好且易用，仅在无法安装 PyTorch 时使用无 PyTorch 版本。

**Q: 动态 shape 和静态 shape 如何选择？**
A: 静态 shape 性能更好，动态 shape 更灵活。生产环境推荐静态 shape。

**Q: 如何优化推理速度？**
A: 启用 FP16、使用批处理、选择合适的输入分辨率、使用最新驱动和库。

---

**文档版本**: 1.0  
**最后更新**: 2025-12-29  
**维护者**: YOLOv8-TensorRT 项目团队
