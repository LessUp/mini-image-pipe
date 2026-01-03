# Mini-ImagePipe

基于任务图（Task Graph）的异构图像处理流水线框架，专为高分辨率视频流处理场景设计。

## 特性

- **GPU 加速**: 全 CUDA 实现，支持异步执行
- **DAG 调度**: 基于有向无环图的任务依赖管理
- **多流并发**: 支持多 CUDA 流并行执行独立任务
- **内存优化**: Pinned Memory 池化管理，减少分配开销
- **可分离滤波**: 高斯模糊使用可分离滤波优化
- **共享内存**: 卷积操作使用 GPU 共享内存加速

## 图像处理算子

| 算子 | 功能 | 特性 |
|------|------|------|
| GaussianBlur | 高斯模糊 | 3x3/5x5/7x7 可分离滤波，反射边界填充 |
| Sobel | 边缘检测 | 3x3 Sobel 核，梯度幅值输出 |
| Resize | 图像缩放 | 双线性/最近邻插值 |
| ColorConvert | 颜色转换 | RGB↔Gray, BGR↔RGB, RGBA→RGB |

## 构建

### 依赖

- CMake >= 3.18
- CUDA Toolkit >= 11.0
- GTest (可选，用于测试)

### 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行示例

```bash
./demo_pipeline
```

### 运行测试

```bash
./mini_image_pipe_tests
```

## 使用示例

```cpp
#include "pipeline.h"
#include "operators/resize.h"
#include "operators/color_convert.h"
#include "operators/gaussian_blur.h"
#include "operators/sobel.h"

using namespace mini_image_pipe;

int main() {
    // 创建 Pipeline
    PipelineConfig config;
    config.numStreams = 4;
    Pipeline pipeline(config);

    // 添加算子
    auto resize = std::make_shared<ResizeOperator>(320, 240, InterpolationMode::BILINEAR);
    auto gray = std::make_shared<ColorConvertOperator>(ColorConversionType::RGB_TO_GRAY);
    auto blur = std::make_shared<GaussianBlurOperator>(GaussianKernelSize::KERNEL_5x5);
    auto sobel = std::make_shared<SobelOperator>();

    int n1 = pipeline.addOperator("Resize", resize);
    int n2 = pipeline.addOperator("Gray", gray);
    int n3 = pipeline.addOperator("Blur", blur);
    int n4 = pipeline.addOperator("Sobel", sobel);

    // 连接算子: Resize -> Gray -> Blur -> Sobel
    pipeline.connect(n1, n2);
    pipeline.connect(n2, n3);
    pipeline.connect(n3, n4);

    // 设置输入
    pipeline.setInput(n1, d_input, width, height, channels);

    // 执行
    pipeline.execute();

    // 获取输出
    void* output = pipeline.getOutput(n4);
    
    return 0;
}
```

## 项目结构

```
mini-image-pipe/
├── include/
│   ├── types.h              # 数据类型定义
│   ├── operator.h           # IOperator 基类
│   ├── memory_manager.h     # 内存管理器
│   ├── task_graph.h         # DAG 任务图
│   ├── scheduler.h          # CUDA 流调度器
│   ├── pipeline.h           # Pipeline 构建器
│   └── operators/
│       ├── color_convert.h
│       ├── resize.h
│       ├── sobel.h
│       └── gaussian_blur.h
├── src/
│   ├── memory_manager.cu
│   ├── task_graph.cpp
│   ├── scheduler.cu
│   ├── pipeline.cpp
│   └── operators/
│       ├── color_convert.cu
│       ├── resize.cu
│       ├── sobel.cu
│       └── gaussian_blur.cu
├── tests/                   # 属性测试
├── examples/
│   └── demo_pipeline.cpp
└── CMakeLists.txt
```

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                      Pipeline API                        │
├─────────────────────────────────────────────────────────┤
│  TaskGraph  │  DAGScheduler  │  MemoryManager           │
├─────────────────────────────────────────────────────────┤
│  Operators: Gaussian │ Sobel │ Resize │ ColorConvert    │
├─────────────────────────────────────────────────────────┤
│  CUDA Streams  │  CUDA Events  │  Shared Memory         │
└─────────────────────────────────────────────────────────┘
```

## License

MIT
