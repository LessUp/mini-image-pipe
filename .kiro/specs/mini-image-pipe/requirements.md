# Requirements Document

## Introduction

Mini-ImagePipe 是一个基于任务图（Task Graph）的异构图像处理流水线框架，专为高分辨率视频流处理场景设计。该框架支持全 GPU 流程执行，适用于自动驾驶感知、医疗影像处理和嵌入式 AI 等工业场景。

## Glossary

- **Task_Graph**: 有向无环图（DAG）结构，用于表示图像处理任务之间的依赖关系
- **Operator**: 图像处理算子，执行特定的图像变换操作
- **Scheduler**: 任务调度器，负责管理和调度 DAG 中的任务执行
- **CUDA_Stream**: CUDA 流，用于实现 GPU 上的异步并发执行
- **Pinned_Memory**: 页锁定内存，用于优化 Host-Device 数据传输
- **Separable_Filter**: 可分离滤波器，将二维卷积分解为两个一维卷积以提高性能
- **Halo_Region**: 光晕区域，用于处理卷积边界的额外数据区域
- **Shared_Memory**: GPU 共享内存，用于加速线程块内的数据访问

## Requirements

### Requirement 1: Gaussian Blur Operator

**User Story:** As a developer, I want to apply Gaussian blur to images, so that I can reduce noise and smooth images in the processing pipeline.

#### Acceptance Criteria

1. WHEN a Gaussian blur operation is requested, THE Operator SHALL apply a configurable kernel size (3x3, 5x5, 7x7) to the input image
2. WHEN processing large images, THE Operator SHALL use separable filter optimization to decompose 2D convolution into two 1D passes
3. WHEN executing on GPU, THE Operator SHALL utilize shared memory with halo regions for efficient boundary handling
4. WHEN the input image has edges, THE Operator SHALL handle boundary pixels using reflection padding
5. THE Operator SHALL support single-channel (grayscale) and multi-channel (RGB/RGBA) images

### Requirement 2: Sobel Edge Detection Operator

**User Story:** As a developer, I want to detect edges in images, so that I can identify object boundaries for downstream processing.

#### Acceptance Criteria

1. WHEN a Sobel operation is requested, THE Operator SHALL compute horizontal and vertical gradients using 3x3 Sobel kernels
2. WHEN computing edge magnitude, THE Operator SHALL calculate the gradient magnitude as sqrt(Gx² + Gy²)
3. WHEN executing on GPU, THE Operator SHALL use shared memory to minimize global memory access
4. THE Operator SHALL output gradient magnitude as a single-channel image

### Requirement 3: Resize Operator

**User Story:** As a developer, I want to resize images to different resolutions, so that I can adapt images for various processing stages.

#### Acceptance Criteria

1. WHEN a resize operation is requested, THE Operator SHALL support bilinear interpolation for smooth scaling
2. WHEN downscaling images, THE Operator SHALL support nearest-neighbor interpolation for fast processing
3. WHEN the target size is specified, THE Operator SHALL correctly compute output pixel coordinates from input coordinates
4. THE Operator SHALL support arbitrary scale factors (both upscaling and downscaling)

### Requirement 4: Color Conversion Operator

**User Story:** As a developer, I want to convert images between color spaces, so that I can prepare images for different processing algorithms.

#### Acceptance Criteria

1. WHEN a color conversion is requested, THE Operator SHALL support RGB to Grayscale conversion using standard luminance weights
2. WHEN converting RGB to Grayscale, THE Operator SHALL use the formula: Y = 0.299*R + 0.587*G + 0.114*B
3. WHEN a BGR to RGB conversion is requested, THE Operator SHALL correctly swap channel order
4. THE Operator SHALL preserve alpha channel when present during color space conversion

### Requirement 5: DAG Task Scheduler

**User Story:** As a developer, I want to define processing pipelines as directed acyclic graphs, so that I can express complex task dependencies and enable parallel execution.

#### Acceptance Criteria

1. WHEN tasks are added to the scheduler, THE Scheduler SHALL validate that no circular dependencies exist
2. WHEN executing the task graph, THE Scheduler SHALL respect all dependency constraints between tasks
3. WHEN multiple tasks have no dependencies on each other, THE Scheduler SHALL enable concurrent execution
4. WHEN a task completes, THE Scheduler SHALL notify dependent tasks and trigger their execution when ready
5. IF a task fails during execution, THEN THE Scheduler SHALL propagate the error and halt dependent tasks
6. THE Scheduler SHALL support topological sorting to determine valid execution order

### Requirement 6: CUDA Streams Concurrency

**User Story:** As a developer, I want to process multiple video streams concurrently, so that I can maximize GPU utilization and throughput.

#### Acceptance Criteria

1. WHEN multiple independent tasks are ready, THE Scheduler SHALL assign them to different CUDA streams for concurrent execution
2. WHEN a task depends on another task in a different stream, THE Scheduler SHALL use CUDA events for synchronization
3. WHEN processing multiple video streams, THE Scheduler SHALL enable overlapping of upload, compute, and download operations
4. THE Scheduler SHALL support configurable number of CUDA streams (default: 4)
5. WHEN all tasks complete, THE Scheduler SHALL synchronize all streams before returning results

### Requirement 7: Pinned Memory Management

**User Story:** As a developer, I want optimized host-device data transfer, so that I can achieve maximum bandwidth for video stream processing.

#### Acceptance Criteria

1. WHEN allocating host memory for data transfer, THE Memory_Manager SHALL use cudaHostAlloc for pinned memory allocation
2. WHEN transferring data to GPU, THE Memory_Manager SHALL use asynchronous memory copies with CUDA streams
3. WHEN pinned memory allocation fails, THE Memory_Manager SHALL fall back to pageable memory with a warning
4. THE Memory_Manager SHALL provide a memory pool to reuse pinned memory allocations and reduce allocation overhead
5. WHEN the pipeline shuts down, THE Memory_Manager SHALL properly free all pinned memory resources

### Requirement 8: Pipeline Integration

**User Story:** As a developer, I want to chain multiple operators into a complete processing pipeline, so that I can build end-to-end image processing workflows.

#### Acceptance Criteria

1. WHEN building a pipeline, THE Pipeline SHALL allow operators to be connected in sequence or parallel branches
2. WHEN executing a pipeline, THE Pipeline SHALL automatically manage intermediate buffer allocation
3. WHEN the same intermediate result is used by multiple downstream operators, THE Pipeline SHALL avoid redundant computation
4. THE Pipeline SHALL support runtime configuration of operator parameters without rebuilding the graph
5. WHEN processing video streams, THE Pipeline SHALL support batch processing of multiple frames
