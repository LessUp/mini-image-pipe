# Implementation Plan: Mini-ImagePipe

## Overview

This implementation plan builds the Mini-ImagePipe framework incrementally, starting with core infrastructure (memory management, base interfaces), then implementing operators, followed by the scheduler and pipeline integration. Property-based tests are placed close to their implementations to catch errors early.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - [x] 1.1 Create CMake project structure with CUDA support
    - Set up CMakeLists.txt with CUDA language support
    - Configure include directories and library targets
    - _Requirements: Project setup_

  - [x] 1.2 Implement ImageBuffer and KernelConfig data structures
    - Create `include/types.h` with ImageBuffer, KernelConfig, PipelineConfig structs
    - _Requirements: Data Models_

  - [x] 1.3 Implement IOperator base interface
    - Create `include/operator.h` with abstract IOperator class
    - _Requirements: Component Interfaces_

- [x] 2. Implement Memory Manager
  - [x] 2.1 Implement MemoryManager singleton with pinned memory allocation
    - Create `src/memory_manager.cu` with cudaHostAlloc/cudaFree
    - Implement allocatePinned, freePinned, allocateDevice, freeDevice
    - Implement async copy functions with CUDA streams
    - _Requirements: 7.1, 7.2_

  - [x] 2.2 Implement memory pool for pinned memory reuse
    - Add MemoryPool struct with free block tracking
    - Implement block reuse logic in allocate/free
    - _Requirements: 7.4_

  - [x] 2.3 Implement fallback to pageable memory
    - Add fallback logic when cudaHostAlloc fails
    - Log warning on fallback
    - _Requirements: 7.3_

  - [x] 2.4 Implement shutdown and cleanup
    - Free all tracked allocations on shutdown
    - _Requirements: 7.5_

  - [x] 2.5 Write property test for memory pool reuse
    - **Property 17: Memory Pool Reuse**
    - **Validates: Requirements 7.4**

  - [x] 2.6 Write property test for memory cleanup
    - **Property 18: Memory Cleanup**
    - **Validates: Requirements 7.5**

- [x] 3. Checkpoint - Memory Manager
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Color Conversion Operator
  - [x] 4.1 Implement ColorConvertOperator class
    - Create `src/operators/color_convert.cu`
    - Implement RGB_TO_GRAY, BGR_TO_RGB, RGBA_TO_RGB conversions
    - Use luminance formula Y = 0.299*R + 0.587*G + 0.114*B
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 4.2 Implement alpha channel preservation
    - Preserve alpha channel during conversions when present
    - _Requirements: 4.4_

  - [x] 4.3 Write property test for RGB to Grayscale formula
    - **Property 8: RGB to Grayscale Formula**
    - **Validates: Requirements 4.2**

  - [x] 4.4 Write property test for BGR to RGB channel swap
    - **Property 9: BGR to RGB Channel Swap**
    - **Validates: Requirements 4.3**

  - [x] 4.5 Write property test for alpha channel preservation
    - **Property 10: Alpha Channel Preservation**
    - **Validates: Requirements 4.4**

- [x] 5. Implement Resize Operator
  - [x] 5.1 Implement ResizeOperator class with bilinear interpolation
    - Create `src/operators/resize.cu`
    - Implement bilinear interpolation kernel
    - _Requirements: 3.1_

  - [x] 5.2 Implement nearest-neighbor interpolation
    - Add NEAREST mode to resize kernel
    - _Requirements: 3.2_

  - [x] 5.3 Implement coordinate mapping and arbitrary scale factors
    - Compute src coordinates from dst coordinates
    - Support both upscaling and downscaling
    - _Requirements: 3.3, 3.4_

  - [x] 5.4 Write property test for resize coordinate mapping
    - **Property 6: Resize Coordinate Mapping**
    - **Validates: Requirements 3.1, 3.2, 3.3**

  - [x] 5.5 Write property test for arbitrary scale factors
    - **Property 7: Resize Arbitrary Scale Factors**
    - **Validates: Requirements 3.4**

- [x] 6. Implement Sobel Edge Detection Operator
  - [x] 6.1 Implement SobelOperator class
    - Create `src/operators/sobel.cu`
    - Implement 3x3 Sobel kernels for Gx and Gy
    - Use shared memory for efficient access
    - _Requirements: 2.1, 2.3_

  - [x] 6.2 Implement gradient magnitude computation
    - Compute magnitude as sqrt(Gx² + Gy²)
    - Output single-channel result
    - _Requirements: 2.2, 2.4_

  - [x] 6.3 Write property test for Sobel gradient computation
    - **Property 4: Sobel Gradient Computation**
    - **Validates: Requirements 2.1, 2.2**

  - [x] 6.4 Write property test for Sobel single-channel output
    - **Property 5: Sobel Single-Channel Output**
    - **Validates: Requirements 2.4**

- [x] 7. Checkpoint - Basic Operators
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Gaussian Blur Operator
  - [x] 8.1 Implement GaussianBlurOperator class with separable filter
    - Create `src/operators/gaussian_blur.cu`
    - Generate 1D Gaussian kernels for horizontal and vertical passes
    - Support kernel sizes 3x3, 5x5, 7x7
    - _Requirements: 1.1, 1.2_

  - [x] 8.2 Implement shared memory with halo regions
    - Load tile + halo into shared memory
    - Handle boundary with reflection padding
    - _Requirements: 1.3, 1.4_

  - [x] 8.3 Implement multi-channel support
    - Support 1, 3, and 4 channel images
    - _Requirements: 1.5_

  - [x] 8.4 Write property test for Gaussian blur multi-channel support
    - **Property 1: Gaussian Blur Multi-Channel Support**
    - **Validates: Requirements 1.1, 1.5**

  - [x] 8.5 Write property test for separable filter equivalence
    - **Property 2: Separable Filter Equivalence**
    - **Validates: Requirements 1.2**

  - [x] 8.6 Write property test for reflection padding boundary handling
    - **Property 3: Reflection Padding Boundary Handling**
    - **Validates: Requirements 1.4**

- [x] 9. Checkpoint - All Operators
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Task Graph
  - [x] 10.1 Implement TaskNode and TaskGraph classes
    - Create `src/task_graph.cpp`
    - Implement addTask, addDependency methods
    - Track node states (PENDING, READY, RUNNING, COMPLETED, FAILED)
    - _Requirements: 5.1_

  - [x] 10.2 Implement cycle detection
    - Use DFS-based cycle detection in addDependency
    - Reject edges that would create cycles
    - _Requirements: 5.1_

  - [x] 10.3 Implement topological sorting
    - Implement getTopologicalOrder using Kahn's algorithm
    - Implement getReadyTasks for scheduler
    - _Requirements: 5.6_

  - [x] 10.4 Write property test for DAG cycle detection
    - **Property 11: DAG Cycle Detection**
    - **Validates: Requirements 5.1**

- [x] 11. Implement DAG Scheduler
  - [x] 11.1 Implement DAGScheduler class with CUDA streams
    - Create `src/scheduler.cu`
    - Create configurable number of CUDA streams
    - Implement stream assignment for tasks
    - _Requirements: 6.1, 6.4_

  - [x] 11.2 Implement dependency-based execution
    - Execute tasks in topological order
    - Respect all dependency constraints
    - Trigger dependents when task completes
    - _Requirements: 5.2, 5.4_

  - [x] 11.3 Implement CUDA event synchronization
    - Insert events for cross-stream dependencies
    - Synchronize all streams on completion
    - _Requirements: 6.2, 6.5_

  - [x] 11.4 Implement error propagation
    - Mark failed tasks and halt dependents
    - Invoke error callback on failure
    - _Requirements: 5.5_

  - [x] 11.5 Write property test for dependency ordering
    - **Property 12: Dependency Ordering**
    - **Validates: Requirements 5.2, 5.4, 5.6**

  - [x] 11.6 Write property test for error propagation
    - **Property 13: Error Propagation**
    - **Validates: Requirements 5.5**

  - [x] 11.7 Write property test for stream assignment and synchronization
    - **Property 14: Stream Assignment and Synchronization**
    - **Validates: Requirements 6.1, 6.2**

  - [x] 11.8 Write property test for stream synchronization on completion
    - **Property 15: Stream Synchronization on Completion**
    - **Validates: Requirements 6.5**

- [x] 12. Checkpoint - Scheduler
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement Pipeline Builder
  - [x] 13.1 Implement Pipeline class
    - Create `src/pipeline.cpp`
    - Implement addOperator, connect methods
    - Wire to TaskGraph and DAGScheduler
    - _Requirements: 8.1_

  - [x] 13.2 Implement automatic intermediate buffer allocation
    - Allocate buffers based on operator output dimensions
    - Manage buffer lifecycle
    - _Requirements: 8.2_

  - [x] 13.3 Implement shared output for multiple dependents
    - Ensure single execution for nodes with multiple dependents
    - Share output buffer reference
    - _Requirements: 8.3_

  - [x] 13.4 Implement runtime parameter configuration
    - Add setParameter method for runtime updates
    - Apply without graph reconstruction
    - _Requirements: 8.4_

  - [x] 13.5 Implement batch processing
    - Implement executeBatch for multiple frames
    - _Requirements: 8.5_

  - [x] 13.6 Write property test for pipeline topology and buffer management
    - **Property 19: Pipeline Topology and Buffer Management**
    - **Validates: Requirements 8.1, 8.2**

  - [x] 13.7 Write property test for no redundant computation
    - **Property 20: No Redundant Computation**
    - **Validates: Requirements 8.3**

  - [x] 13.8 Write property test for runtime parameter configuration
    - **Property 21: Runtime Parameter Configuration**
    - **Validates: Requirements 8.4**

  - [x] 13.9 Write property test for batch processing
    - **Property 22: Batch Processing**
    - **Validates: Requirements 8.5**

- [x] 14. Final Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Integration and wiring
  - [x] 15.1 Create example pipeline demonstrating all operators
    - Create `examples/demo_pipeline.cpp`
    - Chain Resize → ColorConvert → GaussianBlur → Sobel
    - _Requirements: 8.1_

  - [x] 15.2 Write integration tests for end-to-end pipeline
    - Test complete pipeline execution
    - Verify output correctness
    - _Requirements: All_

## Notes

- All tasks including property tests are required for comprehensive coverage
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- CUDA code files use `.cu` extension, C++ files use `.cpp`
