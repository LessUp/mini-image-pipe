#pragma once

#include "task_graph.h"
#include "scheduler.h"
#include "memory_manager.h"
#include "operators/resize.h"
#include "operators/gaussian_blur.h"
#include "types.h"
#include <unordered_map>
#include <any>

namespace mini_image_pipe {

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config = PipelineConfig());
    ~Pipeline();

    // Add operator to pipeline, returns node ID
    int addOperator(const std::string& name, OperatorPtr op);

    // Connect operators: output of 'from' feeds into 'to'
    bool connect(int from, int to);

    // Set input source for a node
    void setInput(int nodeId, void* data, int width, int height, int channels);

    // Get output from a node
    void* getOutput(int nodeId);

    // Execute pipeline
    cudaError_t execute();

    // Execute batch of frames
    cudaError_t executeBatch(
        const std::vector<void*>& inputs,
        std::vector<void*>& outputs,
        int width, int height, int channels
    );

    // Update operator parameters at runtime
    template<typename T>
    void setParameter(int nodeId, const std::string& param, T value);

    // Get task graph (for testing)
    TaskGraph& getTaskGraph() { return graph_; }
    const TaskGraph& getTaskGraph() const { return graph_; }

    // Get scheduler (for testing)
    DAGScheduler& getScheduler() { return scheduler_; }

    // Reset pipeline state
    void reset();

private:
    PipelineConfig config_;
    TaskGraph graph_;
    DAGScheduler scheduler_;
    MemoryManager& memMgr_;

    std::unordered_map<int, void*> intermediateBuffers_;
    std::unordered_map<int, size_t> bufferSizes_;
    std::vector<int> inputNodes_;  // Nodes with external input
    std::vector<int> outputNodes_; // Nodes with no dependents

    // Parameter storage
    std::unordered_map<int, std::unordered_map<std::string, std::any>> parameters_;

    void allocateIntermediateBuffers();
    void freeIntermediateBuffers();
    void setupBufferConnections();
    void findInputOutputNodes();
};

// Template implementation
template<typename T>
void Pipeline::setParameter(int nodeId, const std::string& param, T value) {
    parameters_[nodeId][param] = value;

    // Apply parameter to operator if it supports it
    TaskNode* task = graph_.getTask(nodeId);
    if (!task || !task->op) return;

    // Handle specific operator types dynamically
    // ResizeOperator parameters
    if (param == "targetWidth" || param == "targetHeight") {
        auto* resizeOp = dynamic_cast<ResizeOperator*>(task->op.get());
        if (resizeOp) {
            if (param == "targetWidth") {
                resizeOp->setTargetSize(static_cast<int>(value), resizeOp->getTargetHeight());
            } else {
                resizeOp->setTargetSize(resizeOp->getTargetWidth(), static_cast<int>(value));
            }
        }
    }
    
    // GaussianBlurOperator parameters
    if (param == "sigma") {
        auto* blurOp = dynamic_cast<GaussianBlurOperator*>(task->op.get());
        if (blurOp) {
            blurOp->setSigma(static_cast<float>(value));
        }
    }
}

} // namespace mini_image_pipe
