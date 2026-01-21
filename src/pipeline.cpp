#include "pipeline.h"
#include <algorithm>
#include <iostream>

namespace mini_image_pipe {

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config)
    , scheduler_(config.numStreams)
    , memMgr_(MemoryManager::getInstance()) {}

Pipeline::~Pipeline() {
    freeIntermediateBuffers();
}

int Pipeline::addOperator(const std::string& name, OperatorPtr op) {
    return graph_.addTask(name, op);
}

bool Pipeline::connect(int from, int to) {
    return graph_.addDependency(from, to);
}

void Pipeline::setInput(int nodeId, void* data, int width, int height, int channels) {
    TaskNode* task = graph_.getTask(nodeId);
    if (!task) return;

    task->inputBuffer = data;
    task->width = width;
    task->height = height;
    task->channels = channels;

    // Calculate output dimensions
    if (task->op) {
        task->op->getOutputDimensions(
            width, height, channels,
            task->outputWidth, task->outputHeight, task->outputChannels
        );
    } else {
        task->outputWidth = width;
        task->outputHeight = height;
        task->outputChannels = channels;
    }

    // Mark as input node
    if (std::find(inputNodes_.begin(), inputNodes_.end(), nodeId) == inputNodes_.end()) {
        inputNodes_.push_back(nodeId);
    }
}

void* Pipeline::getOutput(int nodeId) {
    TaskNode* task = graph_.getTask(nodeId);
    if (!task) return nullptr;
    return task->outputBuffer;
}

void Pipeline::findInputOutputNodes() {
    inputNodes_.clear();
    outputNodes_.clear();

    for (const auto& task : graph_.getTasks()) {
        // Input nodes have no dependencies
        if (task.dependencies.empty()) {
            inputNodes_.push_back(task.id);
        }
        // Output nodes have no dependents
        if (task.dependents.empty()) {
            outputNodes_.push_back(task.id);
        }
    }
}

void Pipeline::allocateIntermediateBuffers() {
    // First, propagate dimensions through the graph
    std::vector<int> order = graph_.getTopologicalOrder();

    for (int taskId : order) {
        TaskNode* task = graph_.getTask(taskId);
        if (!task) continue;

        // If this task has dependencies, get input from first dependency's output
        if (!task->dependencies.empty()) {
            int depId = task->dependencies[0];
            TaskNode* dep = graph_.getTask(depId);
            if (dep) {
                task->inputBuffer = dep->outputBuffer;
                task->width = dep->outputWidth;
                task->height = dep->outputHeight;
                task->channels = dep->outputChannels;
            }
        }

        // Calculate output dimensions
        if (task->op && task->width > 0 && task->height > 0) {
            task->op->getOutputDimensions(
                task->width, task->height, task->channels,
                task->outputWidth, task->outputHeight, task->outputChannels
            );
        }

        // Allocate or resize output buffer if needed
        if (task->outputWidth > 0 && task->outputHeight > 0 && task->outputChannels > 0) {
            size_t bufferSize = static_cast<size_t>(task->outputWidth) *
                               task->outputHeight * task->outputChannels;

            auto it = intermediateBuffers_.find(taskId);
            if (it != intermediateBuffers_.end()) {
                size_t currentSize = 0;
                auto sizeIt = bufferSizes_.find(taskId);
                if (sizeIt != bufferSizes_.end()) {
                    currentSize = sizeIt->second;
                }

                if (currentSize >= bufferSize) {
                    task->outputBuffer = it->second;
                    continue;
                }

                memMgr_.freeDevice(it->second);
                intermediateBuffers_.erase(it);
                bufferSizes_.erase(taskId);
            }

            void* buffer = memMgr_.allocateDevice(bufferSize);
            if (buffer) {
                intermediateBuffers_[taskId] = buffer;
                bufferSizes_[taskId] = bufferSize;
                task->outputBuffer = buffer;
            }
        }
    }
}

void Pipeline::freeIntermediateBuffers() {
    for (auto& pair : intermediateBuffers_) {
        memMgr_.freeDevice(pair.second);
    }
    intermediateBuffers_.clear();
    bufferSizes_.clear();
}

void Pipeline::setupBufferConnections() {
    // Connect output buffers to input buffers for dependent tasks
    for (auto& task : graph_.getTasks()) {
        if (!task.dependencies.empty()) {
            // Use first dependency's output as input
            int depId = task.dependencies[0];
            TaskNode* dep = graph_.getTask(depId);
            if (dep) {
                task.inputBuffer = dep->outputBuffer;
                task.width = dep->outputWidth;
                task.height = dep->outputHeight;
                task.channels = dep->outputChannels;
            }
        }
    }
}

cudaError_t Pipeline::execute() {
    graph_.reset();

    // Validate graph
    if (!graph_.validate()) {
        return cudaErrorInvalidValue;
    }

    // Find input/output nodes
    findInputOutputNodes();

    for (const auto& task : graph_.getTasks()) {
        if (!task.op) {
            return cudaErrorInvalidValue;
        }
    }

    for (int nodeId : inputNodes_) {
        TaskNode* task = graph_.getTask(nodeId);
        if (!task || !task->inputBuffer || task->width <= 0 || task->height <= 0 || task->channels <= 0) {
            return cudaErrorInvalidValue;
        }
    }

    // Allocate intermediate buffers
    allocateIntermediateBuffers();

    // Setup buffer connections
    setupBufferConnections();

    for (const auto& task : graph_.getTasks()) {
        if (!task.op) {
            continue;
        }
        if (!task.inputBuffer || task->width <= 0 || task->height <= 0 || task->channels <= 0) {
            return cudaErrorInvalidValue;
        }
        if (task->outputWidth <= 0 || task->outputHeight <= 0 || task->outputChannels <= 0) {
            return cudaErrorInvalidValue;
        }
        if (!task->outputBuffer) {
            return cudaErrorMemoryAllocation;
        }
    }

    // Execute the graph
    cudaError_t err = scheduler_.execute(graph_);

    return err;
}

cudaError_t Pipeline::executeBatch(
    const std::vector<void*>& inputs,
    std::vector<void*>& outputs,
    int width, int height, int channels
) {
    if (inputs.empty()) {
        return cudaErrorInvalidValue;
    }

    if (config_.maxBatchSize > 0 && inputs.size() > static_cast<size_t>(config_.maxBatchSize)) {
        return cudaErrorInvalidValue;
    }

    findInputOutputNodes();
    if (inputNodes_.empty()) {
        return cudaErrorInvalidValue;
    }

    outputs.assign(inputs.size(), nullptr);
    cudaError_t lastError = cudaSuccess;

    // Process each frame
    for (size_t i = 0; i < inputs.size(); i++) {
        // Reset graph state
        graph_.reset();

        // Set input for all input nodes
        for (int nodeId : inputNodes_) {
            setInput(nodeId, inputs[i], width, height, channels);
        }

        // Execute pipeline
        cudaError_t err = execute();
        if (err != cudaSuccess) {
            lastError = err;
        }

        // Get output from first output node
        if (!outputNodes_.empty()) {
            outputs[i] = getOutput(outputNodes_[0]);
        }
    }

    return lastError;
}

void Pipeline::reset() {
    graph_.reset();
}

} // namespace mini_image_pipe
