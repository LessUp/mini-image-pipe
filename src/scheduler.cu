#include "scheduler.h"
#include <iostream>
#include <algorithm>

namespace mini_image_pipe {

DAGScheduler::DAGScheduler(int numStreams)
    : numStreams_(numStreams) {
    // Create CUDA streams
    streams_.resize(numStreams_);
    for (int i = 0; i < numStreams_; i++) {
        cudaStreamCreate(&streams_[i]);
    }
}

DAGScheduler::~DAGScheduler() {
    // Destroy CUDA streams
    for (auto& stream : streams_) {
        cudaStreamDestroy(stream);
    }

    // Destroy events
    for (auto& event : taskEvents_) {
        cudaEventDestroy(event);
    }
}

void DAGScheduler::setErrorCallback(std::function<void(int taskId, cudaError_t)> cb) {
    errorCallback_ = cb;
}

int DAGScheduler::assignStream(TaskNode& task, const TaskGraph& graph) {
    // If task has no dependencies, assign to stream based on task ID
    if (task.dependencies.empty()) {
        return task.id % numStreams_;
    }

    // Try to find a stream different from dependencies for parallelism
    std::vector<int> depStreams;
    for (int depId : task.dependencies) {
        auto it = taskStreamMap_.find(depId);
        if (it != taskStreamMap_.end()) {
            depStreams.push_back(it->second);
        }
    }

    // Find a stream not used by dependencies
    for (int s = 0; s < numStreams_; s++) {
        if (std::find(depStreams.begin(), depStreams.end(), s) == depStreams.end()) {
            return s;
        }
    }

    // All streams used by dependencies, use the first dependency's stream
    return depStreams.empty() ? 0 : depStreams[0];
}

void DAGScheduler::insertSynchronization(int fromTask, int toTask, TaskGraph& graph) {
    auto fromIt = taskStreamMap_.find(fromTask);
    auto toIt = taskStreamMap_.find(toTask);

    if (fromIt == taskStreamMap_.end() || toIt == taskStreamMap_.end()) {
        return;
    }

    int fromStream = fromIt->second;
    int toStream = toIt->second;

    // Only need synchronization if tasks are on different streams
    if (fromStream != toStream) {
        // Record event on source stream
        if (fromTask < static_cast<int>(taskEvents_.size())) {
            cudaStreamWaitEvent(streams_[toStream], taskEvents_[fromTask], 0);
            synchronizations_.push_back({fromTask, toTask});
        }
    }
}

cudaError_t DAGScheduler::executeTask(TaskNode& task, cudaStream_t stream) {
    if (!task.op) {
        return cudaErrorInvalidValue;
    }

    return task.op->execute(
        task.inputBuffer,
        task.outputBuffer,
        task.width,
        task.height,
        task.channels,
        stream
    );
}

void DAGScheduler::propagateFailure(int taskId, TaskGraph& graph) {
    TaskNode* task = graph.getTask(taskId);
    if (!task) return;

    for (int depId : task->dependents) {
        TaskNode* dep = graph.getTask(depId);
        if (dep && dep->state.load() == TaskState::PENDING) {
            dep->state.store(TaskState::FAILED);
            propagateFailure(depId, graph);
        }
    }
}

cudaError_t DAGScheduler::execute(TaskGraph& graph) {
    // Clear previous state
    taskStreamMap_.clear();
    synchronizations_.clear();

    // Create events for each task
    size_t numTasks = graph.size();
    while (taskEvents_.size() < numTasks) {
        cudaEvent_t event;
        cudaEventCreate(&event);
        taskEvents_.push_back(event);
    }

    // Get topological order
    std::vector<int> order = graph.getTopologicalOrder();
    if (order.empty() && numTasks > 0) {
        return cudaErrorInvalidValue;  // Cycle detected
    }

    cudaError_t lastError = cudaSuccess;

    // Execute tasks in topological order
    for (int taskId : order) {
        TaskNode* task = graph.getTask(taskId);
        if (!task) continue;

        // Check if any dependency failed
        bool depFailed = false;
        for (int depId : task->dependencies) {
            TaskNode* dep = graph.getTask(depId);
            if (dep && dep->state.load() == TaskState::FAILED) {
                depFailed = true;
                break;
            }
        }

        if (depFailed) {
            task->state.store(TaskState::FAILED);
            continue;
        }

        // Assign stream
        int streamIdx = assignStream(*task, graph);
        taskStreamMap_[taskId] = streamIdx;
        task->assignedStream = streamIdx;

        // Insert synchronization for dependencies on different streams
        for (int depId : task->dependencies) {
            insertSynchronization(depId, taskId, graph);
        }

        // Execute task
        task->state.store(TaskState::RUNNING);
        graph.incrementExecutionCount(taskId);

        cudaError_t err = executeTask(*task, streams_[streamIdx]);

        // Record event after task completion
        cudaEventRecord(taskEvents_[taskId], streams_[streamIdx]);

        if (err != cudaSuccess) {
            task->state.store(TaskState::FAILED);
            lastError = err;

            if (errorCallback_) {
                errorCallback_(taskId, err);
            }

            // Propagate failure to dependents
            propagateFailure(taskId, graph);
        } else {
            task->state.store(TaskState::COMPLETED);
        }
    }

    // Synchronize all streams
    for (auto& stream : streams_) {
        cudaStreamSynchronize(stream);
    }

    return lastError;
}

int DAGScheduler::getTaskStream(int taskId) const {
    auto it = taskStreamMap_.find(taskId);
    return (it != taskStreamMap_.end()) ? it->second : -1;
}

bool DAGScheduler::hasSynchronization(int fromTask, int toTask) const {
    for (const auto& sync : synchronizations_) {
        if (sync.first == fromTask && sync.second == toTask) {
            return true;
        }
    }
    return false;
}

} // namespace mini_image_pipe
