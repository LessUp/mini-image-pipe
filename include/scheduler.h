#pragma once

#include "task_graph.h"
#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <unordered_map>

namespace mini_image_pipe {

class DAGScheduler {
public:
    explicit DAGScheduler(int numStreams = 4);
    ~DAGScheduler();

    // Execute the task graph
    cudaError_t execute(TaskGraph& graph);

    // Set error callback
    void setErrorCallback(std::function<void(int taskId, cudaError_t)> cb);

    // Get number of streams
    int getNumStreams() const { return numStreams_; }

    // Get stream for a task (for testing)
    int getTaskStream(int taskId) const;

    // Check if synchronization was used between tasks (for testing)
    bool hasSynchronization(int fromTask, int toTask) const;

private:
    int numStreams_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> taskEvents_;  // One event per task
    std::function<void(int, cudaError_t)> errorCallback_;

    std::unordered_map<int, int> taskStreamMap_;
    std::vector<std::pair<int, int>> synchronizations_;  // (from, to) pairs

    // Assign stream to task based on dependencies
    int assignStream(TaskNode& task, const TaskGraph& graph);

    // Insert synchronization between streams
    void insertSynchronization(int fromTask, int toTask, TaskGraph& graph);

    // Execute a single task
    cudaError_t executeTask(TaskNode& task, cudaStream_t stream);

    // Propagate failure to dependent tasks
    void propagateFailure(int taskId, TaskGraph& graph);
};

} // namespace mini_image_pipe
