#pragma once

#include "operator.h"
#include "types.h"
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <unordered_set>

namespace mini_image_pipe {

// Represents a single task in the DAG
struct TaskNode {
    int id = -1;
    std::string name;
    OperatorPtr op;
    std::vector<int> dependencies;  // IDs of upstream tasks
    std::vector<int> dependents;    // IDs of downstream tasks
    
    std::atomic<TaskState> state{TaskState::PENDING};
    
    void* inputBuffer = nullptr;
    void* outputBuffer = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    int outputChannels = 0;
    
    int assignedStream = -1;  // CUDA stream index
    
    TaskNode() = default;
    TaskNode(int id, const std::string& name, OperatorPtr op)
        : id(id), name(name), op(op) {}
    
    // Copy constructor for atomic member
    TaskNode(const TaskNode& other)
        : id(other.id)
        , name(other.name)
        , op(other.op)
        , dependencies(other.dependencies)
        , dependents(other.dependents)
        , state(other.state.load())
        , inputBuffer(other.inputBuffer)
        , outputBuffer(other.outputBuffer)
        , width(other.width)
        , height(other.height)
        , channels(other.channels)
        , outputWidth(other.outputWidth)
        , outputHeight(other.outputHeight)
        , outputChannels(other.outputChannels)
        , assignedStream(other.assignedStream) {}
};

class TaskGraph {
public:
    TaskGraph() = default;
    ~TaskGraph() = default;

    // Add a task node, returns task ID
    int addTask(const std::string& name, OperatorPtr op);

    // Add dependency: 'from' must complete before 'to' starts
    bool addDependency(int from, int to);

    // Validate graph has no cycles
    bool validate() const;

    // Get topologically sorted execution order
    std::vector<int> getTopologicalOrder() const;

    // Get tasks with no pending dependencies
    std::vector<int> getReadyTasks() const;

    // Get task by ID
    TaskNode* getTask(int id);
    const TaskNode* getTask(int id) const;

    // Get all tasks
    std::vector<TaskNode>& getTasks() { return nodes_; }
    const std::vector<TaskNode>& getTasks() const { return nodes_; }

    // Get number of tasks
    size_t size() const { return nodes_.size(); }

    // Reset all task states to PENDING
    void reset();

    // Check if two tasks are independent (no path between them)
    bool areIndependent(int taskA, int taskB) const;

    // Get execution count for a task (for testing redundant computation)
    int getExecutionCount(int taskId) const;
    void incrementExecutionCount(int taskId);
    void resetExecutionCounts();

private:
    std::vector<TaskNode> nodes_;
    std::vector<int> executionCounts_;

    // DFS-based cycle detection
    bool hasCycle() const;
    bool hasCycleUtil(int node, std::vector<bool>& visited, 
                      std::vector<bool>& recStack) const;
    
    // Check if there's a path from src to dst
    bool hasPath(int src, int dst) const;
};

} // namespace mini_image_pipe
