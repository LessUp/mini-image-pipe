#include "task_graph.h"
#include <queue>
#include <algorithm>

namespace mini_image_pipe {

int TaskGraph::addTask(const std::string& name, OperatorPtr op) {
    int id = static_cast<int>(nodes_.size());
    nodes_.emplace_back(id, name, op);
    executionCounts_.push_back(0);
    return id;
}

bool TaskGraph::addDependency(int from, int to) {
    // Validate task IDs
    if (from < 0 || from >= static_cast<int>(nodes_.size()) ||
        to < 0 || to >= static_cast<int>(nodes_.size())) {
        return false;
    }

    // Self-dependency is not allowed
    if (from == to) {
        return false;
    }

    // Check if dependency already exists
    auto& deps = nodes_[to].dependencies;
    if (std::find(deps.begin(), deps.end(), from) != deps.end()) {
        return true;  // Already exists, not an error
    }

    // Temporarily add the edge
    nodes_[to].dependencies.push_back(from);
    nodes_[from].dependents.push_back(to);

    // Check for cycles
    if (hasCycle()) {
        // Remove the edge
        nodes_[to].dependencies.pop_back();
        nodes_[from].dependents.pop_back();
        return false;
    }

    return true;
}

bool TaskGraph::validate() const {
    return !hasCycle();
}

bool TaskGraph::hasCycle() const {
    std::vector<bool> visited(nodes_.size(), false);
    std::vector<bool> recStack(nodes_.size(), false);

    for (size_t i = 0; i < nodes_.size(); i++) {
        if (!visited[i]) {
            if (hasCycleUtil(static_cast<int>(i), visited, recStack)) {
                return true;
            }
        }
    }
    return false;
}

bool TaskGraph::hasCycleUtil(int node, std::vector<bool>& visited,
                             std::vector<bool>& recStack) const {
    visited[node] = true;
    recStack[node] = true;

    for (int dependent : nodes_[node].dependents) {
        if (!visited[dependent]) {
            if (hasCycleUtil(dependent, visited, recStack)) {
                return true;
            }
        } else if (recStack[dependent]) {
            return true;
        }
    }

    recStack[node] = false;
    return false;
}

std::vector<int> TaskGraph::getTopologicalOrder() const {
    std::vector<int> order;
    std::vector<int> inDegree(nodes_.size(), 0);

    // Calculate in-degree for each node
    for (const auto& node : nodes_) {
        inDegree[node.id] = static_cast<int>(node.dependencies.size());
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (size_t i = 0; i < nodes_.size(); i++) {
        if (inDegree[i] == 0) {
            queue.push(static_cast<int>(i));
        }
    }

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        order.push_back(current);

        for (int dependent : nodes_[current].dependents) {
            inDegree[dependent]--;
            if (inDegree[dependent] == 0) {
                queue.push(dependent);
            }
        }
    }

    // If order doesn't contain all nodes, there's a cycle
    if (order.size() != nodes_.size()) {
        return {};  // Return empty vector to indicate error
    }

    return order;
}

std::vector<int> TaskGraph::getReadyTasks() const {
    std::vector<int> ready;

    for (const auto& node : nodes_) {
        if (node.state.load() != TaskState::PENDING) {
            continue;
        }

        bool allDepsCompleted = true;
        for (int depId : node.dependencies) {
            if (nodes_[depId].state.load() != TaskState::COMPLETED) {
                allDepsCompleted = false;
                break;
            }
        }

        if (allDepsCompleted) {
            ready.push_back(node.id);
        }
    }

    return ready;
}

TaskNode* TaskGraph::getTask(int id) {
    if (id < 0 || id >= static_cast<int>(nodes_.size())) {
        return nullptr;
    }
    return &nodes_[id];
}

const TaskNode* TaskGraph::getTask(int id) const {
    if (id < 0 || id >= static_cast<int>(nodes_.size())) {
        return nullptr;
    }
    return &nodes_[id];
}

void TaskGraph::reset() {
    for (auto& node : nodes_) {
        node.state.store(TaskState::PENDING);
    }
    resetExecutionCounts();
}

bool TaskGraph::areIndependent(int taskA, int taskB) const {
    if (taskA < 0 || taskA >= static_cast<int>(nodes_.size()) ||
        taskB < 0 || taskB >= static_cast<int>(nodes_.size())) {
        return false;
    }

    // Two tasks are independent if there's no path between them in either direction
    return !hasPath(taskA, taskB) && !hasPath(taskB, taskA);
}

bool TaskGraph::hasPath(int src, int dst) const {
    if (src == dst) return true;

    std::vector<bool> visited(nodes_.size(), false);
    std::queue<int> queue;
    queue.push(src);
    visited[src] = true;

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        for (int dependent : nodes_[current].dependents) {
            if (dependent == dst) {
                return true;
            }
            if (!visited[dependent]) {
                visited[dependent] = true;
                queue.push(dependent);
            }
        }
    }

    return false;
}

int TaskGraph::getExecutionCount(int taskId) const {
    if (taskId < 0 || taskId >= static_cast<int>(executionCounts_.size())) {
        return 0;
    }
    return executionCounts_[taskId];
}

void TaskGraph::incrementExecutionCount(int taskId) {
    if (taskId >= 0 && taskId < static_cast<int>(executionCounts_.size())) {
        executionCounts_[taskId]++;
    }
}

void TaskGraph::resetExecutionCounts() {
    std::fill(executionCounts_.begin(), executionCounts_.end(), 0);
}

} // namespace mini_image_pipe
