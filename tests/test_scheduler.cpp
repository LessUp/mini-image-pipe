#include <gtest/gtest.h>
#include "scheduler.h"
#include "task_graph.h"
#include "operators/color_convert.h"
#include "memory_manager.h"
#include <random>
#include <set>
#include <vector>

using namespace mini_image_pipe;

// Mock operator for testing
class MockOperator : public IOperator {
public:
    MockOperator(bool shouldFail = false) : shouldFail_(shouldFail) {}
    
    cudaError_t execute(
        const void* input,
        void* output,
        int width,
        int height,
        int channels,
        cudaStream_t stream
    ) override {
        if (shouldFail_) {
            return cudaErrorInvalidValue;
        }
        // Simple copy operation
        size_t size = width * height * channels;
        return cudaMemcpyAsync(output, input, size, cudaMemcpyDeviceToDevice, stream);
    }
    
    void getOutputDimensions(
        int inputWidth, int inputHeight, int inputChannels,
        int& outputWidth, int& outputHeight, int& outputChannels
    ) const override {
        outputWidth = inputWidth;
        outputHeight = inputHeight;
        outputChannels = inputChannels;
    }
    
    const char* getName() const override { return "Mock"; }
    
private:
    bool shouldFail_;
};

// Feature: mini-image-pipe, Property 13: Error Propagation
// Validates: Requirements 5.5
TEST(SchedulerPropertyTest, ErrorPropagation) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    for (int iter = 0; iter < 100; iter++) {
        TaskGraph graph;
        DAGScheduler scheduler(4);
        
        // Create a chain: A -> B -> C -> D
        // Make B fail, verify C and D are marked as failed
        auto opA = std::make_shared<MockOperator>(false);
        auto opB = std::make_shared<MockOperator>(true);  // This one fails
        auto opC = std::make_shared<MockOperator>(false);
        auto opD = std::make_shared<MockOperator>(false);
        
        int a = graph.addTask("A", opA);
        int b = graph.addTask("B", opB);
        int c = graph.addTask("C", opC);
        int d = graph.addTask("D", opD);
        
        graph.addDependency(a, b);
        graph.addDependency(b, c);
        graph.addDependency(c, d);
        
        // Allocate buffers
        size_t bufferSize = 64 * 64;
        void* buffer = mgr.allocateDevice(bufferSize);
        
        // Set up task buffers
        TaskNode* taskA = graph.getTask(a);
        TaskNode* taskB = graph.getTask(b);
        TaskNode* taskC = graph.getTask(c);
        TaskNode* taskD = graph.getTask(d);
        
        taskA->inputBuffer = buffer;
        taskA->outputBuffer = buffer;
        taskA->width = 64;
        taskA->height = 64;
        taskA->channels = 1;
        
        taskB->inputBuffer = buffer;
        taskB->outputBuffer = buffer;
        taskB->width = 64;
        taskB->height = 64;
        taskB->channels = 1;
        
        taskC->inputBuffer = buffer;
        taskC->outputBuffer = buffer;
        taskC->width = 64;
        taskC->height = 64;
        taskC->channels = 1;
        
        taskD->inputBuffer = buffer;
        taskD->outputBuffer = buffer;
        taskD->width = 64;
        taskD->height = 64;
        taskD->channels = 1;
        
        bool errorCallbackCalled = false;
        scheduler.setErrorCallback([&](int taskId, cudaError_t err) {
            errorCallbackCalled = true;
            EXPECT_EQ(taskId, b);
        });
        
        // Execute
        cudaError_t err = scheduler.execute(graph);
        
        // Verify error propagation
        EXPECT_NE(err, cudaSuccess);
        EXPECT_TRUE(errorCallbackCalled);
        
        // A should complete, B should fail, C and D should be failed due to propagation
        EXPECT_EQ(taskA->state.load(), TaskState::COMPLETED);
        EXPECT_EQ(taskB->state.load(), TaskState::FAILED);
        EXPECT_EQ(taskC->state.load(), TaskState::FAILED);
        EXPECT_EQ(taskD->state.load(), TaskState::FAILED);
        
        mgr.freeDevice(buffer);
    }
}

// Feature: mini-image-pipe, Property 14: Stream Assignment and Synchronization
// Validates: Requirements 6.1, 6.2
TEST(SchedulerPropertyTest, StreamAssignmentAndSynchronization) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    for (int iter = 0; iter < 100; iter++) {
        TaskGraph graph;
        DAGScheduler scheduler(4);
        
        // Create independent tasks that should be assigned to different streams
        // A and B are independent
        // C depends on both A and B
        auto opA = std::make_shared<MockOperator>();
        auto opB = std::make_shared<MockOperator>();
        auto opC = std::make_shared<MockOperator>();
        
        int a = graph.addTask("A", opA);
        int b = graph.addTask("B", opB);
        int c = graph.addTask("C", opC);
        
        graph.addDependency(a, c);
        graph.addDependency(b, c);
        
        // Verify A and B are independent
        EXPECT_TRUE(graph.areIndependent(a, b));
        
        // Allocate buffers
        size_t bufferSize = 64 * 64;
        void* bufferA = mgr.allocateDevice(bufferSize);
        void* bufferB = mgr.allocateDevice(bufferSize);
        void* bufferC = mgr.allocateDevice(bufferSize);
        
        // Set up task buffers
        TaskNode* taskA = graph.getTask(a);
        TaskNode* taskB = graph.getTask(b);
        TaskNode* taskC = graph.getTask(c);
        
        taskA->inputBuffer = bufferA;
        taskA->outputBuffer = bufferA;
        taskA->width = 64;
        taskA->height = 64;
        taskA->channels = 1;
        
        taskB->inputBuffer = bufferB;
        taskB->outputBuffer = bufferB;
        taskB->width = 64;
        taskB->height = 64;
        taskB->channels = 1;
        
        taskC->inputBuffer = bufferC;
        taskC->outputBuffer = bufferC;
        taskC->width = 64;
        taskC->height = 64;
        taskC->channels = 1;
        
        // Execute
        cudaError_t err = scheduler.execute(graph);
        EXPECT_EQ(err, cudaSuccess);
        
        // Verify stream assignment
        int streamA = scheduler.getTaskStream(a);
        int streamB = scheduler.getTaskStream(b);
        int streamC = scheduler.getTaskStream(c);
        
        // A and B should potentially be on different streams (for parallelism)
        // This is not strictly required but is the expected behavior
        EXPECT_GE(streamA, 0);
        EXPECT_GE(streamB, 0);
        EXPECT_GE(streamC, 0);
        
        // If A and B are on different streams than C, synchronization should exist
        if (streamA != streamC) {
            EXPECT_TRUE(scheduler.hasSynchronization(a, c));
        }
        if (streamB != streamC) {
            EXPECT_TRUE(scheduler.hasSynchronization(b, c));
        }
        
        mgr.freeDevice(bufferA);
        mgr.freeDevice(bufferB);
        mgr.freeDevice(bufferC);
    }
}

// Feature: mini-image-pipe, Property 15: Stream Synchronization on Completion
// Validates: Requirements 6.5
TEST(SchedulerPropertyTest, StreamSynchronizationOnCompletion) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    MemoryManager& mgr = MemoryManager::getInstance();
    
    for (int iter = 0; iter < 100; iter++) {
        TaskGraph graph;
        DAGScheduler scheduler(4);
        
        // Create multiple independent tasks
        std::vector<int> taskIds;
        std::vector<void*> buffers;
        
        int numTasks = 4;
        for (int i = 0; i < numTasks; i++) {
            auto op = std::make_shared<MockOperator>();
            int id = graph.addTask("Task" + std::to_string(i), op);
            taskIds.push_back(id);
            
            void* buffer = mgr.allocateDevice(64 * 64);
            buffers.push_back(buffer);
            
            TaskNode* task = graph.getTask(id);
            task->inputBuffer = buffer;
            task->outputBuffer = buffer;
            task->width = 64;
            task->height = 64;
            task->channels = 1;
        }
        
        // Execute
        cudaError_t err = scheduler.execute(graph);
        EXPECT_EQ(err, cudaSuccess);
        
        // After execute returns, all tasks should be completed
        for (int id : taskIds) {
            TaskNode* task = graph.getTask(id);
            EXPECT_EQ(task->state.load(), TaskState::COMPLETED)
                << "Task " << id << " not completed after execute()";
        }
        
        // Cleanup
        for (void* buffer : buffers) {
            mgr.freeDevice(buffer);
        }
    }
}
