#include <gtest/gtest.h>
#include "memory_manager.h"
#include <random>
#include <vector>
#include <cstring>

using namespace mini_image_pipe;

// Feature: mini-image-pipe, Property 17: Memory Pool Reuse
// Validates: Requirements 7.4
TEST(MemoryManagerPropertyTest, MemoryPoolReuse) {
    MemoryManager& mgr = MemoryManager::getInstance();
    
    // Run 100 iterations with random sizes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> sizeDist(1024, 1024 * 1024);
    
    for (int iter = 0; iter < 100; iter++) {
        size_t size = sizeDist(gen);
        
        // Allocate
        void* ptr1 = mgr.allocatePinned(size);
        ASSERT_NE(ptr1, nullptr);
        
        // Free
        mgr.freePinned(ptr1);
        
        // Allocate again - should reuse
        void* ptr2 = mgr.allocatePinned(size);
        ASSERT_NE(ptr2, nullptr);
        
        // Free again
        mgr.freePinned(ptr2);
    }
    
    // Verify reuse happened
    EXPECT_GT(mgr.getPinnedReuseCount(), 0);
}

// Feature: mini-image-pipe, Property 18: Memory Cleanup
// Validates: Requirements 7.5
TEST(MemoryManagerPropertyTest, MemoryCleanup) {
    // Create a fresh instance scenario by tracking allocations
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> sizeDist(1024, 64 * 1024);
    std::uniform_int_distribution<int> countDist(5, 20);
    
    for (int iter = 0; iter < 100; iter++) {
        int allocCount = countDist(gen);
        std::vector<void*> ptrs;
        
        // Allocate multiple buffers
        for (int i = 0; i < allocCount; i++) {
            size_t size = sizeDist(gen);
            void* ptr = mgr.allocatePinned(size);
            if (ptr) {
                ptrs.push_back(ptr);
            }
        }
        
        // Free all
        for (void* ptr : ptrs) {
            mgr.freePinned(ptr);
        }
    }
    
    // Shutdown should free all
    mgr.shutdown();
    
    // After shutdown, active allocations should be 0
    EXPECT_EQ(mgr.getActiveAllocations(), 0);
}

// Feature: mini-image-pipe, Property 16: Pinned Memory Async Transfer
// Validates: Requirements 7.1, 7.2
TEST(MemoryManagerPropertyTest, PinnedMemoryAsyncTransfer) {
    MemoryManager& mgr = MemoryManager::getInstance();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> sizeDist(1024, 256 * 1024);
    std::uniform_int_distribution<uint8_t> dataDist(0, 255);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (int iter = 0; iter < 100; iter++) {
        size_t size = sizeDist(gen);
        
        // Allocate host and device memory
        void* h_src = mgr.allocatePinned(size);
        void* h_dst = mgr.allocatePinned(size);
        void* d_buf = mgr.allocateDevice(size);
        
        ASSERT_NE(h_src, nullptr);
        ASSERT_NE(h_dst, nullptr);
        ASSERT_NE(d_buf, nullptr);
        
        // Fill source with random data
        uint8_t* src = static_cast<uint8_t*>(h_src);
        for (size_t i = 0; i < size; i++) {
            src[i] = dataDist(gen);
        }
        
        // Transfer to device and back
        cudaError_t err = mgr.copyToDeviceAsync(d_buf, h_src, size, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        err = mgr.copyToHostAsync(h_dst, d_buf, size, stream);
        EXPECT_EQ(err, cudaSuccess);
        
        cudaStreamSynchronize(stream);
        
        // Verify data integrity
        uint8_t* dst = static_cast<uint8_t*>(h_dst);
        for (size_t i = 0; i < size; i++) {
            EXPECT_EQ(src[i], dst[i]) << "Mismatch at index " << i;
        }
        
        // Cleanup
        mgr.freePinned(h_src);
        mgr.freePinned(h_dst);
        mgr.freeDevice(d_buf);
    }
    
    cudaStreamDestroy(stream);
}
