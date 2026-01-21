#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <cstddef>

namespace mini_image_pipe {

class MemoryManager {
public:
    static MemoryManager& getInstance();

    // Disable copy and move
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    // Allocate pinned host memory
    void* allocatePinned(size_t size);

    // Free pinned memory
    void freePinned(void* ptr);

    // Allocate device memory
    void* allocateDevice(size_t size);

    // Free device memory
    void freeDevice(void* ptr);

    // Async copy host to device
    cudaError_t copyToDeviceAsync(
        void* dst, const void* src, size_t size, cudaStream_t stream
    );

    // Async copy device to host
    cudaError_t copyToHostAsync(
        void* dst, const void* src, size_t size, cudaStream_t stream
    );

    // Release all resources
    void shutdown();

    // Check if using pinned memory (for testing)
    bool isUsingPinnedMemory() const { return usePinnedMemory_; }

    // Get allocation statistics (for testing)
    size_t getPinnedAllocCount() const { return pinnedAllocCount_; }
    size_t getPinnedReuseCount() const { return pinnedReuseCount_; }
    size_t getActiveAllocations() const;

private:
    MemoryManager();
    ~MemoryManager();

    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool isPinned;
    };

    struct MemoryPool {
        std::vector<MemoryBlock> freeBlocks;
        std::mutex mutex;
    };

    MemoryPool pinnedPool_;
    MemoryPool devicePool_;
    
    std::unordered_set<void*> activePinnedAllocs_;
    std::unordered_set<void*> activeDeviceAllocs_;
    std::unordered_map<void*, size_t> pinnedSizes_;  // Track sizes for reuse
    std::unordered_map<void*, bool> pinnedFlags_;    // Track if pinned
    std::unordered_map<void*, size_t> deviceSizes_;  // Track sizes for reuse
    std::mutex allocMutex_;
    
    bool usePinnedMemory_ = true;
    size_t pinnedAllocCount_ = 0;
    size_t pinnedReuseCount_ = 0;

    void* findOrAllocatePinned(size_t size);
    void* findOrAllocateDevice(size_t size);
};

} // namespace mini_image_pipe
