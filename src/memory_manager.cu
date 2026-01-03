#include "memory_manager.h"
#include <iostream>
#include <algorithm>
#include <cstring>

namespace mini_image_pipe {

MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager() : usePinnedMemory_(true) {}

MemoryManager::~MemoryManager() {
    shutdown();
}

void* MemoryManager::allocatePinned(size_t size) {
    if (size == 0) return nullptr;
    return findOrAllocatePinned(size);
}

void* MemoryManager::findOrAllocatePinned(size_t size) {
    std::lock_guard<std::mutex> poolLock(pinnedPool_.mutex);
    std::lock_guard<std::mutex> allocLock(allocMutex_);

    // Try to find a suitable block in the pool (best fit)
    auto bestIt = pinnedPool_.freeBlocks.end();
    size_t bestSize = SIZE_MAX;
    
    for (auto it = pinnedPool_.freeBlocks.begin(); it != pinnedPool_.freeBlocks.end(); ++it) {
        if (it->size >= size && it->size < bestSize) {
            bestIt = it;
            bestSize = it->size;
        }
    }
    
    if (bestIt != pinnedPool_.freeBlocks.end()) {
        void* ptr = bestIt->ptr;
        size_t blockSize = bestIt->size;
        bool isPinned = bestIt->isPinned;
        pinnedPool_.freeBlocks.erase(bestIt);
        activePinnedAllocs_.insert(ptr);
        pinnedSizes_[ptr] = blockSize;
        pinnedFlags_[ptr] = isPinned;
        pinnedReuseCount_++;
        return ptr;
    }

    // Allocate new pinned memory
    void* ptr = nullptr;
    cudaError_t err = cudaSuccess;
    bool isPinned = false;

    if (usePinnedMemory_) {
        err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
        if (err == cudaSuccess) {
            isPinned = true;
        } else {
            // Fallback to pageable memory
            std::cerr << "Warning: cudaHostAlloc failed, falling back to pageable memory" << std::endl;
            ptr = malloc(size);
            isPinned = false;
            usePinnedMemory_ = false;
        }
    } else {
        ptr = malloc(size);
        isPinned = false;
    }

    if (ptr) {
        activePinnedAllocs_.insert(ptr);
        pinnedSizes_[ptr] = size;
        pinnedFlags_[ptr] = isPinned;
        pinnedAllocCount_++;
    }

    return ptr;
}

void MemoryManager::freePinned(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> poolLock(pinnedPool_.mutex);
    std::lock_guard<std::mutex> allocLock(allocMutex_);

    auto it = activePinnedAllocs_.find(ptr);
    if (it == activePinnedAllocs_.end()) {
        std::cerr << "Warning: Attempting to free unknown pinned memory" << std::endl;
        return;
    }

    activePinnedAllocs_.erase(it);

    // Get stored size and pinned flag
    size_t blockSize = pinnedSizes_[ptr];
    bool isPinned = pinnedFlags_[ptr];
    
    // Add to pool for reuse
    MemoryBlock block{ptr, blockSize, isPinned};
    pinnedPool_.freeBlocks.push_back(block);
}

void* MemoryManager::allocateDevice(size_t size) {
    if (size == 0) return nullptr;
    return findOrAllocateDevice(size);
}

void* MemoryManager::findOrAllocateDevice(size_t size) {
    std::lock_guard<std::mutex> poolLock(devicePool_.mutex);
    std::lock_guard<std::mutex> allocLock(allocMutex_);

    // Try to find a suitable block in the pool
    for (auto it = devicePool_.freeBlocks.begin(); it != devicePool_.freeBlocks.end(); ++it) {
        if (it->size >= size) {
            void* ptr = it->ptr;
            devicePool_.freeBlocks.erase(it);
            activeDeviceAllocs_.insert(ptr);
            return ptr;
        }
    }

    // Allocate new device memory
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    activeDeviceAllocs_.insert(ptr);
    return ptr;
}

void MemoryManager::freeDevice(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> poolLock(devicePool_.mutex);
    std::lock_guard<std::mutex> allocLock(allocMutex_);

    auto it = activeDeviceAllocs_.find(ptr);
    if (it == activeDeviceAllocs_.end()) {
        std::cerr << "Warning: Attempting to free unknown device memory" << std::endl;
        return;
    }

    activeDeviceAllocs_.erase(it);

    // Add to pool for reuse
    MemoryBlock block{ptr, 0, false};
    devicePool_.freeBlocks.push_back(block);
}

cudaError_t MemoryManager::copyToDeviceAsync(
    void* dst, const void* src, size_t size, cudaStream_t stream
) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

cudaError_t MemoryManager::copyToHostAsync(
    void* dst, const void* src, size_t size, cudaStream_t stream
) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

void MemoryManager::shutdown() {
    std::lock_guard<std::mutex> pinnedLock(pinnedPool_.mutex);
    std::lock_guard<std::mutex> deviceLock(devicePool_.mutex);
    std::lock_guard<std::mutex> allocLock(allocMutex_);

    // Free all pinned memory in pool
    for (auto& block : pinnedPool_.freeBlocks) {
        if (block.isPinned) {
            cudaFreeHost(block.ptr);
        } else {
            free(block.ptr);
        }
    }
    pinnedPool_.freeBlocks.clear();

    // Free all active pinned allocations
    for (void* ptr : activePinnedAllocs_) {
        auto flagIt = pinnedFlags_.find(ptr);
        if (flagIt != pinnedFlags_.end() && flagIt->second) {
            cudaFreeHost(ptr);
        } else {
            free(ptr);
        }
    }
    activePinnedAllocs_.clear();
    pinnedSizes_.clear();
    pinnedFlags_.clear();

    // Free all device memory in pool
    for (auto& block : devicePool_.freeBlocks) {
        cudaFree(block.ptr);
    }
    devicePool_.freeBlocks.clear();

    // Free all active device allocations
    for (void* ptr : activeDeviceAllocs_) {
        cudaFree(ptr);
    }
    activeDeviceAllocs_.clear();

    pinnedAllocCount_ = 0;
    pinnedReuseCount_ = 0;
    usePinnedMemory_ = true;
}

size_t MemoryManager::getActiveAllocations() const {
    return activePinnedAllocs_.size() + activeDeviceAllocs_.size();
}

} // namespace mini_image_pipe
