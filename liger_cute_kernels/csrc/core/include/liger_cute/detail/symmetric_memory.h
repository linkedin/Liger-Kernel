// symmetric_memory.h — CORE-INTERNAL symmetric/device allocators.
//
// NOT part of the flat ABI: this header pulls in <nvshmem.h>/<cuda_runtime.h>
// and uses STL freely. It is compiled only into liger_cute_kernels.so, where
// libstdc++ is linked statically and all non-liger_cute_* symbols are hidden,
// so the std::map/std::stack/std::string instantiations here stay local and
// never reach the boundary. Frontends must use the flat liger_cute_* control
// entry points or the TVM FFI facade rather than including this header.
//
// Ported from LigerCommKernels' utils/buffer_pool.cuh; the torch-free change is
// TORCH_CHECK/abort -> LIGER_CHECK (throws liger_cute::Error, caught at the
// boundary).
#pragma once

#include <cuda_runtime.h>
#include <nvshmem.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <stack>
#include <string>

#include "liger_cute/check.h"
#include "liger_cute/detail/status.h"

namespace liger_cute {
namespace detail {

// Caches device (cudaMalloc) and symmetric (nvshmem_malloc) allocations by
// name, growing the device buffers and pinning the symmetric ones.
class BufferPool {
 public:
  ~BufferPool() { clear(); }

  // Get or allocate a device buffer (cudaMalloc). Newly-allocated buffers are
  // zero-initialized so callers can rely on a fresh slot starting at 0 —
  // important for counters / signals. Cached reuse does NOT re-zero; the prior
  // kernel's writes persist (intentional — e.g. monotonic counters).
  void* get_device(const std::string& name, size_t bytes) {
    auto it = device_bufs_.find(name);
    if (it != device_bufs_.end() && it->second.bytes >= bytes) {
      return it->second.ptr;
    }
    // Free old if it exists. cudaFree's implicit sync only covers CUDA's
    // dependency tracking — NVSHMEM proxy / IBGDA ops that still hold the
    // buffer aren't visible to CUDA, so drain the device before freeing.
    // Otherwise a prior kernel's in-flight remote puts/gets can land in the
    // newly-allocated region (or fault when the old region is unmapped).
    if (it != device_bufs_.end()) {
      if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
        LIGER_FAIL_CUDA("BufferPool: sync before growing '", name,
                        "' failed: ", cudaGetErrorString(e));
      }
      if (cudaError_t e = cudaFree(it->second.ptr); e != cudaSuccess) {
        LIGER_FAIL_CUDA("BufferPool: cudaFree before growing '", name,
                        "' failed: ", cudaGetErrorString(e));
      }
      device_bufs_.erase(it);
    }
    void* ptr = nullptr;
    if (cudaError_t e = cudaMalloc(&ptr, bytes); e != cudaSuccess) {
      LIGER_FAIL_CUDA("BufferPool: cudaMalloc failed for '", name, "' (",
                      bytes, " bytes): ", cudaGetErrorString(e));
    }
    if (cudaError_t e = cudaMemset(ptr, 0, bytes); e != cudaSuccess) {
      cudaFree(ptr);
      LIGER_FAIL_CUDA("BufferPool: cudaMemset failed for '", name, "' (",
                      bytes, " bytes): ", cudaGetErrorString(e));
    }
    device_bufs_[name] = {ptr, bytes};
    return ptr;
  }

  // Get or allocate a symmetric buffer (nvshmem_malloc). Symmetric memory is
  // collective — all PEs must allocate the same sizes. Reallocation after first
  // use risks desync, so an existing-but-too-small buffer is a fatal error
  // (configure max sizes upfront). Newly-allocated buffers are zero-initialized.
  void* get_symmetric(const std::string& name, size_t bytes) {
    auto it = symm_bufs_.find(name);
    if (it != symm_bufs_.end() && it->second.bytes >= bytes) {
      return it->second.ptr;
    }
    LIGER_CHECK(it == symm_bufs_.end(),
                "BufferPool: symmetric buffer '", name, "' needs ", bytes,
                " bytes but was allocated with ",
                it == symm_bufs_.end() ? size_t{0} : it->second.bytes,
                ". Configure max sizes upfront.");
    void* ptr = nvshmem_malloc(bytes);
    LIGER_CHECK(ptr != nullptr, "BufferPool: nvshmem_malloc failed for '", name,
                "' (", bytes, " bytes)");
    cudaMemset(ptr, 0, bytes);
    symm_bufs_[name] = {ptr, bytes};
    return ptr;
  }

  void clear() {
    for (auto& [name, buf] : device_bufs_) {
      cudaFree(buf.ptr);
    }
    device_bufs_.clear();
    for (auto& [name, buf] : symm_bufs_) {
      nvshmem_free(buf.ptr);
    }
    symm_bufs_.clear();
  }

 private:
  struct Buffer {
    void* ptr;
    size_t bytes;
  };
  // std::map (vs. unordered_map) so clear() iteration order is lexicographic
  // and identical across PEs — required for collective nvshmem_free to stay in
  // lockstep.
  std::map<std::string, Buffer> device_bufs_;
  std::map<std::string, Buffer> symm_bufs_;
};

// Global singleton.
inline BufferPool& global_buffer_pool() {
  static BufferPool pool;
  return pool;
}

// LIFO allocator over the symmetric heap: holds X / Y / offset buffers that
// must persist across forward calls until the matching backward, with a free
// stack for reuse.
class SymmetricMemoryStack {
 public:
  SymmetricMemoryStack() = default;

  ~SymmetricMemoryStack() { clear(); }

  void clear() {
    for (auto& [name, x] : symm_stack_) {
      while (!x.empty()) {
        nvshmem_free(x.top());
        x.pop();
      }
    }
    symm_stack_.clear();
    for (auto& [name, x] : symm_free_stack_) {
      while (!x.empty()) {
        nvshmem_free(x.top());
        x.pop();
      }
    }
    symm_free_stack_.clear();
    symm_sizes_.clear();
  }

  // Sets the per-allocation size for `name`. Must be called collectively on
  // every PE with the same value before the first put(name). The recorded size
  // is immutable once set: a smaller request is a no-op (the existing larger
  // allocation already covers it), and a larger request throws — resizing would
  // invalidate already-allocated symmetric buffers and risk heap desync.
  void set_size(const std::string& name, std::size_t size) {
    auto sz_it = symm_sizes_.find(name);
    if (sz_it == symm_sizes_.end()) {
      symm_sizes_[name] = size;
      return;
    }
    if (size <= sz_it->second) {
      return;
    }
    LIGER_CHECK(false, "SymmetricMemoryStack::set_size() refuses to grow '", name,
                "': sizes are immutable once set. Configure the max size upfront.");
  }

  // Allocates (or reuses a freed) symmetric buffer for `name` and pushes it onto
  // the live stack. Must be called collectively. Callers MUST keep
  // put/pop/set_size sequences identical on every PE, otherwise the free-stack
  // reuse vs. nvshmem_malloc decision diverges and the heap desyncs silently.
  void* put(const std::string& name) {
    auto stack_it = symm_free_stack_.find(name);
    if (stack_it != symm_free_stack_.end() && !stack_it->second.empty()) {
      void* ptr = stack_it->second.top();
      stack_it->second.pop();
      symm_stack_[name].push(ptr);
      return ptr;
    }

    auto it = symm_sizes_.find(name);
    LIGER_CHECK(it != symm_sizes_.end(),
                "SymmetricMemoryStack::put() called before set_size() for name: ", name);

    void* ptr = nvshmem_malloc(it->second);
    LIGER_CHECK(ptr != nullptr, "SymmetricMemoryStack: nvshmem_malloc failed for '",
                name, "' (", it->second, " bytes)");
    // Zero-initialize on fresh allocation so callers see a clean slot on first
    // use (mirrors BufferPool).
    cudaMemset(ptr, 0, it->second);
    symm_stack_[name].push(ptr);
    return ptr;
  }

  // Returns the pointer at the top of the live stack for `name` without removing
  // it. Useful for the bwd path to read what fwd put.
  void* top(const std::string& name) const {
    auto it = symm_stack_.find(name);
    LIGER_CHECK(it != symm_stack_.end() && !it->second.empty(),
                "SymmetricMemoryStack::top() called with empty stack for name: ", name);
    return it->second.top();
  }

  // Pops the top of the live stack for `name` and moves it to the free stack for
  // reuse. Returns the popped pointer. Must be called collectively in the same
  // order on every PE.
  void* pop(const std::string& name) {
    auto it = symm_stack_.find(name);
    LIGER_CHECK(it != symm_stack_.end() && !it->second.empty(),
                "SymmetricMemoryStack::pop() called with empty stack for name: ", name);
    void* ptr = it->second.top();
    it->second.pop();
    symm_free_stack_[name].push(ptr);
    return ptr;
  }

  // Disable copy — owns NVSHMEM allocations.
  SymmetricMemoryStack(const SymmetricMemoryStack&) = delete;
  SymmetricMemoryStack& operator=(const SymmetricMemoryStack&) = delete;

 private:
  // std::map (vs. unordered_map) so clear() iteration order is lexicographic and
  // identical across PEs — required for collective nvshmem_free to stay in
  // lockstep across PEs.
  std::map<std::string, std::size_t> symm_sizes_;
  std::map<std::string, std::stack<void*>> symm_stack_;
  std::map<std::string, std::stack<void*>> symm_free_stack_;
};

// Global singleton — holds X / Y / offset symmetric buffers that persist across
// forward calls until the matching backward.
inline SymmetricMemoryStack& global_symmetric_stack() {
  static SymmetricMemoryStack stack;
  return stack;
}

}  // namespace detail
}  // namespace liger_cute
