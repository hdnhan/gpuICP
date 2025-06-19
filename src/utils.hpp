#pragma once

#include <thrust/device_allocator.h>

constexpr int error_exit_code = 1;

#ifdef HIP_FOUND
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpy hipMemcpy

#define GPU_CHECK(condition)                                                                                 \
    {                                                                                                        \
        const hipError_t error = condition;                                                                  \
        if (error != hipSuccess) {                                                                           \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " << __FILE__      \
                      << ':' << __LINE__ << std::endl;                                                       \
            std::exit(error_exit_code);                                                                      \
        }                                                                                                    \
    }

#elif defined(CUDA_FOUND)
#define GPU_CHECK(condition)                                                                                 \
    {                                                                                                        \
        const cudaError_t error = condition;                                                                 \
        if (error != cudaSuccess) {                                                                          \
            std::cerr << "An error encountered: \"" << cudaGetErrorString(error) << "\" at " << __FILE__     \
                      << ':' << __LINE__ << std::endl;                                                       \
            std::exit(error_exit_code);                                                                      \
        }                                                                                                    \
    }
#endif

// https://forums.developer.nvidia.com/t/is-there-a-similar-temporary-allocation-feature-like-cub-for-thrusts-thrust-sort-by-key/312583
// https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu
template <typename T> class CachingAllocator : public thrust::device_allocator<T> {
  public:
    CachingAllocator() {}

    ~CachingAllocator() {
        if (_ptr)
            thrust::device_allocator<T>::deallocate(_ptr, _size);
        _size = 0;
        _ptr = nullptr;
    }

    thrust::device_ptr<T> allocate(size_t n) {
        if (_ptr && _size >= n)
            return _ptr;
        if (_ptr)
            thrust::device_allocator<T>::deallocate(_ptr, _size);
        _size = n;
        _ptr = thrust::device_allocator<T>::allocate(n);
        return _ptr;
    }

    void deallocate(thrust::device_ptr<T> p, size_t n) {
        // Do not deallocate memory here, we will manage it ourselves
    }

  private:
    size_t _size = 0;
    thrust::device_ptr<T> _ptr = nullptr;
};