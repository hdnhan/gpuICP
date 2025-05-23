#pragma once

constexpr int error_exit_code = -1;

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

#endif

#ifdef CUDA_FOUND
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