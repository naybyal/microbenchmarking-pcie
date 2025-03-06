// benchmark.cu
// A simple PCIe Bandwidth Benchmark using CUDA runtime API.
//
// This program allocates a 64 MB pinned host buffer and a device buffer,
// then performs multiple iterations of data transfers in both directions.
// It uses CUDA events to time the transfers and calculates the average bandwidth.
//
// Usage:
//   ./cuda_pcie_bw [iterations] [buffer_size_in_MB]
//   Default iterations: 10, default buffer size: 64 MB

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define DEFAULT_ITERATIONS 10
#define DEFAULT_BUF_SIZE_MB 64

// Macro for checking CUDA errors.
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main(int argc, char* argv[]) {
    int iterations = DEFAULT_ITERATIONS;
    size_t bufSize = DEFAULT_BUF_SIZE_MB * 1024 * 1024; // 64 MB

    if(argc >= 2) {
        iterations = atoi(argv[1]);
        if(iterations <= 0) iterations = DEFAULT_ITERATIONS;
    }
    if(argc >= 3) {
        bufSize = atol(argv[2]) * 1024 * 1024;
        if(bufSize == 0) bufSize = DEFAULT_BUF_SIZE_MB * 1024 * 1024;
    }

    printf("CUDA PCIe Bandwidth Benchmark\n");
    printf("Buffer Size: %zu bytes (%zu MB), Iterations: %d\n\n",
           bufSize, bufSize / (1024 * 1024), iterations);
    cudaCheckError( cudaSetDevice(0) );
    // Allocate pinned host memory for optimal transfer performance.
    void* hostData;
    cudaCheckError( cudaMallocHost(&hostData, bufSize) );
    memset(hostData, 0xA5, bufSize);

    // Allocate device memory.
    void* deviceData;
    cudaCheckError( cudaMalloc(&deviceData, bufSize) );

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaCheckError( cudaEventCreate(&start) );
    cudaCheckError( cudaEventCreate(&stop) );

    float totalH2D_ms = 0.0f, totalD2H_ms = 0.0f;
    float elapsed_ms = 0.0f;

    // Benchmark Host-to-Device transfers.
    for (int i = 0; i < iterations; i++) {
        cudaCheckError( cudaEventRecord(start, 0) );
        cudaCheckError( cudaMemcpy(deviceData, hostData, bufSize, cudaMemcpyHostToDevice) );
        cudaCheckError( cudaEventRecord(stop, 0) );
        cudaCheckError( cudaEventSynchronize(stop) );
        cudaCheckError( cudaEventElapsedTime(&elapsed_ms, start, stop) );
        totalH2D_ms += elapsed_ms;
    }

    // Benchmark Device-to-Host transfers.
    for (int i = 0; i < iterations; i++) {
        cudaCheckError( cudaEventRecord(start, 0) );
        cudaCheckError( cudaMemcpy(hostData, deviceData, bufSize, cudaMemcpyDeviceToHost) );
        cudaCheckError( cudaEventRecord(stop, 0) );
        cudaCheckError( cudaEventSynchronize(stop) );
        cudaCheckError( cudaEventElapsedTime(&elapsed_ms, start, stop) );
        totalD2H_ms += elapsed_ms;
    }

    // Compute average times and bandwidths.
    float avgH2D_sec = (totalH2D_ms / iterations) / 1000.0f;
    float avgD2H_sec = (totalD2H_ms / iterations) / 1000.0f;
    // Convert bytes to gigabytes.
    double gbTransferred = (double)bufSize / (1LL << 30);
    double h2dBandwidth = gbTransferred / avgH2D_sec;
    double d2hBandwidth = gbTransferred / avgD2H_sec;

    printf("Average Host->Device Bandwidth: %.2f GB/s\n", h2dBandwidth);
    printf("Average Device->Host Bandwidth: %.2f GB/s\n", d2hBandwidth);

    // Cleanup resources.
    cudaCheckError( cudaEventDestroy(start) );
    cudaCheckError( cudaEventDestroy(stop) );
    cudaCheckError( cudaFree(deviceData) );
    cudaCheckError( cudaFreeHost(hostData) );

    return 0;
}

