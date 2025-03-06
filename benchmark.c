// pcie_bw_benchmark.c
//
// A simple PCIe Bandwidth Benchmark using OpenCL.
// It enumerates all available GPU devices and for each device performs
// host-to-device and device-to-host memory copies using clEnqueueWriteBuffer
// and clEnqueueReadBuffer (with profiling enabled) to compute the effective bandwidth.
//
// This code is intended for Linux systems with OpenCL drivers installed for both
// a discrete NVIDIA GTX 1650 Mobile Refresh and an AMD Radeon integrated GPU.
//
// Compile with:
//   gcc -O2 benchmark.c -lOpenCL -o benchmark
//
// Run as:
//   ./pcie_bw_benchmark [iterations] [buffer_size_in_MB]
//
// Default iterations: 10, default buffer size: 64 MB

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <time.h>

#define DEFAULT_ITERATIONS 10
#define DEFAULT_BUF_SIZE_MB 64

// Helper function to check error codes.
void checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    cl_int err;
    int iterations = DEFAULT_ITERATIONS;
    size_t buf_size = DEFAULT_BUF_SIZE_MB * 1024 * 1024; // bytes

    if (argc >= 2) {
        iterations = atoi(argv[1]);
        if (iterations <= 0) iterations = DEFAULT_ITERATIONS;
    }
    if (argc >= 3) {
        buf_size = atol(argv[2]) * 1024 * 1024;
        if (buf_size == 0) buf_size = DEFAULT_BUF_SIZE_MB * 1024 * 1024;
    }

    printf("PCIe Bandwidth Benchmark\n");
    printf("Buffer Size: %zu bytes (%zu MB), Iterations: %d\n\n",
           buf_size, buf_size/(1024*1024), iterations);

    // Get available platforms
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    checkErr(err, "clGetPlatformIDs");
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return EXIT_FAILURE;
    }
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    checkErr(err, "clGetPlatformIDs");

    // Loop through platforms and devices
    for (cl_uint p = 0; p < num_platforms; p++) {
        // Get GPU devices only
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            continue; // no GPU devices on this platform
        }
        cl_device_id *devices = malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        checkErr(err, "clGetDeviceIDs");

        for (cl_uint d = 0; d < num_devices; d++) {
            // Retrieve device name
            char device_name[128];
            err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            checkErr(err, "clGetDeviceInfo");
            printf("Device: %s\n", device_name);

            // Create context and command queue (with profiling enabled)
            cl_context context = clCreateContext(NULL, 1, &devices[d], NULL, NULL, &err);
            checkErr(err, "clCreateContext");
            cl_command_queue queue = clCreateCommandQueue(context, devices[d], CL_QUEUE_PROFILING_ENABLE, &err);
            checkErr(err, "clCreateCommandQueue");

            // Allocate host memory (normal malloc here)
            void *host_data = malloc(buf_size);
            if (!host_data) {
                fprintf(stderr, "Failed to allocate host memory.\n");
                exit(EXIT_FAILURE);
            }
            // Initialize host data
            memset(host_data, 0xA5, buf_size);

            // Create device buffer (non-pinned memory)
            cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
            checkErr(err, "clCreateBuffer (device)");

            double total_h2d_ns = 0.0;
            double total_d2h_ns = 0.0;

            // Benchmark iterations for Host-to-Device (Write)
            for (int i = 0; i < iterations; i++) {
                cl_event event;
                err = clEnqueueWriteBuffer(queue, device_buffer, CL_TRUE, 0, buf_size, host_data, 0, NULL, &event);
                checkErr(err, "clEnqueueWriteBuffer");
                // Ensure the command has finished
                clFinish(queue);

                cl_ulong start, end;
                err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                checkErr(err, "clGetEventProfilingInfo (start)");
                err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                checkErr(err, "clGetEventProfilingInfo (end)");
                total_h2d_ns += (double)(end - start);
                clReleaseEvent(event);
            }

            // Benchmark iterations for Device-to-Host (Read)
            for (int i = 0; i < iterations; i++) {
                cl_event event;
                err = clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, buf_size, host_data, 0, NULL, &event);
                checkErr(err, "clEnqueueReadBuffer");
                clFinish(queue);

                cl_ulong start, end;
                err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                checkErr(err, "clGetEventProfilingInfo (start)");
                err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                checkErr(err, "clGetEventProfilingInfo (end)");
                total_d2h_ns += (double)(end - start);
                clReleaseEvent(event);
            }

            double avg_h2d_sec = (total_h2d_ns / iterations) * 1e-9;
            double avg_d2h_sec = (total_d2h_ns / iterations) * 1e-9;
            double h2d_bw = ((double)buf_size / avg_h2d_sec) / 1e9; // in GB/s
            double d2h_bw = ((double)buf_size / avg_d2h_sec) / 1e9; // in GB/s

            printf("  Host->Device Bandwidth: %.2f GB/s\n", h2d_bw);
            printf("  Device->Host Bandwidth: %.2f GB/s\n\n", d2h_bw);

            // Cleanup per-device
            free(host_data);
            clReleaseMemObject(device_buffer);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        free(devices);
    }
    free(platforms);
    return 0;
}

