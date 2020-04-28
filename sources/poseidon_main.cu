#include "cuda_structs.h"
#include "poseidon.h"

#include <chrono>
#include <stdlib.h>

#include <stdint.h>
#include <vector>
#include <iostream>

#include <stdio.h>
#include <time.h>

# define BENCH_SIZE (1000000)


struct Geometry
{
    int grid_size;
    int block_size;
};


template<typename T>
Geometry find_optimal_geometry(T func)
{
    int gridSize;
    int blockSize;
    int maxActiveBlocks;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t smCount = prop.multiProcessorCount;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, 0);
    gridSize = maxActiveBlocks * smCount;

    return Geometry{ gridSize, blockSize };
}


int main(int argc, char* argv[])
{
    embedded_field* host_arr = nullptr;
    embedded_field* device_arr = nullptr;
    curandState* dev_states = nullptr;

    int return_error_code = 0;

    std::chrono::high_resolution_clock::time_point start, end;
    std::int64_t duration;
    cudaError_t cudaStatus;
    
    bool result = CUDA_init();
    if (!result)
    {
        fprintf(stderr, "error on cuda init");
        return_error_code = -1;
        goto Error;
    }
    get_device_info();

    std::cout << "RUNNING POSEIDON BENCHMARK with  " << BENCH_SIZE << " ELEMENTS" << std::endl << std::endl;

    cudaStatus = cudaMalloc(&device_arr, BENCH_SIZE * sizeof(embedded_field));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (device_memory) failed!\n");
        return_error_code = -1;
        goto Error;
    }

    // we generate all the random elements on the device 
    {
        Geometry rand_gm = find_optimal_geometry(gen_random_array_kernel<embedded_field>);
        
        cudaStatus = cudaMalloc((void**)&dev_states, rand_gm.grid_size * rand_gm.block_size * sizeof(curandState));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc (devStates) failed!\n");
            return_error_code = -1;
            goto Error;
        }
        
        long ltime = time(NULL);
        unsigned int stime = (unsigned int)ltime / 2;
        srand(stime);

        gen_random_array_kernel << < rand_gm.grid_size, rand_gm.block_size >> > (device_arr, BENCH_SIZE, dev_states, rand());

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "random elements generator kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return_error_code = -1;
            goto Error;
        }
    }

    // although we run our randomizer generator on the device we want to simulate the "real" flow of execution:
    // cpu -> device -> cpu
    // that's why we copy created elements to CPU (and then we will copy them back, but benchmarking the time)

    host_arr = (embedded_field*)malloc(BENCH_SIZE * sizeof(embedded_field));
    
    cudaStatus = cudaMemcpy(host_arr, device_arr, BENCH_SIZE * sizeof(embedded_field), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (device -> host) failed!\n");
        return_error_code = -1;
        goto Error;
    }

    {
        start = std::chrono::high_resolution_clock::now();

        cudaStatus = cudaMemcpy(device_arr, host_arr, BENCH_SIZE * sizeof(embedded_field), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy (host -> device) failed!\n");
            return_error_code = -1;
            goto Error;
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "MEMCOPY: HOST -> DEVICE took " << std::dec << duration << "ns." << std::endl << std::endl;
    }

    {
        Geometry poseidon_gm = find_optimal_geometry(poseidon);     
        start = std::chrono::high_resolution_clock::now();

        poseidon<< <poseidon_gm.grid_size, poseidon_gm.block_size >> > (device_arr, BENCH_SIZE);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "poseidon kerkem failed with error code!\n", cudaStatus);
            return_error_code = -1;
            goto Error;
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "POSEIDON on GPU took " << std::dec << duration << "ns." << std::endl << std::endl;
    }

    {
        start = std::chrono::high_resolution_clock::now();

        cudaStatus = cudaMemcpy(host_arr, device_arr, BENCH_SIZE * sizeof(embedded_field), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy (device -> host) failed!\n");
            return_error_code = -1;
            goto Error;
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "MEMCOPY: DEVICE -> HOST took " << std::dec << duration << "ns." << std::endl << std::endl;
    }

Error:
    cudaFree(device_arr);
    cudaFree(dev_states);

    free(host_arr);

    return return_error_code;
}

