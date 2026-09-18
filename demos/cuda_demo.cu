#include<cuda_runtime.h>
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<algorithm>
#include<math.h>

#define checkcuda(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char * file, int line)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error:%s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define N (1024 * 1024 / sizeof(float))
#define K 1204

__global__ void my_kenel(float * d_in, float * d_out, int n, int repeats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    float val = d_in[idx];
    for(int i = 0; i < repeats; i ++)
    {
        val = val * 0.999f + 1.0001f;
    }
    d_out[idx] = val;
}

void cpu_compute(float * h_in, float * h_out, int n, int repeats)
{
    for(int i = 0; i < n; i ++)
    {
        float val = h_in[i];
        for(int j = 0; j < repeats; j ++)
        {
            val = val * 0.999f + 1.0001f;
        }
        h_out[i] = val;
    }
    return ;
}

int main()
{
    cudaEvent_t start, stop;
    checkcuda(cudaEventCreate(&start));
    checkcuda(cudaEventCreate(&stop));
    float gpu_time_ms = 0.00f;

    float * h_in, * h_out_cpu, *h_out_gpu;
    checkcuda(cudaMallocHost((void **)&h_in, N * sizeof(float)));
    checkcuda(cudaMallocHost((void **)&h_out_cpu, N * sizeof(float)));
    checkcuda(cudaMallocHost((void **)&h_out_gpu, N * sizeof(float)));

    float * d_in, *d_out;
    checkcuda(cudaMalloc((void **)&d_in, N * sizeof(float)));
    checkcuda(cudaMalloc((void **)&d_out, N * sizeof(float)));

    for(int i = 0; i < N; i ++)
    {
        h_in[i] = (float)(rand() % 100) / 10.0f;
    }
    
    //===========================GPU version=========================
    printf("\n===================GPU version==================\n");
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    //H2D, kenel, D2H
    checkcuda(cudaEventRecord(start));
    checkcuda(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    my_kenel<<<blocks, threads>>>(d_in, d_out, N, K);
    checkcuda(cudaGetLastError());
    checkcuda(cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    checkcuda(cudaEventRecord(stop));
    checkcuda(cudaEventSynchronize(stop));
    checkcuda(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    printf("kernel execution:     %.3f ms (blocks = %d, thread = %d)\n", gpu_time_ms, blocks, threads);

    //===========================CPU version=========================
    printf("\n===================CPU version==================\n");
    clock_t cpu_start = clock();
    cpu_compute(h_in, h_out_cpu, N, K);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Serial compute:      %.3f ms\n", cpu_time_ms);

    //===========================Verification=========================
    printf("\n===================Verification==================\n");
    int error_count = 0;
    for(int i = 0; i < N; i++)
    {
        if(fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-3f)
        {
            if(error_count < 5)
            {
                printf("Mismatch at %d: GPU=%.6f, CPU=%.6f\n", i, h_out_gpu[i], h_out_cpu[i]);
            }
            error_count++;
        }
    }
    if(error_count == 0)
        printf("Results match! All %d elements verified.\n", N);
    else
        printf("Total mismatches: %d (%.4f%%)\n", error_count, 100.0f * error_count / N);

    //===========================Summary=========================
    printf("\n===================Summary==================\n");
    printf("Data size:               %.2f MB (%d floats)\n", N * sizeof(float) / (1024.0*1024.0), N);
    printf("Iterations per element:  %d\n", K);
    printf("CPU Time:                %.3f ms\n", cpu_time_ms);
    printf("GPU Total Time:          %.3f ms\n", gpu_time_ms);
    
    // ====================Free Resouce====================
    checkcuda(cudaFree(d_in));
    checkcuda(cudaFree(d_out));
    checkcuda(cudaFreeHost(h_in));
    checkcuda(cudaFreeHost(h_out_gpu));
    checkcuda(cudaFreeHost(h_out_cpu));
    checkcuda(cudaEventDestroy(start));
    checkcuda(cudaEventDestroy(stop));

    return 0;
}
