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

#define N 1024 * 50
#define K 200
#define STREAMS 2

__global__ void my_kenel(float * d_in, int n, int repeats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    float val = d_in[idx];
    for(int i = 0; i < repeats; i ++)
    {
        val = val * 0.999f + 1.0001f;
    }
    d_in[idx] = val;
}

void init_matrix(float *m, int n) 
{
    for (int i = 0; i < n; i++) 
        m[i] = (float)(rand() % 10) / 10.0f;
}

int main()
{
    float *h[STREAMS], *d[STREAMS];
    cudaStream_t s[STREAMS];
    dim3 blocks((N + 255) / 256);
    dim3 threads(256);
    float sequential_time, streams_time;
    cudaEvent_t start, stop;

    checkcuda(cudaEventCreate(&start));
    checkcuda(cudaEventCreate(&stop));

    for(int i = 0; i < STREAMS; i ++)
    {
        checkcuda(cudaMallocHost(&h[i], N * sizeof(float)));
        checkcuda(cudaMalloc(&d[i], N * sizeof(float)));
        checkcuda(cudaStreamCreate(&s[i]));
        init_matrix(h[i], N);
    }

    //----------------------------no streams-------------------------
    checkcuda(cudaEventRecord(start));
    for(int i = 0; i < STREAMS; i ++)
    {
        checkcuda(cudaMemcpy(d[i], h[i], N * sizeof(float), cudaMemcpyHostToDevice));
        my_kenel<<<blocks, threads>>>(d[i], N, K);
        checkcuda(cudaMemcpy(h[i], d[i], N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkcuda(cudaDeviceSynchronize());
    checkcuda(cudaEventRecord(stop));
    checkcuda(cudaEventSynchronize(stop));
    checkcuda(cudaEventElapsedTime(&sequential_time, start, stop));

    //----------------------------streams-------------------------------
    checkcuda(cudaEventRecord(start));
    for(int i = 0; i < STREAMS; i ++)
    {
        checkcuda(cudaMemcpyAsync(d[i], h[i], N * sizeof(float), cudaMemcpyHostToDevice, s[i]));
        my_kenel<<<blocks, threads, 0, s[i]>>>(d[i], N, K);
        checkcuda(cudaMemcpyAsync(h[i], d[i], N * sizeof(float), cudaMemcpyDeviceToHost, s[i]));
    }
    checkcuda(cudaDeviceSynchronize());
    checkcuda(cudaEventRecord(stop));
    checkcuda(cudaEventSynchronize(stop));
    checkcuda(cudaEventElapsedTime(&streams_time, start, stop));

    //---------------------------summary-----------------------------------
    printf("------------------summary-----------------------\n");
    printf("Data size:  %.2f MB (%d floats)\n", N * sizeof(float) / (1024.0*1024.0), N);
    printf("sequential time : %.3f \n", sequential_time);
    printf("streams time: %.3f \n", streams_time);
    printf("accelerate: %.3f \n", sequential_time / streams_time);

    //----------------------------free resource--------------------------------------
    checkcuda(cudaEventDestroy(start));
    checkcuda(cudaEventDestroy(stop));
    for(int i = 0; i < STREAMS; i ++)
    {
        checkcuda(cudaFreeHost(h[i]));
        checkcuda(cudaFree(d[i]));
        checkcuda(cudaStreamDestroy(s[i]));
    }

    return 0;
}
    