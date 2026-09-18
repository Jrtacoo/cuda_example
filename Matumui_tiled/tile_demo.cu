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

#define N 9000
#define TILE_SIZE 32

__global__ void matmul_simt(float * a, float * b, float * c, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < n && row < n)
    {
        float sum = 0.00f;
        for(int i = 0; i < n; i ++)
        {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(float *a, float *b, float *c, int n)
{
    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.00f;

    for(int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile ++)
    {
        int a_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if(a_row < n && a_col < n)
            sa[threadIdx.y][threadIdx.x] = a[a_row * n + a_col];
        else sa[threadIdx.y][threadIdx.x] = 0.00f;

        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;
        if(b_row < n && b_col < n)
            sb[threadIdx.y][threadIdx.x] = b[b_row * n + b_col];
        else sb[threadIdx.y][threadIdx.x] = 0.00f;

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k ++)
            sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        
        __syncthreads();
    }

    if(row < n && col < n)
        c[row * n + col] = sum;
}

void init_matrix(float *m, int n) 
{
    for (int i = 0; i < n * n; i++) 
        m[i] = (float)(rand() % 10) / 10.0f;
}

int main()
{
    size_t M = N * N * sizeof(float);
    float *h_a, *h_b, *h_c1, *h_c2;
    float *d_a, *d_b, *d_c;
    cudaEvent_t start, stop;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    float gpu_time1, gpu_time2;

    //-------------------------initial-------------------------------
    checkcuda(cudaMallocHost((void **)&h_a, M));
    checkcuda(cudaMallocHost((void **)&h_b, M));
    checkcuda(cudaMallocHost((void **)&h_c1, M));
    checkcuda(cudaMallocHost((void **)&h_c2, M));
    init_matrix(h_a, N); 
    init_matrix(h_b, N);

    checkcuda(cudaMalloc((void **)&d_a, M));
    checkcuda(cudaMalloc((void **)&d_b, M));
    checkcuda(cudaMalloc((void **)&d_c, M));

    checkcuda(cudaMemcpy(d_a, h_a, M, cudaMemcpyHostToDevice));
    checkcuda(cudaMemcpy(d_b, h_b, M, cudaMemcpyHostToDevice));

    checkcuda(cudaEventCreate(&start));
    checkcuda(cudaEventCreate(&stop));

    //-------------------------simt----------------------------------
    checkcuda(cudaEventRecord(start));
    matmul_simt<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    checkcuda(cudaGetLastError());
    checkcuda(cudaEventRecord(stop));
    checkcuda(cudaEventSynchronize(stop));
    checkcuda(cudaMemcpy(h_c1, d_c, M, cudaMemcpyDeviceToHost));
    checkcuda(cudaEventElapsedTime(&gpu_time1, start, stop));

    //-------------------------tiled---------------------------------
    checkcuda(cudaEventRecord(start));
    matmul_tiled<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    checkcuda(cudaGetLastError());
    checkcuda(cudaEventRecord(stop));
    checkcuda(cudaEventSynchronize(stop));
    checkcuda(cudaMemcpy(h_c2, d_c, M, cudaMemcpyDeviceToHost));
    checkcuda(cudaEventElapsedTime(&gpu_time2, start, stop));

    //===========================Verification=========================
    printf("\n===================Verification==================\n");
    int error_count = 0;
    for(int i = 0; i < N * N; i ++)
    {
        if(fabs(h_c1[i] - h_c2[i]) > 1e-5f)
        {
            if(error_count < 5)
            {
                printf("Mismatch at %d: h_c1=%.6f, h_c2=%.6f\n", i, h_c1[i], h_c2[i]);
            }
            error_count++;
        }
    }
    if(error_count == 0)
        printf("Results match! All %d elements verified.\n", N * N);
    else
        printf("Total mismatches: %d (%.4f%%)\n", error_count, 100.0f * error_count / N);

    //------------------------summary---------------------------------
    printf("------------------summary-----------------------\n");
    printf("Data size:  %.2f MB (%d floats)\n", M / (1024.0*1024.0), N * N);
    printf("Tile size:  %d  \n", TILE_SIZE);
    printf("simt time : %.3f \n", gpu_time1);
    printf("tiled time: %.3f \n", gpu_time2);
    printf("Accelerate: %.3f \n", gpu_time1 / gpu_time2);

    //------------------------ free------------------------------------
    checkcuda(cudaFree(d_a));
    checkcuda(cudaFree(d_b));
    checkcuda(cudaFree(d_c));
    checkcuda(cudaFreeHost(h_a));
    checkcuda(cudaFreeHost(h_b));
    checkcuda(cudaFreeHost(h_c1));
    checkcuda(cudaFreeHost(h_c2));
    checkcuda(cudaEventDestroy(start));
    checkcuda(cudaEventDestroy(stop));

    return 0;
}