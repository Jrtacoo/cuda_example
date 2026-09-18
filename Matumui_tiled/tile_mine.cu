#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <math.h>

#define checkcuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error:%s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define N 9000
#define TILE_SIZE 32

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void matmul_simt(float *a, float *b, float *c, int n)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < n && col < n)
    {
        float sum = 0.00f;
        for(int i = 0; i < n; i ++)
        {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmul_tile_reg(const float * __restrict__ a, 
                                const float *__restrict__ b, 
                                float * __restrict__ c, int n)
{
    __shared__ float sa[BK][BM + 1];
    __shared__ float sb[BK][BN + 4];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockm = blockIdx.y * BM;
    int blockn = blockIdx.x * BN;

    int row_a = tid / 4, col_a = (tid % 4) * 4;
    int row_b = tid / 16, col_b = (tid % 16) * 4;

    float acc[TM][TN] = {0.0f};  

    for(int k = 0; k < (n + BK - 1) / BK; k ++)
    {
        // carry A(64 * 16)
        int gr_a = blockm + row_a;
        int gc_a = k * BK + col_a;

        float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
        if(gr_a < n && gc_a + 3 < n)
            tmp = *reinterpret_cast<const float4*>(a + (long long)gr_a * n + gc_a);
        else if(gr_a < n)
            for(int t = 0; t < 4 && gc_a + t < n; t ++)
                (&tmp.x)[t] = a[(long long)gr_a * n + gc_a + t];

        sa[col_a + 0][row_a] = tmp.x;
        sa[col_a + 1][row_a] = tmp.y;
        sa[col_a + 2][row_a] = tmp.z;
        sa[col_a + 3][row_a] = tmp.w;

        // carry B(16 * 64)
        int gr_b = k * BK + row_b;
        int gc_b = blockn + col_b;
        float4 tmp1 = make_float4(0.f, 0.f, 0.f, 0.f);
        if(gr_b < n && gc_b + 3 < n)
            tmp1 = *reinterpret_cast<const float4*>(b + (long long)gr_b * n + gc_b);
        else if(gr_b < n)
            for(int t = 0; t < 4 && gc_b + t < n; t ++)
                (&tmp1.x)[t] = b[(long long)gr_b * n + gc_b + t];
        *reinterpret_cast<float4*>(&sb[row_b][col_b]) = tmp1;

        __syncthreads();
        
        // accumulate result to acc
        #pragma unroll
        for(int kk = 0; kk < BK; kk ++)
        {
            float ra[TM], rb[TN];
            #pragma unroll
            for(int i = 0; i < TM; i ++) ra[i] = sa[kk][i + ty * TM];
            #pragma unroll
            for(int i = 0; i < TN; i ++) rb[i] = sb[kk][i + tx * TN];
            #pragma unroll
            for(int i = 0; i < TM; i ++)
                #pragma unroll
                for(int j = 0; j < TN; j ++)
                    acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < TM; i ++)
    {
        int row = blockm + ty * TM + i;
        if(row >= n) continue;
        #pragma unroll
        for(int j = 0; j < TN; j ++)
        {
            int col = blockn + tx * TN + j;
            if(col < n) c[row * n + col] = acc[i][j];
        }
    }
}

void init_matrix(float* m, int n)
{
    for (int i = 0; i < n * n; i++)
        m[i] = (float)(rand() % 10) / 10.0f;
}

template <typename KernelFunc>
float time_kernel(KernelFunc kernel,
                  float* a, float* b, float* c, int n,
                  dim3 grid, dim3 block, int iters)
{
    cudaEvent_t start, stop;
    checkcuda(cudaEventCreate(&start));
    checkcuda(cudaEventCreate(&stop));

    // warmup
    kernel<<<grid, block>>>(a, b, c, n);
    checkcuda(cudaGetLastError());
    checkcuda(cudaDeviceSynchronize());

    checkcuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        kernel<<<grid, block>>>(a, b, c, n);
    checkcuda(cudaEventRecord(stop));
    checkcuda(cudaEventSynchronize(stop));

    float ms;
    checkcuda(cudaEventElapsedTime(&ms, start, stop));
    checkcuda(cudaEventDestroy(start));
    checkcuda(cudaEventDestroy(stop));
    return ms / iters;
}

int main()
{
    size_t M = (size_t)N * N * sizeof(float);
    float *h_a, *h_b, *h_c1, *h_c2;
    float *d_a, *d_b, *d_c;

    //========================initial==================================
    checkcuda(cudaMallocHost((void**)&h_a, M));
    checkcuda(cudaMallocHost((void**)&h_b, M));
    checkcuda(cudaMallocHost((void**)&h_c1, M));
    checkcuda(cudaMallocHost((void**)&h_c2, M));
    srand(42);
    init_matrix(h_a, N);
    init_matrix(h_b, N);

    checkcuda(cudaMalloc((void**)&d_a, M));
    checkcuda(cudaMalloc((void**)&d_b, M));
    checkcuda(cudaMalloc((void**)&d_c, M));
    checkcuda(cudaMemcpy(d_a, h_a, M, cudaMemcpyHostToDevice));
    checkcuda(cudaMemcpy(d_b, h_b, M, cudaMemcpyHostToDevice));

    //=========================simt===================================
    dim3 block_simt(TILE_SIZE, TILE_SIZE);
    dim3 gird_simt((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    float gpu_time1 = time_kernel(matmul_simt, d_a, d_b, d_c, N, gird_simt, block_simt, 3);
    checkcuda(cudaMemcpy(h_c1, d_c, M, cudaMemcpyDeviceToHost));

    //=========================tiled==================================
    dim3 block_tiled(BN / TN, BM / TM);   // (16, 16)
    dim3 gird_tiled((N + BN - 1) / BN, (N + BM - 1) / BM);
    float gpu_time2 = time_kernel(matmul_tile_reg, d_a, d_b, d_c, N, gird_tiled, block_tiled, 3);
    checkcuda(cudaMemcpy(h_c2, d_c, M, cudaMemcpyDeviceToHost));

    //===========================Verification=========================
    printf("\n===================Verification==================\n");
    int error_count = 0;
    for (long long i = 0; i < (long long)N * N; i++)
    {
        float rel = fabs(h_c1[i] - h_c2[i]) / (fabs(h_c1[i]) + 1e-3f);
        if (rel > 1e-4f)
        {
            if (error_count < 5)
                printf("Mismatch at %lld: h_c1=%.6f, h_c2=%.6f\n", i, h_c1[i], h_c2[i]);
            error_count++;
        }
    }
    if (error_count == 0)
        printf("Results match! All %d elements verified.\n", N * N);
    else
        printf("Total mismatches: %d (%.4f%%)\n", error_count,
               100.0f * error_count / ((double)N * N));

    //============================summary================================
    double gflop = 2.0 * N * N * N / 1e9;
    printf("------------------summary-----------------------\n");
    printf("Data size:  %.2f MB (%d floats)\n", M / (1024.0 * 1024.0), N * N);
    printf("simt time : %.3f ms (%.1f GFLOPS)\n", gpu_time1, gflop / (gpu_time1 / 1000.0));
    printf("tiled time: %.3f ms (%.1f GFLOPS)\n", gpu_time2, gflop / (gpu_time2 / 1000.0));
    printf("Accelerate: %.3f x\n", gpu_time1 / gpu_time2);

    //=============================free===================================
    checkcuda(cudaFree(d_a));
    checkcuda(cudaFree(d_b));
    checkcuda(cudaFree(d_c));
    checkcuda(cudaFreeHost(h_a));
    checkcuda(cudaFreeHost(h_b));
    checkcuda(cudaFreeHost(h_c1));
    checkcuda(cudaFreeHost(h_c2));

    return 0;
}
