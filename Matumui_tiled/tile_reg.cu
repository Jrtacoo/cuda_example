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

// ---------------- 改进版 tile 配置 ----------------
#define BM 64       // block tile: M
#define BN 64       // block tile: N
#define BK 16       // K 方向厚度
#define TM 4        // 每个线程负责的 M 方向元素数
#define TN 4        // 每个线程负责的 N 方向元素数
// blockDim = (16, 16) = 256 线程

__global__ void matmul_simt(float* a, float* b, float* c, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n)
    {
        float sum = 0.00f;
        for (int i = 0; i < n; i++)
            sum += a[row * n + i] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int n)
{
    __shared__ float sa[BK][BM + 1];
    __shared__ float sb[BK][BN + 4];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255
    const int tx  = threadIdx.x;   // 0..15, 本线程负责的 N 子块起点 = tx*TN
    const int ty  = threadIdx.y;   // 0..15, 本线程负责的 M 子块起点 = ty*TM

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    const int a_row = tid / 4,  a_col = (tid % 4)  * 4;   // a_row: 0..63  a_col: 0, 4, 8, 12 (4)
    const int b_row = tid / 16, b_col = (tid % 16) * 4;   // b_row: 0..15  b_col: 0, 4, 8, 12 ... 56, 60 (16)
 
    float acc[TM][TN] = {0.0f};   // 寄存器里的 4x4 累加器

    for (int tile = 0; tile < (n + BK - 1) / BK; tile++)
    {
        // ---- 协作加载 A (BM x BK)，转置写入 sa，越界补 0 ----
        {
            int gr = block_m + a_row;
            int gc = tile * BK + a_col;
            float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < n && gc + 3 < n)
                tmp = *reinterpret_cast<const float4*>(a + (long long)gr * n + gc);
            else if (gr < n)   // 尾列零散处理
                for (int t = 0; t < 4 && gc + t < n; t++)
                    (&tmp.x)[t] = a[(long long)gr * n + gc + t];
            sa[a_col + 0][a_row] = tmp.x;
            sa[a_col + 1][a_row] = tmp.y;
            sa[a_col + 2][a_row] = tmp.z;
            sa[a_col + 3][a_row] = tmp.w;
        }
        // ---- 协作加载 B (BK x BN)，直接写入 sb，越界补 0 ----
        {
            int gr = tile * BK + b_row;
            int gc = block_n + b_col;
            float4 tmp = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < n && gc + 3 < n)
                tmp = *reinterpret_cast<const float4*>(b + (long long)gr * n + gc);
            else if (gr < n)
                for (int t = 0; t < 4 && gc + t < n; t++)
                    (&tmp.x)[t] = b[(long long)gr * n + gc + t];
            *reinterpret_cast<float4*>(&sb[b_row][b_col]) = tmp;
        }
        __syncthreads();

        // ---- 内层计算：4x4 寄存器分块 ----
        #pragma unroll
        for (int k = 0; k < BK; k++)
        {
            float ra[TM], rb[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) ra[i] = sa[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) rb[j] = sb[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += ra[i] * rb[j];   // 8 次读 -> 16 次 FMA
        }
        __syncthreads();
    }

    // ---- 写回（带边界检查）----
    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int row = block_m + ty * TM + i;
        if (row >= n) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++)
        {
            int col = block_n + tx * TN + j;
            if (col < n) c[(long long)row * n + col] = acc[i][j];
        }
    }
}

void init_matrix(float* m, int n)
{
    for (int i = 0; i < n * n; i++)
        m[i] = (float)(rand() % 10) / 10.0f;
}

// 计时封装：warmup + 多次平均
template <typename KernelFunc>
float time_kernel(KernelFunc kernel,
                  float* a, float* b, float* c, int n,
                  dim3 grid, dim3 block, int iters)
{
    cudaEvent_t start, stop;
    checkcuda(cudaEventCreate(&start));
    checkcuda(cudaEventCreate(&stop));

    kernel<<<grid, block>>>(a, b, c, n);   // warmup
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
    float gpu_time2 = time_kernel(matmul_tiled, d_a, d_b, d_c, N, gird_tiled, block_tiled, 3);
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
               100.0f * error_count / ((double)N * N));   // 修正: 除以 N*N

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