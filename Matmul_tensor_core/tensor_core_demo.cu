#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<mma.h>
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<algorithm>
#include<math.h>

#define checkCuda(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char * file, int line)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error:%s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

__global__ void wmma_fp16_gemm(half *C, const half *A, const half *B, int M, int N, int K, half alpha, half beta)
{
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    if(warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for(int k = 0; k < (K + WMMA_K - 1)  / WMMA_K; k ++)
    {
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k * WMMA_K, K);
        wmma::load_matrix_sync(b_frag, B + k * WMMA_K * N + warpN * WMMA_N, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * c_frag.x[i] + beta * __half(0.0f);
    }
    
    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N,
                            c_frag, N, wmma::mem_row_major);
}

double test_wmma(int M, int N, int K, int iters)
{
    half *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, M * K * sizeof(half)));
    checkCuda(cudaMalloc(&d_B, K * N * sizeof(half)));
    checkCuda(cudaMalloc(&d_C, M * N * sizeof(half)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    dim3 blocks((M + WMMA_M - 1) / (WMMA_M * 8), (N + WMMA_N - 1) / WMMA_N);
    dim3 threads(256, 1);

    for(int i = 0; i < 10; i ++)
    {
        wmma_fp16_gemm<<<blocks, threads>>>(d_C, d_A, d_B, M, N, K, __float2half(1.0f), __float2half(0.0f));
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for(int i = 0; i < iters; i ++)
    {
        wmma_fp16_gemm<<<blocks, threads>>>(d_C, d_A, d_B, M, N, K, __float2half(1.0f), __float2half(0.0f));
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * M * N * K * iters;
    double tflops = (flops / (ms / 1000.0)) / 1e12;

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    return tflops;
}

int main()
{
    int device;
    cudaDeviceProp prop;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("  GEMM size (M=N=K) |   TFLOPS (FP16) |  %% of peak dense\n");
    printf("  ------------------|-----------------|-----------------\n");

    int sizes[] = {512, 1024, 2048, 4096, 8192};
    float peak_tflops = 312.0f;  

    for (int si = 0; si < 5; si++) {
        int s = sizes[si];
        int iters = (s <= 2048) ? 50 : 10;
        double tflops = test_wmma(s, s, s, iters);
        printf("  %16d | %13.1f | %13.1f%%\n", s, tflops, 100.0 * tflops / peak_tflops);
    }

    return 0;
}