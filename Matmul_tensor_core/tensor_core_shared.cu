#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <math.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error:%s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

using namespace nvcuda;

// -------------------- Tile 配置 --------------------
#define BM 128      // block 负责的 C tile: M 方向
#define BN 128      // block 负责的 C tile: N 方向
#define BK 32       // K 方向每次搬运的厚度
#define WM 32       // 每个 warp 负责的 M 高度 (2 个 16x16 fragment)
#define WN 64       // 每个 warp 负责的 N 宽度 (4 个 16x16 fragment)
#define PAD 8       // 共享内存 padding，消除 bank conflict

// 256 线程 = 8 个 warp，按 4(M) x 2(N) 排布
// 要求 M、N 是 128 的倍数，K 是 32 的倍数（测试尺寸均满足）
__global__ void wmma_fp16_gemm(half* __restrict__ C,
                               const half* __restrict__ A,
                               const half* __restrict__ B,
                               int M, int N, int K)
{
    __shared__ half As[BK][BM + PAD];   // A 按 (K x M) 转置存放
    __shared__ half Bs[BK][BN + PAD];   // B 按 (K x N) 存放

    const int tid     = threadIdx.x;        // 0..255
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / 2;        // 0..3
    const int warp_n  = warp_id % 2;        // 0..1

    const int block_m = blockIdx.x * BM;
    const int block_n = blockIdx.y * BN;

    // FP32 累加器（精度比 half 累加器好）
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WM/16][WN/16];
    for (int i = 0; i < WM/16; i++)
        for (int j = 0; j < WN/16; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // 协作加载用的线程映射（每线程一个 float4 = 8 个 half）
    const int a_row = tid / 4,  a_col = (tid % 4)  * 8;   // A: 128 x 32, 搬运的一轮共512个float4(128 * 32)， 每行4个float4(32half)， 一个线程负责两个float4，因此两轮覆盖
    const int b_row = tid / 16, b_col = (tid % 16) * 8;   // B: 32 x 128

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // ---- 协作加载 A (BM x BK)，转置写入 As (BK x BM) ----
        #pragma unroll
        for (int r = 0; r < BM; r += 64)
        {
            int gr = block_m + a_row + r;
            int gc = k0 + a_col;
            float4 tmp = *reinterpret_cast<const float4*>(A + (long long)gr * K + gc);
            const half* h = reinterpret_cast<const half*>(&tmp);
            #pragma unroll
            for (int c = 0; c < 8; c++) As[a_col + c][a_row + r] = h[c];
        }
        // ---- 协作加载 B (BK x BN)，直接写入 Bs ----
        #pragma unroll
        for (int r = 0; r < BK; r += 16)
        {
            int gr = k0 + b_row + r;
            int gc = block_n + b_col;
            float4 tmp = *reinterpret_cast<const float4*>(B + (long long)gr * N + gc);
            *reinterpret_cast<float4*>(&Bs[b_row + r][b_col]) = tmp;
        }
        __syncthreads();

        // ---- 内层 K 循环：全部从共享内存读 ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[WM/16];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WN/16];

            #pragma unroll
            for (int i = 0; i < WM/16; i++)
                wmma::load_matrix_sync(a_frag[i],
                    &As[kk][warp_m * WM + i * 16], BM + PAD);
            #pragma unroll
            for (int j = 0; j < WN/16; j++)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                wmma::load_matrix_sync(b_frag[j],
                    &Bs[kk][warp_n * WN + j * 16], BN + PAD);

            #pragma unroll
            for (int i = 0; i < WM/16; i++)
                #pragma unroll
                for (int j = 0; j < WN/16; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        __syncthreads();
    }

    // ---- FP32 累加结果转 FP16 写回 ----
    const int c_m = block_m + warp_m * WM;
    const int c_n = block_n + warp_n * WN;
    for (int i = 0; i < WM/16; i++)
    {
        for (int j = 0; j < WN/16; j++)
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h_frag;
            for (int e = 0; e < h_frag.num_elements; e++)
                h_frag.x[e] = __float2half(c_frag[i][j].x[e]);
            wmma::store_matrix_sync(C + (long long)(c_m + i * 16) * N + c_n + j * 16,
                                    h_frag, N, wmma::mem_row_major);
        }
    }
}

// -------------------- 正确性检查: CPU 对拍 --------------------
void cpu_gemm_ref(const half* A, const half* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            C[i * N + j] = acc;
        }
}

double test_wmma(int M, int N, int K, int iters)
{
    half *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, M * K * sizeof(half)));
    checkCuda(cudaMalloc(&d_B, K * N * sizeof(half)));
    checkCuda(cudaMalloc(&d_C, M * N * sizeof(half)));

    // 初始化输入（随机数，范围小一点避免 FP16 累加误差过大）
    half* h_A = (half*)malloc(M * K * sizeof(half));
    half* h_B = (half*)malloc(K * N * sizeof(half));
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half((rand() % 200 - 100) / 100.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half((rand() % 200 - 100) / 100.0f);
    checkCuda(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 grid(M / BM, N / BN);
    dim3 blocks(256);

    // ---- 正确性验证（只在第一个尺寸做一次，CPU 参考很慢）----
    static bool verified = false;
    if (!verified)
    {
        verified = true;
        wmma_fp16_gemm<<<grid, blocks>>>(d_C, d_A, d_B, M, N, K);
        checkCuda(cudaDeviceSynchronize());
        half* h_C = (half*)malloc(M * N * sizeof(half));
        float* ref = (float*)malloc(M * N * sizeof(float));
        checkCuda(cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
        cpu_gemm_ref(h_A, h_B, ref, M, N, K);
        double max_rel = 0.0;
        for (int i = 0; i < M * N; i++)
        {
            double diff = fabs(__half2float(h_C[i]) - ref[i]);
            double rel  = diff / (fabs(ref[i]) + 1e-3);
            if (rel > max_rel) max_rel = rel;
        }
        printf("  [check] M=N=K=%d, max relative error = %.4f %s\n",
               M, max_rel, max_rel < 0.05 ? "(PASS)" : "(FAIL!)");
        free(h_C);
        free(ref);
    }

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < 10; i++)
        wmma_fp16_gemm<<<grid, blocks>>>(d_C, d_A, d_B, M, N, K);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        wmma_fp16_gemm<<<grid, blocks>>>(d_C, d_A, d_B, M, N, K);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));

    double flops  = 2.0 * M * N * K * iters;
    double tflops = (flops / (ms / 1000.0)) / 1e12;

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);

    return tflops;
}

int main()
{
    int device;
    cudaDeviceProp prop;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("  GEMM size (M=N=K) |   TFLOPS (FP16)\n");
    printf("  ------------------|----------------\n");

    int sizes[] = {512, 1024, 2048, 4096, 8192};

    for (int si = 0; si < 5; si++)
    {
        int s = sizes[si];
        int iters = (s <= 2048) ? 50 : 10;
        double tflops = test_wmma(s, s, s, iters);
        printf("  %16d | %13.1f\n", s, tflops);
    }

    return 0;
}