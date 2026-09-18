#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <math.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4*>(&(value))[0])

#define checkCuda(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char * file, int line)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error:%s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// -------------------- 基础数学函数 --------------------

__host__ __device__ __forceinline__ float gelu_f(float x) {
    const float cbrt = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float t = sqrt_2_over_pi * (x + cbrt * x3);
    float tanh_t = tanhf(t);
    return 0.5f * x * (1.0f + tanh_t);
}

// -------------------- 未融合版本：分步实现 --------------------
// 修复：原代码只做 warp 内归约，跨 warp 部分和被丢弃，这里补上 block 级归约

// Step 1: 计算每行的 mean
__global__ void row_mean_kernel(const float* input, float* mean, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    const float* row_ptr = input + row * cols;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += row_ptr[i];
    }

    // Warp 内归约
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Block 级归约：各 warp leader 写入 shared，再由 warp 0 汇总
    __shared__ float s_warp[32];  // 最多支持 1024 线程
    if (lane_id == 0) s_warp[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? s_warp[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (tid == 0) mean[row] = sum / cols;
    }
}

// Step 2: 计算 variance
__global__ void row_variance_kernel(const float* input, const float* mean,
                                    float* variance, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    const float* row_ptr = input + row * cols;
    float m = mean[row];

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_ptr[i] - m;
        sum += diff * diff;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float s_warp[32];
    if (lane_id == 0) s_warp[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? s_warp[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (tid == 0) variance[row] = sum / cols;
    }
}

// Step 3: Normalize + Scale + Shift + GELU
__global__ void layernorm_gelu_kernel_split(const float* input, const float* mean,
                                             const float* variance, const float* gamma,
                                             const float* beta, float* output,
                                             int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float m = mean[row];
    float var = variance[row];
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < cols; i += blockDim.x) {
        float x = input[row * cols + i];
        float norm = (x - m) * inv_std;
        float scaled = norm * gamma[i] + beta[i];
        output[row * cols + i] = gelu_f(scaled);
    }
}

// -------------------- 融合版本：单 Kernel 完成所有操作 --------------------

__global__ void fused_layernorm_gelu_kernel(const float* input, const float* gamma,
                                             const float* beta, float* output,
                                             int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    const float* row_ptr = input + row * cols;

    extern __shared__ float s_mem[];
    float* s_warp_mean = s_mem;
    float* s_warp_var = s_mem + (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Phase 1: Mean
    float local_sum = 0.0f;
    int vec_cols = cols / 4 * 4;
    for (int i = tid * 4; i < vec_cols; i += blockDim.x * 4) {
        float4 val4 = FLOAT4_CONST(row_ptr[i]);
        local_sum += val4.x + val4.y + val4.z + val4.w;
    }
    for (int i = vec_cols + tid; i < cols; i += blockDim.x) {
        local_sum += row_ptr[i];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane_id == 0) s_warp_mean[warp_id] = local_sum;
    __syncthreads();

    float mean = 0.0f;
    if (warp_id == 0) {
        mean = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? s_warp_mean[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            mean += __shfl_down_sync(0xFFFFFFFF, mean, offset);
        }
        if (tid == 0) s_warp_mean[0] = mean / cols;
    }
    __syncthreads();
    mean = s_warp_mean[0];

    // Phase 2: Variance
    float local_sq_diff = 0.0f;
    for (int i = tid * 4; i < vec_cols; i += blockDim.x * 4) {
        float4 val4 = FLOAT4_CONST(row_ptr[i]);
        float d0 = val4.x - mean;
        float d1 = val4.y - mean;
        float d2 = val4.z - mean;
        float d3 = val4.w - mean;
        local_sq_diff += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    for (int i = vec_cols + tid; i < cols; i += blockDim.x) {
        float diff = row_ptr[i] - mean;
        local_sq_diff += diff * diff;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sq_diff += __shfl_down_sync(0xFFFFFFFF, local_sq_diff, offset);
    }
    if (lane_id == 0) s_warp_var[warp_id] = local_sq_diff;
    __syncthreads();

    float variance = 0.0f;
    if (warp_id == 0) {
        variance = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? s_warp_var[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            variance += __shfl_down_sync(0xFFFFFFFF, variance, offset);
        }
        if (tid == 0) s_warp_var[0] = variance / cols;
    }
    __syncthreads();
    variance = s_warp_var[0];

    float inv_std = rsqrtf(variance + eps);

    // Phase 3: Normalize + Scale + Shift + GELU
    for (int i = tid * 4; i < vec_cols; i += blockDim.x * 4) {
        float4 val4 = FLOAT4_CONST(row_ptr[i]);
        float4 gamma4 = FLOAT4_CONST(gamma[i]);
        float4 beta4 = FLOAT4_CONST(beta[i]);

        float4 out4;
        out4.x = gelu_f(((val4.x - mean) * inv_std) * gamma4.x + beta4.x);
        out4.y = gelu_f(((val4.y - mean) * inv_std) * gamma4.y + beta4.y);
        out4.z = gelu_f(((val4.z - mean) * inv_std) * gamma4.z + beta4.z);
        out4.w = gelu_f(((val4.w - mean) * inv_std) * gamma4.w + beta4.w);

        FLOAT4(output[row * cols + i]) = out4;
    }
    for (int i = vec_cols + tid; i < cols; i += blockDim.x) {
        float x = row_ptr[i];
        float norm = (x - mean) * inv_std;
        float scaled = norm * gamma[i] + beta[i];
        output[row * cols + i] = gelu_f(scaled);
    }
}

// -------------------- Host 调用包装 --------------------

// 融合版本
void fused_layernorm_gelu(const float* d_input, const float* d_gamma,
                          const float* d_beta, float* d_output,
                          int rows, int cols, float eps = 1e-5f) {
    int threads = 256;
    int warps = threads / WARP_SIZE;
    size_t smem_size = 2 * warps * sizeof(float);
    fused_layernorm_gelu_kernel<<<rows, threads, smem_size>>>(
        d_input, d_gamma, d_beta, d_output, rows, cols, eps);
}

// 未融合版本：3 个 kernel 串行启动，中间结果（mean/variance）走显存
void split_layernorm_gelu(const float* d_input, const float* d_gamma,
                          const float* d_beta, float* d_output,
                          float* d_mean, float* d_variance,
                          int rows, int cols, float eps = 1e-5f) {
    int threads = 256;
    row_mean_kernel<<<rows, threads>>>(d_input, d_mean, rows, cols);
    row_variance_kernel<<<rows, threads>>>(d_input, d_mean, d_variance, rows, cols);
    layernorm_gelu_kernel_split<<<rows, threads>>>(
        d_input, d_mean, d_variance, d_gamma, d_beta, d_output, rows, cols, eps);
}

// -------------------- CPU 验证 --------------------

void cpu_layernorm_gelu(const float* input, const float* gamma, const float* beta,
                        float* output, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        float mean = 0, var = 0;
        for (int c = 0; c < cols; c++) mean += input[r * cols + c];
        mean /= cols;
        for (int c = 0; c < cols; c++) {
            float diff = input[r * cols + c] - mean;
            var += diff * diff;
        }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < cols; c++) {
            float norm = (input[r * cols + c] - mean) * inv_std;
            float scaled = norm * gamma[c] + beta[c];
            output[r * cols + c] = gelu_f(scaled);
        }
    }
}

float max_error(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

// -------------------- 测试主函数 --------------------

int main() {
    const int rows = 1024;
    const int cols = 768;
    const int N = rows * cols;
    const float eps = 1e-5f;
    const int warmup = 10;    // 预热次数（让 GPU 频率、cache 稳定）
    const int iters = 100;    // 计时迭代次数

    // Host 内存
    float *h_input = new float[N];
    float *h_gamma = new float[cols];
    float *h_beta = new float[cols];
    float *h_out_split = new float[N];
    float *h_out_fused = new float[N];
    float *h_ref = new float[N];

    for (int i = 0; i < N; i++) h_input[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    for (int i = 0; i < cols; i++) h_gamma[i] = 1.0f;
    for (int i = 0; i < cols; i++) h_beta[i] = 0.0f;

    // Device 内存
    float *d_input, *d_gamma, *d_beta, *d_output, *d_mean, *d_variance;
    checkCuda(cudaMalloc(&d_input, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_gamma, cols * sizeof(float)));
    checkCuda(cudaMalloc(&d_beta, cols * sizeof(float)));
    checkCuda(cudaMalloc(&d_output, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_mean, rows * sizeof(float)));      // 未融合版本的中间缓冲
    checkCuda(cudaMalloc(&d_variance, rows * sizeof(float)));

    checkCuda(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_gamma, h_gamma, cols * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_beta, h_beta, cols * sizeof(float), cudaMemcpyHostToDevice));

    // ========== 正确性验证 ==========
    cpu_layernorm_gelu(h_input, h_gamma, h_beta, h_ref, rows, cols, eps);

    split_layernorm_gelu(d_input, d_gamma, d_beta, d_output,
                         d_mean, d_variance, rows, cols, eps);
    checkCuda(cudaMemcpy(h_out_split, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    fused_layernorm_gelu(d_input, d_gamma, d_beta, d_output, rows, cols, eps);
    checkCuda(cudaMemcpy(h_out_fused, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("=== Verification ===\n");
    printf("Split  vs CPU  max error: %.6e\n", max_error(h_out_split, h_ref, N));
    printf("Fused  vs CPU  max error: %.6e\n", max_error(h_out_fused, h_ref, N));
    printf("Split  vs Fused max error: %.6e\n\n", max_error(h_out_split, h_out_fused, N));

    // ========== 性能对比 ==========
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    float ms = 0.0f;

    // --- 未融合版本 ---
    for (int i = 0; i < warmup; i++) {
        split_layernorm_gelu(d_input, d_gamma, d_beta, d_output,
                             d_mean, d_variance, rows, cols, eps);
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        split_layernorm_gelu(d_input, d_gamma, d_beta, d_output,
                             d_mean, d_variance, rows, cols, eps);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    float t_split = ms / iters;

    // --- 融合版本 ---
    for (int i = 0; i < warmup; i++) {
        fused_layernorm_gelu(d_input, d_gamma, d_beta, d_output, rows, cols, eps);
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        fused_layernorm_gelu(d_input, d_gamma, d_beta, d_output, rows, cols, eps);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    float t_fused = ms / iters;

    // 有效带宽估算：理论最小访问量 = 读 input 1 次 + 写 output 1 次
    double min_bytes = 2.0 * N * sizeof(float);

    printf("=== performance comparison (%d x %d, Take the average of %d iterations) ===\n", rows, cols, iters);
    printf("Split (3 kernels): %.4f ms  | Effective Bandwidth %.1f GB/s\n",
           t_split, min_bytes / (t_split * 1e-3) / 1e9);
    printf("Fused (1 kernel ): %.4f ms  | Effective Bandwidth %.1f GB/s\n",
           t_fused, min_bytes / (t_fused * 1e-3) / 1e9);
    printf("Speedup ratio: %.2fx\n", t_split / t_fused);

    // 清理
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    delete[] h_input; delete[] h_gamma; delete[] h_beta;
    delete[] h_out_split; delete[] h_out_fused; delete[] h_ref;
    cudaFree(d_input); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_output); cudaFree(d_mean); cudaFree(d_variance);

    return 0;
}