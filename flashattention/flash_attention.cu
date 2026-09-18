// flash_attn_v1_fwd.cu —— 教学版：每个线程负责 Q 块中的一行
// grid = (cdiv(N, B_r), batch * heads), block = (B_r)   // 32 线程 = 1 warp

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

constexpr int B_r = 32;      // Q 行块
constexpr int B_c = 32;      // K/V 列块
constexpr int D   = 64;      // head_dim（编译期固定，便于寄存器分配）

extern "C" __global__ void flash_attn_v1_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int N, float scale)
{
    __shared__ float sQ[B_r][D];
    __shared__ float sK[B_c][D];
    __shared__ float sV[B_c][D];

    const int row_l = threadIdx.x;                    // 本线程负责的 Q 行（局部）
    const int g_row = blockIdx.x * B_r + row_l;       // 全局行
    const int bh    = blockIdx.y;
    const long base = (long)bh * N * D;

    // 1) 加载 Q 块（常驻 SRAM）
    if (g_row < N)
        for (int c = 0; c < D; c++) sQ[row_l][c] = Q[base + (long)g_row * D + c];
    else
        for (int c = 0; c < D; c++) sQ[row_l][c] = 0.f;
    __syncthreads();

    // 2) 在线 softmax 状态
    float m = -FLT_MAX;          // 运行行最大值
    float l = 0.f;               // 运行分母
    float acc[D];                // O 的未归一化累加器（寄存器）
    for (int c = 0; c < D; c++) acc[c] = 0.f;

    // 3) 遍历 K/V 列块
    for (int j = 0; j < N; j += B_c) {

        // 3a) warp 协作加载 K_j, V_j
        for (int idx = row_l; idx < B_c * D; idx += B_r) {
            int r = idx / D, c = idx % D;
            int g_col = j + r;
            sK[r][c] = (g_col < N) ? K[base + (long)g_col * D + c] : 0.f;
            sV[r][c] = (g_col < N) ? V[base + (long)g_col * D + c] : 0.f;
        }
        __syncthreads();

        // 3b) 本线程算自己这一行的 S 分块: s[t] = Q_row · K_t^T * scale
        float s[B_c];
        float m_tile = -FLT_MAX;
        for (int t = 0; t < B_c; t++) {
            float dot = 0.f;
            for (int c = 0; c < D; c++)
                dot += sQ[row_l][c] * sK[t][c];
            s[t] = (j + t < N) ? dot * scale : -FLT_MAX;   // 尾部 mask
            m_tile = fmaxf(m_tile, s[t]);
        }

        // 3c) 在线 softmax 更新
        float m_new = fmaxf(m, m_tile);
        float alpha = __expf(m - m_new);          // 旧累加器的修正因子
        float p[B_c], l_tile = 0.f;
        for (int t = 0; t < B_c; t++) {
            p[t] = (s[t] == -FLT_MAX) ? 0.f : __expf(s[t] - m_new);
            l_tile += p[t];
        }
        l = alpha * l + l_tile;

        // 3d) 更新输出累加器: acc = alpha * acc + P @ V
        for (int c = 0; c < D; c++) {
            float pv = 0.f;
            for (int t = 0; t < B_c; t++)
                pv += p[t] * sV[t][c];
            acc[c] = alpha * acc[c] + pv;
        }
        m = m_new;
        __syncthreads();                          // 准备覆写 sK/sV
    }

    // 4) 归一化并写回
    if (g_row < N) {
        float inv = (l > 0.f) ? 1.f / l : 0.f;
        for (int c = 0; c < D; c++)
            O[base + (long)g_row * D + c] = acc[c] * inv;
        LSE[bh * N + g_row] = (l > 0.f) ? m + __logf(l) : -FLT_MAX;  // 反向用
    }
}