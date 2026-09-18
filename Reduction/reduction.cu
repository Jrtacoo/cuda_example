#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<algorithm>

namespace cg = cooperative_groups;

#define checkCuda(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char * file, int line)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error:%s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

const int M = 5;

__global__ void reduce0(float *d_in, float *d_out, int n)
{
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[tid] = (i < n) ? d_in[i] : 0.0f;
    __syncthreads();

    for(int s = 1; s < blockDim.x; s <<= 1)
    {
        if((tid % (2 * s)) == 0) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    
    if(tid == 0) d_out[blockIdx.x] = s_data[0];
}

__global__ void reduce1(float *d_in, float *d_out, int n)
{
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[tid] = (i < n) ? d_in[i] : 0.0f;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if(tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    
    if(tid < 32)
    {
        s_data[tid] += s_data[tid + 32]; __syncwarp();
        s_data[tid] += s_data[tid + 16]; __syncwarp();
        s_data[tid] += s_data[tid + 8]; __syncwarp();
        s_data[tid] += s_data[tid + 4]; __syncwarp();
        s_data[tid] += s_data[tid + 2]; __syncwarp();
        s_data[tid] += s_data[tid + 1]; __syncwarp();
    }

    if(tid == 0) d_out[blockIdx.x] = s_data[0];
}

__global__ void reduce2(float *d_in, float *d_out, int n)
{
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (i < n) ? d_in[i] : 0.0f;
    s_data[tid] = sum;
    __syncthreads();

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for(int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if(tid < s) 
        {
            sum += s_data[tid + s];
            s_data[tid] = sum;
        }
        __syncthreads();
    }

    if(cta.thread_rank() < 32)
    {
        if(blockDim.x >= 64) sum += s_data[tid + 32];
        for(int offset = 16; offset > 0; offset >>= 1)
            sum += tile32.shfl_down(sum, offset);
    }

    if(tid == 0) d_out[blockIdx.x] = sum;
}

template <unsigned int blockSize>
__global__ void reduce3(float *d_in, float *d_out, int n)
{
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int gridsize = blockSize * 2 * gridDim.x;
    int i = blockIdx.x * blockSize * 2 + tid;
    float sum = 0;

    while(i < n)
    {
        sum += d_in[i];
        if(i + blockSize < n) sum += d_in[i + blockSize];
        i += gridsize;
    }

    s_data[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { s_data[tid] = sum = sum + s_data[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { s_data[tid] = sum = sum + s_data[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { s_data[tid] = sum = sum + s_data[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
        if (blockSize >=  64) { sum += s_data[tid + 32]; }
        for (int offset = 16; offset > 0; offset /= 2)
            sum += tile32.shfl_down(sum, offset);
    }
    if (tid == 0) d_out[blockIdx.x] = sum;
}

template <unsigned int blockSize>
__global__ void reduce4(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + tid;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        if (blockSize >=  64) sdata[tid] += sdata[tid + 32]; __syncwarp();
        if (blockSize >=  32) sdata[tid] += sdata[tid + 16]; __syncwarp();
        if (blockSize >=  16) sdata[tid] += sdata[tid +  8]; __syncwarp();
        if (blockSize >=   8) sdata[tid] += sdata[tid +  4]; __syncwarp();
        if (blockSize >=   4) sdata[tid] += sdata[tid +  2]; __syncwarp();
        if (blockSize >=   2) sdata[tid] += sdata[tid +  1]; __syncwarp();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================
// bench_kernel: 对 reduce3 单独使用 numBlocksReduce3
// ============================================
double bench_kernel(int kernel_id, int blockSize, float *d_idata, float *d_odata,
                    int numBlocks, int numBlocksReduce3, int n, int iters, int sharedMem)
{
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    int launchBlocks = (kernel_id == 3) ? numBlocksReduce3 : numBlocks;

    for(int i = 0; i < 3; i++) 
    {
        switch (kernel_id){
            case 0: reduce0<<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 1: reduce1<<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 2: reduce2<<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 3: reduce3<256><<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 4: reduce4<256><<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
        }
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for(int i = 0; i < iters; i++)
    {
        switch (kernel_id){
            case 0: reduce0<<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 1: reduce1<<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 2: reduce2<<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 3: reduce3<256><<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 4: reduce4<256><<<launchBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
        }
    }
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    return ms / iters;
}

float check_result(int kernel_id, float *h_odata, float *d_odata, int numBlocks, int numBlocksReduce3)
{
    int launchBlocks = (kernel_id == 3) ? numBlocksReduce3 : numBlocks;
    checkCuda(cudaMemcpy(h_odata, d_odata, launchBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float final_sum = 0;
    for(int i = 0; i < launchBlocks; i++) final_sum += h_odata[i];

    return final_sum;
}

int main()
{
    int device; 
    cudaDeviceProp prop;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaGetDeviceProperties(&prop, device));

    //--------------------------------------initial-----------------------------------------
    int N = 1048576;  
    int maxThreads = (prop.maxThreadsPerBlock > 256) ? 256 : prop.maxThreadsPerBlock;
    
    // ---- reduce0/1/2: 需要足够多的 block 覆盖整个数组 ----
    int numBlocks = (N + maxThreads - 1) / (maxThreads);
    if (numBlocks > prop.maxGridSize[0]) numBlocks = prop.maxGridSize[0];
    
    // ---- reduce3: 固定少量 block，利用 grid-stride loop 遍历全数组 ----
    int numBlocksReduce3 = prop.multiProcessorCount * 32;
    if (numBlocksReduce3 < 128) numBlocksReduce3 = 128;
    
    int maxNumBlocks = std::max(numBlocks, numBlocksReduce3);
    int sharedMem = maxThreads * sizeof(float);
    int iters = 100;

    printf("Device: %s\n", prop.name);
    printf("N = %d, blockSize = %d\n", N, maxThreads);
    printf("reduce0/1/2/4 numBlocks = %d\n", numBlocks);
    printf("reduce3 numBlocks     = %d (fixed, SM=%d)\n\n", numBlocksReduce3, prop.multiProcessorCount);

    //--------------------------------------malloc--------------------------------------------
    float *d_idata, *d_odata;
    float *h_idata = (float*)malloc(N * sizeof(float));
    float *h_odata = (float*)malloc(maxNumBlocks * sizeof(float));
    checkCuda(cudaMalloc(&d_idata, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_odata, maxNumBlocks * sizeof(float)));

    for (int i = 0; i < N; i++) h_idata[i] = 1.0f;

    checkCuda(cudaMemcpy(d_idata, h_idata, N * sizeof(float), cudaMemcpyHostToDevice));

    //--------------------------------------test-----------------------------------------------
    double t[M];
    float result[M];
    for(int i = 0; i < M; i ++)
    {
        t[i] = bench_kernel(i, maxThreads, d_idata, d_odata, numBlocks, numBlocksReduce3, N, iters, sharedMem);
        result[i] = check_result(i, h_odata, d_odata, numBlocks, numBlocksReduce3);
    }
    
    //------------------------------------verification------------------------------------------
    printf("Kernel   Time(ms)   Bandwidth(KB/s)   Result\n");
    printf("------------------------------------------------\n");
    for(int i = 0; i < M; i ++)    printf("reduce%d  %8.4f   %8.2f          %.1f (expected: %d)\n", i, t[i], (N * sizeof(float)) / (t[i] * 1e3 * pow(2, 10)), result[i], N);
    
    //-------------------------------------free------------------------------------------------
    free(h_idata);
    free(h_odata);
    checkCuda(cudaFree(d_idata));
    checkCuda(cudaFree(d_odata));

    return 0;
}