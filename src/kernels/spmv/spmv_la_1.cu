#pragma once
/**
 * @file spmv_la_1.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from TemplateKernelContext
 * @version 0.1
 * @date 2022-04-07
 * 
 * This SpMV kernel is to test using a version of latency amortization
 * that is fairly simple: loading in chunks of the vector into cache. 
 *
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <typeinfo>   // operator typeid
#include <filesystem>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <crs_mat.h>
#include <kernels/spmv/spmv_base.cu>

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;
namespace fs = std::filesystem;

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif

template <typename vt=double>
__forceinline__ __host__ __device__
void force_global_load(vt* arr, uint offset, uint m) {
    if(offset >= m) return; 
    vt tmp_vec;
    // https://www.cplusplus.com/reference/typeinfo/type_info/operator==/
    if constexpr(std::is_same<vt,double>()) {
        asm volatile("ld.global.f64 %0, [%1];"
                    : "=d"(tmp_vec) : "l"((vt *)(arr + offset)));
    } else if constexpr(std::is_same<vt,float>()) {
        asm volatile("ld.global.f32 %0, [%1];"
                    : "=f"(tmp_vec) : "l"((vt *)(arr + offset)));
    } else {
        static_assert(std::is_same<vt,double>()); // Know will fail at this point, but needed to get around ill-formed argument https://stackoverflow.com/questions/38304847/constexpr-if-and-static-assert
    }
    return;
}

#define MAX_THREADS_PER_BLOCK 64
#define MIN_BLOCKS_PER_MP     2

template <typename it=int, typename vt=double, int ILP = 1>
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
// __forceinline__ __host__ __device__ 
__global__ 
void spmv_kernel_latency_amortization_1(vt* product, CRSMat_gpu matrix, vt* vec) {
    uint g_t_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint warp_id = g_t_id / WARP_SIZE;
    if(warp_id >= matrix.m) return;
    // uint stride = 2 * 32 / sizeof(vt);
    uint lane = threadIdx.x % WARP_SIZE; 
    // assume vector is preloaded into cache

#if __CUDA_ARCH__ >= 700
    uint stride = 2 * WARP_SIZE / sizeof(vt); // 2 sectors
#else
    uint stride = 1 * WARP_SIZE / sizeof(vt); // 1 sector
#endif

    // uint row_id = warp_id;
    uint start = matrix.offsets[warp_id];
    uint stop =  matrix.offsets[warp_id + 1];
    uint vals_processed = stop - start;

    uint chunk_parts = 2;
    uint chunk_size = WARP_SIZE * chunk_parts;
    int num_chunks = (vals_processed + chunk_size) / chunk_size;

    vt t_sum = 0;
    for(int chunk=0; chunk < num_chunks; chunk++) {
        uint local_start = start + chunk * chunk_size;

        uint local_start_col_idx = matrix.indices[local_start];
        uint local_stop_col_idx = max(matrix.indices[min(local_start + chunk_size, stop)-1], local_start_col_idx + 1);
        uint cur_preload_start_idx = local_start_col_idx;
        while(cur_preload_start_idx < local_stop_col_idx) {
            force_global_load<vt>(vec, cur_preload_start_idx + lane*stride, local_stop_col_idx);
            cur_preload_start_idx += WARP_SIZE * stride;
        }
    
    
        for(uint part=0; part < chunk_parts; part++) {
            uint immediate_idx = local_start + part*WARP_SIZE + lane;
            if(immediate_idx >= stop) break;
            vt val = matrix.values[immediate_idx];
            it col = matrix.indices[immediate_idx];
            t_sum += val * vec[col];
        }
    }
    // if (lane == 0) { product[warp_id] = local_stop_col_idx;} return;
    
    // Final parallel reduce
    unsigned m = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        t_sum += __shfl_down_sync(m, t_sum, offset);
    }
    if (lane == 0) {
        product[warp_id] = t_sum;  // Single thread writing single value...
    }
    return;
}

template <typename it=int, typename vt=double>
struct SpmvKernelLAv1 : SpmvKernel<it, vt> {
   public:
    typedef SpmvKernel<it, vt> super;


    SpmvKernelLAv1(int bs, device_context* d_ctx, int shd_mem_alloc = 0, int matrix_file_id=0) 
    : super(bs, d_ctx, shd_mem_alloc, matrix_file_id) {
        this->name = "SpmvKernelLAv1";
    }
    ~SpmvKernelLAv1() {}

    void output_config_info() override {
        cout << "SpMV Latency Amortization V1 with : "
                << " Bsz=" << this->Bsz 
                << " Blocks used ="<< this->Gsz
                << " occupancy=" << this->get_occupancy() << endl;

    }

    float local_execute() override {  
        if(this->dev_ctx->props_.major >= 7) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (void *) spmv_kernel_latency_amortization_1<int, double>);
            int shmem = this->dev_ctx->props_.sharedMemPerMultiprocessor-1024-attr.sharedSizeBytes;
            cudaFuncSetAttribute((void *) spmv_kernel_latency_amortization_1<int, double>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
            cudaFuncSetAttribute((void *) spmv_kernel_latency_amortization_1<int, double>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            cudaPrintLastError();
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        
        cudaEventRecord(start);
        // dense_vector_cache_preload<<<preload_blocks, Bsz, shared_memory_usage>>>(gpu_vector, gpu_matrix.m);
        // cudaDeviceSynchronize();
        // cudaPrintLastError();
        spmv_kernel_latency_amortization_1<it,vt><<<this->Gsz, this->Bsz, this->shared_memory_usage>>>(this->gpu_results, this->gpu_matrix, this->gpu_vector);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaPrintLastError();

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);

        return time; 
    }

    // No change
    void local_compute_register_usage(bool& pass) override {
        // Kernel Registers
        struct cudaFuncAttributes funcAttrib;
        cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *spmv_kernel_latency_amortization_1<it, vt>), "getting function attributes (for # registers)", pass);
        if (!pass) {
            this->okay = false;
            return;
        }
        this->register_usage = funcAttrib.numRegs;
    }

    string get_local_extra_config_parameters() override { 
        return "";
    
    string get_local_extra_config_values() { 
        return "";
    } 

};
