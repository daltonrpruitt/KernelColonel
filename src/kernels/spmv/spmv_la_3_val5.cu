#pragma once
/**
 * @file spmv_la_3_val5.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from SpmvKernelLAv2 (spmv_la_2_val4.cu)
 * @version 0.1
 * @date 2022-04-29
 * 
 * This SpMV kernel is meant to make use of an assumption 
 * of a constant valence of 5 assuming a tetrahedral mesh. 
 * To have consistent comparisons between different orderings, 
 * should only run matrices derived from actual tetrahedral
 * meshes, as these would conform to the assumption. 
 * A more relaxed version of this could be generated that would
 * allow for variable variance within some tolerance, but that 
 * would possible run into the same possible load balancing 
 * issues I saw for the SpmvKernelLAv1 in spmv_la_1.cu ...
 *
 * From the assumption of a valence of 5 (meaning 5 nonzeros in 
 * each row), I can process 6 rows (equations) for each warp, 
 * but this is more complicated that the SpmvKernelLAv2 kernel.
 * 
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
#include <local_basic_utils.h>
#include <crs_mat.h>
#include <kernels/spmv/spmv_base.cu>

// #define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;
namespace fs = std::filesystem;

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif


#define MAX_THREADS_PER_BLOCK 64
#define MIN_BLOCKS_PER_MP     2

template <typename it=int, typename vt=double, bool preload=false, bool include_preload_arith=false, int chunk_parts=1>
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
// __forceinline__ __host__ __device__ 
__global__ 
void spmv_kernel_latency_amortization_3(vt* product, CRSMat_gpu<it,vt> matrix, vt* vec) {
    __shared__ vt sums[30];
    unsigned int g_t_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = g_t_id / WARP_SIZE;
    if(warp_id*6 >= matrix.m) return;
    // unsigned int stride = 2 * 32 / sizeof(vt);
    unsigned int lane = threadIdx.x % WARP_SIZE; 
    // assume vector is preloaded into cache

#if __CUDA_ARCH__ >= 700
    unsigned int stride = 2 * 32 / sizeof(vt); // 2 sectors
#else
    unsigned int stride = 1 * 32 / sizeof(vt); // 1 sector
#endif

    // unsigned int row_id = warp_id;
    unsigned int start = matrix.offsets[warp_id*6*chunk_parts];
    unsigned int stop =  matrix.offsets[(warp_id+1)*6*chunk_parts];
    unsigned int vals_processed = stop - start; // should be equal to 32*chunk_parts always
    // if (lane == 0) { product[warp_id] = vals_processed; } return; 

    // can probably get rid of the above steps, since already know start/stop from warp_id

    // unsigned int chunk_parts = 2;
    unsigned int chunk_size = WARP_SIZE * chunk_parts;
    // int num_chunks = (vals_processed + chunk_size) / chunk_size;

    // vt t_sum = 0;
    //    for(int chunk=0; chunk < num_chunks; chunk++) {
    unsigned int local_start = start;// + chunk * chunk_size;

        unsigned int local_start_col_idx=0, local_stop_col_idx=0, cur_preload_start_idx=local_start_col_idx;

        if constexpr(include_preload_arith || preload) {
            local_start_col_idx = matrix.indices[local_start];
            local_stop_col_idx = max(matrix.indices[min(local_start + chunk_size, stop)-1], local_start_col_idx + 1);
            cur_preload_start_idx = local_start_col_idx;
        }
        if constexpr(include_preload_arith || preload) {
            while(cur_preload_start_idx < local_stop_col_idx) {
                if constexpr(preload){
                    force_global_load<vt>(vec, cur_preload_start_idx + lane*stride, local_stop_col_idx);
                }
                cur_preload_start_idx += WARP_SIZE * stride;
            }
        }
    
    
        // for(unsigned int part=0; part < chunk_parts; part++) {
            if(lane >= 30) return; // cannot work with last two threads in warp (except for preloading)
            unsigned int immediate_idx = local_start + part*WARP_SIZE + lane;
            if(immediate_idx >= stop) break;
            vt val = matrix.values[immediate_idx];
            it col = matrix.indices[immediate_idx];
            sums[lane] = val * vec[col];
        // }
	}
    // if (lane == 0) { product[warp_id] = local_stop_col_idx;} return;
    
    // Final parallel reduce
    // unsigned m = 0xffffffff;
    // for (int offset = 2; offset > 0; offset /= 2) {
    //     t_sum += __shfl_down_sync(m, t_sum, offset);
    // }

    // Only first 30 threads/lanes at this point !!!
    unsigned int logical_local_lane = lane % 5;
    if( (logical_local_lane < 4)  && (logical_local_lane % 2 == 0) ) {
        sums[lane] += sums[lane+1];
    } 
    if( logical_local_lane == 0 ) {
        sums[lane] += sums[lane+2];
        sums[lane] += sums[lane+4];
    }

    if (lane % 4 == 0) {
        product[warp_id * 6 + lane/5] = t_sum;
    }
    return;
}

template <typename it=int, typename vt=double, bool preload=false, bool include_preload_arith=false, int chunk_parts=1>
struct SpmvKernelLAv3 : SpmvKernel<it, vt, 5> {
   public:
    typedef SpmvKernel<it, vt, 5 > super;


  SpmvKernelLAv3(int bs, GpuDeviceContext* d_ctx, string mtx_filename, int shd_mem_alloc = 0) 
    : super(bs, d_ctx, mtx_filename, shd_mem_alloc) {
        this->name = "SpmvKernelLAv3";
        static_assert(!( preload && !include_preload_arith)); // if preload, must include arith (just to reduce test space complexity)
    }
    ~SpmvKernelLAv3() {}

    void output_config_info() override {
        cout << "SpMV Latency Amortization V3 (Val=5) with : "
                << "\n\t Bsz=" << this->Bsz 
                << "\n\t Blocks used ="<< this->Gsz
                << "\n\t matrix file="<< fs::path(this->matrix_filename).filename()
                << "\n\t occupancy=" << this->get_occupancy()
                << "\n\t preload=" << bool_to_string(preload)
                << "\n\t include_preload_arithmetic=" << bool_to_string(include_preload_arith)
                // << "\n\t chunk_parts=" << chunk_parts 
                << endl;
    }

    float local_execute() override {  
        if(this->dev_ctx->props_.major >= 7) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, 
                (void *) spmv_kernel_latency_amortization_3<it, vt, preload, include_preload_arith, chunk_parts>);
            int shmem = this->dev_ctx->props_.sharedMemPerMultiprocessor-1024-attr.sharedSizeBytes;
            cudaFuncSetAttribute(
                (void *) spmv_kernel_latency_amortization_3<it, vt, preload, include_preload_arith, chunk_parts>, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
            cudaFuncSetAttribute(
                (void *) spmv_kernel_latency_amortization_3<it, vt, preload, include_preload_arith, chunk_parts>, 
                cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            cudaPrintLastError();
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        
        cudaEventRecord(start);
        // dense_vector_cache_preload<<<preload_blocks, Bsz, shared_memory_usage>>>(gpu_vector, gpu_matrix.m);
        // cudaDeviceSynchronize();
        // cudaPrintLastError();
        spmv_kernel_latency_amortization_3<it,vt,preload,include_preload_arith,chunk_parts>
            <<<this->Gsz, this->Bsz, this->shared_memory_usage>>>(this->gpu_results, this->gpu_matrix, this->gpu_vector);
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
        cudaErrChk(cudaFuncGetAttributes(&funcAttrib, spmv_kernel_latency_amortization_3<it,vt,preload,include_preload_arith,chunk_parts>), "getting function attributes (for # registers)", pass);
        if (!pass) {
            this->okay = false;
            return;
        }
        this->register_usage = funcAttrib.numRegs;
    }

    string get_local_extra_config_parameters() override { 
        return "preload,include_preload_arith,chunk_parts";
    }
    
    string get_local_extra_config_values() { 
        stringstream out;
        out << bool_to_string(preload) << "," << bool_to_string(include_preload_arith) << "," << chunk_parts;
        return out.str();
    } 

};
