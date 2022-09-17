#pragma once
/**
 * @file uncoalesced_reuse_gen_single_ILP.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Based on UncoalescedReuseGeneralSingleElementContext; 
 *      testing for cache reducing impact of noncoalesced accesses  
 * @version 0.1
 * @date 2022-03-10
 * 
 * Adding ILP (instruction-level parallelism) to uncoalesced reuse kernel. 
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>
#include <local_basic_utils.h>

using std::string;
using std::stringstream;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;


#define SEPARATE_IDX_COMPUTATIONS_FROM_READS

template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts, int shuffle_size, int ILP>
__forceinline__ __host__ __device__        
void uncoalesced_reuse_gen_single_ilp_kernel(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
    // idx = blockIdx.x * blockDim.x + threadIdx.x; 

    // unsigned int Sz = shuffle_size; 
    unsigned int warp_size = 32; // I do not think the algorithm I have is acutally general enough for different warp sizes
    // unsigned int num_warps = shuffle_size / 32;
    unsigned int warps_per_shuffle = shuffle_size / warp_size;
    unsigned int warps_per_shuffle_scan = warps_per_shuffle / warp_size;
    // unsigned int scans_per_shuffle = warp_size;

    unsigned int warp_t_idx = threadIdx.x % warp_size;
    unsigned int logical_start_t_idx = idx + blockIdx.x * blockDim.x * (ILP-1);  

    for(int i=0; i < ILP; ++i) {

        // Preload data
        if constexpr(preload_for_reuse) {
            vt tmp = gpu_in[logical_start_t_idx + i*blockDim.x];
            if(tmp < 0) return; // all values should be > 0; this is just to ensure this write is not removed
        }

    }

#ifdef SEPARATE_IDX_COMPUTATIONS_FROM_READS
    it access_idxs[ILP];
    for(int i=0; i < ILP; ++i) {
        unsigned long long local_logical_t_idx = logical_start_t_idx + i*blockDim.x;
        unsigned int shuffle_block_idx = local_logical_t_idx / shuffle_size;
        // unsigned int shuffle_t_idx = local_logical_t_idx % shuffle_size;
        // unsigned int start_idx = shuffle_b_idx * Sz;
        unsigned int shuffle_warp_idx = ( local_logical_t_idx % shuffle_size ) / warp_size;
        unsigned int shuffle_scan_id = shuffle_warp_idx / warps_per_shuffle_scan;
        unsigned int shuffle_scan_warp_id = shuffle_warp_idx % warps_per_shuffle_scan;
        unsigned int scan_local_start_idx = shuffle_scan_warp_id * shuffle_size / warps_per_shuffle_scan;
        
        it shuffle_block_start_idx = shuffle_block_idx * shuffle_size;

        int warp_local_logical_t_idx_offset;
        if constexpr(!avoid_bank_conflicts) {
            warp_local_logical_t_idx_offset = ( shuffle_scan_id ) % warp_size + warp_t_idx*warp_size;
        } else {
            warp_local_logical_t_idx_offset = (warp_t_idx + shuffle_scan_id) % warp_size + warp_t_idx*warp_size;
        }
        it final_idx = shuffle_block_start_idx + scan_local_start_idx + warp_local_logical_t_idx_offset;

        access_idxs[i] = final_idx;
    }
#endif

    vt data_vals[ILP];

    for(int i=0; i < ILP; ++i) {
        
#ifdef SEPARATE_IDX_COMPUTATIONS_FROM_READS
        data_vals[i] = gpu_in[access_idxs[i]];
#else
        return;
        // unsigned long long local_logical_t_idx = logical_start_t_idx + i*blockDim.x;
        // unsigned int shuffle_b_idx = local_logical_t_idx / Sz;
        // unsigned int shuffle_t_idx = local_logical_t_idx % Sz;
        // unsigned long long access_idx;
        // unsigned int start_idx = shuffle_b_idx * Sz;
        // unsigned long long access_idx; 
        // if constexpr(!avoid_bank_conflicts) {
        //     access_idx = ( shuffle_t_idx % num_warps) * 32 + shuffle_t_idx / num_warps + start_idx;
        // } else {
        //     access_idx = ( (shuffle_t_idx % 32) * 32 + (shuffle_t_idx % 32 + shuffle_t_idx / num_warps ) % 32) % Sz + start_idx;
        // }
        // data_vals[i] = gpu_in[access_idx];
#endif

    }

    for(int i=0; i < ILP; ++i) {
        gpu_out[logical_start_t_idx + i*blockDim.x] = data_vals[i];
    }
}


template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts, int shuffle_size, int ILP>
__global__        
void kernel_for_regs_reuse_gen_single_ilp(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
        extern __shared__ int dummy[];
        uncoalesced_reuse_gen_single_ilp_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts, shuffle_size, ILP>(idx, gpu_in, gpu_out, N);
}

template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts, int shuffle_size, int ILP>
struct UncoalescedReuseGenSingleILPContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        unsigned long long N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];

        int data_reads_per_element = 1; // only one valualbe read?
        int index_reads_per_element = 0;
        int writes_per_element = 1;
        struct gpu_ctx {
            vt * gpu_in;
            vt * gpu_out;
            unsigned long long N;

            __device__        
            void operator() (unsigned int idx){
                extern __shared__ int dummy[];
                uncoalesced_reuse_gen_single_ilp_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts, shuffle_size, ILP>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        UncoalescedReuseGenSingleILPContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            this->name = "UncoalescedReuseGenSingleILP"; 
            this->Gsz /= ILP;
            assert(this->Gsz > 0);
            // if constexpr(preload_for_reuse) {
            //     data_reads_per_element += 1;
            // }

            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~UncoalescedReuseGenSingleILPContext(){}

        void init_inputs(bool& pass) override {
            for(int i=0; i<N; ++i){
                in.push_back(i);
                out.push_back(0);
            }
        }

        void init_indices(bool& pass) override {}

        void set_dev_ptrs() override {
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
            ctx.N = N;
        }

        void output_config_info() override {
            cout << this->name << " with : "
                 <<" preloading?=" << preload_for_reuse 
                 << " avoiding bank conflicts?=" << avoid_bank_conflicts 
                 << " shuffle size=" << shuffle_size 
                 << " ILP=" << ILP 
                 << " occupancy=" << this->get_occupancy() <<  endl;
        }

        float local_execute() override {
            return local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        bool local_check_result() override {
            bool pass = true;
            // int num_warps = shuffle_size / 32;
            it global_t_idx = 0;
            int warps_per_shuffle = shuffle_size / warp_size;
            int warps_per_shuffle_scan = warps_per_shuffle / warp_size;
            int scans_per_shuffle = warp_size;
            for(int shuffle_block_idx=0; shuffle_block_idx < N / shuffle_size; ++shuffle_block_idx) {
                if(!pass) break;
                int shuffle_block_start_idx = shuffle_block_idx * shuffle_size;
            
                for(int shuffle_scan_id=0; shuffle_scan_id<scans_per_shuffle; shuffle_scan_id++) {
                    if(!pass) break;
                    
                    for(int shuffle_scan_warp_id=0; shuffle_scan_warp_id<warps_per_shuffle_scan; shuffle_scan_warp_id++) {
                        if(!pass) break;
                        it scan_local_start_idx = shuffle_scan_warp_id * shuffle_size / warps_per_shuffle_scan;

                        for(int warp_t_idx=0; warp_t_idx<warp_size; ++warp_t_idx) {
                            global_t_idx = shuffle_block_start_idx + (shuffle_scan_id * warps_per_shuffle_scan + shuffle_scan_warp_id)*warp_size + warp_t_idx; 
                            
                            int warp_local_idx_offset;
                            if constexpr(!avoid_bank_conflicts) {
                                warp_local_idx_offset = ( shuffle_scan_id ) % warp_size + warp_t_idx*warp_size;
                            } else {
                                warp_local_idx_offset = (warp_t_idx + shuffle_scan_id) % warp_size + warp_t_idx*warp_size;
                            }
                            
                            it final_idx = shuffle_block_start_idx + scan_local_start_idx + warp_local_idx_offset;
                            // indxs[global_t_idx] = final_idx;
                            if (out[global_t_idx] != in[final_idx]) {
                                cout << "Validation Failed at " << "t_idx=" << global_t_idx << ", accesses at "<< final_idx 
                                    << " : in="<<in[final_idx] << " out="<< out[global_t_idx] << endl;
                                pass = false;
                                break;
                            }

                            // if(output_sample) print_indices_sample(indxs, shuffle_size, idx);
                        }
                    }
                }
            }

            if(!pass) {
                cout << "Debug dump of in and out array: " << endl;
                cout << std::setw(10) << "IN" << "  |" << std::setw(10) << "OUT " << endl; 
                int output_size = 10;
                unsigned long long j = max((int)0, (int)(global_t_idx - output_size/2));
                for(int k=0; k < output_size; ++k, ++j) { 
                    cout << std::setw(10) << in[j] <<"  |" <<std::setw(10)<<out[j] << endl; 
                }
            }
            return pass;
        }

        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, kernel_for_regs_reuse_gen_single_ilp<vt,it,preload_for_reuse,avoid_bank_conflicts,shuffle_size,ILP>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

    string get_extra_config_parameters() override { return "preload,avoid_bank_conflicts,shuffle_size,ILP";}
    string get_extra_config_values() override { 
        stringstream out; 
        out << bool_to_string(preload_for_reuse) << "," << bool_to_string(avoid_bank_conflicts) << "," << to_string(shuffle_size) << "," << to_string(ILP);
        return out.str();
    }

};
