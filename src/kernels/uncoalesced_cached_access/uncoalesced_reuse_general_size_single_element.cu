#pragma once
/**
 * @file uncoalesced_reuse_general_size_single_element.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from UncoalescedReuseGeneralContext; testing for coalescing's impact on cache 
 * @version 0.1
 * @date 2022-03-02
 * 
 * Meant to only test single load performance.
 * Performance is still in the only used loads; this may need to change?
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <locale>
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


template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts, int shuffle_size>
__forceinline__ __host__ __device__        
void uncoalesced_reuse_general_single_kernel(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){

    unsigned int Sz = shuffle_size; 
    unsigned int shuffle_b_idx = idx / Sz;
    unsigned int shuffle_t_idx = idx % Sz;

    unsigned int num_warps = shuffle_size / 32;
    
    // Preload data
    if constexpr(preload_for_reuse) {
        vt tmp = gpu_in[idx];
        if(tmp < 0) return; // all values should be > 0; this is just to ensure this write is not removed
    }

    int start_idx = shuffle_b_idx * Sz;

    unsigned long long access_idx;
    if constexpr(!avoid_bank_conflicts) {
        access_idx = ( shuffle_t_idx % num_warps) * 32 + shuffle_t_idx / num_warps + start_idx;
    } else {
        access_idx = ( (shuffle_t_idx % 32) * 32 + (shuffle_t_idx % 32 + shuffle_t_idx / num_warps ) % 32) % Sz + start_idx;
    }

    gpu_out[idx] = gpu_in[access_idx];
}


template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts, int shuffle_size>
__global__        
void kernel_for_regs_reuse_single(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
        extern __shared__ int dummy[];
        uncoalesced_reuse_general_single_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts, shuffle_size>(idx, gpu_in, gpu_out, N);
}

template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts, int shuffle_size>
struct UncoalescedReuseGeneralSingleElementContext : public KernelCPUContext<vt, it> {
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
                uncoalesced_reuse_general_single_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts, shuffle_size>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        UncoalescedReuseGeneralSingleElementContext(int n, int bs, GpuDeviceContext* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            this->name = "UncoalescedReuseGeneralSingleElement"; 
            assert(this->Gsz > 0);
            // if constexpr(preload_for_reuse) {
            //     data_reads_per_element += 1;
            // }

            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~UncoalescedReuseGeneralSingleElementContext(){}

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
                 << " occupancy=" << this->get_occupancy() <<  endl;
        }

        float local_execute() override {
            return local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        bool local_check_result() override {
            bool pass = true;
            int num_warps = shuffle_size / 32;
            unsigned long long global_tidx = 0;
            for(int i=0; i < this->N / shuffle_size; ++i) {
                int start_idx = i * shuffle_size;
                for (int j=0; j < shuffle_size; ++j){
                    global_tidx = start_idx + j;

                    unsigned int shuffle_t_idx = global_tidx % shuffle_size;
                    vt val = 0;
                    if constexpr(!avoid_bank_conflicts) {
                        val = in[( shuffle_t_idx % num_warps) * 32 + shuffle_t_idx / num_warps + start_idx];
                    } else {
                        val = in[( (shuffle_t_idx % 32) * 32 + (shuffle_t_idx % 32 + shuffle_t_idx / num_warps ) % 32) % shuffle_size + start_idx];

                    }
                    if (out[global_tidx] != val) {
                        cout << "Validation Failed at " << global_tidx << ": in="<<in[global_tidx] << " out="<< out[global_tidx] << endl;
                        pass = false;
                        break;
                    }
                }
                if(!pass) break;
            }

            if(!pass) {
                cout << "Debug dump of in and out array: " << endl;
                cout << std::setw(10) << "IN" << "  |" << std::setw(10) << "OUT " << endl; 
                int output_size = 10;
                unsigned long long j = max((int)0, (int)(global_tidx - output_size/2));
                for(int k=0; k < output_size; ++k, ++j) { 
                    cout << std::setw(10) << in[j] <<"  |" <<std::setw(10)<<out[j] << endl; 
                }
            }
            return pass;
        }

        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, kernel_for_regs_reuse_single<vt,it,preload_for_reuse,avoid_bank_conflicts,shuffle_size>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

    string get_extra_config_parameters() override { return "preload,avoid_bank_conflicts,shuffle_size";}
    string get_extra_config_values() override { 
        stringstream out; 
        out << bool_to_string(preload_for_reuse) << "," << bool_to_string(avoid_bank_conflicts) << "," << to_string(shuffle_size);
        return out.str();
    }

};
