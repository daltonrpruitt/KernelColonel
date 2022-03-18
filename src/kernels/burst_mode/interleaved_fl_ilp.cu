#pragma once
/**
 * @file interleaved_fl_ilp.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from InterleavedCopyFullLifeContext; testing for burst mode 
 * @version 0.1
 * @date 2022-03-09
 * 
 * New functionality to allow for use of instruction-level parallelism (ILP). 
 * Essentially, now a thread/warp can issue multiple reads or writes before moving 
 * to next thread/warp.
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>

using std::string;
using std::stringstream;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;

template<typename vt, typename it, int elements, int ILP>
__forceinline__ __host__ __device__        
void interleaved_fl_ilp_kernel(uint idx, vt* gpu_in, vt* gpu_out, unsigned long long N){

    // unsigned long long b_idx = blockIdx.x;
    // unsigned long long t_idx = threadIdx.x;
    // unsigned long long Bsz = blockDim.x;
    // unsigned long long Gsz = gridDim.x;
    
    // int block_life = N / gridDim.x / elements; 
    unsigned long long start_idx = blockIdx.x * blockDim.x * elements + threadIdx.x;
    uint cycle_offset = gridDim.x * blockDim.x * elements;
    // uint ILP_loop_offset = blockDim.x * ILP;

    vt vals[ILP];

    for(int i=0; i < N / ( gridDim.x * blockDim.x * ILP) ; ++i) {
        // uint x = i / elements;
        // uint y = i % elements;

        // for(int y=0; y < elements / ILP; ++y) {
            
            for(int k=0; k < ILP; ++k){
                uint x = (i * ILP + k) / elements;
                uint y = (i * ILP + k) % elements;
                unsigned long long data_idx = start_idx + cycle_offset * x + blockDim.x * y ;
                if(data_idx >= N) continue;
                vals[k] = gpu_in[data_idx];
            }
            
            for(int k=0; k < ILP; ++k){
                uint x = (i * ILP + k) / elements;
                uint y = (i * ILP + k) % elements;
                unsigned long long data_idx = start_idx + cycle_offset * x + blockDim.x * y ;
                gpu_out[data_idx] = vals[k];
                if(data_idx >= N) continue;
                // {
                //     gpu_out[idx] = data_idx;
                //     // return; 
                // }                    
            }
        // }
    }
}


template<typename vt, typename it, int elements, int ILP>
__global__        
void interleaved_fl_ilp_kernel_for_regs(uint idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
        extern __shared__ int dummy[];
        interleaved_fl_ilp_kernel<vt, it, elements, ILP>(idx, gpu_in, gpu_out, N);
}

template<typename vt, typename it, int elements, int ILP, bool match_ilp>
struct InterleavedFullLifeILPContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        // name = "Array_Copy";
        unsigned long long N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];

        int data_reads_per_element = 1;
        int index_reads_per_element = 0;
        int writes_per_element = 1;
        struct gpu_ctx {
            vt * gpu_in;
            vt * gpu_out;
            unsigned long long N;

            __device__        
            void operator() (uint idx){
                extern __shared__ int dummy[];
                interleaved_fl_ilp_kernel<vt, it, elements, ILP>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        // bool match_ilp;

        InterleavedFullLifeILPContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            assert(N % elements == 0);
            assert(N % ILP == 0);
            this->name = "InterleavedFullLifeILP";

            if(match_ilp) {
                float desired_occupany = 1.0 / float(ILP);
                int shdmem = this->get_sharedmemory_from_occupancy(desired_occupany);
                if (shdmem == -1) {
                    cerr << "Could not set sharedmem for occupancy!" << endl;
                    this->okay = false;
                    return;
                }
                this->shared_memory_usage = shdmem;
            }
            
            int occupancy_blocks = int(this->get_occupancy() * float(this->dev_ctx->props_.maxThreadsPerMultiProcessor)) / this->Bsz;
            cout << "Occupancy = " << this->get_occupancy() << endl;
            this->Gsz = this->dev_ctx->props_.multiProcessorCount * occupancy_blocks;
            if(N < (elements * this->Bsz * this->Gsz ) ) {
                cout << "Elements=" <<elements << " is too large!" << endl;
                this->okay = false;
                return;
            }
            assert(this->Gsz > 0);
           
            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~InterleavedFullLifeILPContext(){}

        void set_config_bool(bool val) override {
            // match_ilp = val;
        }

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
            cout << "InterleavedCopyFullLife with : "
                 << " Elements/cycle=" << elements 
                 << " ILP=" << ILP 
                 << " Blocks used="<< this->Gsz << endl;
        }

        float local_execute() override {
            return local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        bool local_check_result() override {
            bool pass = true;
            unsigned long long i = 0;
            for(; i<N; ++i){
                if(in[i] != out[i]){
                    cout << "Validation Failed at " << i << ": in="<<in[i] << " out="<< out[i] << endl;
                    pass = false;
                    break;
                }
            }

            if(!pass) {
                cout << "Debug dump of in and out array: " << endl;
                cout << std::setw(12) << "IN"<<"|" <<std::setw(12)<<"OUT " << endl; 
                int output_size = 10;
                unsigned long long j = max((int)0, (int)(i - output_size/2));
                for(int k=0; k < output_size; ++k, ++j) { 
                    cout << std::setw(12) << in[j] <<"|" << std::setw(12) << std::lround(out[j]) << endl; 
                }
            }
            return pass;
        }

        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *interleaved_fl_ilp_kernel_for_regs<vt,it,elements,ILP>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

    string get_extra_config_parameters() override { return "elements,ILP";}
    string get_extra_config_values() override { 
        stringstream out; 
        out << to_string(elements) << "," << to_string(ILP);
        return out.str();
    }
};
