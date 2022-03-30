#pragma once
/**
 * @file expansion_contraction.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-28
 * 
 * Based on indirect_copy.cu 
 * Meant to test different amounts of "expansion" and "contraction" as
 * defined with respect to reads and writes that a kernel does. 
 * Only tests basic copy and/or reduce. No complex arithmetic chains used. 
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>
#include <indices_generation.h>
#include <kernels/indirect/indirect_copy.cu>

#define DEBUG
// #define DEBUG_LEVEL1

using std::string;
using std::stringstream;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;

template<typename vt, typename it, int degree_of_contraction>
__forceinline__ __host__ __device__        
void kernel_contraction(uint idx, vt* in, vt* out, it* indices){
    it indxs[degree_of_contraction];

    uint indices_start_idx = idx + blockIdx.x * blockDim.x * (degree_of_contraction-1) + (threadIdx.x / warpSize) * warpSize * (degree_of_contraction-1);
    for(int i=0; i < degree_of_contraction; ++i) {
        indxs[i] = indices[indices_start_idx + warpSize*i];
    }

    vt arr[degree_of_contraction];
    for(int i=0; i < degree_of_contraction; ++i) {
        arr[i] = in[indxs[i]];
    }

    vt tmp = 0;
    for(int i=0; i < degree_of_contraction; ++i) {
        tmp += arr[i];
    }
    out[idx]  = tmp;
}

template<typename vt, typename it, int reads_per_8_writes, int stream_size>
__global__        
void kernel_for_regs_expansion_contraction(uint idx, vt* in, vt* out, it* indices){
    extern __shared__ int dummy[];
    if constexpr(reads_per_8_writes > 8) {
        kernel_contraction<vt, it, reads_per_8_writes/8>(idx, in, out, indices);
    } else {
        kernel_indirect_copy<vt, it, 8/reads_per_8_writes>(idx, in, out, indices);    
    }
}

template<typename vt, typename it, int reads_per_8_writes, int stream_size>
struct ExpansionContractionContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        // name = "Array_Copy";
        int N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        // Setup inputs/outputs, both data and indicies
        // Can be still in vector form, but is easier to identify explicitly with names   
        // Example
        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vector<it> & indices = super::host_indices[0];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];
        it* & d_indices = super::device_indices_ptrs[0];
        

        // int data_reads_per_element = 1; // Actual number 
        // int writes_per_element = 1; // Actual number
        // int index_reads_per_element = 1; // Actual number

        int degree_of_expansion = -1;
        int degree_of_contraction = -1;

        struct gpu_ctx {
            vt * gpu_in; 
            vt * gpu_out;
            it * gpu_indices;

            __device__        
            void operator() (uint idx){
                extern __shared__ int dummy[];
                if constexpr(reads_per_8_writes > 8) {
                    kernel_contraction<vt, it, reads_per_8_writes/8>(idx, gpu_in, gpu_out, gpu_indices);
                } else {
                    kernel_indirect_copy<vt, it, 8/reads_per_8_writes>(idx, gpu_in, gpu_out, gpu_indices);    
                }
            }
        } ctx ;

        ExpansionContractionContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 1, n, bs, dev_ctx, shd_mem_alloc) {
            this->name = "ExpansionContraction";
            unsigned long long total_size = 2 * N; 
            
            // max = 64 * 8 * 108 = 55296 ~ 0.6% of 8388608 elements of A100
            int minimum_division = stream_size * dev_ctx->props_.multiProcessorCount; 

            if(reads_per_8_writes == 8) {
                this->input_size = N; 
                this->output_size = N; 
                this->indices_size = N;
            } else if(reads_per_8_writes < 8) {
                degree_of_expansion = 8 / reads_per_8_writes;

                float tmp = float(total_size) / float(1 + 1.0/degree_of_expansion); 
                unsigned long long write_size = ( ((unsigned long long)tmp + minimum_division - 1) / minimum_division) * minimum_division; 

                this->input_size = write_size / degree_of_expansion; 
                this->output_size = write_size; 
                this->indices_size = write_size;
            } else if(reads_per_8_writes > 8) {
                degree_of_contraction = reads_per_8_writes / 8;

                float tmp = float(total_size) / float(1 + degree_of_contraction); 
                unsigned long long write_size = ( ((unsigned long long)tmp + minimum_division - 1) / minimum_division) * minimum_division; 

                this->input_size = write_size * degree_of_contraction; 
                this->output_size = write_size; 
                this->indices_size = write_size * degree_of_contraction;
            }
            this->Gsz = this->output_size / this->Bsz;
            if(degree_of_expansion > 0) {
                this->Gsz /= degree_of_expansion;
            }
            assert(this->Gsz > 0);

            this->total_data_reads = this->input_size;
            this->total_index_reads = this->indices_size;
            this->total_writes = this->output_size;
        }
        ~ExpansionContractionContext(){}

        void init_inputs(bool& pass) override {
            for(int i=0; i<this->input_size; ++i){
                in.push_back(i);
            }
            for(int i=0; i<this->output_size; ++i){
                out.push_back(0);
            }
        }

        void init_indices(bool& pass) override {
            bool debug = false;
// #ifdef DEBUG
//             debug = true;
// #endif
            for(int i=0; i<this->indices_size; ++i){
                indices.push_back(0);
            }
            if( expansion_contraction_indices(indices.data(), this->indices_size, reads_per_8_writes, stream_size, debug) != 0) {
                cerr << "Failed to generate indices!"; 
                this->okay = false;
            }
        }

        void set_dev_ptrs() override {
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
            ctx.gpu_indices = d_indices;
        }

        void output_config_info() override {
            cout << this->name << " with : "
                 << " reads/write =" << float(reads_per_8_writes)/8.0
                 << " stream size=" << stream_size 
                //  << " ILP=" << ILP 
                 << " occupancy=" << this->get_occupancy() <<  endl;
        }

        bool local_check_result() override {
            bool pass = true;
            unsigned long long i=0;
            if(degree_of_contraction > 0){
                for(i=0; i<this->output_size; ++i){
                    unsigned long long warp_id = i / warp_size;
                    vt tmp=0;
                    for(int j=0; j<degree_of_contraction; ++j) {
                        tmp += indices[warp_id * degree_of_contraction * warp_size + warp_size * j + i%warp_size];
                    }
                    if(tmp != out[i]) {
                        pass = false; 
                        break;
                    }

                }
            } else {
                for(i=0; i<this->output_size; ++i){
                    if(in[indices[i]] != out[i]){
                        pass = false;
                        break;
                    }
                }
            }
            if (!pass) {
                cout << "Validation Failed at " << i << ":" << " idx=" << indices[i] << " out="<< out[i] << endl;
                
                cout << "Debug dump of in and out array: " << endl;
                cout << std::setw(10) << "IDX" << "  |" << std::setw(10) << "OUT" << endl; 
                int output_size = 20;
                unsigned long long j = max((int)0, (int)(i - output_size/2));
                for(int k=0; k < output_size; ++k, ++j) { 
                    cout << std::setw(10) << indices[j] <<"  |" << std::setw(10) << out[j] << endl; 
                }    
#ifdef DEBUG
#ifdef DEBUG_LEVEL1
                cout << "Actual output of " << this->name <<":"<<endl;
                for(i=0; i < 1024*2; i+= 1024 ) {
                    for(int j=0; j<64; ++j) {
                        cout << "  " << i+j << ":" << (unsigned long long)out[i+j]; 
                        if(j%16==15) cout << endl;
                        if(j%32==15) cout << "\t";
                    }
                }
                for(i=0; i<128; ++i) cout <<  "-";
                cout << endl;
                for(i=this->output_size - (1024 + 64); i < this->output_size; i+= 1024 ) {
                    for(int j=0; j<64; ++j) {
                        cout << "  " << i+j << ":" <<  (unsigned long long)out[i+j]; 
                        if(j%16==15) cout << endl;
                        if(j%32==15) cout << "\t";
                    }
                }
                for(i=1024; i < this->output_size; ++i) {
                    if(out[i] == 0) {
                        cout << "Failed at " << i << ":" << out[i] << endl;
                        for(int j=-32; j<32; ++j) {
                            cout << "  " << i+j << ":" <<  (unsigned long long)out[i+j]; 
                            if(j%16==15) cout << endl;
                        }
                        pass = false; 
                        break;
                    }
                }
#endif
#endif
            }
            return pass;
        }

        // No change
        float local_execute() override {
            return local_execute_template<gpu_ctx>(this->output_size, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        // No change
        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *kernel_for_regs_expansion_contraction<vt,it,reads_per_8_writes,stream_size>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

    string get_extra_config_parameters() override { return "reads_per_write,stream_size";}
    string get_extra_config_values() override { 
        stringstream out; 
        out << to_string(float(reads_per_8_writes)/8.0) << "," << to_string(stream_size);
        return out.str();
    }

};
