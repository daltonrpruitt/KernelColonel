#pragma once
/**
 * @file spmv_preload.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from TemplateKernelContext
 * @version 0.1
 * @date 2022-02-08
 * 
 * This application is for performing sparse matrix-vector 
 * multiplication (SpMV) operations.
 *
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <crs_mat.h>
#include <kernel_context.cu>

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;

template <typename it=int, typename vt=double, int ILP = 1>
// __forceinline__ __host__ __device__
__global__ 
void dense_vector_cache_preload(vt* vector, int m) {
    uint g_t_id = blockIdx.x * blockDim.x + threadIdx.x;
    // uint warp_id = g_t_id / warpSize;
    uint stride = 2 * 32 / sizeof(vt);

    // assume m % stride == 0
    if (g_t_id < m / stride) {
        vt tmp_vec;  // = vector[g_t_id*stride];
        asm volatile("ld.global.f64 %0, [%1];"
                     : "=d"(tmp_vec) : "l"((double *)(vector + g_t_id * stride)));
        //if(tmp_vec == -0.0001) return;  // not needed if asm volatile (?)
    }
    return;
}

template <typename it=int, typename vt=double, int ILP = 1>
// __forceinline__ __host__ __device__ 
__global__ 
void spmv_kernel(vt* product, CRSMat_gpu matrix, vt* vector) { //}, int max_nz_row) {
    uint g_t_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint warp_id = g_t_id / warpSize;
    if(warp_id >= matrix.m) return;
    // uint stride = 2 * 32 / sizeof(vt);
    uint lane = threadIdx.x % warpSize; 
    // assume vector is preloaded into cache

    // uint row_id = warp_id;
    uint start = matrix.offsets[warp_id];
    uint stop =  matrix.offsets[warp_id + 1];
    uint vals_processed = stop - start;

    // if (lane == 0) {
    //     product[warp_id] =  (vals_processed / warpSize) + 1;
    // }
    // return;

    vt t_sum = 0;
    // assume n >> blockDim.x (5000 >> 128 or 256-ish)
    for (int i = 0; i < (vals_processed / warpSize) + 1; ++i) {
        if (lane + i * warpSize >= vals_processed) break;
        vt val = matrix.values[ start + i * warpSize + lane];
        it col = matrix.indices[start + i * warpSize + lane];
        t_sum += val * vector[col];
    }
    unsigned m = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        t_sum += __shfl_down_sync(m, t_sum, offset);
    }
    if (lane == 0) {
        product[warp_id] = t_sum;  // Single thread writing single value...
    }
    return;
}

std::vector<string> matrix_filenames= {
    "../../matrices/bcsstk13.mtx"
};


template <typename it=int, typename vt=double>
class SpmvKernel {
   public:
    // typedef KernelCPUContext<vt, it> super;
    string name = "spmv";
    bool initialized = false, okay = true;
    int Bsz, Gsz;


    size_t shared_memory_usage=0;
    int register_usage=-1;
    int max_blocks_simultaneous_per_sm=-1;
    device_context* dev_ctx;


    uint matrix_id;
    CRSMat host_matrix;
    CRSMat_gpu gpu_matrix;
    uint nnz;
    vector<double> host_vector;
    double *gpu_vector;
    vector<double> host_results;
    double *gpu_results;
    
    /*  
  //int N = super::N;
  //int Gsz = super::Gsz;
  //int Bsz = super::Bsz;
  
        vector<vt> & n = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];
  */
    int data_reads_per_element = -1;   // Actual number
    int index_reads_per_element = -1;  // Actual number
    int writes_per_element = -1;       // Actual number

    // struct gpu_ctx {
    //     // Params for kernel; set in set_dev_ptrs()
    //     CRSMat gpu_mat;
    //     vt* gpu_vector;
    //     vt* gpu_results;
    //     vt* gpu_out;

    //     __device__ void operator()(uint idx, vt* product, CRSMat matrix, vt* vector, int max_nz_row) {
    //         extern __shared__ int dummy[];
    //         kernel<vt, it>(idx, product,  matrix, vector, max_nz_row);
    //     }
    // } ctx;

    SpmvKernel(int n, int bs, device_context* d_ctx, int shd_mem_alloc = 0, int run_times = 25, int matrix_file_id=0) 
    : Bsz(bs), Gsz( (n+bs-1)/bs ), dev_ctx(d_ctx), shared_memory_usage(shd_mem_alloc), host_matrix(matrix_filenames[matrix_file_id]) {
        //  : super(2, 1, 2, n, bs, dev_ctx, shd_mem_alloc) {
        //this->name = "SpMV";
        // this->total_data_reads = N * data_reads_per_element;
        // this->total_index_reads = N * index_reads_per_element;
        // this->total_writes = N * writes_per_element;

        if(!check() ) {return;}


    }
    ~SpmvKernel() { uninit(); }

    void init(bool& pass)  {
        // Init Matrix here (host arrays/data)
        // host_matrix = CRSMat();
        if(host_matrix.nnz < 0) {
            pass = false; 
            return;
        }
        #ifdef DEBUG
        host_matrix.dump();
        #endif

        gpu_matrix.nnz = host_matrix.nnz;
        gpu_matrix.m   = host_matrix.m;
        gpu_matrix.n   = host_matrix.n;

        host_vector.reserve(host_matrix.m);
        for(int i=0; i < host_matrix.m; ++i) { host_vector[i] = (vt)i; }
        host_results.reserve(host_matrix.m);
        for(int i=0; i < host_matrix.m; ++i) { host_results[i] = (vt)0; }

        /* !!!!!!!!!!!!!!!!!!!!!!!*/
        // Allocate gpu arrays/copy host to gpu
        cudaErrChk(cudaMalloc((void **)&gpu_matrix.values,gpu_matrix.nnz * sizeof(double)),"gpu_matrix.values mem allocation", pass);
        if(pass){
            cudaErrChk(cudaMalloc((void **)&gpu_matrix.indices,gpu_matrix.nnz * sizeof(int)),"gpu_matrix.indices mem allocation", pass);
        }	
        if(pass){
            cudaErrChk(cudaMalloc((void **)&gpu_matrix.offsets,(gpu_matrix.m+1) * sizeof(int)),"gpu_matrix.offsets mem allocation", pass);
        }
        if(pass){
            cudaErrChk(cudaMalloc((void **)&gpu_vector,(gpu_matrix.m) * sizeof(double)),"gpu_vector mem allocation", pass);
        }
        if(pass){
            cudaErrChk(cudaMalloc((void **)&gpu_results,(gpu_matrix.m) * sizeof(double)),"gpu_vector mem allocation", pass);
        }
        
        // cudaMemCpy 
        if(pass) {
            cudaErrChk(
                cudaMemcpy(gpu_matrix.values, host_matrix.values,gpu_matrix.nnz * sizeof(double), cudaMemcpyHostToDevice),
                "copy host_matrix.values to gpu_matrix.values", pass
                );
        }
        
        if(pass){
            cudaErrChk(
                cudaMemcpy(gpu_matrix.indices, host_matrix.indices, gpu_matrix.nnz * sizeof(int), cudaMemcpyHostToDevice),
                "copy host_matrix.indices to gpu_matrix.indices", pass
                );
        }	
        if(pass){
            cudaErrChk(
                cudaMemcpy(gpu_matrix.offsets, host_matrix.offsets, (gpu_matrix.m+1) * sizeof(int), cudaMemcpyHostToDevice),
                "copy host_matrix.offsets to gpu_matrix.offsets", pass
                );
        }
        if(pass){
            cudaErrChk(
                cudaMemcpy(gpu_vector,host_vector.data(),gpu_matrix.m * sizeof(double), cudaMemcpyHostToDevice),
                "copy host_vector to gpu_vector", pass
                );
        }
        if(pass){
            cudaErrChk(
                cudaMemset(gpu_results, 0, gpu_matrix.m *sizeof(double)),
                "initializing gpu_results to 0", pass
                );
        }


        if (!pass) {
            cerr << "Could not initialize " << name << "!" << endl;
            return;
        }

        initialized = true;
        return;
    }

    void uninit() {
        // 8. Uninitialize data on device and host
        if (!initialized) {
            return;
        }

        // CudaFree gpu memory

        // Update this section!
        delete host_matrix.values; host_matrix.values = nullptr;
        delete host_matrix.indices; host_matrix.indices = nullptr;
        delete host_matrix.offsets; host_matrix.offsets = nullptr;
        vector<double>().swap(host_vector); 
        vector<double>().swap(host_results); 

        cudaFree(gpu_matrix.values); gpu_matrix.values = nullptr;
        cudaFree(gpu_matrix.indices); gpu_matrix.indices = nullptr;
        cudaFree(gpu_matrix.offsets); gpu_matrix.offsets = nullptr;
        cudaFree(gpu_vector); gpu_vector = nullptr;
        cudaFree(gpu_results); gpu_results = nullptr;

        initialized = false;
    }

    bool local_check_result() {
        // Perform matrix multiply here
        bool debug = false;
#ifdef DEBUG
        debug = true;
#endif
        vector<double> cpu_results;
        for(int i=0; i < host_matrix.m; ++i) { cpu_results.push_back(0); }
        
        int cur_row_start = 0;
        for(int i=0; i < host_matrix.m; i++) {
            if(debug && i < 10) cout << "Row " << i << ": "; 
            double result = 0;
            int start = host_matrix.offsets[i];
            int end   = host_matrix.offsets[i+1];
            int row_nz = end - start;
            for(int j=0; j < row_nz; j++) {
                int cur_pos = cur_row_start + j; 
                int col = host_matrix.indices[cur_pos];
                double val = host_matrix.values[cur_pos];
                double vec_val = host_vector[col];
                
                if(debug && i < 10 && j < 32) cout <<  std::setprecision(2) << val << "*" << std::setprecision(2) << vec_val << ", "; 
                result += val * vec_val; 

            }
            if(debug && i < 10) cout <<  "...  = " << std::setprecision(2) << result << endl; 
            cpu_results[i] = result;
            
            cur_row_start += row_nz;
        }

        for(int i=0; i < host_matrix.m; i++) {
            if(abs(cpu_results[i] - host_results[i])/cpu_results[i] > 1e-3 ) {
                cout << "Results are incorrect at " << i << ": host=" << std::setprecision(3) << cpu_results[i] 
                    << " device="  << std::setprecision(3) << host_results[i] << endl;
                int output_num = 10;
                int print_start = max(0, i-output_num / 2);
                int print_end = min(host_matrix.m, i+output_num / 2);
                for(int j=print_start; j < print_end; ++j) {
                    cout << "\t" << j << ": " << std::setprecision(3) << cpu_results[j] 
                                    << " =?= "  << std::setprecision(3) << host_results[j] << endl;  
                }
                return false;
            }
        
        }

        // Check resulting vector with output of kernel
/*
        for (int i = 0; i < N; ++i) {
            if (in[i] != out[i]) {
                cout << "Validation Failed at " << i << ": in=" << in[i] << " out=" << out[i] << endl;
                return false;
            }
        }
        */
        return true;
    }
    
    bool check() {
        if(!initialized) {
            bool pass = true;
            init(pass);
            if(!pass) return false;
        }
        if(execute() < 0) return false;
        float time = local_check_result(); 
        uninit();
        return time;
    }

    float local_execute() {
        //  Need to update since will be using two separate kernels.

        //  return  local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
   
        if(dev_ctx->props_.major >= 7) {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, (void *) dense_vector_cache_preload<int, double>);
            int shmem = dev_ctx->props_.sharedMemPerMultiprocessor-1024-attr.sharedSizeBytes;
            cudaFuncSetAttribute((void *) dense_vector_cache_preload<int, double>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
            cudaFuncSetAttribute((void *) dense_vector_cache_preload<int, double>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            cudaPrintLastError();

            cudaFuncGetAttributes(&attr, (void *) spmv_kernel<int, double>);
            shmem = dev_ctx->props_.sharedMemPerMultiprocessor-1024-attr.sharedSizeBytes;
            cudaFuncSetAttribute((void *) spmv_kernel<int, double>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
            cudaFuncSetAttribute((void *) spmv_kernel<int, double>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
            cudaPrintLastError();


        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        int stride = 4;
        if(dev_ctx->props_.major >= 7) {
            stride = 8;
        }
        int preload_blocks = host_matrix.m / (Bsz / warp_size * stride) + 1; 
        int spmv_blocks = host_matrix.m / (Bsz / warp_size) + 1;
        
        cudaEventRecord(start);
        dense_vector_cache_preload<<<preload_blocks, Bsz, shared_memory_usage>>>(gpu_vector, gpu_matrix.m);
        cudaDeviceSynchronize();
        cudaPrintLastError();
        spmv_kernel<<<spmv_blocks, Bsz, shared_memory_usage>>>(gpu_results, gpu_matrix, gpu_vector);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaPrintLastError();

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);

        return time; 
    }

    void copy_result_to_host(bool &pass) {
        cout << "Before copy to host :" ; 
        for(int i=0; i < 64; ++i) { cout << " " << std::setprecision(2) << host_results[i]; if(i %32 == 31) { cout << endl << "\t";} }
        cout << endl;

        cudaErrChk(cudaMemcpy(host_results.data(), gpu_results, host_matrix.m * sizeof(double), cudaMemcpyDeviceToHost),"copying results from gpu to host", pass);

        cout << "After copy to host :" ; 
        for(int i=0; i < 64; ++i) { cout << " " << std::setprecision(2) << host_results[i]; if(i %32 == 31) { cout << endl << "\t";} }
        cout << endl;

    }   

    float execute() {
        if(!okay) return -1.0;

        float time = local_execute();
        bool pass = true;
        copy_result_to_host(pass);
        
        if(!pass) {uninit(); okay = false; time = -1.0;}
        return time;
    }

    float run() {
        if(!initialized) {
            bool pass = true;
            init(pass);
            if(!pass) return -1.0;
        }
        return execute();
    }

    // No change
    void local_compute_register_usage(bool& pass) {
        // Kernel Registers
        struct cudaFuncAttributes funcAttrib;
        cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *spmv_kernel<vt, it>), "getting function attributes (for # registers)", pass);
        if (!pass) {
            this->okay = false;
            return;
        }
        this->register_usage = funcAttrib.numRegs;
    }

    void compute_max_simultaneous_blocks(bool& pass) {
        local_compute_register_usage(pass);
        if (!pass) {
            okay = false;
            return;
        }
        int due_to_block_size = (int)floor(dev_ctx->props_.maxThreadsPerMultiProcessor / Bsz);
        int due_to_registers = (int)floor(dev_ctx->props_.regsPerMultiprocessor / (register_usage * Bsz));
        max_blocks_simultaneous_per_sm = std::min({due_to_block_size,
                                                   due_to_registers, dev_ctx->props_.maxBlocksPerMultiProcessor});
    }

    vector<int> shared_memory_allocations() {
        return vector<int>(0);
        /*
        vector<int> alloc_amounts;
        bool pass = true;
        if (max_blocks_simultaneous_per_sm < 0) compute_max_simultaneous_blocks(pass);
        if (!pass) {
            okay = false;
            alloc_amounts.push_back(-1);
            return alloc_amounts;
        }
        int max_shd_mem_per_block = dev_ctx->props_.sharedMemPerBlock;
        int max_shd_mem_per_proc = dev_ctx->props_.sharedMemPerMultiprocessor;
        if (dev_ctx->props_.major == 8) {
            max_shd_mem_per_block = max_shd_mem_per_proc;
        }
        int min_blocks_due_to_shd_mem = max_shd_mem_per_proc / max_shd_mem_per_block;

        for (int i = min_blocks_due_to_shd_mem; i < max_blocks_simultaneous_per_sm; i *= 2) {
            int sm_alloc = std::min((max_shd_mem_per_proc / i) / 256 * 256, max_shd_mem_per_block);
            if (dev_ctx->props_.major == 8) {
                sm_alloc -= 1024;
            }
            if (std::find(alloc_amounts.begin(), alloc_amounts.end(), sm_alloc) == alloc_amounts.end()) {
                alloc_amounts.push_back(sm_alloc);
            }
        }
        return alloc_amounts;
        */
    }

    float get_occupancy() {
        bool pass = true;
        if (max_blocks_simultaneous_per_sm < 0) compute_max_simultaneous_blocks(pass);
        if (!pass) {
            okay = false;
            return -1.0;
        }

        int max_blocks_shared_mem;
        if (shared_memory_usage == 0) {
            max_blocks_shared_mem = dev_ctx->props_.maxBlocksPerMultiProcessor;
        } else {
            max_blocks_shared_mem = dev_ctx->props_.sharedMemPerMultiprocessor / shared_memory_usage;
        }
        int max_blocks_simul = std::min(max_blocks_simultaneous_per_sm, max_blocks_shared_mem);
        int num_threads_simul = max_blocks_simul * Bsz;
        return float(num_threads_simul) / float(dev_ctx->props_.maxThreadsPerMultiProcessor);
    }

    int get_sharedmemory_from_occupancy(float occupancy) {
        bool pass = true;
        if (max_blocks_simultaneous_per_sm < 0) compute_max_simultaneous_blocks(pass);
        if (!pass) {
            okay = false;
            return -1;
        }

        int blocks = float(dev_ctx->props_.maxThreadsPerMultiProcessor / Bsz) * occupancy;
        if (blocks > max_blocks_simultaneous_per_sm) {
            cerr << "Try to get occupancy higher than architecture allows!" << endl;
            return -1;
        }

        int max_shd_mem_per_block = dev_ctx->props_.sharedMemPerBlock;
        int max_shd_mem_per_proc = dev_ctx->props_.sharedMemPerMultiprocessor;
        if (dev_ctx->props_.major == 8) {
            max_shd_mem_per_block = max_shd_mem_per_proc;
        }

        int shdmem = max_shd_mem_per_proc / blocks;

        if (shdmem > max_shd_mem_per_block) {
            cerr << "Cannot set shared memory high enough to match occupancy of " << occupancy << "!" << endl;
            shdmem = max_shd_mem_per_block;
        }
        if (dev_ctx->props_.major == 8) {
            shdmem -= 1024;
        }
        return shdmem;
    }

    void print_register_usage() {
        bool pass = true;
        if (register_usage < 0) {
            local_compute_register_usage(pass);
        }
        if (!pass) {
            cerr << "Cannot get register usage for " << name << "!" << endl;
        } else {
            cout << name << " register usage = " << register_usage << endl;
        }
    }

    unsigned long long get_total_bytes_processed() {
        //  return ( total_data_reads+ total_writes)*sizeof(vt) +  total_index_reads*sizeof(it);
    }
};
