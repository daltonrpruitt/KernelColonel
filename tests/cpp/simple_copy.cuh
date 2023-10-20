/**
 * @brief Test file for experimenting with Jitify's file inclusion capability
 * 
 */


template<typename value_t, typename index_t> 
struct SimpleKernelData_gpu_data_s
{
    value_t* input = nullptr;
    value_t* output = nullptr;
    index_t* indices = nullptr;
};

template<typename vt, typename it>
__global__
void simple_copy_kernel(unsigned int N, SimpleKernelData_gpu_data_s<vt,it> gpu_data) {
    for( int i=0; i<N; ++i ) {
        gpu_data.output[i] = gpu_data.input[i];
        // if(i<10) printf("At i=%d input[i]=%f, output[i]=%f\n",i, gpu_data.input[i], gpu_data.output[i]);
    }
}
