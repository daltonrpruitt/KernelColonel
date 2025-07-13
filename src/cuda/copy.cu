#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>

template<typename gpu_data_t>
__global__
void copy(unsigned long long size, gpu_data_t data_struct) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    auto input = data_struct.input;
    auto output = data_struct.output;
    output[idx] = input[idx];
}
