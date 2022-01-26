// kernel_test.cu
// Testing simple kernel

#include <cuda.h>


template<typename vt>
__global__
void array_copy(vt * in, vt * out, int size){
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >=size) return;
  out[idx] = in[idx];   
}
