#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      FAIL() << "Experienced above CUDA error!";                          \
    }                                                                     \
  } while (0)


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorName(code), file, line);
      fprintf(stderr," full err: %s\n", cudaGetErrorString(code));
      std::cout << "Error code " << code << std::endl;
      if (abort) exit(code);
   }
}
