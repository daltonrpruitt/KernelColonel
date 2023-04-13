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
