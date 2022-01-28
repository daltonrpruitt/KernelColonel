#pragma once
#include <cuda.h>
// #include <cuda_runtime_api.h>
#include <string>
using std::string;


void cudaErrChk(cudaError_t status, string msg, bool & pass)
{
  if (status != cudaSuccess)
  {
    printf("Error with %s!\n", msg.c_str());
    pass = false;
  }
}


