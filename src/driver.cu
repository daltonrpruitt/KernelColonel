// driver.cpp
// simple driver file for kernel testing

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <kernel_test.cu> // std::cout

using vt = double;
using std::string;
using std::cout;
using std::endl;


void cudaErrChk(cudaError_t status, string msg, bool & pass)
{
  if (status != cudaSuccess)
  {
    printf("Error with %s!\n", msg.c_str());
    pass = false;
  }
}



int main() {

  int N = 32*32*32;
  int block_size = 128;
  int grid_size = N / block_size;

  vt *a, *d_a, *b, *d_b;
  a = new vt[N];
  b = new vt[N];
  // Initialize host arrays
  for (int i = 0; i < N; i++)
  {
    a[i] = (vt)4;
    b[i] = (vt)0;
  }

  bool pass = true;
  cudaErrChk(cudaMalloc((void **)&d_a, N * sizeof(vt)),
              "d_a mem allocation", pass);

  cudaErrChk(cudaMalloc((void **)&d_b, N * sizeof(vt)),
              "d_b mem allocation", pass);


  cudaErrChk(cudaMemcpy(d_a, a, N * sizeof(vt), cudaMemcpyHostToDevice),
              "copying a to device", pass);

    if (!pass)
    {
      cudaFree(d_a);
      cudaFree(d_b);
      return -1;
    }

  array_copy<vt><<<grid_size, block_size>>>(d_a, d_b, N);
  // cudaDeviceSynchronize();
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
  {
    printf("Error = %s\n", cudaGetErrorString(status));
    cudaFree(d_a);
    cudaFree(d_b);  
    return -1;
  }


  cudaErrChk(cudaMemcpy(b, d_b, N * sizeof(vt), cudaMemcpyDeviceToHost),
          "copying d_b to b", pass);
  if (pass) { 
    bool match = true;
    for(int i=0; i < N; ++i) {
      if (b[i] != a[i]) {
        match = false;
        cout << "Mismatch at " << i <<": "<<b[i] << " != " << a[i] << "!" << endl;
        break;
      } 
    }
    if(match) { cout << "Success!" << endl;}
    
  } else {
    cout << " Failure! " << endl;
  }
  cudaFree(d_a);
  cudaFree(d_b);

  
  
}