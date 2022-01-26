# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.22 )


project(TEST_DRIVER CUDA CXX)

include(FindCUDAToolkit)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/kernels)


add_executable(driver src/driver.cu)
