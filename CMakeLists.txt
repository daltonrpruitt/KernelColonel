# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.22 )


project(microbenchmarks CUDA CXX)

include(FindCUDAToolkit)
# find_package(Python 3.1 REQUIRED)

add_subdirectory(test)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/utilities/matrices)



# add_executable(driver src/driver.cu)
add_executable(main src/main.cu)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 35 52 72 80) #35-K20m, 52-Quadro M2000, 72-V100, 80-A100  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep -Xptxas -dlcm=cg")

add_executable(spmv src/spmv.cu)
set_property(TARGET spmv PROPERTY CXX_STANDARD 17)
# set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES 35 52 72 80) #35-K20m, 52-Quadro M2000, 72-V100, 80-A100  
# message(STATUS "python exec: ${Python_EXECUTABLE}")

# add_custom_target(
#  gen_kernels ALL
#  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/generate_kernel_template_instances.py ${CMAKE_CURRENT_SOURCE_DIR}
#  BYPRODUCTS src/kernel_types.h
#  COMMENT "Generating kernel template instances"
# )

# add_dependencies(main gen_kernels)
