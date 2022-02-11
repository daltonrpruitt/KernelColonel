# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.22 )


project(TEST_DRIVER CUDA CXX)

include(FindCUDAToolkit)
find_package(Python 3.1 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/kernels)


# add_executable(driver src/driver.cu)
add_executable(ctx_driver src/kernel_context_driver.cu)
set_property(TARGET ctx_driver PROPERTY CXX_STANDARD 17)
set_property(TARGET ctx_driver PROPERTY CUDA_ARCHITECTURES 35 52 80) #35-K20m, 52-Quadro M2000, 80-A100  

message(STATUS "python exec: ${Python_EXECUTABLE}")

add_custom_target(
 gen_kernels ALL
 COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/generate_kernel_template_instances.py ${CMAKE_CURRENT_SOURCE_DIR}
 BYPRODUCTS src/kernel_types.h
 COMMENT "Generating kernel template instances"
)

add_dependencies(ctx_driver gen_kernels)
