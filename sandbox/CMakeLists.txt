# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.20) 


project(sandbox C CXX)

# include(FindCUDAToolkit)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/matrices)

# include_directories(${CMAKE_SOURCE_DIR}/src/kernels)


# set_property(TARGET example_matrix_read PROPERTY CXX_STANDARD 17)

# add_executable(driver src/driver.cu)
add_executable(sandbox main.cpp ${CMAKE_SOURCE_DIR}/src/matrices/mmio.c)
set_property(TARGET sandbox PROPERTY LINKER_LANGUAGE CXX)
