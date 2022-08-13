# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.20) 


project(test C CXX)

# include(FindCUDAToolkit)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/utilities/matrices)

# include_directories(${CMAKE_SOURCE_DIR}/src/kernels)


# set_property(TARGET example_matrix_read PROPERTY CXX_STANDARD 17)

# add_executable(driver src/driver.cu)
add_executable(test main.cpp ${CMAKE_SOURCE_DIR}/src/utilities/matrices/mmio.c)
set_property(TARGET test PROPERTY LINKER_LANGUAGE CXX)
