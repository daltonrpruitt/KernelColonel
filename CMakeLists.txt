# Root CMakeLists.txt

cmake_minimum_required(VERSION 2.8)


project(foo)

message("After 'project':")


include_directories(${CMAKE_SOURCE_DIR}/src)


add_executable(foo src/foo.cpp)

message("Processing CMakeLists.txt")