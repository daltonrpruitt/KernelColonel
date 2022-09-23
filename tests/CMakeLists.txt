# For gtest tests
cmake_minimum_required(VERSION 3.20)


include(CTest REQUIRED)

include(FetchContent REQUIRED)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${CMAKE_SOURCE_DIR}/../src)

add_executable(
  DirectoryFinderTests
  DirectoryFinderTests.cpp
)

target_link_libraries(
  DirectoryFinderTests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(DirectoryFinderTests)

# Testing
enable_testing()

add_test(DirectoryFinderTests DirectoryFinderTests)
