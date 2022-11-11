#include <gtest/gtest.h>

#include "IKernelData.cuh"


// Demonstrate some basic assertions.
TEST(IKernelDataTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
