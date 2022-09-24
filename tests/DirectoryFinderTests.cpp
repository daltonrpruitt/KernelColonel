
#include "DirectoryFinder.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <filesystem>
#include <iostream>


namespace KernelColonel {

namespace utilities {


namespace fs = std::filesystem;
using ::testing::HasSubstr;

class DirectoryFinderTest : public ::testing::Test {
  protected:
   DirectoryFinderTest() {}

   ~DirectoryFinderTest() override {}

   void SetUp() override {
      ASSERT_THAT(fs::current_path().string(), HasSubstr("KernelColonel"));
   }

   void TearDown() override {}
};

TEST_F(DirectoryFinderTest, FailToFindParent) {
  
    // Arrange
    fs::path foundDir;
    // Act
    EXPECT_THROW({
        try
        {
            foundDir = find_parent_dir_by_name("non_existent_directory");
        }
        catch( const DirectoryError& e )
        {
            EXPECT_THAT( e.what(), HasSubstr("not a parent of current directory"));
            throw;
        }
    }, DirectoryError );
    
    // Assert  
    ASSERT_TRUE(foundDir.empty());
    // probably not a great assertion, but oh well.
}
/*
TEST(DirectoryFinderTest, BasicAssertions) {

}

TEST(DirectoryFinderTest, BasicAssertions) {

}
*/
} // namespace utilities

} // namespace KernelColonel