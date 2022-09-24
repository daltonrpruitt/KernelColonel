
#include "output.h"

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

TEST_F(DirectoryFinderTest, FailsToFind) {
  
    // Arrange
    
    // Act
        EXPECT_THROW({
        try
        {
            DirectoryFinder(std::vector<std::string>("non_existant_directory"), "should_not_create");
        }
        catch( const DirectoryFinderException& e )
        {
            // and this tests that it has the correct message
            EXPECT_STREQ( "Sibling directories could not be found", e.what() );
            throw;
        }
    }, DirectoryFinderException );
    
    // Assert  
    ASSERT_FALSE(std::filesystem::exists("non_existant_directory"));
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