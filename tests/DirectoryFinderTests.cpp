
#include "DirectoryFinder.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <string>
#include <string_view>
#include <filesystem>
#include <iostream>


namespace KernelColonel {

namespace utilities {

constexpr std::string_view PROJECT_BASE_DIR {"KernelColonel"};
namespace fs = std::filesystem;
using ::testing::HasSubstr;

class DirectoryFinderTest : public ::testing::Test {
  protected:
   DirectoryFinderTest() {}

   ~DirectoryFinderTest() override {
        if (fs::exists(starting_working_dir)) fs::current_path(starting_working_dir);
    }

    // will create subdirectories "./parent/child" and move to "child" as starting point
   void SetUp() override {
        ASSERT_THAT(starting_working_dir.string(), HasSubstr(PROJECT_BASE_DIR));
        while(fs::current_path().parent_path().string().find(PROJECT_BASE_DIR) != std::string::npos) {
            fs::current_path(fs::current_path().parent_path());
        }
        fs::current_path(fs::current_path() / "build" / "tests" / "Debug");
        if ( ! fs::exists(dummy_parent_dir)) fs::create_directory(dummy_parent_dir);
        if ( ! fs::exists(dummy_child_dir)) fs::create_directory(dummy_child_dir);
        ASSERT_TRUE(fs::exists(dummy_child_dir));
        fs::current_path(dummy_child_dir);

    }

    void TearDown() override {
        fs::current_path(starting_working_dir);
        ASSERT_EQ(fs::current_path(), starting_working_dir);
    }

    std::string dummy_parent_dir_name = "test_parent_directory";
    std::string dummy_child_dir_name = "test_child_directory";
    fs::path starting_working_dir = fs::current_path();
    fs::path dummy_parent_dir = starting_working_dir / fs::path(dummy_parent_dir_name);
    fs::path dummy_child_dir = dummy_parent_dir / fs::path(dummy_child_dir_name);
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