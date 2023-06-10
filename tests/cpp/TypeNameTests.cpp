/**
 * @file TypeNameTests.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Some testing of type_name utility file (not currently working - 2023-06-10)
 * @version 0.1
 * @date 2022-11-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <string>
#include <typeinfo>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "utils/type_name.hpp"

// #include "IKernelExecution.hpp"
// using namespace KernelColonel;

template<typename value_t, typename index_t> 
struct gpu_data_s
{
    value_t* input = nullptr;
    value_t* output = nullptr;
    index_t* indices = nullptr;
};


template<typename ...kernel_param_types>
class TypeNameTest // : public IKernelExecution<...>
{
  public:
    // using TypeNameTest_t = TypeNameTest<kernel_param_types...>;
    // using super = IKernelExecution<kenrel_param_types...>
    TypeNameTest() = default; //: super(n) {}
    ~TypeNameTest() = default; //: super(n) {}
    
  private:
  
  template<typename... types, //typename... tail_types, 
    typename = std::enable_if_t<sizeof...(types)==0, bool>>
    void innerPrintTypes(std::ostream &os) const {}
    
    template<typename head_type, typename... tail_types>
    void innerPrintTypes(std::ostream &os) const {
        if(!os.tellp() == 0) 
            os << ", ";
        os << type_name<head_type>();
        innerPrintTypes<tail_types...>(os);
    }

  public: 
    // template<typename head_type, typename... tail_types>
    template<typename ...ts> 
    friend std::ostream& operator<<(std::ostream& os, const TypeNameTest<ts...>& exec);
    
    void PrintTypes() const {
        std::stringbuf buf;
        std::ostream s(&buf);
        innerPrintTypes<kernel_param_types...>(s);
        std::cout << "type of *this = " << type_name<decltype(this)>() << std::endl; 
        std::cout << "\ttemplated on: < " << buf.str() << " >" << std::endl;
    }
    
};

TEST(TypeNameTests, Construct) {
    using IKernelExecution_t = TypeNameTest<double, double, unsigned long long, bool, std::string>;
    IKernelExecution_t kernel;
    kernel.PrintTypes();
    ASSERT_TRUE(true);

}

/*

TEST(TypeNameTests, Initialize) {
    using IKernelExecution_t = TypeNameTest<float, int>;
    size_t data_size = 4;
    IKernelExecution_t data(data_size);
    
    ASSERT_TRUE(data.init(0));

    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();

    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), data_size);
    EXPECT_EQ(cpu_data_vector[1].size(), data_size);

    ASSERT_EQ(gpu_data_ptrs_vector.size(), 2);
    EXPECT_NE(gpu_data_ptrs_vector[0], nullptr);
    EXPECT_NE(gpu_data_ptrs_vector[1], nullptr);

    ASSERT_EQ(gpu_indices_ptrs_vector.size(), 1);
    EXPECT_NE(gpu_indices_ptrs_vector[0], nullptr);

    EXPECT_EQ(data.gpu_named_data.input, gpu_data_ptrs_vector[0]);
    EXPECT_EQ(data.gpu_named_data.output, gpu_data_ptrs_vector[1]);
    EXPECT_EQ(data.gpu_named_data.indices, gpu_indices_ptrs_vector[0]);
}

TEST(TypeNameTests, Uninitialize) {
    using IKernelExecution_t = TypeNameTest<float, int>;
    size_t data_size = 4;
    IKernelExecution_t data(data_size);
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();

    ASSERT_TRUE(data.init(0));
    data.uninit();
    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);

    ASSERT_EQ(cpu_indices_vector.size(), 1);
    EXPECT_EQ(cpu_indices_vector[0].size(), 0);

    ASSERT_EQ(gpu_data_ptrs_vector.size(), 2);
    EXPECT_EQ(gpu_data_ptrs_vector[0], nullptr);
    EXPECT_EQ(gpu_data_ptrs_vector[1], nullptr);

    ASSERT_EQ(gpu_indices_ptrs_vector.size(), 1);
    EXPECT_EQ(gpu_indices_ptrs_vector[0], nullptr);

    EXPECT_EQ(data.gpu_named_data.input, nullptr);
    EXPECT_EQ(data.gpu_named_data.output, nullptr);
    EXPECT_EQ(data.gpu_named_data.indices, nullptr);
}

*/