#pragma once 
/**
 * @file IKernelData.cuh
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides a wrapper surrounding the data inputs/outputs and indices 
 *          GPU kernel execution (in IKernelContext)
 * 
 */

#include <vector>
#include <algorithm>
#include <exception>


/**
 * @brief Container for CPU/GPU data for different situations; used for reducing `cudaMemCpy()` calls
 * 
 * Necessary because different kernel contexts may use the same input data, but will the data
 * will be removed and re-computed and copied between separate kernel context instance executions.
 * 
 * @tparam value_t Value type (data arrays)
 * @tparam it Index type (indirection arrays)
 */
template<typename value_t, 
         typename index_t, 
         unsigned int num_in_data, 
         unsigned num_out_data, 
         unsigned int num_indices>
class IKernelData { 
  public:
    IKernelData(unsigned long long n);
    ~IKernelData();

    enum struct DataState { PREINIT = 0, INIT, UNINIT };

    /**
     * @brief Setup all CPU and GPU data/index arrays
     * 
     * Setup on CPU side, allocate GPU memory, copy data over to GPU. 
     * 
     * @return true Prematurely return if already initialized
     * @return false Failed to initialize properly (handling taken care of by owner of object)
     */
    bool init(int dev_ctx_id);

    /**
     * @brief Free relevant structures (CPU and GPU)
     */
    void uninit();
    
    void copyOutputToDevice();

  private: 
    /**
     * @brief Placeholder for user-defined input data array(s) initialization
     * 
     * Throws an exception if fails.
     */
    virtual void initInputsCpu() = 0;

    /**
     * @brief Placeholder for user-defined indices array(s) initialization
     * 
     * Throws an exception if fails.
     */
    virtual void initIndicesCpu() = 0;

    /**
     * @brief Placeholder for user-defined gpu data structure initialization
     * 
     * Simple aliasing of pointers, so should not fail.
     */
     virtual void setGpuNamedData() = 0;

    /**
     * @brief Free GPU memory
     */
    void freeGpuData();
          
    /**
     * @brief Free CPU memory
     */
    void freeCpuData();

    static constexpr unsigned int value_t_size = sizeof(value_t);
    static constexpr unsigned int index_t_size = sizeof(index_t);
    static constexpr unsigned int num_total_data = num_in_data + num_out_data;
    string name;
    
    bool okay = true;
    DataState state = PREINIT;
    
    int gpu_device_id;
protected:
    unsigned long long N=0;
    unsigned long long input_size=0;
    unsigned long long output_size=0;
    unsigned long long indices_size=0;

    vector<vector<value_t>> host_data{(unsigned long)num_total_data};
    vector<value_t *> device_data_ptrs{(unsigned long)num_total_data};
    
    vector<vector<index_t>> host_indices{(unsigned long)num_indices};
    vector<index_t *> device_indices_ptrs{(unsigned long)num_indices};
};
