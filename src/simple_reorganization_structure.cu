#include <iostream>
#include <fstream>
#include <string>


/**
 * @brief current (2023-04-04) plan:
 * - Assume a common simple kernel signature : single input, single output; both doubles
 * - Configuration allowed is basically the launch parameters (block size/grid size/ input data setup category)
 * - the framework then : 
 *      1. takes some config file input
 *      2. configures the list of kernel runs from the file and places into some sorted queue based on 
 *              the configuration of the input data 
 *      3. runs in the "sorted queue" are grouped into batches to be run on the GPU
 *      4. for each batch:
 *          1. Start by allocating the required memory on the GPU, computing the data somehow, then copying the data to the GPU
 *          2. perform checks of each configured run
 *                  loop over the configurations:
 *                      execute the kernel
 *                      copy output data back to host
 *                      verify output data with CPU computation/comparison
 *                  if fails any, record why (if can), throw and stop any remaining runs (need all to pass)
 *                  This step is to confirm that every configuration is valid before getting performance data.
 *          3. Actual performance runs:
 *             for i in repeat_times:
 *                  loop over the configurations:
 *                      start timer
 *                      execute the kernel
 *                      stop timer
 *                      record time result to vector of times for this configuration
 *          4. perform statistics on recorded data vectors
 *          5. record data out to files for post-processing
 */

class IData
{
  public:
    data() : m_initialized(false) {}
    void init() {
        if(m_initialized) return;
        // perform initialization
    }
    void uninit() {
        if(!m_initialized){
            std::cout << "In Data: Trying to uninit() when not initialized!" << std::endl;
            return;
        } 
        // perform uninitialization
    }

  private:
    bool m_initialized;
};

struct base_kernel_config {
    int exec_times;
}

template<typeanme value_t, typename index_t>
class IKernel
{
    // std::shared_ptr<IData> m_data;
    std::vector<float> m_exec_times;
    std::string m_kernel_name = "IKernel";

    std::vector<std::string> m_extra_columns;


  public:
    IKernel(base_kernel_config config) : m_exec_times(exec_times) {}
    ~IKernel() = default;

    std::string get_kernel_name() { return m_kernel_name; }
    size_t get_sizeof_kernel_object() { return sizeof(this); }

    bool check(std::shared_ptr<IData> data) {
        ///
    }

    float run(std::shared_ptr<IData> data) {
        ///
    }

    void run_and_record_single(std::shared_ptr<IData> data) {
        m_exec_times.push_back(run(data));
    }

    void add_header_to_output_file(std::ofstream& output_file) {
        static std::vector<std::string> column_headers{
            "Kernel_type","some_other_compile_type_param...","runtime_param..."};
        // assume first to access data file
        std::ofstream output_filestream(filename, std::ios_base::write);
        std::string header = ""; // do the comma between items thing from original version
        // assert file's first line does not begin with typical header...
        output_filestream << header;
    } 

    void output_stats_to_filestream(std::ofstream& output_filestream) {
        stats = compute_basic_stats(m_exec_times);
        filename << stats;
    }

    virtual void output_extra_data_to_file(std::ofstream& output_filestream) = 0; 
    void output_data_to_directory(std::ofstream& output_filestream) {
        output_stats_to_filestream(output_filestream);
        output_extra_data_to_file(output_filestream);
    }

};


class driver
{
    vector<std::shared_ptr<IKernel>> kernels;
    std::shared_ptr<IData> m_data;

    void set_data(std::shared_ptr<IData> data){
        if(m_data) std::cout << "In Driver: Trying to set_data() after is already set!" << std::endl;
        m_data = data;
    }
    
    void init_data(){
        if(!m_data) throw std::runtime_error("In Driver: Trying to init_data() before is set!");
        m_data->init();
    }
    void uninit_data(){
        if(!m_data) throw std::runtime_error("In Driver: Trying to uninit_data() before is set!");
        m_data->uninit();
    }

    void add_kernel(std::shared_ptr<IKernel> kernel) {
        kernels.push_back(kernel);
    }

    void perform_checks_and_runs(int num_runs = 1) {
        if(num_runs == 1) 
            std::cout << "In Driver::perform_checks_and_runs() : num_runs = 1; is this really desired?" << std::endl;
        init_data();
        for(k : kernels) {
            k->check(m_data); // should be fast...?
        }
        for(int i=0; i < num_runs; ++i) {
            for(k : kernels) {
                k->run_and_record_single(m_data);
            }
        }
    }

    void full_execution_and_output(std::path output_dir) {
        perform_checks_and_runs();

        std::path output_filename = output_dir + kernels[0]->get_kernel_name() + ".csv";
        std::ofstream output_filestream(filename, std::ios_base::out);

        kernels[0]->add_header_to_output_file(output_filestream);
        for(k : kernels) {
            k->output_data_to_dir(output_dir);
        }
    }

};
