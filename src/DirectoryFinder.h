/**
 * @file output.h
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Simple structure for creating/holding output directory
 * @version 0.1
 * @date 2022-02-15
 * 
 * 1. Finds the output folder (error if does not exist)
 * 2. Evaluates the node name and adds the first name to the base directory
 * 3. Evaluates the current time and appends a stringified version to base directory
 * 4. Keeps this directory for when later csv files are output 
 * 
 */

#include <stdlib.h>
#include <errno.h>
#include <ctime>
#include <iostream>
#include <string>
#include <filesystem>
#include <sstream>

#if defined(__linux__) 
#include <sys/utsname.h>
#elif defined(_WIN64)
#include <windows.h>
#define INFO_BUFFER_SIZE 32767
#endif

using std::ostream;
using std::string;
using std::stringstream;
using std::cout;
using std::cerr;
using std::endl;

namespace KernelColonel {
namespace utilities {

namespace fs = std::filesystem;
    
class DirectoryError : public std::exception {
private:
    std::string message;

public:
    DirectoryError(std::string msg) : message(msg) {}
    const char * what () const {
        return message.c_str();
    }
};


/**
 * @brief Finds one of the parent directories via name
 * 
 * @param dir_name Name of directory to search for
 * @return std::filesystem::path The path to the found directory
 */
fs::path find_parent_dir_by_name(const std::string& dir_name) {
    bool atBase = false;
    fs::path currPath = fs::current_path();
    if(currPath.string().find(dir_name) == std::string::npos) {
        throw DirectoryError("Directory '"+dir_name +"' is not a parent of current directory '"+currPath.string() +"'");
    }

    while (currPath != currPath.root_path()) {
        if (currPath.stem().string().compare(dir_name) == 0) {
            return currPath; 
        }
        currPath = currPath.parent_path();
    }
    throw DirectoryError("Reached supposedly unreachable point of find_parent_dir_by_name()");    
} 

fs::path add_directory_to_path(const fs::path& start_dir, const std::string& dir_name) {
    if (!fs::exists(start_dir)) {
        throw DirectoryError("Directory '" + start_dir.string() + "' does not exist! Curr work dir='" +fs::current_path().string()+"'");
    }
    
    fs::path output_path = start_dir / fs::path(dir_name);
    if(fs::exists(output_path)) {
        throw DirectoryError("Directory '" + output_path.string() + "' already exists!");
    }
    if ( ! fs::create_directory(output_path) ) {
        throw DirectoryError("Failed to create directory '" + output_path.string() + "'!");
        output_path = fs::path("");
    }
    
    return output_path;
}


/*
std::string find_directory_by_siblings(std::vector<std::string> sibling_dirs, 
                                       std::string base_dir,
                                       std::string dir_to_add, 
                                       int max_depth=4)
    fs::path foundDir;
    bool atBase = false;
    fs::path currPath = fs::current_path();
    while (!at_base) {
        bool found_src=false, found_output=false;
        for (const auto & entry : fs::directory_iterator(curr_path)) {
            string name = entry.path().filename().string();
            if(name.compare("src") == 0) { found_src = true; }
            if(name.compare("output") == 0) { found_output = true; }
            if(found_src && found_output) {
                at_base = true;
                break;
            } 
        }
        if(at_base) { break; }
            if(curr_path.has_relative_path()) { 
                curr_path = curr_path.parent_path(); 
            } else {
                cerr << "Error: Could not find base directory for output files!";
                curr_path = fs::path(""); break;
            }
        }
        if(at_base) {
            base_dir = curr_path.append("output");
        }
        
        string node_first_name = get_node_first_name(); 
        if(node_first_name.empty()) {
            base_dir = fs::path(""); 
            return;
        }
        base_dir.append(node_first_name);
        
        string curr_time_str = get_current_time(); 
        base_dir.append(curr_time_str);

        if(fs::create_directories(base_dir)) {
            cout << "Created " << base_dir.string() <<  endl;
        } else {
            base_dir = fs::path("");
        }
    }
*/

    string get_node_first_name() {
        // https://stackoverflow.com/questions/3596310/c-how-to-use-the-function-uname
        
        string node_full;

        #if defined(__linux__)

        struct utsname buffer;
        errno = 0;
        if (uname(&buffer) < 0) {
            perror("uname");
            throw std::exception("uname"); 
            // base_dir = fs::path("");
            return "";
        }
        node_full = buffer.nodename;

        #elif defined(_WIN64)

        TCHAR  infoBuf[INFO_BUFFER_SIZE];
        DWORD  bufCharCount = INFO_BUFFER_SIZE;

        // Based on https://stackoverflow.com/questions/27914311/get-computer-name-and-logged-user-name
        // Get and display the name of the computer.
        if( !GetComputerName( infoBuf, &bufCharCount ) )
        throw std::exception("GetComputerName"); 

        node_full = std::string(infoBuf);

        #endif
        
        
        int first_dot = node_full.find(".");
        string node_first = node_full.substr(0, first_dot);
        // cout << "Node first name = " << node_first << endl;
        return node_first;
    }

    string get_current_time() {
        // https://stackoverflow.com/questions/997946/how-to-get-current-time-and-date-in-c
        std::time_t t = std::time(0);   // get time now
        std::tm* now = std::localtime(&t);
        char buffer[80];
        strftime(buffer, 80, "%F_%H-%M-%S", now );
        return string(buffer);

    }


} // namespace utilities 
} // namespace KernelColonel
