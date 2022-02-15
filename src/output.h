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
#include <sys/utsname.h>
#include <ctime>
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using std::ostream;
using std::string;
using std::stringstream;
using std::cout;
using std::endl;

class Output {
  public: 
    fs::path base_dir; 
    Output() {
        fs::path curr_path = fs::current_path();
        bool at_base = false;
        while (!at_base) {
            bool found_src=false, found_output=false;
            for (const auto & entry : fs::directory_iterator(curr_path)) {
                string name = entry.path().filename();
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
    ~Output() {}

    string get_node_first_name() {
        // https://stackoverflow.com/questions/3596310/c-how-to-use-the-function-uname
        struct utsname buffer;
        errno = 0;
        if (uname(&buffer) < 0) {
            perror("uname");
            // base_dir = fs::path("");
            return "";
        }
        string node_full = buffer.nodename;
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

    bool empty() { return base_dir.empty(); }

    // operator fs::path const & () const noexcept { return base_dir; }
    // operator fs::path       & ()     & noexcept { return base_dir; }

    string operator+(string filename) {
        stringstream out; 
        out << base_dir.string() << "/" << filename;
        return out.str();
    }

    string operator+(const char * filename) {
        stringstream out; 
        out << base_dir.string() << "/" << filename;
        return out.str();
    }

};

ostream& operator<<(ostream& os, const Output& output){
    os << output.base_dir.string();
    return os;
}
