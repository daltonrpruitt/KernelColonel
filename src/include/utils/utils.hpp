#pragma once
/**
 * @file utils.h
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Utility functions
 * @version 0.1
 * @date 2022-02-16
 * 
 *  Copied from other project
 */

#include <time.h>
#include <algorithm>
#include <string>
#include <iostream>

#include <vector_types.h> // for dim3

inline double elapsed_time_ms(timespec startTime, timespec endTime){
    return (endTime.tv_sec - startTime.tv_sec) * 1000.0 +
                    (endTime.tv_nsec - startTime.tv_nsec) * 1e-6;
}

inline std::string bool_to_string(bool b){
    // Python capitalization style
    return b ? "True" : "False";
}

/**
 * @brief From https://stackoverflow.com/a/24315631
 * 
 * @param str String we are modifying
 * @param from String to remove
 * @param to String to put in
 */
static inline void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

namespace std
{
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec){
        os << "< " << vec[0];
            for (int i=1; i<vec.size() && i < 10; ++i) { os << ", " << vec[i]; }
        if(vec.size() > 10) { os << " ... "; }
        os << " >" << std::endl;
        return os;
    }

    inline ostream& operator<<(std::ostream& stream, dim3 d) {
        if (d.y == 1 && d.z == 1) {
            stream << d.x;
        } else {
            stream << "(" << d.x << "," << d.y << "," << d.z << ")";
        }
        return stream;
    }

} // namespace std
