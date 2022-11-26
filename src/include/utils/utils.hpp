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
#include <string>
#include <iostream>

double elapsed_time_ms(timespec startTime, timespec endTime){
    return (endTime.tv_sec - startTime.tv_sec) * 1000.0 +
                    (endTime.tv_nsec - startTime.tv_nsec) * 1e-6;
}

std::string bool_to_string(bool b){
    // Python capitalization style
    return b ? "True" : "False";
}
