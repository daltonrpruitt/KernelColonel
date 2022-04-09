#pragma once
#include <algorithm>
#include <string>
#include <vector>

using std::string;

template<typename T> 
using vec = std::vector<T>; 

template <typename T>
vec<T> min_med_max(vec<T> data) {
    std::sort(data.begin(), data.end());

    T min, med, max;
    min = data[0];
    int med_idx = data.size() / 2;
    if (data.size() % 2) {
        med = data[med_idx];
    } else {
        med = (data[med_idx - 1] + data[med_idx]) / 2.0;
    }
    max = data[data.size() - 1];
    return vec<T>{min, med, max};
}

template <typename T>
vec<T> avg_stddev(vec<T>& data) {
    T total, avg, sum_of_square_deviation = 0, std_dev;
    for (T val : data) total += val;
    avg = total / (T)data.size();
    for (T val : data) sum_of_square_deviation += pow(avg - val, 2);

    std_dev = sqrt(sum_of_square_deviation / (T)(data.size()-1));
    return vec<T>{avg, std_dev};
}

template <typename T>
vec<T> stats_from_vec(vec<T>& data) {
    vec<T> avg_stddev_vals = avg_stddev(data);
    vec<T> all_vals = min_med_max(data);
    all_vals.insert(all_vals.end(), avg_stddev_vals.begin(), avg_stddev_vals.end());
    return all_vals;
}
