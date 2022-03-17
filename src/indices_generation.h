// Index array generation functions
#pragma once
#include <iostream>
#include <algorithm>
#include <cassert>

using std::cout;
using std::endl;

template<typename it>
void print_indices_sample(it* indxs, int block_size, unsigned long long idx) {
    if(idx % block_size >= block_size / 2 - 2 &&
            idx % block_size <= block_size / 2 + 2 &&
            idx / block_size < 2) {
        cout << " " << idx << ":" << indxs[idx];
    } else if(idx % block_size == block_size / 2 + 3 &&
            idx / block_size < 2) {
        cout << " ... | ";
    } else if((idx % block_size >= block_size - 2 && idx / block_size < 2 ) || 
            (idx % block_size <= 2 && idx / block_size < 3)){
        cout << " " << idx << ":" << indxs[idx];
    } else if(idx % block_size == 3 &&
            idx / block_size < 2) {
        cout << " ... | ";
    } else if(idx / block_size == 2 && idx % block_size == 3 ) {
        cout << " ... ... ... ";
    }
}

template<typename it>
int sequential_indices(it* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){

    if(output_sample) cout << "sequential indices: ";
    for(unsigned long long i=0; i < N; i++) {
        indxs[i] = i;
        if(output_sample) print_indices_sample(indxs, block_size, i); 
    }
    if(output_sample) cout << endl;
    return 0;
}


/**
 * \brief Strides a warp's accesses to go across the other warps in the block
 * 
 *  If a single array was 
 *      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 *  with a blocksize of 8 and warpsize of 4
 * The shuffled indices would be
 *      0 4 1 5 2 6 3 7 8 12 9 13 10 14 11 15
 * For an array
 *      0 1  2  3  ... 30 31 32 33 ... 62 63  64 65 ... ... 124 125 126 127 128 129 130 ... ... 252 253 254 255
 * with a blocksize of 128, warpsize of 32 
 * the shuffled indices would be 
 *      0 32 64 96 ... 71 103 8 40 ... 79 111 16 48 ... ... 31  63  95  127 128 160 192 ... ... 159 191 223 255
 * 
 */
int strided_indices(int* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){
    if(output_sample) cout << "strided indices (Bsz="<<block_size<<",shuffle sz="<<shuffle_size<< "): ";
    int num_warps = block_size / 32;
    for(unsigned long long i=0; i < N/block_size; i++) {
        int start_idx = i * block_size;
        for(unsigned long long j=0; j < block_size; j++) {
            unsigned long long idx = start_idx + j;
            indxs[idx] = (j%num_warps) * 32 + j / num_warps + start_idx;
            if(output_sample) print_indices_sample(indxs, block_size, idx);
        }
    }

    if(output_sample) cout << endl;
    return 0;
}

int strided_no_conflict_indices(int* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){
    if(output_sample) cout << "strided no conflict indices (Bsz="<<block_size<<",shuffle sz="<<shuffle_size<< "): ";
    int num_warps = block_size / 32;
    for(int i=0; i < N/block_size; i++) {
        int start_idx = i * block_size;
        for(int j=0; j < block_size; j++) {
            int idx = start_idx + j;
            indxs[idx] = ( (j % 32) * 32 + (j % 32 + j / num_warps ) % 32) % block_size + start_idx; 
            if(output_sample) print_indices_sample(indxs, block_size, idx);
        }
    }

    if(output_sample) cout << endl;
    return 0;
}

template<typename it, bool avoid_bank_conflicts>
int uncoalesced_access_shuffle_size(it* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){
    assert(N % shuffle_size == 0);
    if(output_sample) cout << "uncoalesced indices (shuffle sz="<<shuffle_size<< ","<< (avoid_bank_conflicts?"no bank conflicts":"with bank conflicts")<<"): ";
    int num_warps = shuffle_size / 32;
    for(int i=0; i < N / shuffle_size; ++i) {
        int start_idx = i * shuffle_size;
        for (int j=0; j < shuffle_size; ++j){
            unsigned long long idx = start_idx + j;
            uint shuffle_t_idx = idx % shuffle_size;
            if constexpr(!avoid_bank_conflicts) {
                indxs[idx] = ( shuffle_t_idx % num_warps) * 32 + shuffle_t_idx / num_warps + start_idx;
            } else {
                indxs[idx] = ( (shuffle_t_idx % 32) * 32 + (shuffle_t_idx % 32 + shuffle_t_idx / num_warps ) % 32) % shuffle_size + start_idx;
            }
            if(output_sample) print_indices_sample(indxs, shuffle_size, idx);
        }
    }
    if(output_sample) cout << endl;
    return 0;
}

int random_indices(int* indxs, int N, int block_size, int shuffle_size, bool output_sample = false){
    if(output_sample) cout << "random indices : ";
    for(int i=0; i < N; i++) {
        indxs[i] = i;                                                                    
    }
    for(int i=0; i < N; i+=shuffle_size) {
        std::random_shuffle(indxs+i, indxs + i + shuffle_size);
    }

    for(int i=0; i < N) {
        if(output_sample) print_indices_sample(indxs, shuffle_size, idx);
    }
    return 0;
}
