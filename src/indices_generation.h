// Index array generation functions
#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cassert>

using std::cout;
using std::endl;
using std::vector;
using std::string;

int warp_size = 32;

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
template<typename it>
int strided_indices(it* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){
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

template<typename it>
int strided_no_conflict_indices(it* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){
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

    int warps_per_shuffle = shuffle_size / warp_size;
    int warps_per_shuffle_scan = warps_per_shuffle / warp_size;
    int scans_per_shuffle = warp_size;
    for(int shuffle_block_idx=0; shuffle_block_idx < N / shuffle_size; ++shuffle_block_idx) {
        int shuffle_block_start_idx = shuffle_block_idx * shuffle_size;
        
        for(int shuffle_scan_id=0; shuffle_scan_id<scans_per_shuffle; shuffle_scan_id++) {
            
            for(int shuffle_scan_warp_id=0; shuffle_scan_warp_id<warps_per_shuffle_scan; shuffle_scan_warp_id++) {
                it scan_local_start_idx = shuffle_scan_warp_id * shuffle_size / warps_per_shuffle_scan;

                for(int warp_t_idx=0; warp_t_idx<warp_size; ++warp_t_idx) {
                    it global_t_idx = shuffle_block_start_idx + (shuffle_scan_id * warps_per_shuffle_scan + shuffle_scan_warp_id)*warp_size + warp_t_idx; 
                    
                    int warp_local_idx_offset;
                    if constexpr(!avoid_bank_conflicts) {
                        warp_local_idx_offset = ( shuffle_scan_id ) % warp_size + warp_t_idx*warp_size;
                    } else {
                        warp_local_idx_offset = (warp_t_idx + shuffle_scan_id) % warp_size + warp_t_idx*warp_size;
                    }

                    it final_idx = shuffle_block_start_idx + scan_local_start_idx + warp_local_idx_offset;
                    indxs[global_t_idx] = final_idx;
                    if(output_sample) print_indices_sample(indxs, shuffle_size, global_t_idx);
                }
            }
        }
    }
    if(output_sample) cout << endl;
    return 0;
}

template<typename vt, typename it, bool avoid_bank_conflicts>
int sector_based_uncoalesced_access(it* indxs, unsigned long long N, int compute_capability_major, int shuffle_size, bool output_sample = false){
    assert(N % shuffle_size == 0);
    int sector_size = 32; // Bytes
    int sectors_per_transaction = 1; // Before Volta
    if(compute_capability_major >= 7) { sectors_per_transaction = 2;} // Volta and after
    int stride = sector_size * sectors_per_transaction / 

    if(output_sample) cout << "sector-based uncoalesced pattern (shuffle sz="<<shuffle_size<< ", stride="<<stride
        << ", "<< (avoid_bank_conflicts?"no bank conflicts":"with bank conflicts")<<"): ";
    
    int warps_per_shuffle = shuffle_size / warp_size;
    int warps_per_shuffle_scan = warps_per_shuffle / stride;
    int scans_per_shuffle = stride;
    for(int shuffle_block_idx=0; shuffle_block_idx < N / shuffle_size; ++shuffle_block_idx) {
        int shuffle_block_start_idx = shuffle_block_idx * shuffle_size;
        
        for(int shuffle_scan_id=0; shuffle_scan_id<scans_per_shuffle; shuffle_scan_id++) {
            
            for(int shuffle_scan_warp_id=0; shuffle_scan_warp_id<warps_per_shuffle_scan; shuffle_scan_warp_id++) {
                it scan_local_start_idx = shuffle_scan_warp_id * shuffle_size / warps_per_shuffle_scan;

                for(int warp_t_idx=0; warp_t_idx<warp_size; ++warp_t_idx) {
                    it global_t_idx = shuffle_block_start_idx + (shuffle_scan_id * warps_per_shuffle_scan + shuffle_scan_warp_id)*warp_size + warp_t_idx; 
                    
                    int warp_local_idx_offset;
                    if constexpr(!avoid_bank_conflicts) {
                        warp_local_idx_offset = ( shuffle_scan_id ) % stride + warp_t_idx*stride;
                    } else {
                        warp_local_idx_offset = (warp_t_idx + shuffle_scan_id) % stride + warp_t_idx*stride;
                    }

                    it final_idx = shuffle_block_start_idx + scan_local_start_idx + warp_local_idx_offset;
                    indxs[global_t_idx] = final_idx;
                    if(output_sample) print_indices_sample(indxs, shuffle_size, global_t_idx);
                }
            }
        }
    }
    if(output_sample) cout << endl;
    return 0;
}

template<typename it>
int random_indices(it* indxs, unsigned long long N, int block_size, int shuffle_size, bool output_sample = false){
    if(output_sample) cout << "random indices : ";
    for(int i=0; i < N; i++) {
        indxs[i] = i;                                                                    
    }
    for(int i=0; i < N; i+=shuffle_size) {
        std::random_shuffle(indxs+i, indxs + i + shuffle_size);
    }

    for(int i=0; i < N; i++) {
        if(output_sample) print_indices_sample(indxs, shuffle_size, i);
    }
    return 0;
}

// using it = unsigned long long; // declared in main...
typedef int func_t(it*, unsigned long long, int, int, bool);
typedef func_t* pfunc_t;


static const vector<pfunc_t> index_patterns = {
    sequential_indices<it>,
    strided_indices<it>,
    strided_no_conflict_indices<it>,
    uncoalesced_access_shuffle_size<it, false>,
    uncoalesced_access_shuffle_size<it, true>,
    sector_based_uncoalesced_access<vt, it, false>,
    sector_based_uncoalesced_access<vt, it, true>,
    random_indices<it>
};

enum indices_pattern {
    SEQUENTIAL = 0,
    STRIDED_BSZ = 1,
    STRIDED_BLOCKSZ_NO_BANK_CONFLICTS = 2,
    UNCOALESCED_SHUFFLESZ = 3,
    UNCOALESCED_SHUFFLESZ_NO_BANK_CONFLICTS = 4,
    SECTOR_BASED_UNCOALESCED_SHUFFLESZ = 5,
    SECTOR_BASED_UNCOALESCED_SHUFFLESZ_NO_BANK_CONFLICTS = 6,
    RANDOM_BLOCKED_SHUFFLESZ = 7
};

static const vector<string> index_pattern_strings {
    "sequential",
    "strided_block_size",
    "strided_block_size_no_bank_conflicts",
    "uncoalesced_shuffle_size",
    "uncoalesced_shuffle_size_no_bank_conflicts",
    "sector_based_uncoalesced_access",
    "sector_based_uncoalesced_access_no_bank_conflicts",
    "random_blocked_shuffle_size"
};
