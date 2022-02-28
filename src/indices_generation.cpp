// Index array generation functions

#include <stdio.h>
#include <algorithm>

int sequential_indices(int* indxs, int N, int block_size, int shuffle_size, bool output_sample = false){

    if(output_sample) cout << "sequential indices: ");
    for(int i=0; i < N; i++) {
        indxs[i] = i;
        if(output_sample && (i < 10 || (i > 1022 && i < 1028)) ) cout << i <<":"<<indxs[i];
    }
    if(output_sample) cout << endl;
    return 0;
}


int reverse_indices(int* indxs, int N, int block_size, int shuffle_size){
#ifdef DEBUG
    printf("indices: ");
#endif
    for(int i=0; i < N; i++) {
        indxs[i] = (i/block_size +1)* block_size - (i%block_size + 1); // Reverse                                                                      
#ifdef DEBUG
        if(i < 10 || (i > 1022 && i < 1028)) printf("%d:%d ",i,indxs[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif
    return 0;
}

/**
 * \brief Splits array into first half indexing the evens, second half the odds
 * 
 *  If a single array was 
 *      0 1 2 3 
 *  with a shuffle size of 4 (= N = size of the array)
 * The shuffled indices would be
 *      0 2 1 3 
 * For an array
 *      0 1 2 3 ... 30 31 32 33 ... 60 61 62 63 
 * with a shuffle size of 32 (half of N)
 * the shuffled indices would be 
 *      0 2 4 6 ... 29 31 32 34... 57 59 61 63
 *
 * Update 8/23/21: This keeps a constant stride of 2. This is obvious, but I was 
 *    unaware of the impact it has on performance.  
 * 
 */
int evens_odds(int* indxs, int N, int block_size, int shuffle_size){
#ifdef DEBUG
    printf("indices: ");
#endif
    if(N % shuffle_size != 0) { // Not appropriate sizes
        printf("Error cannot shuffle via 'evens_odds()' using N=%d, shuffle_size=%d !\n",
            N, shuffle_size);
        return -1;
    }
    if(block_size % shuffle_size != 0) { // Not appropriate sizes
        printf("Error cannot shuffle via 'evens_odds()' using block_size=%d, shuffle_size=%d !\n", 
            block_size, shuffle_size);
        return -1;
    }
    for(int i=0; i < N/shuffle_size; i++) {
        int start_idx = i * shuffle_size;
        for(int j=0; j < shuffle_size; j++) {
            int idx = start_idx + j;
            indxs[idx] = (2*j + (j < shuffle_size/2 ? 0 : 1) ) % shuffle_size + start_idx;
#ifdef DEBUG 
            if(idx % shuffle_size >= shuffle_size / 2 - 2 && 
                    idx % shuffle_size <= shuffle_size / 2 + 2 && 
                    idx / shuffle_size < 2)      
                printf("%d:%d | ",idx,indxs[idx]); 
            else if(idx % shuffle_size == shuffle_size / 2 + 3 && 
                    idx / shuffle_size < 2)                               
                printf(" ... | ");
            else if((idx % shuffle_size >= shuffle_size - 2 && idx / shuffle_size < 2 ) || 
                    (idx % shuffle_size <= 2 && idx / shuffle_size < 3))   
                printf("%d:%d | ",idx,indxs[idx]); 
            else if(idx % shuffle_size == 3 && 
                    idx / shuffle_size < 2)                               
                printf(" ... | ");
            else if(idx / shuffle_size == 2 && idx % shuffle_size == 3 )                                   
                printf(" ... ... ... ");
#endif
        }
    }

#ifdef DEBUG
    printf("\n");
#endif 
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
int strided_access(int* indxs, int N, int block_size, int shuffle_size){
#ifdef DEBUG
    printf("indices: ");
#endif
    int num_warps = block_size / 32;
    for(int i=0; i < N/block_size; i++) {
        int start_idx = i * block_size;
        for(int j=0; j < block_size; j++) {
            int idx = start_idx + j;
            indxs[idx] = (j%num_warps) * 32 + j / num_warps + start_idx; 
#ifdef DEBUG 
            if(idx % block_size >= block_size / 2 - 2 && 
                    idx % block_size <= block_size / 2 + 2 && 
                    idx / block_size < 2)      
                printf("%d:%d | ",idx,indxs[idx]); 
            else if(idx % block_size == block_size / 2 + 3 && 
                    idx / block_size < 2)                               
                printf(" ... | ");
            else if((idx % block_size >= block_size - 2 && idx / block_size < 2 ) || 
                    (idx % block_size <= 2 && idx / block_size < 3))   
                printf("%d:%d | ",idx,indxs[idx]); 
            else if(idx % block_size == 3 && 
                    idx / block_size < 2)                               
                printf(" ... | ");
            else if(idx / block_size == 2 && idx % block_size == 3 )                                   
                printf(" ... ... ... ");
#endif
        }
    }

#ifdef DEBUG
    printf("\n");
#endif 
    return 0;
}

int strided_access_no_conflicts(int* indxs, int N, int block_size, int shuffle_size){
#ifdef DEBUG
    printf("indices: ");
#endif
    int num_warps = block_size / 32;
    for(int i=0; i < N/block_size; i++) {
        int start_idx = i * block_size;
        for(int j=0; j < block_size; j++) {
            int idx = start_idx + j;
            indxs[idx] = ( (j % 32) * 32 + (j % 32 + j / num_warps ) % 32) % block_size + start_idx; 
#ifdef DEBUG 
            if(idx % block_size >= block_size / 2 - 2 && 
                    idx % block_size <= block_size / 2 + 2 && 
                    idx / block_size < 2)      
                printf("%d:%d | ",idx,indxs[idx]); 
            else if(idx % block_size == block_size / 2 + 3 && 
                    idx / block_size < 2)                               
                printf(" ... | ");
            else if((idx % block_size >= block_size - 2 && idx / block_size < 2 ) || 
                    (idx % block_size <= 2 && idx / block_size < 3))   
                printf("%d:%d | ",idx,indxs[idx]); 
            else if(idx % block_size == 3 && 
                    idx / block_size < 2)                               
                printf(" ... | ");
            else if(idx / block_size == 2 && idx % block_size == 3 )                                   
                printf(" ... ... ... ");
#endif
        }
    }

#ifdef DEBUG
    printf("\n");
#endif 
    return 0;
}


int random_indices(int* indxs, int N, int block_size){
#ifdef DEBUG
    printf("indices: ");
#endif
    for(int i=0; i < N; i++) {
        indxs[i] = i;                                                                    
    }
    std::random_shuffle(indxs, indxs + N);
    
    for(int i=0; i < N; i++) {
#ifdef DEBUG
        if(i < 10 || (i > 1022 && i < 1028)) printf("%d:%d ",i,indxs[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif
    return 0;
}
