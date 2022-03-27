import numpy as np
from icecream import ic


warp_size = 32

def sequential_indices(indices, N, stream_size):
    for i in range(N):
        indices[i] = i


def expansion_indices(indices, N, stream_size, degree_of_expansion):
    assert(N % stream_size == 0)
    warps_per_stream = stream_size // warp_size
    for i in range(N):
        warp_id = i // warp_size
        stream_warp_id = warp_id % warps_per_stream
        stream_id = warp_id // (warps_per_stream * degree_of_expansion)
        thread_idx = i % warp_size
        access_idx = stream_id * stream_size  + stream_warp_id * warp_size + thread_idx
        indices[i] = int(access_idx)



def contraction_indices(indices, N, stream_size, degree_of_contraction):
    assert(N % stream_size == 0)
    warps_per_stream = stream_size // warp_size
    for i in range(N):
        warp_id = i // warp_size
        thread_idx = i % warp_size
        start_idx = warp_id // warps_per_stream * stream_size
        for j in range(degree_of_contraction):
            access_idx = start_idx + j*stream_size+ thread_idx
            indices[i*degree_of_contraction*warp_size + j*warp_size] = access_idx


def expansion_contraction_indices(indices, N, stream_size, reads_per_8_writes):
    if reads_per_8_writes == 8:
        return sequential_indices(indices, N, stream_size)
    elif reads_per_8_writes < 8:
        return expansion_indices(indices, N, stream_size, 8/reads_per_8_writes)
    elif reads_per_8_writes > 8:
        return contraction_indices(indices, N, stream_size, reads_per_8_writes/8)
    else:
        print("Error!")


def check_expansion(N, stream_size, degree_of_expansion, debug=False):
    indices = np.zeros(N, dtype=np.int32)
    expansion_contraction_indices(indices, N, stream_size, 8/degree_of_expansion)
    sorted_indices = np.sort(indices)
    unique, counts = np.unique(sorted_indices, return_counts=True)
    start, stop = 0, 10
    if(debug): print(f"indices[{start}:{stop}] = \n{np.reshape(indices, newshape=(-1, 32))[start:stop]}")
    for i in range(len(unique)):
        if i != unique[i]:
            print("Missed a value!")
            print(f"indices[{max(0,i-5)}:{i+5}] = \n{unique[max(0,i-5):i+5]}")   
            return False
    for i in range(len(counts)):
        if degree_of_expansion != counts[i]:
            print("Missed a value!")
            print(f"indices[{max(0,i-5)}:{i+5}] = \n{unique[max(0,i-5):i+5]}")   
            return False

def check_contraction(N, stream_size, degree_of_contraction, debug=False):
    indices = np.zeros(N, dtype=np.int32)
    expansion_contraction_indices(indices, N, stream_size, degree_of_contraction*8)
    sorted_indices = np.sort(indices)
    unique, counts = np.unique(sorted_indices, return_counts=True)
    start, stop = 0, 10
    if(debug): print(f"indices[{start}:{stop}] = \n{np.reshape(indices, newshape=(-1, 32))[start:stop]}")
    # for i in range(len(unique)):
    #     if i != unique[i]:
    #         print("Missed a value!")
    #         print(f"indices[{max(0,i-5)}:{i+5}] = \n{unique[max(0,i-5):i+5]}")   
    #         return False
    # for i in range(len(counts)):
    #     if degree_of_contraction != counts[i]:
    #         print("Missed a value!")
    #         print(f"indices[{max(0,i-5)}:{i+5}] = \n{unique[max(0,i-5):i+5]}")   
    #         return False



np.set_printoptions(linewidth=np.inf)
def main():

    N = 2**20
    # indices = np.zeros(N, dtype=np.int32)
    # reads_per_8_writes = 4
    # stream_size = 128
    # check_expansion(N, 128, 2, True)
    check_contraction(N, 64, 4, True)
    


if __name__ == "__main__":
    main()