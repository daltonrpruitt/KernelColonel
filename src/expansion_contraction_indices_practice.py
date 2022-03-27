import numpy as np
from icecream import ic


warp_size = 32

def sequential_indices(indices, N, use_warp_locality, stream_size):
    for i in range(N):
        indices[i] = i


def expansion_indices(indices, N, use_warp_locality, stream_size, degree_of_expansion):
    if use_warp_locality:
        stream_size = warp_size #(32)
    assert(N % stream_size == 0)
    warps_per_stream = stream_size // warp_size
    for i in range(N):
        warp_id = i // warp_size
        warp_stream_id = warp_id % warps_per_stream
        thread_idx = i % warp_size
        access_idx = warp_id // (warps_per_stream * degree_of_expansion) * stream_size + warp_stream_id * stream_size + thread_idx
        indices[i] = int(access_idx)



def contraction_indices(indices, N, use_warp_locality, stream_size, degree_of_contraction):
    if use_warp_locality:
        stream_size = warp_size #(32)
    assert(N % stream_size == 0)
    warps_per_stream = stream_size // warp_size
    for i in range(N):
        warp_id = i // warp_size
        thread_idx = i % warp_size
        start_idx = warp_id // warps_per_stream * stream_size
        for j in range(degree_of_contraction):
            access_idx = start_idx + j*stream_size+ thread_idx
            indices[i*degree_of_contraction*warp_size + j*warp_size] = access_idx


def expansion_contraction_indices(indices, N, use_warp_locality, stream_size, reads_per_8_writes):
    if reads_per_8_writes == 8:
        return sequential_indices(indices, N, use_warp_locality, stream_size)
    elif reads_per_8_writes < 8:
        return expansion_indices(indices, N, use_warp_locality, stream_size, 8/reads_per_8_writes)
    elif reads_per_8_writes < 8:
        return contraction_indices(indices, N, use_warp_locality, stream_size, reads_per_8_writes/8)
    else:
        print("Error!")


np.set_printoptions(linewidth=np.inf)
def main():

    N = 2**20
    indices = np.zeros(N, dtype=np.int32)
    reads_per_8_writes = 4
    stream_size = 1024
    use_warp_locality = False

    expansion_contraction_indices(indices, N, use_warp_locality, stream_size, reads_per_8_writes)
    for i in range(reads_per_8_writes*4):
        print(f"indices[{i*stream_size}:{i*stream_size + 4}] = \n{np.reshape(indices, newshape=(-1, 32))[i*stream_size:i*stream_size + 4]}")
    indices = np.sort(indices)
    for i in range(reads_per_8_writes*4):
        print(f"indices[{i*stream_size}:{i*stream_size + 4}] = \n{np.reshape(indices, newshape=(-1, 32))[i*stream_size:i*stream_size + 4]}")

    # expansion_contraction_indices(indices, N, False, 1024, 1)
    


if __name__ == "__main__":
    main()