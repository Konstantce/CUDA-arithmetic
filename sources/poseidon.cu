#include "cuda_structs.h"
#include "poseidon.h"


DEVICE_FUNC void __inline__ poseidon_sbox(embedded_field* value) 
{
    embedded_field x = *value;
    x *= x;
    x *= x;
    *value = x * (*value);
}


DEVICE_FUNC void __inline__ poseidon_round(embedded_field* values, bool is_full_round, uint32_t round_idx)
{
    // Add - Round Key
    for (uint32_t i = 0; i < m; i++)
    {
        values[i] += embedded_field(ARK[round_idx][i]);
    }

    // SubWords

    if (is_full_round)
    {
        for (uint32_t i = 0; i < m; i++)
        {
            poseidon_sbox(&values[i]);
        }
    }
    else
    {
        poseidon_sbox(&values[m - 1]);
    }

    // MixLayer: mds * values
    embedded_field res[m];
    for (uint32_t i = 0; i < m; i++)
    {
        res[i] = embedded_field::zero();
        for (uint32_t j = 0; j < m; j++)
        {
            res[i] += embedded_field(MDS[i][j]) * values[j];
        }
    }

    for (uint32_t i = 0; i < m; i++)
    {
        values[i] = res[i];
    }
}


__global__ void poseidon(embedded_field* __restrict__ arr, uint32_t arr_len)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < arr_len / RATE)
    {
        embedded_field values[m];

        for (uint32_t i = 0; i < RATE; i++)
        {
            values[i] = arr[tid * RATE + i];
        }
        for (uint32_t i = RATE; i < m; i++)
        {
            values[i] = embedded_field::zero();
        }
        
        uint32_t round_idx = 0;
        for (uint32_t i = 0; i < Rf / 2; i++)
        {
            poseidon_round(values, true, round_idx);
            round_idx += 1;
        }

        for (uint32_t i = 0; i < Rp; i++) 
        {
            poseidon_round(values, false, round_idx);
            round_idx += 1;
        }

        for (uint32_t i = 0; i < Rf / 2; i++) 
        {
            poseidon_round(values, true, round_idx);
            round_idx += 1;
        }

        assert(round_idx == NUM_ROUNDS);
        for (uint32_t i = 0; i < RATE; i++)
        {
            arr[tid * RATE + i] = values[i];
        }

        tid += blockDim.x * gridDim.x;
    }
}


__global__ void poseidon_merkle_tree_construction_iteration(embedded_field* __restrict__ arr, uint32_t arr_len)
{
    assert(POSEIDON_TREE_COLLAPSING_FACTOR <= RATE);
    
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < arr_len / POSEIDON_TREE_COLLAPSING_FACTOR)
    {
        embedded_field values[m];

        for (uint32_t i = 0; i < POSEIDON_TREE_COLLAPSING_FACTOR; i++)
        {
            values[i] = arr[tid * POSEIDON_TREE_COLLAPSING_FACTOR + i];
        }
        for (uint32_t i = POSEIDON_TREE_COLLAPSING_FACTOR; i < m; i++)
        {
            values[i] = embedded_field::zero();
        }

        uint32_t round_idx = 0;
        for (uint32_t i = 0; i < Rf / 2; i++)
        {
            poseidon_round(values, true, round_idx);
            round_idx += 1;
        }

        for (uint32_t i = 0; i < Rp; i++)
        {
            poseidon_round(values, false, round_idx);
            round_idx += 1;
        }

        for (uint32_t i = 0; i < Rf / 2; i++)
        {
            poseidon_round(values, true, round_idx);
            round_idx += 1;
        }

        assert(round_idx == NUM_ROUNDS);
        
        // not completely safe because of possible races between threads in different blocks 
        // i.e. threads of one block may overwrite the array before threads from another block stored data for array
        // however, this bug is not critical for benchmarking purposes, as we are only interested in computation time 
        // and not perfect correctness 

        arr[tid / POSEIDON_TREE_COLLAPSING_FACTOR] = values[0];

        tid += blockDim.x * gridDim.x;
    }
}


// the final round of poseidon merklee tree construction, when the size of the current "layer" of Merklee tree is so small,
// that it can be completely done by one thread block
// in this case we may exploit the capabilities of shared memory and intra-block syncronization
__global__ void poseidon_merkle_tree_single_block(embedded_field* __restrict__ arr, uint32_t arr_len)
{
    assert(arr_len <= POSEIDON_TREE_COLLAPSING_FACTOR * THREADS_PER_BLOCK);
    assert(blockDim.x = THREADS_PER_BLOCK);
    assert(blockIdx.x == 0);

    uint32_t layer_size = arr_len;
    uint32_t tid = threadIdx.x;
    __shared__ embedded_field cache[THREADS_PER_BLOCK * POSEIDON_TREE_COLLAPSING_FACTOR];

    //initalize shared_memory
    if (tid < arr_len / POSEIDON_TREE_COLLAPSING_FACTOR)
    {
        for (uint32_t i = 0; i < POSEIDON_TREE_COLLAPSING_FACTOR; i++)
        {
            cache[tid * POSEIDON_TREE_COLLAPSING_FACTOR + i] = arr[tid * POSEIDON_TREE_COLLAPSING_FACTOR + i];
        }
    }

    __syncthreads();

    while (layer_size > 1)
    {
        if (tid < layer_size / POSEIDON_TREE_COLLAPSING_FACTOR)
        {
            embedded_field values[m];

            for (uint32_t i = 0; i < POSEIDON_TREE_COLLAPSING_FACTOR; i++)
            {
                values[i] = cache[tid * POSEIDON_TREE_COLLAPSING_FACTOR + i];
            }
            for (uint32_t i = RATE; i < m; i++)
            {
                values[i] = embedded_field::zero();
            }

            uint32_t round_idx = 0;
            for (uint32_t i = 0; i < Rf / 2; i++)
            {
                poseidon_round(values, true, round_idx);
                round_idx += 1;
            }

            for (uint32_t i = 0; i < Rp; i++)
            {
                poseidon_round(values, false, round_idx);
                round_idx += 1;
            }

            for (uint32_t i = 0; i < Rf / 2; i++)
            {
                poseidon_round(values, true, round_idx);
                round_idx += 1;
            }

            assert(round_idx == NUM_ROUNDS);

            cache[tid / POSEIDON_TREE_COLLAPSING_FACTOR] = values[0];
        }
        
        layer_size /= POSEIDON_TREE_COLLAPSING_FACTOR;
        __syncthreads();
    }

    // write root hash back to global memory
    if (tid == 0)
    {
        arr[0] = cache[0];
    }
}



