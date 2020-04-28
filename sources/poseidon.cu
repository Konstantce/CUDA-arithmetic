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
            values[i] = arr[tid + i];
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
            arr[tid+i] = values[i];
        }

        tid += blockDim.x * gridDim.x;
    }
}
