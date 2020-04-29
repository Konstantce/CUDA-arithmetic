#ifndef POSEIDON_H
#define POSEIDON_H

#include "cuda_structs.h"

#define CAPACITY 1
#define RATE 2
#define Rf 8
#define Rp 83
#define NUM_ROUNDS (Rf + Rp)
#define m (CAPACITY + RATE)

// 2 -> 1 tree
#define POSEIDON_TREE_COLLAPSING_FACTOR 2
// "standard" number of threads per block
#define THREADS_PER_BLOCK 512

extern DEVICE_VAR CONST_MEMORY uint256_g ARK[NUM_ROUNDS][m];
extern DEVICE_VAR CONST_MEMORY uint256_g MDS[m][m];

__global__ void poseidon(embedded_field* __restrict__ arr, uint32_t arr_len);
__global__ void poseidon_merkle_tree_construction_iteration(embedded_field* __restrict__ arr, uint32_t arr_len);
__global__ void poseidon_merkle_tree_single_block(embedded_field* __restrict__ arr, uint32_t arr_len);

#endif