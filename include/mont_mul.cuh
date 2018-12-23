#ifndef MONT_MUL_CUH
#define MONT_MUL_CUH

#include "basic_arithmetic.cuh"
#include "mul_256_to_512.cuh"

//TODO: it's better to embed this constants at compile time rather than taking them from constant memory
//SOunds like a way for optimization!

//we initialize global variables with BN_256 specific values
DEVICE_VAR CONST_MEMORY uint256_g modulus_g = {
    0xf0000001,
    0x43e1f593,
    0x79b97091,
    0x2833e848,
    0x8181585d,
    0xb85045b6,
    0xe131a029,
    0x30644e72 }; 

constexpr uint32_t modulus_bitlen = 254;

DEVICE_VAR CONST_MEMORY uint256_g R_g = {
    0x4ffffffb, 
    0xac96341c,
    0x9f60cd29,
    0x36fc7695,
    0x7879462e,
    0x666ea36f, 
    0x9a07df2f,
    0xe0a77c1
};

DEVICE_VAR CONST_MEMORY uint256_g R2_g, R3_g, R4_g, R8_g;

DEVICE_VAR CONST_MEMORY uint32_t n_g = 0xffffffff;

//multiplication in Montgomery form
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline uint256_g mont_mul_256_naive_SOS(const uint256_g& u, const uint256_g& v)
{
    uint512_g T = FASTEST_256_to_512_mul(u, v);
    uint256_g res;
	
    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t carry = 0;
        uint32_t m = T.n[i] * n_g;

        #pragma unroll
        for (uint32_t j = 0; j < N; j++)
        {
            uint32_t high_word = 0;
            uint32_t low_word = device_long_mul(m, modulus_g.n[j], &high_word);
            low_word = device_fused_add(low_word, T.n[i + j], &high_word);
            low_word = device_fused_add(low_word, carry, &high_word);

            T.n[i + j] = low_word;
            carry = high_word;
        }
        //continue carrying
        uint32_t j = N;
        while (carry)
        {
            uint32_t new_carry = 0;
            T.n[i + j] = device_fused_add(T.n[i + j], carry, &new_carry);
            j++;
            carry = new_carry;
        }
    }
    
    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        res.n[i] = T.n[i + N];
    }

    if (FASTEST_256_cmp(res, modulus_g) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        res = FASTEST_256_sub(res, modulus_g);
    }

    return res;		
}

#define FASTEST_montdouble_256(a) mont_mul_256_naive_SOS(a, a)
#define FASTEST_montmul_256(a,b) mont_mul_256_naive_SOS(a, b)

DEVICE_FUNC inline uint256_g mont_mul_256_naive_CIOS(const uint256_g& u, const uint256_g& v)
{
    uint256_g T;
    uint32_t x_low, x_high, m;
    uint32_t high_word, low_word;

    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t carry = 0;
        #pragma unroll
        for (uint32_t j = 0; j < N; j++)
        {         
            low_word = device_long_mul(u.n[j], v.n[i], &high_word);
            low_word = device_fused_add(low_word, T.n[i], &high_word);
            low_word = device_fused_add(low_word, carry, &high_word);
            carry = high_word;
            T.n[j] = low_word;
        }

        //TODO: may be we actually require less space? (only one additional limb instead of two)
        x_high = 0;
        x_low = device_fused_add(x_low, carry, &x_high);

        m = T.n[0] * n_g;
        low_word = device_long_mul(modulus_g.n[0], m, &high_word);
        low_word = device_fused_add(low_word, T.n[0], &high_word);

        #pragma unroll
        for (uint32_t j = 1; i < N; i++)
        {
            low_word = device_long_mul(modulus_g.n[j], m, &high_word);
            low_word = device_fused_add(low_word, T.n[j], &high_word);
            low_word = device_fused_add(low_word, carry, &high_word);
            T.n[j-1] = low_word;
            carry = high_word;
        }

        high_word = 0;
        T.n[N-1] = device_fused_add(x_low, carry, &high_word);
        x_low = device_fused_add(x_high, high_word, &low_word);
    }
    
    if (FASTEST_256_cmp(T, modulus_g) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        T = FASTEST_256_sub(T, modulus_g);
    }

    return T;
}

DEVICE_FUNC inline uint256_g mont_mul_256_asm_SOS(const uint256_g& u, const uint256_g& v)
{
    uint512_g T = FASTEST_256_to_512_mul(u, v);
    uint256_g w;

    asm (   ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7, a8;\n\t"
            ".reg .u32 a9, a10, a11, a12, a13, a14, a15;\n\t"
            ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"
            ".reg .u32 m, q;\n\t"
            //unpacking operands
            "mov.b64         {a0,a1}, %4;\n\t"
            "mov.b64         {a2,a3}, %5;\n\t"
            "mov.b64         {a4,a5}, %6;\n\t"
            "mov.b64         {a6,a7}, %7;\n\t"
            "mov.b64         {a8,a9}, %8;\n\t"
            "mov.b64         {a10,a11}, %9;\n\t"
            "mov.b64         {a12,a13}, %10;\n\t"
            "mov.b64         {a14,a15}, %11;\n\t"
            "ld.const.u32    n0, [modulus_g];\n\t"
            "ld.const.u32    n1, [modulus_g + 4];\n\t"
            "ld.const.u32    n2, [modulus_g + 8];\n\t"
            "ld.const.u32    n3, [modulus_g + 12];\n\t"
            "ld.const.u32    n4, [modulus_g + 16];\n\t"
            "ld.const.u32    n5, [modulus_g + 20];\n\t"
            "ld.const.u32    n6, [modulus_g + 24];\n\t"
            "ld.const.u32    n7, [modulus_g + 28];\n\t"
            "ld.const.u32    q, [n_g];\n\t"
            //main routine
            "mul.lo.u32   m, a0, q;\n\t"
            "mad.lo.cc.u32  a0, m, n0, a0;\n\t"
            "madc.lo.cc.u32  a1, m, n1, a1;\n\t"
            "madc.lo.cc.u32  a2, m, n2, a2;\n\t"
            "madc.lo.cc.u32  a3, m, n3, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n4, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n5, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n6, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n7, a7;\n\t"
            "addc.cc.u32  a8, a8, 0;\n\t"
            "addc.cc.u32  a9, a9, 0;\n\t"
            "addc.cc.u32  a10, a10, 0;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a1, m, n0, a1;\n\t"
            "madc.hi.cc.u32  a2, m, n1, a2;\n\t"
            "madc.hi.cc.u32  a3, m, n2, a3;\n\t"
            "madc.hi.cc.u32  a4, m, n3, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n4, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n5, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n6, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n7, a8;\n\t"
            "addc.cc.u32  a9, a9, 0;\n\t"
            "addc.cc.u32  a10, a10, 0;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a1, q;\n\t"
            "mad.lo.cc.u32  a1, m, n0, a1;\n\t"
            "madc.lo.cc.u32  a2, m, n1, a2;\n\t"
            "madc.lo.cc.u32  a3, m, n2, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n3, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n4, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n5, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n6, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n7, a8;\n\t"
            "addc.cc.u32  a9, a9, 0;\n\t"
            "addc.cc.u32  a10, a10, 0;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a2, m, n0, a2;\n\t"
            "madc.hi.cc.u32  a3, m, n1, a3;\n\t"
            "madc.hi.cc.u32  a4, m, n2, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n3, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n4, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n5, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n6, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n7, a9;\n\t"
            "addc.cc.u32  a10, a10, 0;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a2, q;\n\t"
            "mad.lo.cc.u32  a2, m, n0, a2;\n\t"
            "madc.lo.cc.u32  a3, m, n1, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n2, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n3, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n4, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n5, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n6, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n7, a9;\n\t"
            "addc.cc.u32  a10, a10, 0;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a3, m, n0, a3;\n\t"
            "madc.hi.cc.u32  a4, m, n1, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n2, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n3, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n4, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n5, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n6, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n7, a10;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a3, q;\n\t"
            "mad.lo.cc.u32  a3, m, n0, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n1, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n2, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n3, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n4, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n5, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n6, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n7, a10;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a4, m, n0, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n1, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n2, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n3, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n4, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n5, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n6, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n7, a11;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a4, q;\n\t"
            "mad.lo.cc.u32  a4, m, n0, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n1, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n2, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n3, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n4, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n5, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n6, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n7, a11;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a5, m, n0, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n1, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n2, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n3, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n4, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n5, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n6, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n7, a12;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a5, q;\n\t"
            "mad.lo.cc.u32  a5, m, n0, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n1, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n2, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n3, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n4, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n5, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n6, a11;\n\t"
            "madc.lo.cc.u32  a12, m, n7, a12;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a6, m, n0, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n1, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n2, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n3, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n4, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n5, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n6, a12;\n\t"
            "madc.hi.cc.u32  a13, m, n7, a13;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a6, q;\n\t"
            "mad.lo.cc.u32  a6, m, n0, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n1, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n2, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n3, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n4, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n5, a11;\n\t"
            "madc.lo.cc.u32  a12, m, n6, a12;\n\t"
            "madc.lo.cc.u32  a13, m, n7, a13;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a7, m, n0, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n1, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n2, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n3, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n4, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n5, a12;\n\t"
            "madc.hi.cc.u32  a13, m, n6, a13;\n\t"
            "madc.hi.cc.u32  a14, m, n7, a14;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mul.lo.u32   m, a7, q;\n\t"
            "mad.lo.cc.u32  a7, m, n0, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n1, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n2, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n3, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n4, a11;\n\t"
            "madc.lo.cc.u32  a12, m, n5, a12;\n\t"
            "madc.lo.cc.u32  a13, m, n6, a13;\n\t"
            "madc.lo.cc.u32  a14, m, n7, a14;\n\t"
            "add.cc.u32  a15, a15, 0;\n\t"
            "mad.hi.cc.u32  a8, m, n0, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n1, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n2, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n3, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n4, a12;\n\t"
            "madc.hi.cc.u32  a13, m, n5, a13;\n\t"
            "madc.hi.cc.u32  a14, m, n6, a14;\n\t"
            "mad.hi.cc.u32  a15, m, n7, a15;\n\t"
            //pack result back
            "mov.b64         %0, {a8,a9};\n\t"  
            "mov.b64         %1, {a10,a11};\n\t"
            "mov.b64         %2, {a12,a13};\n\t"  
            "mov.b64         %3, {a14,a15};\n\t"
            : "=l"(w.nn[0]), "=l"(w.nn[1]), "=l"(w.nn[2]), "=l"(w.nn[3])
            : "l"(T.nn[0]), "l"(T.nn[1]), "l"(T.nn[2]), "l"(T.nn[3]),
                "l"(T.nn[4]), "l"(T.nn[5]), "l"(T.nn[6]), "l"(T.nn[7]));

    
    if (FASTEST_256_cmp(w, modulus_g) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        w = FASTEST_256_sub(w, modulus_g);
    }
	
    return w;
}


//NB: look carefully on line 11 on page 31 of http://eprints.utar.edu.my/2494/1/CS-2017-1401837-1.pdf
//and find an opportunity for additional speedup
DEVICE_FUNC inline uint256_g mont_mul_256_asm_CIOS(const uint256_g& u, const uint256_g& v)
{
     uint256_g w;

     asm (  ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
            ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
            ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
            ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"
            ".reg .u32 m, q, x, y;\n\t"
            //unpacking operands
            "mov.b64         {a0,a1}, %4;\n\t"
            "mov.b64         {a2,a3}, %5;\n\t"
            "mov.b64         {a4,a5}, %6;\n\t"
            "mov.b64         {a6,a7}, %7;\n\t"
            "mov.b64         {b0,b1}, %8;\n\t"
            "mov.b64         {b2,b3}, %9;\n\t"
            "mov.b64         {b4,b5}, %10;\n\t"
            "mov.b64         {b6,b7}, %11;\n\t"
            "ld.const.u32    n0, [modulus_g];\n\t"
            "ld.const.u32    n1, [modulus_g + 4];\n\t"
            "ld.const.u32    n2, [modulus_g + 8];\n\t"
            "ld.const.u32    n3, [modulus_g + 12];\n\t"
            "ld.const.u32    n4, [modulus_g + 16];\n\t"
            "ld.const.u32    n5, [modulus_g + 20];\n\t"
            "ld.const.u32    n6, [modulus_g + 24];\n\t"
            "ld.const.u32    n7, [modulus_g + 28];\n\t"
            "ld.const.u32    q, [n_g];\n\t"
            //main routine - step 1
            "mul.lo.u32 r0, a0, b0;\n\t"
            "mul.lo.u32 r1, a0, b1;\n\t"
            "mul.lo.u32 r2, a0, b2;\n\t"
            "mul.lo.u32 r3, a0, b3;\n\t"
            "mul.lo.u32 r4, a0, b4;\n\t"
            "mul.lo.u32 r5, a0, b5;\n\t"
            "mul.lo.u32 r6, a0, b6;\n\t"
            "mul.lo.u32 r7, a0, b7;\n\t"
            "mad.hi.cc.u32 r1, a0, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a0, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a0, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a0, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a0, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a0, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a0, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a0, b7, r1;\n\t"
            "addc.u32 y, 0, 0;\n\t"           
            //step - 2
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t"
            //step 3
            "mad.lo.cc.u32 r0, a1, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a1, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a1, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a1, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a1, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a1, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a1, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a1, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a1, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a1, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a1, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a1, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a1, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a1, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a1, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a1, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t" 
            //step 4
            "mad.lo.cc.u32 r0, a2, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a2, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a2, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a2, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a2, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a2, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a2, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a2, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a2, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a2, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a2, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a2, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a2, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a2, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a2, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a2, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t"
            //step 5
            "mad.lo.cc.u32 r0, a3, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a3, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a3, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a3, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a3, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a3, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a3, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a3, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a3, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a3, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a3, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a3, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a3, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a3, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a3, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a3, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t" 
            //step 6
            "mad.lo.cc.u32 r0, a4, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a4, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a4, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a4, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a4, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a4, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a4, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a4, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a4, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a4, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a4, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a4, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a4, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a4, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a4, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a4, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t"  
            //step 7
            "mad.lo.cc.u32 r0, a5, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a5, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a5, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a5, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a5, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a5, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a5, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a5, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a5, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a5, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a5, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a5, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a5, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a5, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a5, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a5, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t"
            //step 8
            "mad.lo.cc.u32 r0, a6, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a6, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a6, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a6, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a6, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a6, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a6, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a6, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a6, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a6, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a6, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a6, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a6, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a6, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a6, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a6, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t" 
            //step 9
            "mad.lo.cc.u32 r0, a7, b0, r0;\n\t"
            "madc.lo.cc.u32 r1, a7, b1, r1;\n\t"
            "madc.lo.cc.u32 r2, a7, b2, r2;\n\t"
            "madc.lo.cc.u32 r3, a7, b3, r3;\n\t"
            "madc.lo.cc.u32 r4, a7, b4, r4;\n\t"
            "madc.lo.cc.u32 r5, a7, b5, r5;\n\t"
            "madc.lo.cc.u32 r6, a7, b6, r6;\n\t"
            "madc.lo.cc.u32 r7, a7, b7, r7;\n\t"
            "addc.cc.u32 x, x, 0;\n\t" 
            "addc.u32 y, 0, 0;\n\t"   
            "mad.hi.cc.u32 r1, a7, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a7, b1, r1;\n\t"
            "madc.hi.cc.u32 r3, a7, b2, r1;\n\t"
            "madc.hi.cc.u32 r4, a7, b3, r1;\n\t"
            "madc.hi.cc.u32 r5, a7, b4, r1;\n\t"
            "madc.hi.cc.u32 r6, a7, b5, r1;\n\t"
            "madc.hi.cc.u32 r7, a7, b6, r1;\n\t"
            "madc.hi.cc.u32 x, a7, b7, r1;\n\t"
            "addc.u32 y, y, 0;\n\t"
            "mul.lo.u32   m, r0, q;\n\t"
            "mad.lo.cc.u32 r0, m, n0, r0;\n\t"
            "madc.hi.cc.u32 r0, m, n0, 0;\n\t"
            "mad.lo.cc.u32 r0, m, n1, r1;\n\t"
            "mad.lo.cc.u32  r1, m, n2, r2;\n\t"
            "madc.lo.cc.u32  r2, m, n3, r3;\n\t"
            "madc.lo.cc.u32  r3, m, n4, r4;\n\t"
            "madc.lo.cc.u32  r4, m, n5, r5;\n\t"
            "madc.lo.cc.u32  r5, m, n6, r6;\n\t"
            "madc.lo.cc.u32  r6, m, n7, r7;\n\t"
            "addc.cc.u32  r7, x, 0;\n\t"
            "addc.u32  x, y, 0;\n\t"
            "mad.hi.cc.u32 r1, m, n1, r1;\n\t"
            "madc.hi.cc.u32 r2, m, n2, r2;\n\t"
            "madc.hi.cc.u32 r3, m, n3, r3;\n\t"
            "madc.hi.cc.u32  r4, m, n4, r4;\n\t"
            "madc.hi.cc.u32  r5, m, n5, r5;\n\t"
            "madc.hi.cc.u32  r6, m, n6, r6;\n\t"
            "madc.hi.cc.u32  r7, m, n7, r7;\n\t"
            "addc.u32  x, x, 0;\n\t"
            //pack result back
            "mov.b64         %0, {r0,r1};\n\t"  
            "mov.b64         %1, {r2,r3};\n\t"
            "mov.b64         %2, {r4,r5};\n\t"  
            "mov.b64         %3, {r6,r7};\n\t"
            : "=l"(w.nn[0]), "=l"(w.nn[1]), "=l"(w.nn[2]), "=l"(w.nn[3])
            : "l"(u.nn[0]), "l"(u.nn[1]), "l"(u.nn[2]), "l"(u.nn[3]),
                "l"(v.nn[0]), "l"(v.nn[1]), "l"(v.nn[2]), "l"(v.nn[3]));
                                   
    if (FASTEST_256_cmp(w, modulus_g) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        w = FASTEST_256_sub(w, modulus_g);
    }
	
    return w;
}


#endif


