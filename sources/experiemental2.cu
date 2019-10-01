#include "cuda_structs.h"

//Yep, I'm going to use it!
#include <boost/preprocessor/control/expr_if.hpp>

/* Here we are going to implement RNS-based multiplication */

//both bases have form {2^k - c} for relatively small c

DEVICE_VAR CONST_MEMORY uint32_t A_RESIDUE_SYSTEM[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};
DEVICE_VAR CONST_MEMORY uint32_t B_RESIDUE_SYSTEM[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

//precomputed values "r" used in Barrett reduction algorithms

DEVICE_VAR CONST_MEMORY uint32_t A_BARETT_PRECOMPUTE[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};
DEVICE_VAR CONST_MEMORY uint32_t B_BARETT_PRECOMPUTE[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

//R = M % N - montgomery modulus

DEVICE_FUNC CONST_MEMORY uint256_g R = { 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

#define THREADS_PER_EXP 8

struct RNS_PAIR
{
    uint32_t a;
    uint32_t b;
};


DEVICE_FUNC __forceinline__ uint32_t get_fiber_idx()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret % THREADS_PER_EXP;
}


DEVICE_FUNC __forceinline__ RNS_PAIR RNS_ADD(const RNS_PAIR& left, const RNS_PAIR& right)
{
    RNS_PAIR res = {left.a + right.a, left.b + right.b};
    uint32_t idx = get_fiber_idx();

    if (res.a > A_RESIDUE_SYSTEN[idx])
        res.a -= A_RESIDUE_SYSTEN[idx];
    if (res.b > B_RESIDUE_SYSTEM[idx])
        res.b -= B_RESIDUE_SYSTEM[idx];
    
    return res;
}


DEVICE_FUNC __forceinline__ RNS_PAIR RNS_SUB(const RNS_PAIR& left, const RNS_PAIR& right)
{
    RNS_PAIR res = {left.a - right.a, left.b - right.b};
    uint32_t idx = get_fiber_idx();

    if (res.a >= A_RESIDUE_SYSTEN[idx])
        res.a += A_RESIDUE_SYSTEN[idx];
    if (res.b >= B_RESIDUE_SYSTEM[idx])
        res.b += B_RESIDUE_SYSTEM[idx];
    
    return res;
}


DEVICE_FUNC __forceinline__ uint32_t RNS_BARRETT_RED(uint64_t x, uint32_t n, uint32_t r)
{
    uint32_t ret;
    
    asm (  "{\n\t"
            ".reg .pred p1, p2;\n\t"
            ".reg .u32 x0, x1, r, n, t0, t1;\n\t"
            "mov.b64         {x0,x1}, %1;\n\t"
            "mov.b32         {r}, %3;\n\t"
            "mov.b32         {n}, %2;\n\t"
            "mul.hi.u32      t0, x0, r;\n\t"
            "madc.lo.cc.u32  t0, x1, r, t0;\n\t"
            "madc.hi.u32     t1, x1, r, 0;\n\t"
            "mul.lo.u32      t0, t1, n;\n\t"
            "mul.hi.u32      t1, x1, n;\n\t"
            "mad.lo.cc.u32   r1, a1, b0, r1;\n\t"
            "sub.cc.u32      t0, x0, t0;\n\t"
         	"subc.u32        t1, x1, t1;\n\t"

            "setp.ne.u32 p1, t1, 0;\n\t"
            "setp.ge.u32 p2, t0, n;\n\t"
            "and.pred p1, p1, p2;\n\t"
            "@p sub t0, t0, n;\n\t"
            "mov.b32 %0, t0;\n\t" 

            : "=r"(ret) : "l"(x), "r"(n), "r"(r));

    return ret;  
}


DEVICE_FUNC __forceinline__ RNS_PAIR RNS_MUL(const RNS_PAIR& left, const RNS_PAIR& right)
{
    uint64_t a, b;
    RNS_PAIR res;

    a.low = device_long_mul(left.a, right.a, &a.high);
    b.low = device_long_mul(left.b, right.b, &b.high);

    //use Barett reduction
    uint32_t idx = get_fiber_idx();

    res.a = RNS_BARRETT_RED(a, A_RESIDUE_SYSTEN[idx], A_BARETT_PRECOMPUTE[idx]);
    res.b = RNS_BARRETT_RED(b, B_RESIDUE_SYSTEN[idx], B_BARETT_PRECOMPUTE[idx]);
    return res;
}

#define ASM_REG(I) BOOST_PP_CAT(" x", I)
#define ASM_REG_WITH_COMMA(I) BOOST_PP_CAT(ASM_REG(I), BOOST_PP_COMMA)

#define ASM_REDUCTION_STEP(I, _) \ 
	BOOST_PP_IF(BOOST_PP_EQUAL(I, 0),
		BOOST_PP_SEQ_CAT(("mul.hi.cc.u32 r0, r0, c\n\t"),\
		BOOST_PP_SEQ_CAT(("madc.hi.u32")(ASM_REG_WITH_COMMA(I))(ASM_REG_WITH_COMMA(I))(" c, 0;\n\t"))\
	BOOST_PP_SEQ_CAT(("madc.lo.cc.u32")(ASM_REG_WITH_COMMA(I))(ASM_REG_WITH_COMMA(BOOST_PP_INC(I)))(" c,")(ASM_REG(I))(";\n\t"))

#define ASM_REDUCTION(N, OVERFLOW_FLAG) \
	"mad.lo.cc.u32 r0, x0, c, r0;\n\t"\
	"addc.u32 r1, r1, 0;\n\t"\
	BOOST_PP_REPEAT(N, ASM_REDUCTION_STEP, 0)\
	BOOST_PP_IF(OVERFOW_FLAG,\ 
		BOOST_PP_SEQ_CAT(("madc.hi.u32")(ASM_REG_WITH_COMMA(N))(ASM_REG_WITH_COMMA(N))(" c, 0;\n\t"),\
		BOOST_PP_EMPTY())

DEVICE_FUNC __forceinline__ uint32_t RNS_RED(const uint256_g& elem, uint32_t modulus)
{
    // algorithm 14.47 from handbook of applied cryptography
    // Algorithm Reduction modulo m = b^t -c
    // INPUT: positive integer x
    // OUTPUT: r = x mod m.
    // 1. q_0 <—[х/(Ь^t)], r_0 = x — q_0 b^t, r = r_0, i = 0.
    // 2. While q_i > 0 do the following:
        // 2.1 q_{i+1} = q_i*c/ (b^t), r_{i+i} = q_i*c - q_{i+i}*b^t.
        // 2.2 i = i + 1, r = r + r_i.
    // 3. While r > m do: r = r — m. (NB: this loop is executed at most 2 times)
    // 4. Return(r).

    //m = 2^32 - c

    uint64_t c = { ~modulus, 0};
    uint64_t ret;
    
    asm (  "{\n\t"
            ".reg .u32 r1, r0, x0, x1, x2, x3, x4, x5, x6, c;\n\t"
            "mov.b64         {r0,x0}, %1;\n\t"
            "mov.b64         {x1,x2}, %2;\n\t"
            "mov.b64         {x3,x4}, %3;\n\t"
            "mov.b64         {x5,x6}, %4;\n\t"
            "mov.b32         {r1, c}, %5;\n\t"

            //we assume that c is 8 bits long maximum!

            ASM_REDUCTION(7, true);
			ASM_REDUCTION(7, false);
            ASM_REDUCTION(6, false);
            ASM_REDUCTION(5, false);
            ASM_REDUCTION(5, true);
            ASM_REDUCTION(4, false);
            ASM_REDUCTION(3, false);
            ASM_REDUCTION(2, false);
            ASM_REDUCTION(1, true);
            ASM_REDUCTION(1, false);

            "mov.b64        %0, {r0, r1}"
            : "=l"(ret) : "l"(x.nn[0]), "l"(x.nn[1]), "l"(x.nn[2]), "l"(x.nn[3]), "l"(x.nn[4]));

    return ret;  
}

#undef ASM_REG
#undef ASM_REG_WITH_COMMA
#undef ASM_REDUCTION_STEP
#undef ASM_REDUCTION


DEVICE_FUNC __inline__ RNS_PAIR to_RNS_repr(const uint256_g& elem)
{
    //NB: we assume all elements are in Montgomery form! (with R = M % N)

    uint32_t idx = get_fiber_idx();
    RNS_PAIR res;

    res.a = RNS_RED(elem, A_RESIDUE_SYSTEN[idx]);
    res.b = RNS_RED(elem, B_RESIDUE_SYSTEN[idx]); 

    return res;
}


DEVICE_FUNC __inline__ uint32_t RNS_BASE_EXT(uint32_t x, uint32_t alpha)
{

}



DEVICE_FUNC __inline__ uint32_t RNS_mont_mul(const uint32_t a, const uint32_t b)
{
    //We use ALGORITHM 1 from
    //"Modular Multiplication and Base Extensions in Residue Number Systems"

    uint32_t q = RES_MUL(a, b);
    q = RES_MUL(q, N_INV_IN_MAIN_RESIDUE_SYSTEM[get_fiver_idx()]);

    //Now we need to convert representation from one residue system to another
    //we use 

}



DEVICE_FUNC __inline__ uint32_t RNS_exp(const ec_point& pt, const uint256_g& power)
{
    //we follow paper: 
    //"RNS-Based Elliptic Curve Point Multiplication for Massive Parallel Architectures"
    //TABLE 2

    RNS_PAIR x = to_RNS_repr(pt.x);
    RNS_PAIR z = to_RNS_repr(pt.y)

    //TODO: how to make it correctly

    x_G = RNS_MUL(x, );
    z_G = to_mont_form(z);

    //Init phase

    RNS_PAIR A = RNS_MUL(x_G, x_G);
    RNS_PAIR B = RNS_MUL(b, x_G);
    A = RNS_REDUCE(A);
    B = RNS_REDUCE(B);

    RNS_PAIR C = RNS_SUB(A, a);
    C = RNS_MUL(C, C);
    A = RNS_MUL(x_G, A);
    A = RNS_REDUCE(A);
    C = RNS_REDUCE(C);
    
    //here I'm going to multiply by constant - which effect will it have?
    X_Q = RNS_SUB(C)
}



