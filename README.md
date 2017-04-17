# vec_expi

Euler's formula is:

    exp(i*x) = cos(x) + i*sin(x)

where i = sqrt(-1).

While this is a very common operation in signal processing, it is not
particularly fast to compute. Also, many implementations don't take
advantage of the fact that computing `sin` and `cos` together is only
slightly more effort than computing one of them individually.

This header-only C++ library provides the function:

    void vec_expi(const float *in, std::complex<float> *out, unsigned count);

which computes Euler's formula on a vector.

    out[0] = std::complex<float>( cosf(in[0]), sinf(in[0]) );
    out[1] = std::complex<float>( cosf(in[1]), sinf(in[1]) );
    out[2] = std::complex<float>( cosf(in[2]), sinf(in[2]) );
    ...
    // up to count-1

This implementation depends on two different header-only libraries to
provide support for x86 SIMD instructions.

 - [sse_mathfun.h](http://gruntthepeon.free.fr/ssemath/) which provides
    several transcendentals implemented in SSE1+MMX or SSE2, and
 - [avx_mathfun.h](http://software-lisc.fbk.eu/avx_mathfun/) which ports
    these same functions to AVX and AVX2

These libraries provide a `sincos` function that computes `sin` and `cos`
together. From the library:

> Since sin_ps and cos_ps are almost identical, sincos_ps could replace
> both of them. It is almost as fast, and gives you a free cosine with
> your sine.

`sse_mathfun.h` is included 'as is' from its website.

`avx_mathfun.h` needed a fair bit of modification to work with both gcc and
clang. It previously worked by defining AVX2 intrinsics (implemented with
SSE2) when AVX2 was not enabled. However, clang defines these intrinsics even
when AVX2 is not enabled, which caused redefinition errors throughout.

This modified version uses AVX2 intrinsics prefixed with `impl` to get a
different symbol name. If AVX2 is defined, then a forwarding function is
defined and the AVX2 intrinsic is called directly. If AVX2 is not defined,
the `impl` functions call SSE2 instructions that implement that AVX2
instruction.
