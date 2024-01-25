#pragma once
#include "./detect_architecture.h"
#include "./simd_flags.h"
#include <array>

// Calculated from "derive_horner_polynomial.cpp"
// STEPS:  1. settings are { grad_t=double, TOTAL_COEFFICIENTS=6, SINE_ROOT=0.5, ...defaults }
//         2. train with   { coefficient_t=double, TOTAL_SAMPLES=256 }
//         3. save coefficients and use them as starting coefficients for next run
//         4. retrain with { coefficient_t=float,  TOTAL_SAMPLES=1024 }
//         5. mean absolute error should be 3.63e-8
constexpr static std::array<float, 6> CHEBYSHEV_POLYNOMIAL_COEFFICIENTS = \
// {-25.13274193f,64.83582306f,-67.07688141f,38.50342560f,-14.10175610f,3.26890683f}
{-25.13274193f,64.83583069f,-67.07687378f,38.50016403f,-14.07150173f,3.20396066f} // improved
// {-25.13274192f,64.83582306f,-67.07662964f,38.49588013f,-14.04966354f,3.16160202f} // original
;

// NOTE: Manually unrolling the loop might be required on some compilers 
//       i.e. For scalar sine MSVC can vectorise manually unrolled code
static float chebyshev_sine(float x) {
    const float z = x*x;
    // g(x) = sum ai*x^2i
    constexpr int N = int(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS.size());
    float g = CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[N-1];
    for (int i = N-2; i >= 0; i--) {
        g = g*z + CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[i];
    }
    // h(x) = x*(x-0.5)*(x+0.5)
    const float h = x*(z-0.25f);
    // f(x) = h(x)*g(x)
    return h*g;
}

// x86
#if defined(__ARCH_X86__)
#include <immintrin.h>

#if defined(__SSE__)
static inline __m128 _mm_chebyshev_sine(__m128 x) {
    #if defined(__FMA__)
        #define __muladd(a,b,c) _mm_fmadd_ps(a,b,c)
    #else
        #define __muladd(a,b,c) _mm_add_ps(_mm_mul_ps(a,b),c)
    #endif
    const __m128 z = _mm_mul_ps(x,x);
    // g(x) = sum ai*x^2i
    constexpr int N = int(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS.size());
    __m128 g = _mm_set1_ps(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[N-1]);
    for (int i = N-2; i >= 0; i--) {
        g = __muladd(g,z,_mm_set1_ps(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[i]));
    }
    #undef __muladd
    // h(x) = x*(x-0.5)*(x+0.5)
    const __m128 h = _mm_mul_ps(x,_mm_sub_ps(z,_mm_set1_ps(0.25f)));
    // f(x) = h(x)*g(x)
    return _mm_mul_ps(h,g);
}
#endif

#if defined(__AVX__)
static inline __m256 _mm256_chebyshev_sine(__m256 x) {
    #if defined(__FMA__)
        #define __muladd(a,b,c) _mm256_fmadd_ps(a,b,c)
    #else
        #define __muladd(a,b,c) _mm256_add_ps(_mm_mul_ps(a,b),c)
    #endif
    const __m256 z = _mm256_mul_ps(x,x);
    // g(x) = sum ai*x^2i
    constexpr int N = int(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS.size());
    __m256 g = _mm256_set1_ps(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[N-1]);
    for (int i = N-2; i >= 0; i--) {
        g = __muladd(g,z,_mm256_set1_ps(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[i]));
    }
    #undef __muladd
    // h(x) = x*(x-0.5)*(x+0.5)
    const __m256 h = _mm256_mul_ps(x,_mm256_sub_ps(z,_mm256_set1_ps(0.25f)));
    // f(x) = h(x)*g(x)
    return _mm256_mul_ps(h,g);
}
#endif
#endif

// aarch64-neon
#if defined(__ARCH_AARCH64__)
#if defined(__SIMD_NEON__)
#include <arm_neon.h>
static inline float32x4_t vsineq_f32(float32x4_t x) {
    const float32x4_t z = vmulq_f32(x,x);
    // g(x) = sum ai*x^2i
    constexpr int N = int(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS.size());
    float32x4_t g = vmovq_n_f32(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[N-1]);
    for (int i = N-2; i >= 0; i--) {
        g = vfmaq_f32(vmovq_n_f32(CHEBYSHEV_POLYNOMIAL_COEFFICIENTS[i]),g,z);
    }
    // h(x) = x*(x-0.5)*(x+0.5)
    const float32x4_t h = vmulq_f32(x,vsubq_f32(z,vmovq_n_f32(0.25f)));
    // f(x) = h(x)*g(x)
    return vmulq_f32(h,g);
}
#endif
#endif

