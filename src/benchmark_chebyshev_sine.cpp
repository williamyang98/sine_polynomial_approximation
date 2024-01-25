#include "./detect_architecture.h"
#include "./simd_flags.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "./aligned_allocator.h"
#include "./chebyshev_sine.h"
#include "./span.h"
#include "./timer.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(__ARCH_X86__)
    #if defined(_MSC_VER) && !defined(__clang__)
        #define IS_INTEL_SVML 1
    #endif
    #if !defined(__x86_64__)
        #define __x86_64__ 1
    #endif
    #if defined(__SSE__)
        #define USE_SSE_AUTO
        #define SSE_MATHFUN_WITH_CODE
        #include "./sse_mathfun.h"
    #endif
    #if defined(__AVX__)
        #include "./avx_mathfun.h"
    #endif
#endif

static float calculate_error(tcb::span<const float> X, tcb::span<const float> Y) {
    assert(X.size() == Y.size());
    const size_t N = X.size();
    float error = 0.0f;
    for (size_t i = 0; i < N; i++) {
        const auto v = X[i]-Y[i];
        error += std::abs(v);
    }
    return error / float(N);
}

int main(int /*argc*/, char** /*argv*/) {
    constexpr size_t TOTAL_SAMPLES = 8192;
    constexpr size_t TOTAL_TRIALS = 10240;
    using Alloc = aligned_allocator<float, 32>; // avx2 32byte alignment
    auto X = std::vector<float, Alloc>(TOTAL_SAMPLES); // sin(2*pi*x)
    auto X_svml = std::vector<float, Alloc>(TOTAL_SAMPLES); // sin(x)
    auto Y_std = std::vector<float, Alloc>(TOTAL_SAMPLES);
    auto Y_benchmark = std::vector<float, Alloc>(TOTAL_SAMPLES);
    {
        const float step = 1.0f/float(TOTAL_SAMPLES);
        constexpr float SCALE = 2.0f*float(M_PI);
        for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
            X[i] = -0.5f + step*float(i);
            X_svml[i] = SCALE*X[i];
        }
    }

    printf("[PROGRESS]\n");
    uint64_t time_std_ns;
    // Reference example that we compare everything to
    {
        printf("o std");
        fflush(stdout);
        Timer timer;
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
                constexpr float SCALE = 2.0f*float(M_PI);
                Y_std[i] = std::sin(SCALE*X[i]);
            }
        }
        time_std_ns = timer.get_delta();
        printf("\r- std\n");
    }

    struct Benchmark {
        std::string name;
        uint64_t time_taken_ns = 0;
        float mean_absolute_error = 0.0f;
    };
    std::vector<Benchmark> benchmarks;

    #define RUN_BENCHMARK(NAME, BLOCK) \
    {\
        printf("o %s", NAME);\
        fflush(stdout);\
        Timer timer;\
        BLOCK;\
        const uint64_t time_ns = timer.get_delta();\
        const float error = calculate_error(Y_std, Y_benchmark);\
        benchmarks.push_back({ NAME, time_ns, error });\
        printf("\r- %s\n", NAME);\
    }

    RUN_BENCHMARK("cheby_scalar", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
                Y_benchmark[i] = chebyshev_sine(X[i]);
            }
        }
    })
    #if defined(__SSE__)
    RUN_BENCHMARK("cheby_sse", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(__m128)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                __m128 x = _mm_load_ps(&X[i]);
                __m128 y = _mm_chebyshev_sine(x);
                _mm_store_ps(&Y_benchmark[i], y);
            }
        }
    })
    #endif
    #if defined(__AVX__)
    RUN_BENCHMARK("cheby_avx", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(__m256)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                __m256 x = _mm256_load_ps(&X[i]);
                __m256 y = _mm256_chebyshev_sine(x);
                _mm256_store_ps(&Y_benchmark[i], y);
            }
        }
    })
    #endif
#if defined(__ARCH_AARCH64__)
    #if defined(__SIMD_NEON__)
    RUN_BENCHMARK("cheby_neon", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(float32x4_t)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                float32x4_t x = vld1q_f32(&X[i]);
                float32x4_t y = vsineq_f32(x);
                vst1q_f32(&Y_benchmark[i], y);
            }
        }
    });
    #endif
#endif
#if IS_INTEL_SVML
    #if defined(__SSE__)
    RUN_BENCHMARK("svml_sse", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(__m128)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                __m128 x = _mm_load_ps(&X_svml[i]);
                __m128 y = _mm_sin_ps(x);
                _mm_store_ps(&Y_benchmark[i], y);
            }
        }
    })
    #endif
    #if defined(__AVX__)
    RUN_BENCHMARK("svml_avx", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(__m256)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                __m256 x = _mm256_load_ps(&X_svml[i]);
                __m256 y = _mm256_sin_ps(x);
                _mm256_store_ps(&Y_benchmark[i], y);
            }
        }
    })
    #endif
#endif
    #if defined(__SSE__)
    RUN_BENCHMARK("mathfun_sse", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(__m128)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                __m128 x = _mm_load_ps(&X_svml[i]);
                __m128 y = sin_ps(x);
                _mm_store_ps(&Y_benchmark[i], y);
            }
        }
    })
    #endif
    #if defined(__AVX__)
    RUN_BENCHMARK("mathfun_avx", {
        for (size_t iter = 0; iter < TOTAL_TRIALS; iter++) {
            constexpr size_t STRIDE = sizeof(__m256)/sizeof(float);
            for (size_t i = 0; i < TOTAL_SAMPLES; i+=STRIDE) {
                __m256 x = _mm256_load_ps(&X_svml[i]);
                __m256 y = sin256_ps(x);
                _mm256_store_ps(&Y_benchmark[i], y);
            }
        }
    })
    #endif

    size_t max_name_length = 3;
    for (auto& benchmark: benchmarks) {
        const size_t len = benchmark.name.length();
        max_name_length = (max_name_length > len) ? max_name_length : len;
    }
    std::vector<char> separators;
    separators.resize(max_name_length+1);
    for (size_t i = 0; i < max_name_length; i++) {
        separators[i] = '-';
    }
    separators[max_name_length] = '\0';

    printf("\n[RESULTS]\n");
    printf("| %*s | Time (ns) | Speed up |    MAE   |\n", int(max_name_length), "Name");
    printf("| %*s | --------- | -------- | -------- |\n", int(max_name_length), separators.data());
    printf("| %*s |  %.2e |          |          |\n", int(max_name_length), "std", float(time_std_ns)/float(TOTAL_TRIALS));
    for (auto& benchmark: benchmarks) {
        const float sample_ns = float(benchmark.time_taken_ns)/float(TOTAL_TRIALS);
        const float ratio = float(time_std_ns)/float(benchmark.time_taken_ns);
        const float error = benchmark.mean_absolute_error;
        printf("| %*s |  %.2e | %8.2f | %.2e |\n", int(max_name_length), benchmark.name.c_str(), sample_ns, ratio, error);
    }
    return 0;
}
