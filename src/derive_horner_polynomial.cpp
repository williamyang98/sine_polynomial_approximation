// Compiling with clang or gcc
// clang main.cpp -o main -march=native -O3 -ffast-math -Wextra -Werror -Wunused-parameter
// Use "-march=native -ffast-math -O3" for faster gradient descent

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <cmath>
#include <vector>

#define USE_SIGNAL_HANDLER 1

#if USE_SIGNAL_HANDLER
static bool is_running = true;
#if _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
BOOL WINAPI sighandler(DWORD signum) {
    if (signum == CTRL_C_EVENT) {
        fprintf(stderr, "Signal caught, exiting!\n");
        is_running = false;
        return TRUE;
    }
    return FALSE;
}
#else
#include <errno.h>
#include <signal.h>
static void sighandler(int signum) {
    fprintf(stderr, "Signal caught, exiting! (%d)\n", signum);
    is_running = false;
}
#endif
#endif

// Configuration
using grad_t = double;
using coefficient_t = double;
constexpr size_t PRINT_ITER = 10'000'000;
constexpr size_t TOTAL_ITER_ERROR_PLATEAU_THRESHOLD = 10'000;

// Learning rate update rule when minimum error reaches a plateau
// - If we update than rescale the learning rate
// - Otherwise we exit early
constexpr bool IS_UPDATE_LEARNING_RATE = true;
constexpr grad_t LEARNING_RATE_RESCALER = 1.1;

// We are solving for a[n] in g(x) = sum a[n]*x^(2n)
// Where f(x) = h(x)*g(x), where: 
//   - f(x) = target function
//   - h(x) = modifier function to reduce complexity of function g(x) needs to solve
//   - g(x) = polynomial that can be calculated using Horner's method
//            https://en.wikipedia.org/wiki/Horner%27s_method

// Gradient descent hyperparameters
// - We are approximating f(x) in the range: x = [X_SAMPLE_START, X_SAMPLE_END] with TOTAL_SAMPLES
// - g(x) = sum a[n]*x^(2n) for n = [0, TOTAL_COEFFICIENTS-1]
// - learning rate: INITIAL_LEARNING_RATE
//   - Decrease learning rate if unstable
// - rescaling gradients for de/da[n] by: x^(n*GRAD_RESCALE_MAGNITUDE + GRAD_RESCALE_BIAS)
//   - NOTE: This drastically improves training speed, you should tweak it
//   - Decrease GRAD_RESCALE_MAGNITUDE if training is unstable
//   - NOTE: You should tweak this value before tweaking learning rate

// Sample functions to approximate
#if 1
// y = sin(2*pi*x)
template <typename T>
inline static T get_y_target(T x) {
    return std::sin(T(M_PI)*2.0*x);
}
template <typename T>
inline static T get_h(T x) {
    // constraint so that roots match sin(2*pi*x)
    constexpr T B0 = T(0.5*0.5);
    const T z = x*x;
    return x*(z-B0);
}
constexpr size_t TOTAL_SAMPLES = 256;
constexpr grad_t X_SAMPLE_START = grad_t(0);
constexpr grad_t X_SAMPLE_END = grad_t(0.5);
constexpr size_t TOTAL_COEFFICIENTS = 6;
constexpr grad_t INITIAL_LEARNING_RATE = grad_t(1e0);
constexpr grad_t GRAD_RESCALE_MAGNITUDE = 4;
constexpr grad_t GRAD_RESCALE_BIAS = 1;
#elif 0
// y = cos(2*pi*x)
template <typename T>
inline static T get_y_target(T x) {
    return std::cos(T(M_PI)*2.0*x);
}
template <typename T>
inline static T get_h(T x) {
    // constraint so that roots match cos(2*pi*x)
    constexpr T B0 = T(0.25*0.25);
    constexpr T B1 = T(0.75*0.75);
    const T z = x*x;
    return (z-B0)*(z-B1);
}
constexpr size_t TOTAL_SAMPLES = 256;
constexpr grad_t X_SAMPLE_START = grad_t(0);
constexpr grad_t X_SAMPLE_END = grad_t(0.5 + 1e-2);
constexpr size_t TOTAL_COEFFICIENTS = 6;
constexpr grad_t INITIAL_LEARNING_RATE = grad_t(1e0);
constexpr grad_t GRAD_RESCALE_MAGNITUDE = 4;
constexpr grad_t GRAD_RESCALE_BIAS = 1;
#else
// y = x^2 * exp(pi*x^2) * sin(2pix)
template <typename T>
inline static T get_y_target(T x) {
    const T z = x*x;
    return z*std::exp(T(M_PI)*z)*std::sin(2*T(M_PI)*x);
}
template <typename T>
inline static T get_h(T x) {
    constexpr T B0 = T(0.5*0.5);
    const T z = x*x;
    return z*x*(z-B0);
}
constexpr size_t TOTAL_SAMPLES = 256;
constexpr grad_t X_SAMPLE_START = grad_t(0);
constexpr grad_t X_SAMPLE_END = grad_t(0.5 + 1e-2);
constexpr size_t TOTAL_COEFFICIENTS = 8;
constexpr grad_t INITIAL_LEARNING_RATE = grad_t(1e0);
constexpr grad_t GRAD_RESCALE_MAGNITUDE = 3.6; // needed to be decreased for stable gradient descent
constexpr grad_t GRAD_RESCALE_BIAS = 1;
#endif

inline static coefficient_t get_horner_polynomial(coefficient_t x, coefficient_t* A) {
    // g(x) = sum a[n]*x^(2n)
    // Use Horner's method to calculate this efficiently
    // https://en.wikipedia.org/wiki/Horner%27s_method
    const coefficient_t z = x*x;
    coefficient_t g = A[TOTAL_COEFFICIENTS-1];
    for (size_t i = 1; i < TOTAL_COEFFICIENTS; i++) {
        size_t j = TOTAL_COEFFICIENTS-i-1;
        g = g*z + A[j];
    }
    return g;
}

int main(int /*argc*/, char** /*argv*/) {
#if USE_SIGNAL_HANDLER
#if _WIN32
    SetConsoleCtrlHandler(sighandler, TRUE);
#else
    struct sigaction sigact;
    sigact.sa_handler = sighandler;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGINT, &sigact, nullptr);
    sigaction(SIGTERM, &sigact, nullptr);
    sigaction(SIGQUIT, &sigact, nullptr);
    sigaction(SIGPIPE, &sigact, nullptr);
#endif
#endif
    coefficient_t poly[TOTAL_COEFFICIENTS] = {coefficient_t(0)};

    static_assert(X_SAMPLE_END > X_SAMPLE_START, "Range of evaluation must be from x_min to x_max");
    constexpr grad_t EVAL_STEP = (X_SAMPLE_END-X_SAMPLE_START)/grad_t(TOTAL_SAMPLES);
    auto X_in = std::vector<grad_t>(TOTAL_SAMPLES);
    auto Y_expected = std::vector<grad_t>(TOTAL_SAMPLES);
    auto H_in = std::vector<grad_t>(TOTAL_SAMPLES);
    for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
        const grad_t x = X_SAMPLE_START + grad_t(i)*EVAL_STEP;
        const grad_t y = get_y_target(x);
        const grad_t h = get_h(x);
        X_in[i] = x;
        Y_expected[i] = y;
        H_in[i] = h;
    }
 
    // normalised gradients for each term
    // de/da[n] = 2*(f-y)*h*x^2n 
    //          = 2*e(x)*h*x^2n
    // Assume e(x) converges to 0, then e_max = e^2n
    // de/da[n] = 2*h*x^4n
    // In reality the x^4n factor is approximated as x^(an+b)
    // where a,b are hyperparameters
    grad_t gradient_scale[TOTAL_COEFFICIENTS] = {0.0};
    for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
        grad_t scale = 0.0;
        for (size_t j = 0; j < TOTAL_SAMPLES; j++) {
            const grad_t x = grad_t(X_in[j]);
            const grad_t h = grad_t(H_in[j]);
            const grad_t n = grad_t(i)*GRAD_RESCALE_MAGNITUDE + GRAD_RESCALE_BIAS;
            const grad_t x_n = std::pow(x, n);
            scale += (h*x_n);
        }
        scale = std::abs(scale);
        scale /= grad_t(TOTAL_SAMPLES);
        gradient_scale[i] = grad_t(1) / scale;
    }
    printf("grad_rescale = [");
    for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
        printf("%.2e,", gradient_scale[i]);
    }
    printf("]\n");
    printf("is_update_learning_rate: %u\n", IS_UPDATE_LEARNING_RATE);
 
    constexpr grad_t NORM_SAMPLES = 1.0f / grad_t(TOTAL_SAMPLES);
    grad_t learning_rate = INITIAL_LEARNING_RATE;
    coefficient_t best_poly[TOTAL_COEFFICIENTS] = {coefficient_t(0)};
    grad_t error_minimum = grad_t(1e6);
    size_t error_minimum_iter = 0;
    size_t total_iter_error_plateau = 0;
    for (size_t iter = 0; ; iter++) { 
        grad_t total_error = grad_t(0);
        grad_t gradients[TOTAL_COEFFICIENTS] = {grad_t(0)};
        for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
            const coefficient_t x = X_in[i];
            const coefficient_t y_given = coefficient_t(H_in[i])*get_horner_polynomial(coefficient_t(x), poly);
            const grad_t y_expected = Y_expected[i];
            const grad_t error = grad_t(y_given) - y_expected;
            // gradient descent
            // g(x) = sum a[n]*x^(2n) 
            // f(x) = g(x)*h(x)
            // e(x) = [f(x) - y]^2
            // de/df = 2*(f-y)
            // df/dg = h
            // de/dg = de/df * df/dg, chain rule
            // de/dg = 2*(f-y)*h
            // de/da[n] = de/dg * dg/da[n]
            // dg/da[n] = x^(2n)
            // de/da[n] = 2*(f-y)*h*x^2n
            const grad_t z = x*x;
            const grad_t h = H_in[i];
            grad_t gradient = error*h*grad_t(2);
            gradients[0] += gradient;
            for (size_t j = 1; j < TOTAL_COEFFICIENTS; j++) {
                gradient *= z;
                gradients[j] += gradient;
            }
            total_error += std::abs(error);
        }

        // normalised error and gradient
        total_error *= NORM_SAMPLES;
        for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
            const grad_t scale = NORM_SAMPLES * gradient_scale[i];
            gradients[i] *= scale;
        }

        // save lowest error polynomial
        if (total_error < error_minimum && !std::isnan(total_error)) {
            error_minimum = total_error;
            error_minimum_iter = iter;
            total_iter_error_plateau = 0;
            for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
                best_poly[i] = poly[i];
            }
        } else {
            total_iter_error_plateau++;
        }

        if (iter % PRINT_ITER == 0) {
            printf("[%zu] error=%.2e, grad=[", iter, total_error);
            for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
                printf("%.2e,", gradients[i]);
            }
            printf("]\n");
        }

        // apply gradients
        bool has_nan = false;
        for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
            const grad_t delta = gradients[i] * learning_rate;
            const grad_t coeff = grad_t(poly[i]) - delta;
            poly[i] = coefficient_t(coeff);
            if (std::abs(coeff) > 1e5f) {
                printf("[%zu] [error] Detected exploding coefficient exiting\n", iter);
                has_nan = true;
                break;
            }
            if (std::isnan(coeff) || std::isinf(coeff)) {
                printf("[%zu] [error] Detected NAN in coefficient exiting\n", iter);
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            printf("[%zu] [error] Early exit due to bad values, printing them now\n", iter);
            printf("  learning_rate: %.2e\n", learning_rate);
            printf("  grad: [");
            for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
                printf("%.2e,", gradients[i]);
            }
            printf("]\n");
            printf("  poly: [");
            for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
                printf("%.2e,", poly[i]);
            }
            printf("]\n");
            break;
        }
        if (total_iter_error_plateau >= TOTAL_ITER_ERROR_PLATEAU_THRESHOLD) {
            if (IS_UPDATE_LEARNING_RATE) {
                total_iter_error_plateau = 0;
                learning_rate *= LEARNING_RATE_RESCALER;
            } else {
                printf("[%zu] [info] Exiting since error has plateaued after %zu iters\n", 
                    iter, TOTAL_ITER_ERROR_PLATEAU_THRESHOLD);
                break;
            }
        }
        if (!is_running) break;
    }
    printf("\n[BEST RESULTS]\n");
    printf("step: %zu\n", error_minimum_iter);
    printf("mean_absolute_error: %.2e\n", error_minimum);
    printf("poly[%zu] = {", TOTAL_COEFFICIENTS);
    for (size_t i = 0; i < TOTAL_COEFFICIENTS; i++) {
        constexpr size_t total_dp = (sizeof(coefficient_t) == 4) ? 10 : 16;
        printf("%.*f", int(total_dp), best_poly[i]);
        if (i != TOTAL_COEFFICIENTS-1) {
            printf(",");
        }
    }
    printf("}\n");
    return 0;
}
