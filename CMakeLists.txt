cmake_minimum_required(VERSION 3.10)
project(simd_testing)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

function(add_example target)
    add_executable(${target} ${SRC_DIR}/${target}.cpp)
    target_include_directories(${target} PRIVATE ${INCLUDE_DIR})
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

add_example(benchmark_chebyshev_sine)
add_example(derive_horner_polynomial)
