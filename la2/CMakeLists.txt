cmake_minimum_required (VERSION 3.1)
project (imc)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-g")
set (CMAKE_CXX_FLAGS "-O3")
add_executable(la2 la2.cpp imc/MultilayerPerceptron.cpp imc/MultilayerPerceptron.h imc/util.h)
