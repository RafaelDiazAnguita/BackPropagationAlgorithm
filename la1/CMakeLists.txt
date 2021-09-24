cmake_minimum_required (VERSION 3.1)
project (imc)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-g")
add_executable(la1 la1.cpp imc/MultilayerPerceptron.cpp imc/MultilayerPerceptron.h imc/util.h)
