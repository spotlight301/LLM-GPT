cmake_minimum_required(VERSION 3.14)

project(libjustlm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LM_BACKEND "llama.cpp" CACHE STRING "The language model backend to use")
set(LM_PYBIND No CACHE BOOL "If python bindings should be build")

if (LM_BACKEND STREQUAL "libnc gpt2")
    add_library(libjustlm STATIC libjustlm_gpt2.cpp gpt2/arith.c gpt2/cp_utils.c gpt2/gpt2tc.c)
    target_link_libraries(libjustlm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpt2/libnc.so pthread)
elseif (LM_BACKEND STREQUAL "llama.cpp")
    add_subdirectory(llama.cpp)
    add_library(libjustlm STATIC libjustlm_llama.cpp)
    target_link_libraries(libjustlm PRIVATE llama)
else()
    message(FATAL_ERROR "LM_BACKEND '${LM_BACKEND}' is unsupported. Please use either 'libnc gpt2' or 'llama.cpp'.")
endif()

if (LM_PYBIND)
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)
    pybind11_add_module(libjustlm_py pybind.cpp)
    target_link_libraries(libjustlm_py PRIVATE libjustlm)
endif()

target_include_directories(libjustlm PUBLIC include/)
