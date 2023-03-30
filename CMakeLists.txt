cmake_minimum_required(VERSION 3.14)

project(libjustlm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(libjustlm STATIC
  libjustlm_core.cpp
  justlm.hpp
)

set(LM_BACKEND "llama.cpp" CACHE STRING "The language model backend to use")

if (LM_BACKEND STREQUAL "libnc gpt2")
    add_library(libjustlm_gpt2 STATIC libjustlm_gpt2.cpp gpt2/arith.c gpt2/cp_utils.c gpt2/gpt2tc.c)
    target_link_libraries(libjustlm_gpt2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gpt2/libnc.so pthread)
elseif (LM_BACKEND STREQUAL "llama.cpp")
    add_subdirectory(llama.cpp)
    add_library(libjustlm_llama STATIC libjustlm_llama.cpp)
    target_link_libraries(libjustlm_llama PRIVATE llama)
else()
    message(FATAL_ERROR "LM_BACKEND '${LM_BACKEND}' is unsupported. Please use either 'libnc gpt2' or 'llama.cpp'.")
endif()
