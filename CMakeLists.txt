cmake_minimum_required(VERSION 3.14)

project(libjustlm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LM_PYBIND No CACHE BOOL "If python bindings should be build")

add_subdirectory(llama.cpp)
add_library(libjustlm STATIC
    include/justlm.hpp justlm.cpp
    justlm_llama.hpp
    justlm_gptj.hpp gptj/gptj.cpp gptj/gptj.hpp gptj/utils.cpp gptj/utils.hpp gptj/additions.hpp gptj/additions.cpp
    include/justlm_pool.hpp justlm_pool.cpp
)
target_link_libraries(libjustlm PRIVATE llama)

if (LM_PYBIND)
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)
    pybind11_add_module(libjustlm_py pybind.cpp)
    target_link_libraries(libjustlm_py PRIVATE libjustlm)
endif()

target_include_directories(libjustlm PUBLIC include/)
