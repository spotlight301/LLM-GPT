cmake_minimum_required(VERSION 3.14)

project(libjustlm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LM_PYBIND No CACHE BOOL "If Libjustlm Python bindings should be build")
set(LM_COSCHED No CACHE BOOL "If Libjustlm should make use of CoSched")
set(LM_NOEXCEPT No CACHE BOOL "If exceptions should be disabled")

if (LM_COSCHED)
    set(CMAKE_CXX_STANDARD 20)
endif()

add_subdirectory(llama.cpp)
add_library(libjustlm STATIC
    include/justlm.hpp justlm.cpp
    justlm_llama.hpp
    justlm_gptj.hpp gptj/gptj.cpp gptj/gptj.hpp gptj/utils.cpp gptj/utils.hpp
    include/justlm_pool.hpp justlm_pool.cpp
)
target_link_libraries(libjustlm PRIVATE llama)

if (LM_COSCHED)
    target_compile_definitions(libjustlm PUBLIC LM_COSCHED)
    target_link_libraries(libjustlm PRIVATE cosched)

    set(LM_COSCHED Yes CACHE BOOL "If Libjustlm should make use of CoSched" FORCE)
endif()

if (LM_NOEXCEPT)
    target_compile_definitions(libjustlm PUBLIC LM_NOEXCEPT)
endif()

if (LM_PYBIND)
    if (LM_COSCHED)
        message(FATAL_ERROR "Pybind can't be enabled in combination with CoSched")
    endif()

    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)
    pybind11_add_module(libjustlm_py pybind.cpp)
    target_link_libraries(libjustlm_py PRIVATE libjustlm)
endif()

target_include_directories(libjustlm PUBLIC include/)
