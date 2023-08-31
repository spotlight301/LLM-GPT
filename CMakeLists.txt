cmake_minimum_required(VERSION 3.18)

project(justlm LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(LM_PYBIND No CACHE BOOL "If justlm Python bindings should be build")
set(LM_COSCHED No CACHE BOOL "If justlm should make use of CoSched")
set(LM_NOEXCEPT No CACHE BOOL "If justlm exceptions should be disabled")
set(LM_LLAMA Yes CACHE BOOL "If LLaMa model support should be built into justlm")
set(LM_GPTJ Yes CACHE BOOL "If GPT-J model support should be built into justlm")
set(LM_MPT Yes CACHE BOOL "If MPT model support should be built into justlm")

if (LM_COSCHED)
    set(CMAKE_CXX_STANDARD 20)
endif()


function(target_justlm_setup TARGET_NAME)
    message(STATUS "Configuring model implementation target ${TARGET_NAME}")
    target_include_directories(${TARGET_NAME} PUBLIC include/)
    if (LM_COSCHED)
        target_compile_definitions(${TARGET_NAME} PUBLIC LM_COSCHED)
        target_link_libraries(${TARGET_NAME} PUBLIC cosched)
    endif()
    if (LM_NOEXCEPT)
        target_compile_definitions(${TARGET_NAME} PUBLIC LM_NOEXCEPT)
    endif()
endfunction()


include(llama.cpp.cmake)

include_ggml(llama.cpp-mainline _mainline Yes)
include_ggml(llama.cpp-alibi _alibi No)


add_library(justlm_g4a_common SHARED g4a_common.cpp g4a_common.hpp)


if (LM_MPT)
    add_library(justlm_mpt SHARED mpt.cpp justlm_mpt.hpp mpt/mpt.cpp mpt/mpt.hpp)
    target_link_libraries(justlm_mpt PRIVATE ggml_alibi justlm_g4a_common)
    target_justlm_setup(justlm_mpt)
endif()

if (LM_GPTJ)
    add_library(justlm_gptj SHARED gptj.cpp justlm_gptj.hpp gptj/gptj.cpp gptj/gptj.hpp)
    target_link_libraries(justlm_gptj PRIVATE ggml_alibi justlm_g4a_common)
    target_justlm_setup(justlm_gptj)
endif()

if (LM_LLAMA)
    add_library(justlm_llama SHARED llama.cpp justlm_llama.hpp)
    target_link_libraries(justlm_llama PRIVATE ggml_mainline llama_mainline)
    target_compile_definitions(justlm_llama PRIVATE LLAMA_DATE=999999)
    target_justlm_setup(justlm_llama)
endif()


add_library(justlm STATIC
    include/justlm.hpp justlm.cpp
    include/justlm_pool.hpp justlm_pool.cpp
    dlhandle.hpp
)
add_library(libjustlm ALIAS justlm)
target_link_libraries(justlm PRIVATE dl)
target_include_directories(justlm PUBLIC include/)
target_compile_definitions(justlm PRIVATE LIB_FILE_EXT="${CMAKE_SHARED_LIBRARY_SUFFIX}")
target_justlm_setup(justlm)

if (LM_PYBIND)
    if (LM_COSCHED)
        message(FATAL_ERROR "Pybind can't be enabled in combination with CoSched")
    endif()

    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG)
    pybind11_add_module(justlm_py pybind.cpp)
    target_link_libraries(justlm_py PRIVATE justlm)
endif()
