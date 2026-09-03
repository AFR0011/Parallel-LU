cmake_minimum_required(VERSION 3.20)

project(ParallelLU VERSION 1.0.0 LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "The maintained MPI-lite path currently requires Windows/Winsock2.")
endif()

option(PARALLEL_LU_BUILD_MSMPI "Build the optional MS-MPI source target" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(lu_common STATIC lu_common.cpp)
target_include_directories(lu_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(lu_common PUBLIC cxx_std_17)

add_executable(lu_driver lu_driver.cpp)
target_link_libraries(lu_driver PRIVATE lu_common OpenMP::OpenMP_CXX ws2_32)

add_executable(lu_worker lu_worker.cpp)
target_link_libraries(lu_worker PRIVATE lu_common OpenMP::OpenMP_CXX ws2_32)

add_executable(lu_common_tests tests/lu_common_tests.cpp)
target_link_libraries(lu_common_tests PRIVATE lu_common ws2_32)

set(PARALLEL_LU_MAINTAINED_TARGETS lu_common lu_driver lu_worker lu_common_tests)
foreach(target IN LISTS PARALLEL_LU_MAINTAINED_TARGETS)
    target_compile_options(${target} PRIVATE /W4 /permissive- /WX)
endforeach()

if(PARALLEL_LU_BUILD_MSMPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_executable(lu_mpi lu_mpi.cpp)
    target_compile_features(lu_mpi PRIVATE cxx_std_17)
    target_compile_options(lu_mpi PRIVATE /W4 /permissive-)
    target_link_libraries(lu_mpi PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME common_contracts COMMAND lu_common_tests)
    add_test(
        NAME mpi_lite_integration
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_smoke.py"
                --driver "$<TARGET_FILE:lu_driver>"
                --worker "$<TARGET_FILE:lu_worker>"
    )
    set_tests_properties(mpi_lite_integration PROPERTIES TIMEOUT 90)
endif()
