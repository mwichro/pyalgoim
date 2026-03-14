cmake_minimum_required(VERSION 3.18)

project(algoim_python_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

enable_testing()

add_library(pyalgoim_backend SHARED python/algoim_pybind.cpp)
target_include_directories(
    pyalgoim_backend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/python
        ${CMAKE_CURRENT_SOURCE_DIR}/algoim
)
set_target_properties(pyalgoim_backend PROPERTIES PREFIX "" OUTPUT_NAME "pyalgoim_backend")

add_executable(algoim_batch_smoke tests/smoke_batch_quadrature.cpp)
target_include_directories(algoim_batch_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_CURRENT_SOURCE_DIR}/algoim)

add_test(NAME algoim_batch_cpp_tests COMMAND algoim_batch_smoke)
add_test(NAME pyalgoim_python_tests COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pyalgoim.py)
set_tests_properties(
    pyalgoim_python_tests
    PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};ALGOIM_PY_BACKEND=$<TARGET_FILE:pyalgoim_backend>"
)