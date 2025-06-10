cmake_minimum_required(VERSION 3.10)
project(pf_frac)

add_executable(pf_frac-opt
  # List source files here; e.g.:
  src/main.C
)

target_include_directories(pf_frac-opt PRIVATE ${CMAKE_SOURCE_DIR}/include/base)