############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2020-03-05          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan, Michele Bianchi </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com jazzinghen@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(cs344 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenCV REQUIRED PATHS "C:\\Users\\Jazzinghen\\Tools\\opencv\\build")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "problem_sets/utils")

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

add_subdirectory ("problem_sets/problem_set_1")
add_subdirectory ("problem_sets/problem_set_2")
add_subdirectory ("problem_sets/problem_set_3")
add_subdirectory ("problem_sets/problem_set_4")
add_subdirectory ("problem_sets/problem_set_5")
add_subdirectory ("problem_sets/problem_set_6")
