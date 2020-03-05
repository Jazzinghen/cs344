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

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

add_subdirectory ("Problem Sets/Problem Set 1")
add_subdirectory ("Problem Sets/Problem Set 2")
add_subdirectory ("Problem Sets/Problem Set 3")
add_subdirectory ("Problem Sets/Problem Set 4")
add_subdirectory ("Problem Sets/Problem Set 5")
add_subdirectory ("Problem Sets/Problem Set 6")
