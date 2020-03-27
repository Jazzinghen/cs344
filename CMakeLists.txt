############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2020-03-05          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan, Michele Bianchi </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com jazzinghen@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(cs344
    DESCRIPTION "Udacity's CS344 course on CUDA"
    LANGUAGES CUDA CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD 14)

    # Let's add the local bin folder for executables output
    set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

    
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

find_package(OpenCV REQUIRED PATHS "C:\\Users\\Jazzinghen\\Tools\\opencv\\build")

add_subdirectory ("problem_sets")