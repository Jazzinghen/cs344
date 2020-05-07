############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

SET(PROJ_HEADERS reference_calc.h)
SET(CUDA_SRC student_func.cu)
SET(HW1_files main.cpp reference_calc.cpp)

add_executable(problem_set_2 ${CUDA_SRC} ${HW1_files})

set_target_properties(
        problem_set_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# We need this directory, and users of our library will need it too
target_include_directories(problem_set_2 PUBLIC  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/problem_sets/utils")

# All users of this library will need at least C++14
target_compile_features(problem_set_2 PUBLIC cxx_std_14)

# Time to add some serious checks to this library!
target_compile_options(problem_set_2 PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Werror>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4 /WX>)

target_link_libraries(
        problem_set_2
        cs344_utils
        ${OpenCV_LIBS}
)