############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

file( GLOB  HEADERS *.hpp *.h )
file( GLOB  CUDA_SRC  *.cu)
SET (HW1_files main.cpp reference_calc.cpp "../utils/compare.cpp")

add_executable(problem_set_1 ${CUDA_SRC} ${HW1_files})

set_target_properties(
        problem_set_1
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(
        problem_set_1
        ${OpenCV_LIBS}
)