############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
# minimum required cmake version

SET (compare_files compare.cpp)

file( GLOB  HEADERS *.hpp *.h )
file( GLOB  CUDA_SRC  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)
    
add_executable(problem_set_3 ${CUDA_SRC} ${HW3_files})

set_target_properties(
        problem_set_3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(
        problem_set_3
        ${OpenCV_LIBS}
)