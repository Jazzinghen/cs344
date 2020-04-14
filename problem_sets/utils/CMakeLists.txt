############################################################################
# <summary> CMakeLists.txt for utils library used by all the problem sets </summary>
# <date>    2020/03/27          </date>
# <author>  Michele Bianchi </author>
# <email>   jazzinghen@gmail.com </email>
############################################################################

# Optionally glob, but only for CMake 3.12 or later:
# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${ModernCMakeExample_SOURCE_DIR}/include/modern/*.hpp")
set(HEADER_LIST
    "compare.h"
    "timer.h"
    "utils.h"
    "loadSaveImage.h")

# Make an automatic library - will be static or dynamic based on user setting
add_library(cs344_utils "compare.cpp" "loadSaveImage.cpp" ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(cs344_utils PUBLIC  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS})

# All users of this library will need at least C++14
target_compile_features(cs344_utils PUBLIC cxx_std_14)

# Time to add some serious checks to this library!
target_compile_options(cs344_utils PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Werror>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4 /WX>)

target_link_libraries(
    cs344_utils
    ${OpenCV_LIBS}
)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/problem_sets/utils" PREFIX "Header Files" FILES ${HEADER_LIST})
