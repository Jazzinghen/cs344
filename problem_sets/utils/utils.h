#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>

#include "cudaException.h"

namespace cs344 {

inline void
checkCudaErrors(const cudaError& err, const char* const file, const int line)
{
    if (err != cudaSuccess) {
        throw cudaException(cudaGetErrorString(err), file, line);
    }
}

template<typename T>
bool
checkResultsExact(const T* const ref, const T* const gpu, size_t numElem)
{
    // check that the GPU result matches the CPU result
    for (size_t i = 0; i < numElem; ++i) {
        if (ref[i] != gpu[i]) {
            std::cerr << "Difference at pos " << i << std::endl;
            // the + is magic to convert char to int without messing
            // with other types
            std::cerr << "Reference: " << std::setprecision(17) << +ref[i]
                      << "\nGPU      : " << +gpu[i] << std::endl;
            return false;
        }
    }

    return true;
}

template<typename T>
bool
checkResultsEps(const T* const ref,
                const T* const gpu,
                size_t numElem,
                double eps1,
                double eps2)
{
    assert(eps1 >= 0 && eps2 >= 0);
    unsigned long long totalDiff = 0;
    unsigned numSmallDifferences = 0;
    for (size_t i = 0; i < numElem; ++i) {
        // subtract smaller from larger in case of unsigned types
        T smaller = std::min(ref[i], gpu[i]);
        T larger = std::max(ref[i], gpu[i]);
        T diff = larger - smaller;
        if (diff > 0 && diff <= eps1) {
            numSmallDifferences++;
        } else if (diff > eps1) {
            std::cerr << "Difference at pos " << +i << " exceeds tolerance of " << eps1
                      << std::endl;
            std::cerr << "Reference: " << std::setprecision(17) << +ref[i]
                      << "\nGPU      : " << +gpu[i] << std::endl;
            return false;
        }
        totalDiff += diff * diff;
    }
    double percentSmallDifferences = (double)numSmallDifferences / (double)numElem;
    if (percentSmallDifferences > eps2) {
        std::cerr << "Total percentage of non-zero pixel difference between the two "
                     "images exceeds "
                  << 100.0 * eps2 << "%" << std::endl;
        std::cerr << "Percentage of non-zero pixel differences: "
                  << 100.0 * percentSmallDifferences << "%" << std::endl;
        return false;
    }

    return true;
}

// Uses the autodesk method of image comparison
// Note the the tolerance here is in PIXELS not a percentage of input pixels
template<typename T>
void
checkResultsAutodesk(const T* const ref,
                     const T* const gpu,
                     size_t numElem,
                     double variance,
                     size_t tolerance)
{

    size_t numBadPixels = 0;
    for (size_t i = 0; i < numElem; ++i) {
        T smaller = std::min(ref[i], gpu[i]);
        T larger = std::max(ref[i], gpu[i]);
        T diff = larger - smaller;
        if (diff > variance)
            ++numBadPixels;
    }

    if (numBadPixels > tolerance) {
        std::cerr << "Too many bad pixels in the image." << numBadPixels << "/"
                  << tolerance << std::endl;
        exit(1);
    }
}
} // namespace cs344