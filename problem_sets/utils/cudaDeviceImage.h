#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaException.h"
#include "utils.h"

namespace cs344 {

template<typename container_type>
class CudaDeviceImage
{
  public:
    CudaDeviceImage();

    explicit CudaDeviceImage(uint32_t rows, uint32_t columns)
      : rows_(rows)
      , cols_(columns)
      , image_area_(rows_ * cols_)
    {
        checkCudaErrors(cudaMalloc(&container_, sizeof(container_type) * image_area_),
                        __FILE__,
                        __LINE__);
    }

    ~CudaDeviceImage()
    {
        if (container_ != nullptr) {
            cudaFree(container_);
        };
    };

    void clear()
    {
        checkCudaErrors(cudaMemset(container_, 0, image_area_ * sizeof(container_type)),
                        __FILE__,
                        __LINE__);
    }

    void copy_to(const container_type* from) { copy_to(from, image_area_); }

    void copy_to(const container_type* from, size_t area)
    {
        checkCudaErrors(
          cudaMemcpy(
            container_, from, sizeof(container_type) * area, cudaMemcpyHostToDevice),
          __FILE__,
          __LINE__);
    }

    void copy_from(const container_type* to)
    {
        checkCudaErrors(cudaMemcpy(to,
                                   container_,
                                   sizeof(container_type) * image_area_,
                                   cudaMemcpyDeviceToHost),
                        __FILE__,
                        __LINE__);
    }

    container_type* get() { return container_; }

    std::pair<uint32_t, uint32_t> dimensions() { return std::make_pair(rows_, cols_); }

  private:
    container_type* container_ = nullptr;

    uint32_t rows_;
    uint32_t cols_;
    size_t image_area_;
};
} // namespace cs344