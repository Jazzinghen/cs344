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
        const auto m_result =
          cudaMalloc(&container_, sizeof(container_type) * image_area_);
        if (m_result != cudaError::cudaSuccess) {
            throw cudaException(cudaGetErrorString(err), __FILE__, __LINE__);
        }
    }

    ~CudaDeviceImage()
    {
        if (container_ != nullptr) {
            cudaFree(container_);
        };
    };

    container_type* get() { return container_; }

  private:
    container_type* container_ = nullptr;

    uint32_t rows_;
    uint32_t cols_;
    size_t image_area_;
};
} // namespace cs344