#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

#include "cudaDeviceImage.h"
#include "utils.h"

namespace cs344 {
class HW1
{

  public:
    HW1(const std::string& input_filename);
    ~HW1() = default;

    void execute_kernel();

    void postProcess(const std::string& output_file, unsigned char* data_ptr);

    void generateReferenceImage(const std::string& input_filename,
                                const std::string& output_filename);

  private:
    cv::Mat imageRGBA;
    cv::Mat imageGrey;

    cv::Mat load_input_image(const std::string& input_filename);

    void your_rgba_to_greyscale(CudaDeviceImage<uchar4>& d_rgbaImage,
                                CudaDeviceImage<uint8_t>& d_greyImage);
};
} // namespace cs344
