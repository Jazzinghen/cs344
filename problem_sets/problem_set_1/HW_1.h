#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

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
};
} // namespace cs344
