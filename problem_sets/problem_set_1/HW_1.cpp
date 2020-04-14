#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include "HW_1.h"
#include "timer.h"
#include "utils.h"

namespace cs344 {

HW1::HW1(const std::string& input_filename)
{
    // make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    const cv::Mat tmp_img = cv::imread(input_filename, cv::IMREAD_COLOR);
    if (imageRGBA.empty()) {
        throw std::invalid_argument("Couldn't open file: " + input_filename);
    }
    cv::cvtColor(tmp_img, imageRGBA, cv::COLOR_BGR2RGBA);
    imageGrey.create(imageRGBA.rows, imageRGBA.cols, CV_8UC1);

    if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
        throw std::runtime_error("OpenCV matrices are not contiguous");
    }
}

// return types are void since any internal error will be handled by quitting
// no point in returning error codes...
// returns a pointer to an RGBA version of the input image
// and a pointer to the single channel grey-scale output
// on both the host and device
void
HW1::execute_kernel()
{

    size_t num_pixels = imageRGBA.rows * imageRGBA.cols;

    std::unique_ptr<uchar4, decltype(cuda_uchar4_deleter)> d_rgbaImage__(
      cuda_uchar4_device_alloc(num_pixels), cuda_uchar4_deleter);
    std::unique_ptr<uint8_t, decltype(cuda_uint8_t_deleter)> d_greyImage__(
      cuda_uint8_t_device_alloc(num_pixels), cuda_uint8_t_deleter);

    // Clean the memory for the grayscale image
    checkCudaErrors(
      cudaMemset(d_greyImage__.get(), 0, num_pixels * sizeof(unsigned char)));

    const uchar4* inputImage = reinterpret_cast<uchar4*>(imageRGBA.ptr<uint8_t>(0));
    // copy input array to the GPU
    checkCudaErrors(cudaMemcpy(d_rgbaImage__.get(),
                               inputImage,
                               sizeof(uchar4) * num_pixels,
                               cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.Start();
    // call the students' code
    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
    timer.Stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
        // Couldn't print! Probably the student closed stdout - bad news
        std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
        exit(1);
    }

    size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMemcpy(h_greyImage,
                               d_greyImage,
                               sizeof(unsigned char) * numPixels,
                               cudaMemcpyDeviceToHost));
}

void
postProcess(const std::string& output_file, unsigned char* data_ptr)
{
    cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

    // output the image
    cv::imwrite(output_file.c_str(), output);
}

void
generateReferenceImage(std::string input_filename, std::string output_filename)
{
    cv::Mat reference = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

    cv::imwrite(output_filename, reference);
}
} // namespace cs344