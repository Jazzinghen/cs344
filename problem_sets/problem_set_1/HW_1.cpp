#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include "HW_1.h"
#include "cudaDeviceImage.h"
#include "cudaException.h"
#include "timer.h"
#include "utils.h"

namespace cs344 {

HW1::HW1(const std::string& input_filename)
{
    // make sure the context initializes ok
    checkCudaErrors(cudaFree(0), __FILE__, __LINE__);

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

    CudaDeviceImage<uchar4> d_rgbaImage(imageRGBA.rows, imageRGBA.cols);
    CudaDeviceImage<uint8_t> d_greyImage(imageRGBA.rows, imageRGBA.cols);

    // Clean the memory for the grayscale image
    d_greyImage.clear();

    const uchar4* inputImage = reinterpret_cast<uchar4*>(imageRGBA.ptr<uint8_t>(0));
    // copy input array to the GPU
    d_rgbaImage.copy_to(inputImage);

    GpuTimer timer;
    timer.Start();
    // call the students' code
    your_rgba_to_greyscale(d_rgbaImage, d_greyImage);
    timer.Stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);

    std::cout << "Your code ran in: " << timer.Elapsed() << " msecs." << std::endl;

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