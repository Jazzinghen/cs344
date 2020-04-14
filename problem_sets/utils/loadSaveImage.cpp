#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "cuda_runtime.h"

//The caller becomes responsible for the returned pointer. This
//is done in the interest of keeping this code as simple as possible.
//In production code this is a bad idea - we should use RAII
//to ensure the memory is freed.  DO NOT COPY THIS AND USE IN PRODUCTION
//CODE!!!
void loadImageHDR(const std::string &filename,
                  std::vector<float> &image_data,
                  size_t &numRows, size_t &numCols)
{
  cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
  if (image.empty())
  {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  if (image.channels() != 3)
  {
    std::cerr << "Image must be color!" << std::endl;
    exit(1);
  }

  if (!image.isContinuous())
  {
    std::cerr << "Image isn't continuous!" << std::endl;
    exit(1);
  }

  image_data.clear();
  image_data.reserve(image.rows * image.cols * image.channels());

  // MEH
  const float *image_ptr = image.ptr<float>(0);

  std::copy(image_ptr, image_ptr + (image.rows * image.cols), image_data.begin());

  numRows = image.rows;
  numCols = image.cols;
}

void loadImageGrey(const std::string &filename,
                   std::vector<unsigned char> image_data,
                   size_t &numRows, size_t &numCols)
{
  cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (image.empty())
  {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  if (image.channels() != 1)
  {
    std::cerr << "Image must be greyscale!" << std::endl;
    exit(1);
  }

  if (!image.isContinuous())
  {
    std::cerr << "Image isn't continuous!" << std::endl;
    exit(1);
  }

  image_data.clear();
  image_data.reserve(image.rows * image.cols);

  unsigned char *image_ptr = image.ptr<unsigned char>(0);

  std::copy(image_ptr, image_ptr + (image.rows * image.cols), image_data.begin());

  numRows = image.rows;
  numCols = image.cols;
}

void loadImageRGBA(const std::string &filename,
                   std::vector<uchar4> &image_data,
                   size_t &numRows, size_t &numCols)
{
  cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  if (image.channels() != 3)
  {
    std::cerr << "Image must be color!" << std::endl;
    exit(1);
  }

  if (!image.isContinuous())
  {
    std::cerr << "Image isn't continuous!" << std::endl;
    exit(1);
  }

  cv::Mat imageRGBA;
  cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

  image_data.clear();
  image_data.reserve(image.rows * image.cols);

  unsigned char *cvPtr = imageRGBA.ptr<unsigned char>(0);
  for (size_t i = 0; i < image.rows * image.cols; ++i)
  {
    image_data[i].x = cvPtr[4 * i + 0];
    image_data[i].y = cvPtr[4 * i + 1];
    image_data[i].z = cvPtr[4 * i + 2];
    image_data[i].w = cvPtr[4 * i + 3];
  }

  numRows = image.rows;
  numCols = image.cols;
}

void saveImageRGBA(const uchar4 *const image,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file)
{
  int sizes[2];
  sizes[0] = static_cast<int>(numRows);
  sizes[1] = static_cast<int>(numCols);
  cv::Mat imageRGBA(2, sizes, CV_8UC4, (void *)image);
  cv::Mat imageOutputBGR;
  cv::cvtColor(imageRGBA, imageOutputBGR, cv::COLOR_RGBA2BGR);
  //output the image
  cv::imwrite(output_file, imageOutputBGR);
}

//output an exr file
//assumed to already be BGR
void saveImageHDR(const float *const image,
                  const size_t numRows, const size_t numCols,
                  const std::string &output_file)
{
  int sizes[2];
  sizes[0] = static_cast<int>(numRows);
  sizes[1] = static_cast<int>(numCols);

  cv::Mat imageHDR(2, sizes, CV_32FC3, (void *)image);

  imageHDR = imageHDR * 255;

  cv::imwrite(output_file, imageHDR);
}
