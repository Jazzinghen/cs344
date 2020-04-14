#ifndef COMPARE_H__
#define COMPARE_H__

#include <string>

bool compareImages(const std::string &reference_filename, const std::string &test_filename, const std::string &diff_filename,
                   bool useEpsCheck, double perPixelError, double globalError);

#endif
