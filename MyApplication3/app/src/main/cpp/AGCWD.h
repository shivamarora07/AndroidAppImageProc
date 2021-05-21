#include <iostream>
#include <opencv2/opencv.hpp>

void AGCWD(const cv::Mat& src, cv::Mat& dst, double alpha = 0.5);
void upscaleAGCWD(const cv::Mat & src, cv::Mat & dst);
void downscaleAGCWD(const cv::Mat & src, cv::Mat & dst);