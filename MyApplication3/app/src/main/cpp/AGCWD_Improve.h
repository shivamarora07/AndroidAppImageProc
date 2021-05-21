#include <iostream>
#include <opencv2/opencv.hpp>

void AGCWD_improve(const cv::Mat& src, cv::Mat& dst, double alpha = 0.7);
void upscaleAGCWD_improve(const cv::Mat & src, cv::Mat & dst);
void downscaleAGCWD_improve(const cv::Mat & src, cv::Mat & dst);