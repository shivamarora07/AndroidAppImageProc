#include <iostream>
#include <opencv2/opencv.hpp>

void  BIMEF(const cv::Mat& input, cv::Mat& output, float mu = 0.5f, float a = -0.3293f, float b = 1.1258f);
void BIMEF(const cv::Mat& input, cv::Mat& output, float k, float mu, float a, float b);
void upscaleBIMEF(const cv::Mat & src, cv::Mat & dst);
void downscaleBIMEF(const cv::Mat & src, cv::Mat & dst);