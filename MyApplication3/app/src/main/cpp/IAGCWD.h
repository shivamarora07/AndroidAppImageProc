#include <iostream>
#include <opencv2/opencv.hpp>

void IAGCWD(const cv::Mat& src, cv::Mat& dst, double alpha_dimmed = 0.75, double alpha_bright = 0.75, int T_t = 112, double tau_t = 0.3, double tau = 0.5);
void upscaleIAGCWD(const cv::Mat & src, cv::Mat & dst);
void downscaIleAGCWD(const cv::Mat & src, cv::Mat & dst);