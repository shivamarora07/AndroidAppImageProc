
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

void Illumination(cv::Mat& src, cv::Mat& out);
void Illumination_filter(cv::Mat& img_in, cv::Mat& img_out);
void lime_enhance(cv::Mat& src, cv::Mat& dst);

