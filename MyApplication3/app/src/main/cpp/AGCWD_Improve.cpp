#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <vector>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include <array>
#include <chrono>
#include <iomanip>
#include <algorithm>


#include "AGCWD_Improve.h"
#include <android/log.h>

#define rdouble register double
#define rint register int

float st[256] = {0.00392157, 0.00784314, 0.01176471, 0.01568627, 0.01960784, 0.02352941, 0.02745098, 0.03137255, 0.03529412, 0.03921569, 0.04313725, 0.04705882, 0.05098039, 0.05490196, 0.05882353, 0.0627451, 0.06666667, 0.07058824, 0.0745098, 0.07843137, 0.08235294, 0.08627451, 0.09019608, 0.09411765, 0.09803922, 0.10196078, 0.10588235, 0.10980392, 0.11372549, 0.11764706, 0.12156863, 0.1254902, 0.12941176, 0.13333333, 0.1372549, 0.14117647, 0.14509804, 0.14901961, 0.15294118, 0.15686275, 0.16078431, 0.16470588, 0.16862745, 0.17254902, 0.17647059, 0.18039216, 0.18431373, 0.18823529, 0.19215686, 0.19607843, 0.2, 0.20392157, 0.20784314, 0.21176471, 0.21568627, 0.21960784, 0.22352941, 0.22745098, 0.23137255, 0.23529412, 0.23921569, 0.24313725, 0.24705882, 0.25098039, 0.25490196, 0.25882353, 0.2627451, 0.26666667, 0.27058824, 0.2745098, 0.27843137, 0.28235294, 0.28627451, 0.29019608, 0.29411765, 0.29803922, 0.30196078, 0.30588235, 0.30980392, 0.31372549, 0.31764706, 0.32156863, 0.3254902, 0.32941176, 0.33333333, 0.3372549, 0.34117647, 0.34509804, 0.34901961, 0.35294118, 0.35686275, 0.36078431, 0.36470588, 0.36862745, 0.37254902, 0.37647059, 0.38039216, 0.38431373, 0.38823529, 0.39215686, 0.39607843, 0.4, 0.40392157, 0.40784314, 0.41176471, 0.41568627, 0.41960784, 0.42352941, 0.42745098, 0.43137255, 0.43529412, 0.43921569, 0.44313725, 0.44705882, 0.45098039, 0.45490196, 0.45882353, 0.4627451, 0.46666667, 0.47058824, 0.4745098, 0.47843137, 0.48235294, 0.48627451, 0.49019608, 0.49411765, 0.49803922, 0.50196078, 0.50588235, 0.50980392, 0.51372549, 0.51764706, 0.52156863, 0.5254902, 0.52941176, 0.53333333, 0.5372549, 0.54117647, 0.54509804, 0.54901961, 0.55294118, 0.55686275, 0.56078431, 0.56470588, 0.56862745, 0.57254902, 0.57647059, 0.58039216, 0.58431373, 0.58823529, 0.59215686, 0.59607843, 0.6, 0.60392157, 0.60784314, 0.61176471, 0.61568627, 0.61960784, 0.62352941, 0.62745098, 0.63137255, 0.63529412, 0.63921569, 0.64313725, 0.64705882, 0.65098039, 0.65490196, 0.65882353, 0.6627451, 0.66666667, 0.67058824, 0.6745098, 0.67843137, 0.68235294, 0.68627451, 0.69019608, 0.69411765, 0.69803922, 0.70196078, 0.70588235, 0.70980392, 0.71372549, 0.71764706, 0.72156863, 0.7254902, 0.72941176, 0.73333333, 0.7372549, 0.74117647, 0.74509804, 0.74901961, 0.75294118, 0.75686275, 0.76078431, 0.76470588, 0.76862745, 0.77254902, 0.77647059, 0.78039216, 0.78431373, 0.78823529, 0.79215686, 0.79607843, 0.8, 0.80392157, 0.80784314, 0.81176471, 0.81568627, 0.81960784, 0.82352941, 0.82745098, 0.83137255, 0.83529412, 0.83921569, 0.84313725, 0.84705882, 0.85098039, 0.85490196, 0.85882353, 0.8627451, 0.86666667, 0.87058824, 0.8745098, 0.87843137, 0.88235294, 0.88627451, 0.89019608, 0.89411765, 0.89803922, 0.90196078, 0.90588235, 0.90980392, 0.91372549, 0.91764706, 0.92156863, 0.9254902, 0.92941176, 0.93333333, 0.9372549, 0.94117647, 0.94509804, 0.94901961, 0.95294118, 0.95686275, 0.96078431, 0.96470588, 0.96862745, 0.97254902, 0.97647059, 0.98039216, 0.98431373, 0.98823529, 0.99215686, 0.99607843, 1.0};

void AGCWD_improve(const cv::Mat & src, cv::Mat & dst, double alpha)
{
    auto starttotal = std::chrono::high_resolution_clock::now();
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();
    int total_pixels = rows * cols;

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat L;
    cv::Mat HSV;
    std::vector<cv::Mat> HSV_channels;
    if (channels == 1) {
        L = src.clone();
    }
    else {
        cv::cvtColor(src, HSV, cv::COLOR_BGR2HSV_FULL);
        cv::split(HSV, HSV_channels);
        L = HSV_channels[2];
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedChannelCreation = finish-start;
    std::cout<<"Elapsed time ChannelCreation: "<<elapsedChannelCreation.count()<<" s\n";

    start = std::chrono::high_resolution_clock::now();
    int histsize = 256;
    float range[] = { 0,256 };
    const float* histRanges = { range };
    int bins = 256;
    cv::Mat hist;
    calcHist(&L, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);

    double total_pixels_inv = 1.0 / total_pixels;
    cv::Mat PDF = cv::Mat::zeros(256, 1, CV_64F);
    for(register int i = 0;i<256; i+=4){
        PDF.at<double>(i) = hist.at<float>(i) * total_pixels_inv;
        PDF.at<double>(i+1) = hist.at<float>(i+1) * total_pixels_inv;
        PDF.at<double>(i+2) = hist.at<float>(i+2) * total_pixels_inv;
        PDF.at<double>(i+3) = hist.at<float>(i+3) * total_pixels_inv;
    }
    finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedPDF = finish-start;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Elapsed time PDF: : %d", (finish-start)/1000000);
    std::cout<<"Elapsed time PDF: "<<elapsedPDF.count()<<" s\n";

    start = std::chrono::high_resolution_clock::now();

    double pdf_min, pdf_max;
    cv::minMaxLoc(PDF, &pdf_min, &pdf_max);
    cv::Mat PDF_w = PDF.clone();
    rdouble pdf_max_min = pdf_max - pdf_min;

    cv::Mat CDF_w = PDF_w.clone();
    double culsum = 0;

    for (register int i = 0; i < 256; i+=4) {
        PDF_w.at<double>(i) = pdf_max * std::pow((PDF_w.at<double>(i) - pdf_min) / (pdf_max_min), alpha);
        culsum += PDF_w.at<double>(i);
        CDF_w.at<double>(i) = culsum;

        PDF_w.at<double>(i+1) = pdf_max * std::pow((PDF_w.at<double>(i+1) - pdf_min) / (pdf_max_min), alpha);
        culsum += PDF_w.at<double>(i+1);
        CDF_w.at<double>(i+1) = culsum;

        PDF_w.at<double>(i+2) = pdf_max * std::pow((PDF_w.at<double>(i+2) - pdf_min) / (pdf_max_min), alpha);
        culsum += PDF_w.at<double>(i+2);
        CDF_w.at<double>(i+2) = culsum;

        PDF_w.at<double>(i+3) = pdf_max * std::pow((PDF_w.at<double>(i+3) - pdf_min) / (pdf_max_min), alpha);
        culsum += PDF_w.at<double>(i+3);
        CDF_w.at<double>(i+3) = culsum;

    }
    finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCDF = finish-start;
    std::cout<<"Elapsed time CDF: "<<elapsedCDF.count()<<" s\n";
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Elapsed time CDF: : %d", (finish-start)/1000000);
    start = std::chrono::high_resolution_clock::now();

    CDF_w /= culsum;

    std::vector<uchar> table(256, 0);
    for (register int i = 1; i < 256; i++) {
        table[i] = cv::saturate_cast<uchar>(255.0 * std::pow(st[i], 1 - CDF_w.at<double>(i)));
    }

    cv::LUT(L, table, L);
    finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedLUT = finish-start;
    std::cout<<"Elapsed time LUT: "<<elapsedLUT.count()<<" s\n";
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Elapsed time LUT: : %d", (finish-start)/1000000);


    start = std::chrono::high_resolution_clock::now();
    if (channels == 1) {
        dst = L.clone();
    }
    else {
        cv::merge(HSV_channels, dst);
        cv::cvtColor(dst, dst, cv::COLOR_HSV2BGR_FULL);
    }

    finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedFusion = finish-start;
    std::cout<<"Elapsed time Fusion: "<<elapsedFusion.count()<<" s\n";
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Elapsed time Fusion: : %d", (finish-start)/1000000);
    auto finishtotal = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Total Time: : %d", (finishtotal-starttotal)/1000000);
    return;
}

void downscaleAGCWD_improve(const cv::Mat & src, cv::Mat & dst)
{
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] AGCWD Downscale src row : %d cols : %d", src.rows,src.cols);
    cv::resize(src, dst, cv::Size(src.rows/2,src.cols/2));
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] AGCWD Downscale dst row : %d cols : %d", dst.rows,dst.cols);
}

void upscaleAGCWD_improve(const cv::Mat & src, cv::Mat & dst)
{
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] AGCWD Upscale src row : %d cols : %d", src.rows,src.cols);
    cv::resize(src, dst, cv::Size(src.rows*2,src.cols*2));
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] AGCWD Upscale dst row : %d cols : %d", dst.rows,dst.cols);
}