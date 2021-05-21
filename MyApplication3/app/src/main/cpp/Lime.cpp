//
// Created by zj on 19-4-18.
//

#include "Lime.h"
#include <vector>
#include <iostream>
#include <android/log.h>

    static inline float compare(float& a, float& b, float& c)
    {
        return fmax(a, fmax(b, c));
    }

    void lime_enhance(cv::Mat& src, cv::Mat& dst) {
        int channel;
        channel = src.channels();
        auto one = std::chrono::high_resolution_clock::now();
        if (channel == 4)
        {
            channel = 3;
            cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);
        }
        auto two = std::chrono::high_resolution_clock::now();
        __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME COLOR_BGRA2BGR: %d", (two-one)/1000000);

        cv::Mat img_norm;
        src.convertTo(img_norm, CV_32F, 1 / 255.0, 0);
        auto three = std::chrono::high_resolution_clock::now();
        __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME convertTo CV_32F: %d", (three-two)/1000000);

        cv::Size sz(img_norm.size());
        cv::Mat out(sz, CV_32F, cv::Scalar::all(0.0));

        auto gammT = out.clone();
        auto four = std::chrono::high_resolution_clock::now();
        __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME clone mat: %d", (four-three)/1000000);

        if (channel == 3)
        {

            Illumination(img_norm, out);
            auto five = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME Illumination Function: %d", (five-four)/1000000);

            Illumination_filter(out, gammT);
            auto six = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME Illumination_filter Function: %d", (six-five)/1000000);


            //lime
            std::vector<cv::Mat> img_norm_rgb;
            cv::Mat img_norm_b, img_norm_g, img_norm_r;

            cv::split(img_norm, img_norm_rgb);

            img_norm_g = img_norm_rgb.at(0);
            img_norm_b = img_norm_rgb.at(1);
            img_norm_r = img_norm_rgb.at(2);
            auto seven = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME Split Mat to g b r : %d", (seven-six)/1000000);

            cv::Mat one = cv::Mat::ones(sz, CV_32F);

            float nameta = 0.9;
            auto g = 1 - ((one - img_norm_g) - (nameta * (one - gammT))) / gammT;
            auto b = 1 - ((one - img_norm_b) - (nameta * (one - gammT))) / gammT;
            auto r = 1 - ((one - img_norm_r) - (nameta * (one - gammT))) / gammT;
            auto eight = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME Working on g b r : %d", (eight-seven)/1000000);


            cv::Mat g1, b1, r1;

            //TODO <=1
            threshold(g, g1, 0.0, 0.0, 3);
            threshold(b, b1, 0.0, 0.0, 3);
            threshold(r, r1, 0.0, 0.0, 3);
            auto nine = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME threshold on g b r : %d", (nine-eight)/1000000);


            img_norm_rgb.clear();
            img_norm_rgb.push_back(g1);
            img_norm_rgb.push_back(b1);
            img_norm_rgb.push_back(r1);

            cv::merge(img_norm_rgb, dst);
            auto ten = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME merge g b r : %d", (ten-nine)/1000000);


            dst.convertTo(dst, CV_8U, 255);
            auto eleven = std::chrono::high_resolution_clock::now();
            __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME convertTo CV_8U : %d", (eleven-ten)/1000000);

        }
        else if (channel == 1)
        {
            Illumination_filter(img_norm, gammT);
            cv::Mat one = cv::Mat::ones(sz, CV_32F);
            float nameta = 0.9;
            //std::cout<<img_norm.at<float>(1,1)<<std::endl;
            auto out = 1 - ((one - img_norm) - (nameta * (one - gammT))) / gammT;

            threshold(out, dst, 0.0, 0.0, 3);

            dst.convertTo(dst, CV_8UC1, 255);

        }

        else
        {
            std::cout << "There is a problem with the channels" << std::endl;
            exit(-1);
        }
    }

    void Illumination_filter(cv::Mat& img_in, cv::Mat& img_out)
    {
        int ksize = 5;
        //mean filter
        blur(img_in, img_out, cv::Size(ksize, ksize));
        //GaussianBlur(img_in,img_mean_filter,Size(ksize,ksize),0,0);

        //gamma
        int row = img_out.rows;
        int col = img_out.cols;
        float tem;
        float gamma = 0.8;
        for (int i = 0;i < row;i++)
        {

            for (int j = 0;j < col;j++)
            {
                tem = pow(img_out.at<float>(i, j), gamma);
                tem = tem <= 0 ? 0.0001 : tem;  //  double epsolon = 0.0001;
                tem = tem > 1 ? 1 : tem;

                img_out.at<float>(i, j) = tem;

            }
        }

    }
    void Illumination(cv::Mat& src, cv::Mat& out)
    {
        int row = src.rows, col = src.cols;

        for (int i = 0;i < row;i++)
        {
            for (int j = 0;j < col;j++)
            {
                out.at<float>(i, j) = compare(src.at<cv::Vec3f>(i, j)[0],
                                                    src.at<cv::Vec3f>(i, j)[1],
                                                    src.at<cv::Vec3f>(i, j)[2]);
            }

        }

    }
