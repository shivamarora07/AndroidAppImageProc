#include <jni.h>
#include <string>
#include "opencv-utils.h"
#include "android/bitmap.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "AGCIE.h"
#include "AGCWD.h"
#include "IAGCWD.h"
#include "BIMEF_Trial.h"
#include "Lime.h"

#include <android/log.h>

#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
void BitmapToMat(JNIEnv * env, jobject bitmap, Mat& dst, jboolean needUnPremultiplyAlpha)
{
    AndroidBitmapInfo  info;
    void*              pixels = 0;

    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        dst.create(info.height, info.width, CV_8UC4);
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

void MatToBitmap(JNIEnv * env, Mat& src, jobject bitmap, jboolean needPremultiplyAlpha)
{
    AndroidBitmapInfo  info;
    void*              pixels = 0;

    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( src.dims == 2 && info.height == (uint32_t)src.rows && info.width == (uint32_t)src.cols );
        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if(src.type() == CV_8UC4){
                if(needPremultiplyAlpha) cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if(src.type() == CV_8UC4){
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_myFlip(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
        Mat src;
        BitmapToMat(env, bitmapIn , src, false);
        myFlip(src);
        MatToBitmap(env,src,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_AGCIE(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat AGCIE_dst;
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    auto start = std::chrono::high_resolution_clock::now();
    AGCIE(src, AGCIE_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCIE is : %d", duration);
    MatToBitmap(env,AGCIE_dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_BIMEF(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat BIMEF_dst;
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    auto start = std::chrono::high_resolution_clock::now();
    BIMEF(src, BIMEF_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF is : %d", duration);
    MatToBitmap(env,BIMEF_dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_LIME(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat LIME_dst;
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    auto start = std::chrono::high_resolution_clock::now();
    lime_enhance(src, LIME_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for LIME is : %d", duration);
    MatToBitmap(env,LIME_dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_AGCIEDSUS(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat AGCIE_dst;
    Mat src;
    Mat dst_ds;
    Mat dst;
    BitmapToMat(env, bitmapIn , src, false);

    auto startds = std::chrono::high_resolution_clock::now();
    downscaleAGCIE(src, dst_ds);
    auto endds = std::chrono::high_resolution_clock::now();
    auto durationds = (endds-startds)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCIE downscale is : %d", durationds);

    auto start = std::chrono::high_resolution_clock::now();
    AGCIE(dst_ds, AGCIE_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCIE is : %d", duration);

    auto startus = std::chrono::high_resolution_clock::now();
    upscaleAGCIE(AGCIE_dst, dst);
    auto endus = std::chrono::high_resolution_clock::now();
    auto durationus = (endus-startus)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCIE upscale is : %d", durationus);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Total Time for AGCIE is : %d", durationus+durationds+duration);
    MatToBitmap(env,dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_AGCWD(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat AGCWD_dst;
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    auto start = std::chrono::high_resolution_clock::now();
    AGCWD(src, AGCWD_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCWD is : %d", duration);
    MatToBitmap(env,AGCWD_dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_AGCWDInput(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut, double alpha) {
    Mat AGCWD_dst;
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    auto start = std::chrono::high_resolution_clock::now();
    AGCWD(src, AGCWD_dst, alpha/100);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] val for AGCWD is : %f", alpha/100);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCWD is : %d", duration);
    MatToBitmap(env,AGCWD_dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_AGCWDDSUS(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat AGCWD_dst;
    Mat src;
    Mat dst_ds;
    Mat dst;
    BitmapToMat(env, bitmapIn , src, false);

    auto startds = std::chrono::high_resolution_clock::now();
    downscaleAGCWD(src, dst_ds);
    auto endds = std::chrono::high_resolution_clock::now();
    auto durationds = (endds-startds)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCWD downscale is : %d", durationds);

    auto start = std::chrono::high_resolution_clock::now();
    AGCWD(dst_ds, AGCWD_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCWD is : %d", duration);

    auto startus = std::chrono::high_resolution_clock::now();
    upscaleAGCWD(AGCWD_dst, dst);
    auto endus = std::chrono::high_resolution_clock::now();
    auto durationus = (endus-startus)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for AGCWD upscale is : %d", durationus);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Total Time for AGCWD is : %d", durationus+durationds+duration);

    MatToBitmap(env,dst,bitmapOut,false);
}


extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_BIMEFDSUS(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut) {
    Mat BIMEF_dst;
    Mat src;
    Mat dst_ds;
    Mat dst;
    BitmapToMat(env, bitmapIn , src, false);

    auto startds = std::chrono::high_resolution_clock::now();
    downscaleBIMEF(src, dst_ds);
    auto endds = std::chrono::high_resolution_clock::now();
    auto durationds = (endds-startds)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF downscale is : %d", durationds);

    auto start = std::chrono::high_resolution_clock::now();
    BIMEF(dst_ds, BIMEF_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF is : %d", duration);

    auto startus = std::chrono::high_resolution_clock::now();
    upscaleBIMEF(BIMEF_dst, dst);
    auto endus = std::chrono::high_resolution_clock::now();
    auto durationus = (endus-startus)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF upscale is : %d", durationus);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Total Time for BIMEF is : %d", durationus+durationds+duration);

    MatToBitmap(env,dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_BIMEFDSUSInput(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut, double mu, double a, double b) {
    Mat BIMEF_dst;
    Mat src;
    Mat dst_ds;
    Mat dst;
    BitmapToMat(env, bitmapIn , src, false);

    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] mu %f a %f b %f  for BIMEF", mu/100,a/100,b/100);
    auto startds = std::chrono::high_resolution_clock::now();
    downscaleBIMEF(src, dst_ds);
    auto endds = std::chrono::high_resolution_clock::now();
    auto durationds = (endds-startds)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF downscale is : %d", durationds);

    auto start = std::chrono::high_resolution_clock::now();
    //BIMEF(dst_ds, BIMEF_dst,mu/100,a/100,b/100);
    BIMEF(dst_ds, BIMEF_dst);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF is : %d", duration);

    auto startus = std::chrono::high_resolution_clock::now();
    upscaleBIMEF(BIMEF_dst, dst);
    auto endus = std::chrono::high_resolution_clock::now();
    auto durationus = (endus-startus)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF upscale is : %d", durationus);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Total Time for BIMEF is : %d", durationus+durationds+duration);

    MatToBitmap(env,dst,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_myBlur(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut, jfloat sigma) {
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    myBlur(src,sigma);
    MatToBitmap(env,src,bitmapOut,false);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_IAGCWD(
        JNIEnv* env,
        jobject /* this */, jobject bitmapIn, jobject bitmapOut,double alpha_dimmed , double alpha_bright , int T_t , double tau_t, double tau ) {
    Mat IAGCWD_dst;
    Mat src;
    BitmapToMat(env, bitmapIn , src, false);
    auto start = std::chrono::high_resolution_clock::now();
    IAGCWD(src, IAGCWD_dst,alpha_dimmed/100,alpha_bright/100,T_t,tau_t/100,tau/100);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end-start)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for IAGCWD is : %d", duration);
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] alpha_dimmed %f alpha_bright %f T_t %d tau_t %f tau %f for IAGCWD", alpha_dimmed/100,alpha_bright/100,T_t,tau_t/100,tau/100);
    MatToBitmap(env,IAGCWD_dst,bitmapOut,false);
}

