
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <array>
#include <iostream>
#include "BIMEF_Trial.h"
#include <android/log.h>

#ifndef HAVE_EIGEN
#define HAVE_EIGEN
#endif

#ifdef HAVE_EIGEN
#include "eigen/Eigen/Sparse"
#include <opencv2/imgproc.hpp>
#endif


#ifdef HAVE_EIGEN
using namespace cv;

/** @addtogroup core_eigen
These functions are provided for OpenCV-Eigen interoperability. They convert `Mat`
objects to corresponding `Eigen::Matrix` objects and vice-versa. Consult the [Eigen
documentation](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html) for
information about the `Matrix` template type.

@note Using these functions requires the `Eigen/Dense` or similar header to be
included before this header.
*/
//! @{

#if defined(OPENCV_EIGEN_TENSOR_SUPPORT) || defined(CV_DOXYGEN)
/** @brief Converts an Eigen::Tensor to a cv::Mat.

The method converts an Eigen::Tensor with shape (H x W x C) to a cv::Mat where:
 H = number of rows
 W = number of columns
 C = number of channels

Usage:
\code
Eigen::Tensor<float, 3, Eigen::RowMajor> a_tensor(...);
// populate tensor with values
Mat a_mat;
eigen2cv(a_tensor, a_mat);
\endcode
*/
template <typename _Tp, int _layout> static inline
void eigen2cv( const Eigen::Tensor<_Tp, 3, _layout> &src, OutputArray dst )
{
    if( !(_layout & Eigen::RowMajorBit) )
    {
        const std::array<int, 3> shuffle{2, 1, 0};
        Eigen::Tensor<_Tp, 3, !_layout> row_major_tensor = src.swap_layout().shuffle(shuffle);
        Mat _src(src.dimension(0), src.dimension(1), CV_MAKETYPE(DataType<_Tp>::type, src.dimension(2)), row_major_tensor.data());
        _src.copyTo(dst);
    }
    else
    {
        Mat _src(src.dimension(0), src.dimension(1), CV_MAKETYPE(DataType<_Tp>::type, src.dimension(2)), (void *)src.data());
        _src.copyTo(dst);
    }
}

/** @brief Converts a cv::Mat to an Eigen::Tensor.

The method converts a cv::Mat to an Eigen Tensor with shape (H x W x C) where:
 H = number of rows
 W = number of columns
 C = number of channels

Usage:
\code
Mat a_mat(...);
// populate Mat with values
Eigen::Tensor<float, 3, Eigen::RowMajor> a_tensor(...);
cv2eigen(a_mat, a_tensor);
\endcode
*/
template <typename _Tp, int _layout> static inline
void cv2eigen( const Mat &src, Eigen::Tensor<_Tp, 3, _layout> &dst )
{
    if( !(_layout & Eigen::RowMajorBit) )
    {
        Eigen::Tensor<_Tp, 3, !_layout> row_major_tensor(src.rows, src.cols, src.channels());
        Mat _dst(src.rows, src.cols, CV_MAKETYPE(DataType<_Tp>::type, src.channels()), row_major_tensor.data());
        if (src.type() == _dst.type())
            src.copyTo(_dst);
        else
            src.convertTo(_dst, _dst.type());
        const std::array<int, 3> shuffle{2, 1, 0};
        dst = row_major_tensor.swap_layout().shuffle(shuffle);
    }
    else
    {
        dst.resize(src.rows, src.cols, src.channels());
        Mat _dst(src.rows, src.cols, CV_MAKETYPE(DataType<_Tp>::type, src.channels()), dst.data());
        if (src.type() == _dst.type())
            src.copyTo(_dst);
        else
            src.convertTo(_dst, _dst.type());
    }
}

/** @brief Maps cv::Mat data to an Eigen::TensorMap.

The method wraps an existing Mat data array with an Eigen TensorMap of shape (H x W x C) where:
 H = number of rows
 W = number of columns
 C = number of channels

Explicit instantiation of the return type is required.

@note Caller should be aware of the lifetime of the cv::Mat instance and take appropriate safety measures.
The cv::Mat instance will retain ownership of the data and the Eigen::TensorMap will lose access when the cv::Mat data is deallocated.

The example below initializes a cv::Mat and produces an Eigen::TensorMap:
\code
float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
Mat a_mat(2, 2, CV_32FC3, arr);
Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> a_tensormap = cv2eigen_tensormap<float>(a_mat);
\endcode
*/
template <typename _Tp> static inline
Eigen::TensorMap<Eigen::Tensor<_Tp, 3, Eigen::RowMajor>> cv2eigen_tensormap(InputArray src)
{
    Mat mat = src.getMat();
    CV_CheckTypeEQ(mat.type(), CV_MAKETYPE(traits::Type<_Tp>::value, mat.channels()), "");
    return Eigen::TensorMap<Eigen::Tensor<_Tp, 3, Eigen::RowMajor>>((_Tp *)mat.data, mat.rows, mat.cols, mat.channels());
}
#endif // OPENCV_EIGEN_TENSOR_SUPPORT

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, OutputArray dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), traits::Type<_Tp>::value,
                 (void*)src.data(), src.outerStride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), traits::Type<_Tp>::value,
                 (void*)src.data(), src.outerStride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src,
               Matx<_Tp, _rows, _cols>& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        dst = Matx<_Tp, _cols, _rows>(static_cast<const _Tp*>(src.data())).t();
    }
    else
    {
        dst = Matx<_Tp, _rows, _cols>(static_cast<const _Tp*>(src.data()));
    }
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    CV_DbgAssert(src.rows == _rows && src.cols == _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void cv2eigen( const Matx<_Tp, _rows, _cols>& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, _rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, _cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

template<typename _Tp>  static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols> static inline
void cv2eigen( const Matx<_Tp, _rows, _cols>& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(_rows, _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, _rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, _cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

template<typename _Tp> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    CV_Assert(src.cols == 1);
    dst.resize(src.rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows> static inline
void cv2eigen( const Matx<_Tp, _rows, 1>& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    dst.resize(_rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(1, _rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, 1, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.copyTo(_dst);
    }
}


template<typename _Tp> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    CV_Assert(src.rows == 1);
    dst.resize(src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

//Matx
template<typename _Tp, int _cols> static inline
void cv2eigen( const Matx<_Tp, 1, _cols>& src,
               Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    dst.resize(_cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, 1, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(1, _cols, traits::Type<_Tp>::value,
                       dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

static void diff(const cv::Mat_<float>& src, Mat_<float>& srcVDiff, Mat_<float>& srcHDiff)
{
    srcVDiff = Mat_<float>(src.size());
    for (int i = 0; i < src.rows; i++)
    {
        if (i < src.rows - 1)
        {
            for (int j = 0; j < src.cols; j++)
            {
                srcVDiff(i, j) = src(i + 1, j) - src(i, j);
            }
        }
        else
        {
            for (int j = 0; j < src.cols; j++)
            {
                srcVDiff(i, j) = src(0, j) - src(i, j);
            }
        }
    }

    srcHDiff = Mat_<float>(src.size());
    for (int j = 0; j < src.cols - 1; j++)
    {
        for (int i = 0; i < src.rows; i++)
        {
            srcHDiff(i, j) = src(i, j + 1) - src(i, j);
        }
    }
    for (int i = 0; i < src.rows; i++)
    {
        srcHDiff(i, src.cols - 1) = src(i, 0) - src(i, src.cols - 1);
    }
}

static void computeTextureWeights(const Mat_<float>& x, float sigma, float sharpness, Mat_<float>& W_h, Mat_<float>& W_v)
{
    Mat_<float> dt0_v, dt0_h;
    diff(x, dt0_v, dt0_h);

    Mat_<float> gauker_h;
    Mat_<float> kernel_h = Mat_<float>::ones(1, static_cast<int>(sigma));
    filter2D(dt0_h, gauker_h, -1, kernel_h, Point(-1, -1), 0, BORDER_CONSTANT);

    Mat_<float> gauker_v;
    Mat_<float> kernel_v = Mat_<float>::ones(static_cast<int>(sigma), 1);
    filter2D(dt0_v, gauker_v, -1, kernel_v, Point(-1, -1), 0, BORDER_CONSTANT);

    W_h = Mat_<float>(gauker_h.size());
    W_v = Mat_<float>(gauker_v.size());

    for (int i = 0; i < gauker_h.rows; i++)
    {
        for (int j = 0; j < gauker_h.cols; j++)
        {
            W_h(i, j) = 1 / (std::abs(gauker_h(i, j)) * std::abs(dt0_h(i, j)) + sharpness);
            W_v(i, j) = 1 / (std::abs(gauker_v(i, j)) * std::abs(dt0_v(i, j)) + sharpness);
        }
    }
}

template <class numeric_t>
static Eigen::SparseMatrix<numeric_t> spdiags(const Eigen::Matrix<numeric_t, -1, -1>& B,
                                              const Eigen::VectorXi& d, int m, int n) {
    typedef Eigen::Triplet<numeric_t> triplet_t;
    std::vector<triplet_t> triplets;
    triplets.reserve(static_cast<size_t>(std::min(m, n) * d.size()));

    for (int k = 0; k < d.size(); ++k) {
        int diag = d(k);  // get diagonal
        int i_start = std::max(-diag, 0); // get row of 1st element
        int i_end = std::min(m, m - diag - (m - n)); // get row of last element
        int j = -std::min(0, -diag); // get col of 1st element
        int B_i; // start index i in matrix B
        if (m < n) {
            B_i = std::max(-diag, 0); // m < n
        }
        else {
            B_i = std::max(0, diag); // m >= n
        }
        for (int i = i_start; i < i_end; ++i, ++j, ++B_i) {
            triplets.push_back({ i, j,  B(B_i,k) });
        }
    }
    Eigen::SparseMatrix<numeric_t> A(m, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}


static Mat solveLinearEquation(const Mat_<float>& img, Mat_<float>& W_h_, Mat_<float>& W_v_, float lambda)
{
    auto one = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXf W_h;
    cv2eigen(W_h_, W_h);
    auto two = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation cv2eigen: %d", (two-one)/1000000);

    Eigen::MatrixXf tempx(W_h.rows(), W_h.cols());
    tempx.block(0, 1, tempx.rows(), tempx.cols() - 1) = W_h.block(0, 0, W_h.rows(), W_h.cols() - 1);
    auto three = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation tempx.block: %d", (three-two)/1000000);

    for (Eigen::Index i = 0; i < tempx.rows(); i++)
    {
        tempx(i, 0) = W_h(i, W_h.cols() - 1);
    }
    auto four = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation tempx.block loop: %d", (four-three)/1000000);

    Eigen::MatrixXf W_v;
    cv2eigen(W_v_, W_v);
    auto five = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation cv2eigen: %d", (five-four)/1000000);

    Eigen::MatrixXf tempy(W_v.rows(), W_v.cols());
    tempy.block(1, 0, tempx.rows() - 1, tempx.cols()) = W_v.block(0, 0, W_v.rows() - 1, W_v.cols());
    auto six = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation tempy.block: %d", (six-five)/1000000);

    for (Eigen::Index j = 0; j < tempy.cols(); j++)
    {
        tempy(0, j) = W_v(W_v.rows() - 1, j);
    }

    auto seven = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation tempy loop: %d", (seven-six)/1000000);


    Eigen::VectorXf dx(W_h.rows() * W_h.cols());
    Eigen::VectorXf dy(W_v.rows() * W_v.cols());

    Eigen::VectorXf dxa(tempx.rows() * tempx.cols());
    Eigen::VectorXf dya(tempy.rows() * tempy.cols());

    //Flatten in a col-major order
    for (Eigen::Index j = 0; j < W_h.cols(); j++)
    {
        for (Eigen::Index i = 0; i < W_h.rows(); i++)
        {
            dx(j * W_h.rows() + i) = -lambda * W_h(i, j);
            dy(j * W_h.rows() + i) = -lambda * W_v(i, j);

            dxa(j * W_h.rows() + i) = -lambda * tempx(i, j);
            dya(j * W_h.rows() + i) = -lambda * tempy(i, j);
        }
    }

    auto eight = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation Flatten in a col-major order: %d", (eight-seven)/1000000);

    tempx.setZero();
    tempx.col(0) = W_h.col(W_h.cols() - 1);

    tempy.setZero();
    tempy.row(0) = W_v.row(W_v.rows() - 1);

    W_h.col(W_h.cols() - 1).setZero();
    W_v.row(W_v.rows() - 1).setZero();

    Eigen::VectorXf dxd1(tempx.rows() * tempx.cols());
    Eigen::VectorXf dyd1(tempy.rows() * tempy.cols());
    Eigen::VectorXf dxd2(W_h.rows() * W_h.cols());
    Eigen::VectorXf dyd2(W_v.rows() * W_v.cols());
    auto nine = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation after Flatten in a col-major order: %d", (nine-eight)/1000000);

    //Flatten in a col-major order
    for (Eigen::Index j = 0; j < tempx.cols(); j++)
    {
        for (Eigen::Index i = 0; i < tempx.rows(); i++)
        {
            dxd1(j * tempx.rows() + i) = -lambda * tempx(i, j);
            dyd1(j * tempx.rows() + i) = -lambda * tempy(i, j);

            dxd2(j * tempx.rows() + i) = -lambda * W_h(i, j);
            dyd2(j * tempx.rows() + i) = -lambda * W_v(i, j);
        }
    }
    auto ten = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation Flatten in a col-major order #1 : %d", (ten-nine)/1000000);


    Eigen::MatrixXf dxd(dxd1.rows(), dxd1.cols() + dxd2.cols());
    dxd << dxd1, dxd2;

    Eigen::MatrixXf dyd(dyd1.rows(), dyd1.cols() + dyd2.cols());
    dyd << dyd1, dyd2;

    const int k = img.rows * img.cols;
    const int r = img.rows;
    Eigen::Matrix<int, 2, 1> diagx_idx;
    diagx_idx << -k + r, -r;
    Eigen::SparseMatrix<float> Ax = spdiags(dxd, diagx_idx, k, k);

    Eigen::Matrix<int, 2, 1> diagy_idx;
    diagy_idx << -r + 1, -1;
    Eigen::SparseMatrix<float> Ay = spdiags(dyd, diagy_idx, k, k);

    auto eleven = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation after Flatten in a col-major order#1 : %d", (eleven-ten)/1000000);

    Eigen::MatrixXf D = (dx + dy + dxa + dya);
    D = Eigen::MatrixXf::Ones(D.rows(), D.cols()) - D;

    Eigen::Matrix<int, 1, 1> diag_idx_zero;
    diag_idx_zero << 0;
    Eigen::SparseMatrix<float> A = (Ax + Ay) + Eigen::SparseMatrix<float>((Ax + Ay).transpose()) + spdiags(D, diag_idx_zero, k, k);

    //CG solver of Eigen
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<float> > cg;
    cg.setTolerance(0.1f);
    cg.setMaxIterations(5);
    cg.compute(A);
    Mat_<float> img_t = img.t();
    Eigen::Map<const Eigen::VectorXf> tin(img_t.ptr<float>(), img_t.rows * img_t.cols);
    Eigen::VectorXf x = cg.solve(tin);
    auto twelve = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation CG solver of Eigen: %d", (twelve-eleven)/1000000);

    Mat_<float> tout(img.rows, img.cols);
    tout.forEach(
            [&](float& pixel, const int* position) -> void
            {
                pixel = x(position[1] * img.rows + position[0]);
            }
    );
    auto thirteen = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation end: %d", (thirteen-twelve)/1000000);

    return tout;
}

static Mat_<float> tsmooth(const Mat_<float>& src, float lambda = 0.01f, float sigma = 3.0f, float sharpness = 0.001f)
{
    Mat_<float> W_h, W_v;

    auto one = std::chrono::high_resolution_clock::now();
    computeTextureWeights(src, sigma, sharpness, W_h, W_v);
    auto two = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth computeTextureWeights : %d", (two-one)/1000000);

    Mat_<float> S = solveLinearEquation(src, W_h, W_v, lambda);
    auto three = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  tsmooth solveLinearEquation : %d", (three-two)/1000000);

    return S;
}

static Mat_<float> rgb2gm(const Mat_<Vec3f>& I)
{
    Mat_<float> gm(I.rows, I.cols);
    gm.forEach(
            [&](float& pixel, const int* position) -> void
            {
                pixel = std::pow(I(position[0], position[1])[0] * I(position[0], position[1])[1] * I(position[0], position[1])[2], 1 / 3.0f);
            }
    );

    return gm;
}

static Mat_<float> applyK(const Mat_<float>& I, float k, float a = -0.3293f, float b = 1.1258f) {
    float beta = std::exp((1 - std::pow(k, a)) * b);
    float gamma = std::pow(k, a);

    Mat_<float> J(I.size());
    pow(I, gamma, J);
    J = J * beta;

    return J;
}

static Mat_<Vec3f> applyK(const Mat_<Vec3f>& I, float k, float a = -0.3293f, float b = 1.1258f, float offset = 0) {
    float beta = std::exp((1 - std::pow(k, a)) * b);
    float gamma = std::pow(k, a);

    Mat_<Vec3f> J(I.size());
    pow(I, gamma, J);

    return J * beta + Scalar::all(offset);
}

static float entropy(const Mat_<float>& I)
{
    Mat_<uchar> I_uchar;
    I.convertTo(I_uchar, CV_8U, 255);

    std::vector<Mat> planes;
    planes.push_back(I_uchar);
    Mat_<float> hist;
    const int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    calcHist(&I_uchar, 1, NULL, Mat(), hist, 1, &histSize, &histRange);

    Mat_<float> hist_norm = hist / cv::sum(hist)[0];

    float E = 0;
    for (int i = 0; i < hist_norm.rows; i++)
    {
        if (hist_norm(i, 0) > 0)
        {
            E += hist_norm(i, 0) * std::log2(hist_norm(i, 0));
        }
    }

    return -E;
}

template <typename T> static int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

static double minimize_scalar_bounded(const Mat_<float>& I, double begin, double end,
                                      double xatol = 1e-4, int maxiter = 500)
{
    // From scipy: https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/optimize.py#L1753-L1894
    //    """
    //    Options
    //    -------
    //    maxiter : int
    //        Maximum number of iterations to perform.
    //    disp: int, optional
    //        If non-zero, print messages.
    //            0 : no message printing.
    //            1 : non-convergence notification messages only.
    //            2 : print a message on convergence too.
    //            3 : print iteration results.
    //    xatol : float
    //        Absolute error in solution `xopt` acceptable for convergence.
    //    """
    double x1 = begin, x2 = end;

    if (x1 > x2) {
        std::runtime_error("The lower bound exceeds the upper bound.");
    }

    double sqrt_eps = std::sqrt(2.2e-16);
    double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
    double a = x1, b = x2;
    double fulc = a + golden_mean * (b - a);
    double nfc = fulc, xf = fulc;
    double rat = 0.0, e = 0.0;
    double x = xf;
    double fx = -entropy(applyK(I, static_cast<float>(x)));
    int num = 1;
    double fu = std::numeric_limits<double>::infinity();

    double ffulc = fx, fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
    double tol2 = 2.0 * tol1;

    for (int iter = 0; iter < maxiter && std::abs(xf - xm) >(tol2 - 0.5 * (b - a)); iter++)
    {
        int golden = 1;
        // Check for parabolic fit
        if (std::abs(e) > tol1) {
            golden = 0;
            double r = (xf - nfc) * (-entropy(applyK(I, static_cast<float>(x))) - ffulc);
            double q = (xf - fulc) * (-entropy(applyK(I, static_cast<float>(x))) - fnfc);
            double p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);

            if (q > 0.0) {
                p = -p;
            }
            q = std::abs(q);
            r = e;
            e = rat;

            // Check for acceptability of parabola
            if (((std::abs(p) < std::abs(0.5 * q * r)) && (p > q * (a - xf)) &
                                                          (p < q * (b - xf)))) {
                rat = (p + 0.0) / q;
                x = xf + rat;

                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    double si = sgn(xm - xf) + ((xm - xf) == 0);
                    rat = tol1 * si;
                }
            }
            else {      // do a golden-section step
                golden = 1;
            }
        }

        if (golden) {  // do a golden-section step
            if (xf >= xm) {
                e = a - xf;
            }
            else {
                e = b - xf;
            }
            rat = golden_mean * e;
        }

        double si = sgn(rat) + (rat == 0);
        x = xf + si * std::max(std::abs(rat), tol1);
        fu = -entropy(applyK(I, static_cast<float>(x)));
        num += 1;

        if (fu <= fx) {
            if (x >= xf) {
                a = xf;
            }
            else {
                b = xf;
            }

            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        }
        else {
            if (x < xf) {
                a = x;
            }
            else {
                b = x;
            }

            if ((fu <= fnfc) || (nfc == xf)) {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            }
            else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                fulc = x;
                ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
        tol2 = 2.0 * tol1;
    }

    return xf;
}

static Mat_<Vec3f> maxEntropyEnhance(const Mat_<Vec3f>& I, const Mat_<uchar>& isBad, float a, float b)
{
    Mat_<Vec3f> input;
    resize(I, input, Size(50, 50));

    Mat_<float> Y = rgb2gm(input);

    Mat_<uchar> isBad_resize;
    resize(isBad, isBad_resize, Size(50, 50));

    std::vector<float> Y_vec;
    for (int i = 0; i < isBad_resize.rows; i++)
    {
        for (int j = 0; j < isBad_resize.cols; j++)
        {
            if (isBad_resize(i, j) >= 0.5)
            {
                Y_vec.push_back(Y(i, j));
            }
        }
    }

    if (Y_vec.empty())
    {
        return I;
    }

    Mat_<float> Y_mat(static_cast<int>(Y_vec.size()), 1, Y_vec.data());
    float opt_k = static_cast<float>(minimize_scalar_bounded(Y_mat, 1, 7));

    return applyK(I, opt_k, a, b, -0.01f);
}

//static void BIMEF_impl(InputArray input_, OutputArray output_, float mu, float* k, float a, float b)
static void BIMEF_impl(const cv::Mat& input, cv::Mat& output, float mu, float* k, float a, float b)
{
    //CV_INSTRUMENT_REGION()
    //Mat input = input_.getMat();
    if (input.empty())
    {
        return;
    }
    auto one = std::chrono::high_resolution_clock::now();
    CV_CheckTypeEQ(input.type(), CV_8UC3, "Input image must be 8-bits color image (CV_8UC3).");
    auto two = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF CV_CheckTypeEQ: %d", (two-one)/1000000);

    Mat_<Vec3f> imgDouble;
    input.convertTo(imgDouble, CV_32F, 1 / 255.0);
    auto three = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF convertTo CV_32F : %d", (three-two)/1000000);

    // t: scene illumination map
    Mat_<float> t_b(imgDouble.size());
    t_b.forEach(
            [&](float& pixel, const int* position) -> void
            {
                pixel = std::max(std::max(imgDouble(position[0], position[1])[0],
                                          imgDouble(position[0], position[1])[1]),
                                 imgDouble(position[0], position[1])[2]);
            }
    );

    auto four = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF scene illumination map : %d", (four-three)/1000000);
    const float lambda = 0.5;
    const float sigma = 5;

    Mat_<float> t_b_resize;
    resize(t_b, t_b_resize, Size(), 0.5, 0.5);
    auto five = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF Resize to Half : %d", (five-four)/1000000);

    Mat_<float> t_our = tsmooth(t_b_resize, lambda, sigma);
    auto six = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF tsmooth : %d", (six-five)/1000000);

    //Mat_<float> t_our = t_b_resize;
    resize(t_our, t_our, t_b.size());
    auto seven = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF Resize back to original : %d", (seven-six)/1000000);

    // k: exposure ratio
    Mat_<Vec3f> J;
    if (k == NULL)
    {
        Mat_<uchar> isBad(t_our.size());
        isBad.forEach(
                [&](uchar& pixel, const int* position) -> void
                {
                    pixel = t_our(position[0], position[1]) < 0.5 ? 1 : 0;
                }
        );

        J = maxEntropyEnhance(imgDouble, isBad, a, b);
    }
    else
    {
        J = applyK(imgDouble, *k, a, b);

        // fix overflow
        J.forEach(
                [](Vec3f& pixel, const int* ) -> void
                {
                    pixel(0) = std::min(1.0f, pixel(0));
                    pixel(1) = std::min(1.0f, pixel(1));
                    pixel(2) = std::min(1.0f, pixel(2));
                }
        );
    }
    auto eight = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF  k: exposure ratio : %d", (eight-seven)/1000000);

    // W: Weight Matrix
    Mat_<float> W(t_our.size());
    pow(t_our, mu, W);
    auto nine = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF W: Weight Matrix : %d", (nine-eight)/1000000);


    //output_.create(input.size(), CV_8UC3);
    output = Mat::zeros( input.size(), CV_8UC3 );
    //Mat output = output_.getMat();
    output.forEach<Vec3b>(
            [&](Vec3b& pixel, const int* position) -> void
            {
                float w = W(position[0], position[1]);
                pixel(0) = saturate_cast<uchar>((imgDouble(position[0], position[1])[0] * w + J(position[0], position[1])[0] * (1 - w)) * 255);
                pixel(1) = saturate_cast<uchar>((imgDouble(position[0], position[1])[1] * w + J(position[0], position[1])[1] * (1 - w)) * 255);
                pixel(2) = saturate_cast<uchar>((imgDouble(position[0], position[1])[2] * w + J(position[0], position[1])[2] * (1 - w)) * 255);
            }
    );
    auto ten = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF getting output Mat : %d", (ten-nine)/1000000);
}
#else
static void BIMEF_impl(cv::InputArray, cv::OutputArray, float, float*, float, float)
{
    std::cout << "This algorithm requires OpenCV built with the Eigen library." << std::endl;

}
#endif

void  BIMEF(const cv::Mat& input, cv::Mat& output, float mu , float a , float b )//;BIMEF(InputArray input, OutputArray output, float mu, float a, float b)
{
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Reached BIMEF  mu a b : %f %f %f", mu,a,b);
    cv::Mat temp;
    auto one = std::chrono::high_resolution_clock::now();
    cv::cvtColor(input,temp,cv::COLOR_BGRA2BGR);
    auto two = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF COLOR_BGRA2BGR: %d", (two-one)/1000000);
    BIMEF_impl(temp, output, mu, NULL, a, b);
    auto three = std::chrono::high_resolution_clock::now();
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF BIMEF_impl: %d", (three-two)/1000000);
}

void BIMEF(const cv::Mat& input, cv::Mat& output, float k, float mu, float a, float b)
{
    auto startBIMEFimp = std::chrono::high_resolution_clock::now();
    BIMEF_impl(input, output, mu, &k, a, b);
    auto endBIMEFimp = std::chrono::high_resolution_clock::now();
    auto duration = (endBIMEFimp-startBIMEFimp)/1000000;
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] Time for BIMEF_impl is : %d", duration);

}

void downscaleBIMEF(const cv::Mat & src, cv::Mat & dst)
{
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] BIMEF Downscale src row : %d cols : %d", src.rows,src.cols);
    cv::resize(src, dst, cv::Size(src.rows/2,src.cols/2));
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] BIMEF Downscale dst row : %d cols : %d", dst.rows,dst.cols);
}

void upscaleBIMEF(const cv::Mat & src, cv::Mat & dst)
{
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] BIMEF Upscale src row : %d cols : %d", src.rows,src.cols);
    cv::resize(src, dst, cv::Size(src.rows*2,src.cols*2));
    __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", " [IMG_PROC] BIMEF Upscale dst row : %d cols : %d", dst.rows,dst.cols);
}



