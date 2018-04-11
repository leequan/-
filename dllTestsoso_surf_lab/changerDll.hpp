
#ifndef _CHANGERDLL_HPP
#define _CHANGERDLL_HPP

//#pragma offload_attribute(push, target(mic))
#include <opencv2/core/core.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <cmath>  
#include <iostream>  
 // #pragma offload_attribute(pop)
using namespace cv;  
using namespace std;  
  
const double PI = 3.14159265;  
#define H_RESOLUTION_PIXEL (1920)
#define V_RESOLUTION_PIXEL (1080)
#define H_RESOLUTION_mm (7.34)
#define V_RESOLUTION_mm (4.13)
#define f_MIN_mm (6.5)
#define arc2deg (57.29577957)
#define deg2arc (0.017453292)
#define MAXGROWS (160000)
typedef struct _RGN{
	int sz;
	int cx;
	int cy;
	int left;
	int top;
	int right;
	int bottom;

}RGN;
typedef struct _DETECT_PARAM{
	int nScale;//检测尺度
	int nSzKernel;//滤波核大小
	int nFilters;//滤波器个数
	int nChangeRnSz;//变化区域最小尺寸
	float fSensitivity;//检测灵敏度，分割阈值
	int nChangeRnSzKH;//颜色变化区域最小尺寸
	float fSensitivityKH;//颜色检测灵敏度，分割阈值
	double whratio1;
	double whratio2;//矩形长宽比阈值
}DETECT_PARAM;
Mat getMyGabor(int width, int height, int U, int V, double Kmax, double f,  
    double sigma, int ktype, const string& kernel_name);  
Mat gabor_filter(Mat& img,int dir,int scale,int kernelSize,int type);  
Mat GetMyHist(Mat image,bool IsShow,int histSize);
RGN  RegionGrow(Mat src, Point2i pt, int th,int label,Mat& matDst) ;
vector<Rect> ImageLabel(Mat srcImg,int szTh, Mat& dstImg);
void colorCorrect(Mat srcImgA,Mat srcImgB,Mat dstImgA,Mat dstImgB);
//void ChangeDetector(Mat srcImgA,Mat srcImgB,DETECT_PARAM params,vector<Rect>& output);
#ifdef __cplusplus
extern "C" {
#endif
void ChangeDetector(Mat srcImgA,Mat srcImgB,DETECT_PARAM params,vector<Rect>& output,vector<float>& outputCorrelationCf,string debugFileName);
void ChangeDetectorKH(Mat srcImgA,Mat srcImgB,DETECT_PARAM params,vector<Rect>& output,vector<float>& outputCorrelationCf);
#ifdef __cplusplus
}
#endif






#endif
