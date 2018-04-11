#ifndef _CHANGERIO_HPP
#define _CHANGERIO_HPP
#include "changerDll.hpp"
extern "C"
{
#include "so_sift.h"
}

#define MAX_MATCHPOINTS (300)
typedef struct _PRESET_POS_INFO{

        float fCarrierAz;// 预置位方位角
        float fCarrierEl;//预置位俯仰角
        float fCarrierHt;//云台高度
        float fFieldViewPP;//预置位视场角


}PRESET_POS_INFO;

typedef struct _CHANGE_RGN{

	float fAz;//变化区域中心方位角0~360
	float fEl;//变化区域中心俯仰角-45~45
        float fDist;////变化区域中心到摄像头斜距
	float fP;//变化的程度0~1

}CHANGE_RGN;
CHANGE_RGN* SceneChangeDetector(char* inputPath1,char* inputPath2,char* outputPath,DETECT_PARAM inputParam1,PRESET_POS_INFO inputParam2,bool flag);
  void CalcFourCorner(CvMat* &H,CvPoint& leftTop,CvPoint& leftBottom, CvPoint& rightTop,CvPoint& rightBottom,IplImage* img2);
  int detectionFeature(IplImage* img,struct feature*& feat);
  void jisuanwh(IplImage* img1,IplImage* img2, int *param_wh,CvPoint *param_pt,CvMat * &H,int *Hflag);
  IplImage* correct(IplImage *img1,IplImage *img2,IplImage *corr_img1,IplImage *corr_img2,int *param_WH,CvPoint *param_pt,CvPoint& origin,CvMat * H);
#endif
