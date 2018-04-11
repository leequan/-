#ifndef _CHANGERIO_HPP
#define _CHANGERIO_HPP
#include "changerDll.hpp"


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
void jisuanwh(Mat img1,Mat img2,Mat &corr_img1,Mat &corr_img2,CvPoint& origin,Mat H,int &Hflag,int MAX_KEYPOINTS,int MAX_GOODMATCHPOINTS);
Mat DrawMaxRectangle(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin);
Mat DrawMaxTriangle(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin);
Mat DrawAllRectangle(Mat image_B,vector<Rect> output,int i,CvPoint origin);
Mat DrawAllTriangle(Mat image_B,vector<Rect> output,int i,CvPoint origin);
Mat DrawMaxDashRectangleString(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin,float fAz,float fEl);
Mat DrawMaxTriangleString(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin,float fAz,float fEl);
vector<Rect> MergeAllRectangle(Mat image_B,vector<Rect> output,CvPoint origin,float whratio1,float whratio2,int MergeThreshold);
Mat DrawMaxMergeRectangle(Mat image_B,vector<Rect> outputtemp);
Mat DrawAllMergeRectangle(Mat image_B,vector<Rect> outputtemp);
Mat DrawMaxMergeTriangleString(Mat image_B,vector<Rect> outputtemp,CvPoint origin,float &fAz,float &fEl,float &fDist);
bool isOverlap( Rect &rc1,  Rect &rc2 ,int MergeThreshold);
Mat DrawAllMergeRectangle(Mat image_B,vector<Rect> output,CvPoint origin);
#endif
