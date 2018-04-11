
#ifndef MYMOSAIC_H
#define MYMOSAIC_H


#include "opencv/cxcore.h"



//在k-d树上进行BBF搜索的最大次数
/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

//目标点与最近邻和次近邻的距离的比值的阈值，若大于此阈值，则剔除此匹配点对
//通常此值取0.6，值越小找到的匹配点对越精确，但匹配数目越少
/* threshold on squared ratio of distances between NN and 2nd NN */
//#define NN_SQ_DIST_RATIO_THR 0.49
#define NN_SQ_DIST_RATIO_THR 0.5

//窗口名字符串
#define IMG1 "图1"
#define IMG2 "图2"
#define IMG1_FEAT "图1特征点"
#define IMG2_FEAT "图2特征点"
#define IMG_MATCH1 "距离比值筛选后的匹配结果"
#define IMG_MATCH2 "RANSAC筛选后的匹配结果"
#define IMG_MOSAIC_TEMP "临时拼接图像"
#define IMG_MOSAIC_SIMPLE "简易拼接图"
#define IMG_MOSAIC_BEFORE_FUSION "重叠区域融合前"
#define IMG_MOSAIC_PROC12 "拼接图1-2"

  
  void CalcFourCorner(CvMat* &H,CvPoint& leftTop,CvPoint& leftBottom, CvPoint& rightTop,CvPoint& rightBottom,IplImage* img2);
  int detectionFeature(IplImage* img,struct feature*& feat);
  IplImage* spliceImage(IplImage* img1,IplImage* img2);
  IplImage* correct(IplImage *img1, IplImage *img2,IplImage *result_img1,IplImage *result_img2);

#endif

