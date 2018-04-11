#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <string>
#include <iostream>
using namespace std;

extern  "C"
{
#include "mymosaic.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "minpq.h"
#include "sift.h"
#include "utils.h"
#include "xform.h"
}

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


//计算图2的四个角经矩阵H变换后的坐标
void CalcFourCorner(CvMat* &H,CvPoint& leftTop,CvPoint& leftBottom, CvPoint& rightTop,CvPoint& rightBottom,IplImage* img2)
{
    //计算图2的四个角经矩阵H变换后的坐标
    double v2[]={0,0,1};//左上角
    double v1[3];//变换后的坐标值
    CvMat V2 = cvMat(3,1,CV_64FC1,v2);
    CvMat V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);//矩阵乘法
    leftTop.x = cvRound(v1[0]/v1[2]);
    leftTop.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,leftTop,7,CV_RGB(255,0,0),2);

    //将v2中数据设为左下角坐标
    v2[0] = 0;
    v2[1] = img2->height;
    V2 = cvMat(3,1,CV_64FC1,v2);
    V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);
    leftBottom.x = cvRound(v1[0]/v1[2]);
    leftBottom.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,leftBottom,7,CV_RGB(255,0,0),2);

    //将v2中数据设为右上角坐标
    v2[0] = img2->width;
    v2[1] = 0;
    V2 = cvMat(3,1,CV_64FC1,v2);
    V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);
    rightTop.x = cvRound(v1[0]/v1[2]);
    rightTop.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,rightTop,7,CV_RGB(255,0,0),2);

    //将v2中数据设为右下角坐标
    v2[0] = img2->width;
    v2[1] = img2->height;
    V2 = cvMat(3,1,CV_64FC1,v2);
    V1 = cvMat(3,1,CV_64FC1,v1);
    cvGEMM(H,&V2,1,0,1,&V1);
    rightBottom.x = cvRound(v1[0]/v1[2]);
    rightBottom.y = cvRound(v1[1]/v1[2]);
    //cvCircle(xformed,rightBottom,7,CV_RGB(255,0,0),2);

}


int detectionFeature(IplImage* img,struct feature*& feat)
{
    int n  = sift_features( img, &feat);//检测图img中的SIFT特征点,n是图的特征点个数
    //export_features("feature.txt",feat,n);//将特征向量数据写入到文件
    return n;
}


IplImage* spliceImage(IplImage* img1,IplImage* img2,IplImage* result_img1,IplImage* result_img2)
{
    struct feature *feat1, *feat2;//feat1：图1的特征点数组，feat2：图2的特征点数组
    int n1, n2;//n1:图1中的特征点个数，n2：图2中的特征点个数
    struct feature *feat;//每个特征点
    struct kd_node *kd_root;//k-d树的树根
    struct feature **nbrs;//当前特征点的最近邻点数组
    CvMat * H = NULL;//RANSAC算法求出的变换矩阵
    struct feature **inliers;//精RANSAC筛选后的内点数组
    int n_inliers;//经RANSAC算法筛选后的内点个数,即feat2中具有符合要求的特征点的个数

    IplImage *xformed = NULL,*xformed_proc = NULL;//xformed临时拼接图，即只将图2变换后的图,xformed_proc是最终合成的图

    //图2的四个角经矩阵H变换后的坐标
    CvPoint leftTop,leftBottom,rightTop,rightBottom;
    ///////////////////////////////////////////////////////////////////

    //特征点检测
    n1 = detectionFeature( img1,feat1 );//检测图1中的SIFT特征点,n1是图1的特征点个数
    //提取并显示第2幅图片上的特征点
    n2 = detectionFeature( img2, feat2 );//检测图2中的SIFT特征点，n2是图2的特征点个数

    //特征匹配
    //方式一：水平排列
    //将2幅图片合成1幅图片,img1在左，img2在右
    //stacked = stack_imgs_horizontal(img1, img2);//合成图像，显示经距离比值法筛选后的匹配结果
    //根据图1的特征点集feat1建立k-d树，返回k-d树根给kd_root
    kd_root = kdtree_build( feat1, n1 );
    CvPoint pt1,pt2;//连线的两个端点
    double d0,d1;//feat2中每个特征点到最近邻和次近邻的距离
    int matchNum = 0;//经距离比值法筛选后的匹配点对的个数
    //遍历特征点集feat2，针对feat2中每个特征点feat，选取符合距离比值条件的匹配点，放到feat的fwd_match域中
    for(int i = 0; i < n2; i++ )
    {
        feat = feat2+i;//第i个特征点的指针
        //在kd_root中搜索目标点feat的2个最近邻点，存放在nbrs中，返回实际找到的近邻点个数
        int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if( k == 2 )
        {
            d0 = descr_dist_sq( feat, nbrs[0] );//feat与最近邻点的距离的平方
            d1 = descr_dist_sq( feat, nbrs[1] );//feat与次近邻点的距离的平方
            //若d0和d1的比值小于阈值NN_SQ_DIST_RATIO_THR，则接受此匹配，否则剔除
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
            {   //将目标点feat和最近邻点作为匹配点对
                pt2.x = cvRound(feat->x);pt2.y = cvRound(feat->y);
                pt1.x = cvRound(nbrs[0]->x); pt1.y = cvRound(nbrs[0]->y);
                pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
                //cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//画出连线
                matchNum++;//统计匹配点对的个数
                feat2[i].fwd_match = nbrs[0];//使点feat的fwd_match域指向其对应的匹配点
            }
        }
        free( nbrs );//释放近邻数组
    }
    //利用RANSAC算法筛选匹配点,计算变换矩阵H，
    //无论img1和img2的左右顺序，H永远是将feat2中的特征点变换为其匹配点，即将img2中的点变换为img1中的对应点
    H = ransac_xform(feat2,n2,FEATURE_FWD_MATCH,lsq_homog,4,0.01,homog_xfer_err,3.0,&inliers,&n_inliers);
 //  fprintf( stderr, "Found %d total matches\n", H->data[1]);
    //若能成功计算出变换矩阵，即两幅图中有共同区域
    IplImage* stacked_ransac;

    //stacked_ransac = stack_imgs(img1, img2);
    stacked_ransac = stack_imgs_horizontal(img1, img2);

    if( H )
    {
      int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图

      //遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线
        for(int i=0; i<n_inliers; i++)
          {
             feat = inliers[i];//第i个特征点
             pt2 = cvPoint(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标
             pt1 = cvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//图1中点的坐标(feat的匹配点)

             //统计匹配点的左右位置关系，来判断图1和图2的左右位置关系
            if(pt2.x > pt1.x)
               invertNum++;

             // pt2.y += img1->height;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
              pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
              cvLine(stacked_ransac,pt1,pt2,CV_RGB(255,0,255),1,8,0);//在匹配图上画出连线
           }
      //    cvNamedWindow(IMG_MATCH2);//创建窗口
       //   cvShowImage(IMG_MATCH2,stacked_ransac);//显示经RANSAC算法筛选后的匹配图
     }

    if( H )
    {
        //全景拼接
        //若能成功计算出变换矩阵，即两幅图中有共同区域，才可以进行全景拼接
        //拼接图像，img1是左图，img2是右图
        CalcFourCorner(H,leftTop,leftBottom,rightTop,rightBottom,img2);//计算图2的四个角经变换后的坐标
        //为拼接结果图xformed分配空间,高度为图1图2高度的较小者，根据图2右上角和右下角变换后的点的位置决定拼接图的宽度
        xformed = cvCreateImage(cvSize(MIN(rightTop.x,rightBottom.x),MIN(img1->height,img2->height)),IPL_DEPTH_8U,3);
        //用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
        cvWarpPerspective(img2,xformed,H,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll(0));

        //处理后的拼接图，克隆自xformed
        xformed_proc = cvCloneImage(xformed);

    //创建重叠区域大小的图像
           int start = MIN(leftTop.x,leftBottom.x) ;//开始位置，即重叠区域的左边界
           double processWidth = img1->width - start;//重叠区域的宽度
        //   int processHeight = xformed_proc->height;//重叠区域高度
           int start2 = MIN(rightTop.y,rightBottom.y);
           int processHeight = xformed_proc->height-abs(start2);
      //IplImage* pImg2 = cvCreateImage(cvGetSize(pImg),pImg->depth,pImg->nChannels);
      //xformed = cvCreateImage(cvSize(MIN(rightTop.x,rightBottom.x),MIN(img1->height,img2->height)),IPL_DEPTH_8U,3);
        IplImage* result_img1=cvCreateImage(cvSize(processWidth ,processHeight ),IPL_DEPTH_8U,3);
        IplImage* result_img2=cvCreateImage(cvSize(processWidth ,processHeight ),IPL_DEPTH_8U,3);


        //图像2
        //设置ROI，是包含重叠区域的矩形
           cvSetImageROI(xformed_proc,cvRect(start,0,processWidth ,processHeight ));
         //复制图像
           cvCopy(xformed_proc,result_img2,0);
           cvResetImageROI(result_img2);
            cvSaveImage("result_img2.jpg",result_img2);//保存目标图像
         //   cvNamedWindow("result_img2");
         //   cvShowImage("result_img2",xformed_proc);

        //图像1
        cvSetImageROI(img1,cvRect(0,0,processWidth ,processHeight ));
        cvCopy(img1 ,result_img1,0);
           cvResetImageROI(result_img1);
            cvSaveImage("result_img1.jpg",result_img1);//保存目标图像
          //  cvNamedWindow("result_img1");
          //  cvShowImage("result_img1", img1);
    }
    else //无法计算出变换矩阵，即两幅图中没有重合区域
    {
        return NULL;
    }

    ///////////////////////////////////////////////////////////////////////////
    kdtree_release(kd_root);//释放kd树

    //只有在RANSAC算法成功算出变换矩阵时，才需要进一步释放下面的内存空间
    if(H)
    {
        cvReleaseMat(&H);//释放变换矩阵H
        free(inliers);//释放内点数组
    }
    if (NULL != xformed)
    {
        cvReleaseImage(&xformed);
    }
/*
if (NULL != result_img1)
       {
        cvReleaseImage(&result_img1);
        }
	if (NULL != result_img2)
       {
        cvReleaseImage(&result_img2);
        }
*/
    return 0;
}



IplImage* correct(IplImage *img1,IplImage *img2,IplImage *result_img1,IplImage *result_img2)
{
    //变量声明
    IplImage* xformed_proc12= NULL;

    int n1,n2;
    struct feature* feat1, * feat2, * feat;
    IplImage *img1_Feat=NULL, *img2_Feat=NULL;
   // IplImage *result_img1=NULL, *result_img2=NULL;

    IplImage* stacked;IplImage* stacked_ransac;
    struct feature** nbrs;
    struct kd_node* kd_root;

    ///////////////////////////////////////////////////


    //sift特征提取
    img1_Feat = cvCloneImage(img1);//复制图1，深拷贝，用来画特征点
    img2_Feat = cvCloneImage(img2);//复制图2，深拷贝，用来画特征点

    //默认提取的是LOWE格式的SIFT特征点
    //提取并显示第1幅图片上的特征点
    n1 = sift_features( img1, &feat1 );//检测图1中的SIFT特征点,n1是图1的特征点个数
   // export_features("featureb2.txt",feat1,n1);//将特征向量数据写入到文件
    draw_features( img1_Feat, feat1, n1 );//画出特征点
 //   cvNamedWindow(IMG1_FEAT);//创建窗口
  //  cvShowImage(IMG1_FEAT,img1_Feat);//显示

    //提取并显示第2幅图片上的特征点
    n2 = sift_features( img2, &feat2 );//检测图2中的SIFT特征点，n2是图2的特征点个数
   // export_features("featureb34.txt",feat2,n2);//将特征向量数据写入到文件
    draw_features( img2_Feat, feat2, n2 );//画出特征点
   // cvNamedWindow(IMG2_FEAT);//创建窗口
   // cvShowImage(IMG2_FEAT,img2_Feat);//显示

    //根据图1的特征点集feat1建立k-d树，返回k-d树根给kd_root
    kd_root = kdtree_build( feat1, n1 );

    CvPoint pt1,pt2;//连线的两个端点
    double d0,d1;//feat2中每个特征点到最近邻和次近邻的距离
    int matchNum = 0;//经距离比值法筛选后的匹配点对的个数

    //stacked = stack_imgs(img1, img2);
    stacked = stack_imgs_horizontal(img1, img2);

    //遍历特征点集feat2，针对feat2中每个特征点feat，选取符合距离比值条件的匹配点，放到feat的fwd_match域中
    for(int i = 0; i < n2; i++ )
    {
        feat = feat2+i;//第i个特征点的指针
        //在kd_root中搜索目标点feat的2个最近邻点，存放在nbrs中，返回实际找到的近邻点个数
        int k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if( k == 2 )
        {
            d0 = descr_dist_sq( feat, nbrs[0] );//feat与最近邻点的距离的平方
            d1 = descr_dist_sq( feat, nbrs[1] );//feat与次近邻点的距离的平方
            //若d0和d1的比值小于阈值NN_SQ_DIST_RATIO_THR，则接受此匹配，否则剔除
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
            {   //将目标点feat和最近邻点作为匹配点对
                pt2 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );//图2中点的坐标
                pt1 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//图1中点的坐标(feat的最近邻点)

                pt2.x += img1->width;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点
                //pt2.y += img1->height;//由于两幅图是左右排列的，pt2的横坐标加上图1的宽度，作为连线的终点

                cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );//画出连线
                matchNum++;//统计匹配点对的个数
                feat2[i].fwd_match = nbrs[0];//使点feat的fwd_match域指向其对应的匹配点
            }
        }
        free( nbrs );//释放近邻数组
    }
  //  fprintf( stderr, "Found %d total matches\n", matchNum );
    //显示并保存经距离比值法筛选后的匹配图
  //  cvNamedWindow(IMG_MATCH1);//创建窗口
  //  cvShowImage(IMG_MATCH1,stacked);//显示
    ////////////////////////////////////////////////////////

    //此处应统一计算特征点，进行匹配，然后统一进行拼接，直接拼接出大图

    ///////////////////////////////////////////////////////

 //   xformed_proc12 = spliceImage(img1,img2);
   spliceImage(img1,img2,result_img1,result_img2);
/*
    if (NULL != xformed_proc12)
    {
        cvNamedWindow("拼接后",1);//创建窗口
        cvShowImage("拼接后",xformed_proc12);//显示处理后的拼接图
        cvSaveImage("gg12345678.jpg",xformed_proc12);
        cvWaitKey(10);
    }
*/
    //////////////////////////////////////////////////////////////////////////////

    cvWaitKey(0);

    if(NULL != img1)
    {
        cvReleaseImage(&img1);
    }

    if(NULL != img2)
    {
        cvReleaseImage(&img2);
    }
/*
    if (NULL != xformed_proc12)
    {
        cvReleaseImage(&xformed_proc12);
        cvDestroyWindow(IMG_MOSAIC_PROC12);//显示处理后的拼接图
    }
*/


    return 0;
}
