
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <iostream>
using namespace std;
using namespace cv;

#include"changerIO.hpp"

CHANGE_RGN* SceneChangeDetector(char* inputPath1,char* inputPath2,char* outputPath,DETECT_PARAM inputParam1,PRESET_POS_INFO inputParam2,bool flag)
{
       
	char fileName1[256];
	char fileName2[256];
	char fileName3[256];

        strcpy(fileName1,inputPath1);
	strcpy(fileName2,inputPath2);
	strcpy(fileName3,outputPath);

    	double scale=1; //设置缩放倍数
	DETECT_PARAM params ;//= {3,7,8,400,0.01,200,1.6,3,0.3};
	params.nScale = inputParam1.nScale;//检测尺度
	params.nSzKernel = inputParam1.nSzKernel;//滤波核大小
	params.nFilters = inputParam1.nFilters;//滤波器个数
	params.nChangeRnSz = (inputParam1.nChangeRnSz)*scale;//变化区域最小尺寸
	params.fSensitivity = inputParam1.fSensitivity;//检测灵敏度，分割阈值
	params.nChangeRnSzKH = (inputParam1.nChangeRnSzKH)*scale;//KH变化区域最小尺寸
	params.fSensitivityKH = inputParam1.fSensitivityKH;//检测灵敏度，分割阈值
	params.whratio1 = inputParam1.whratio1;
        params.whratio2 = inputParam1.whratio2;  //矩形长宽比

	PRESET_POS_INFO presetPosInfo = {0,0,0,0};
	//memcpy(&presetPosInfo,(PRESET_POS_INFO&)inputParam2,sizeof(params));

        presetPosInfo.fCarrierAz = inputParam2.fCarrierAz;// 预置位方位角
        presetPosInfo.fCarrierEl = inputParam2.fCarrierEl;//预置位俯仰角
        presetPosInfo.fCarrierHt = inputParam2.fCarrierHt;//云台高度
        presetPosInfo.fFieldViewPP = inputParam2.fFieldViewPP;//预置位视场角

        int i;
        Mat image_A,image_B; 
        Mat image_A_Color,image_B_Color;

        CHANGE_RGN rtRgn;// = NULL;
        int rtRgnNum = 0;
  	int rtRgnNum1 = 0;
        Rect rcTemp;
	Mat srcImgA; 
	Mat srcImgB;
	vector<Rect> output;//save the changed region
        vector<float> outputCorrelationCf;//save the corresponding correlation coefficient
        image_A = imread(fileName1, 1); // Read the file  
        image_B = imread(fileName2,1);


        if(! image_A.data ) // Check for invalid input  
        {  
            cout << "Could not open or find the image" << std::endl ;  
            return NULL;  
        } 
        if(! image_B.data ) // Check for invalid input  
        {   
            cout << "Could not open or find the image" << std::endl ;  
            return NULL;  
        } 
	if(image_B.size!=image_A.size)
		 return NULL;

        int maxSize = 0;
	int maxSize1 = 0;
	int maxSize2 = 0;
	int maxID = -1;
	int maxID1 = -1;
	int maxID2 = -1;
	int leftLmt = 10*scale;
	int rightLmt = (image_B.cols-10)*scale;
	int topLmt = 150*scale;//字符裁剪位置，垂直方向
	int bottomLmt = (image_B.rows-10)*scale;
	int rightX1,bottomY1;
	int rightX2,bottomY2;
	int sz,sz1,sz2;
	CvPoint origin;//配准后交点坐标
	origin.x = 0;
	origin.y = 0;
 	int Hflag=1;

       if (flag==true)
	{ 
           Mat  H;
           Mat corr_img1,corr_img2;
 
	   jisuanwh(image_A,image_B,corr_img1,corr_img2,origin,H,Hflag);	  //surf
 
              if(Hflag==1)
        	{
	            cvtColor(corr_img1, srcImgA, CV_RGB2GRAY);
		    cvtColor(corr_img2, srcImgB, CV_RGB2GRAY);
		    image_A_Color = corr_img1;
		    image_B_Color = corr_img2;
		}
	 }

	if((flag==false)||(Hflag==0))
	{
		cvtColor(image_A, srcImgA, CV_RGB2GRAY);
		cvtColor(image_B, srcImgB, CV_RGB2GRAY);
		image_A_Color = image_A;
		image_B_Color = image_B;
	}
	// DETECT_PARAM params1={3,7,8,200,1.6};
	 vector<Rect> output1;
	 vector<float> outputCf;
	 //detect the changes from KH space
	 ChangeDetectorKH(image_A_Color,image_B_Color,params,output1,outputCf);
         ChangeDetector(srcImgA,srcImgB, params,output,outputCorrelationCf);//detect the changes
         rtRgnNum1 = output1.size();
	 rtRgnNum = output.size();	
  //all changes start
/*	 
	for(int i = 0;i<output1.size();i++)
	 { 
	        rcTemp = output1.at(i);
	        rcTemp.x = rcTemp.x-origin.x;
		rcTemp.y = rcTemp.y-origin.y;
		rcTemp.x = rcTemp.x<0?0:rcTemp.x;
		rcTemp.y = rcTemp.y<0?0:rcTemp.y;
                rectangle(image_B,rcTemp,Scalar(0,0,255));//label

		rcTemp.x = (rcTemp.x-1)<0?0:(rcTemp.x-1);
		rcTemp.y = (rcTemp.y-1)<0?0:(rcTemp.y-1);
		rcTemp.width = rcTemp.width+2;
		rcTemp.height = rcTemp.height+2;
		rectangle(image_B,rcTemp,Scalar(0,0,0));//label
	 }
	 
	for(int i = 0;i<output.size();i++)
	 { 
	        rcTemp = output.at(i);
	        rcTemp.x = rcTemp.x-origin.x;
		rcTemp.y = rcTemp.y-origin.y;
		rcTemp.x = rcTemp.x<0?0:rcTemp.x;
		rcTemp.y = rcTemp.y<0?0:rcTemp.y;
                rectangle(image_B,rcTemp,Scalar(0,0,255));//label

		rcTemp.x = (rcTemp.x-1)<0?0:(rcTemp.x-1);
		rcTemp.y = (rcTemp.y-1)<0?0:(rcTemp.y-1);
		rcTemp.width = rcTemp.width+2;
		rcTemp.height = rcTemp.height+2;
		rectangle(image_B,rcTemp,Scalar(0,0,0));//label
	 }
	 //all changes finsh
*/	 
	 
     // max changes start
	 
	 for(int i = 0;i<output1.size();i++)
	 {		
		rightX1 = output1.at(i).x+output1.at(i).width;
		bottomY1= output1.at(i).y+output1.at(i).height;
		double wh1=double(output1.at(i).width)/double(output1.at(i).height);	
		if(rightX1<leftLmt||rightX1>rightLmt||bottomY1<topLmt||bottomY1>bottomLmt||wh1>params.whratio1||wh1<params.whratio2)
		{	  
			continue;
		}		
		sz1 = output1.at(i).width*output1.at(i).height;	
		cout<<"sz1="<<sz1<<endl;
                if(maxSize1<sz1)
                {
                   maxSize1 = sz1;
                   maxID1 = i;
               }  
         }
         
	 for(int j = 0;j<output.size();j++)
	 {		
		rightX2 = output.at(j).x+output.at(j).width;
		bottomY2 = output.at(j).y+output.at(j).height;

		double wh2=double(output.at(j).width)/double(output.at(j).height);
		if(rightX2<leftLmt||rightX2>rightLmt||bottomY2<topLmt||bottomY2>bottomLmt||wh2>params.whratio1||wh2<params.whratio2)
		{	  
			continue;
		}		
		sz2 = output.at(j).width*output.at(j).height;
		cout<<"sz2="<<sz2<<endl;
		 if(maxSize2<sz2)
                {
                   maxSize2 = sz2;
                   maxID2 = j;
                }    
         }
		//cout<<"maxSize2="<<maxSize2<<endl;
	if(maxSize1<maxSize2)
	{
		maxSize=maxSize2;
		maxID=maxID2;
		rcTemp = output.at(maxID);
	}
	if(maxSize1>maxSize2)
	{
		maxSize=maxSize1;
		maxID=maxID1;
		rcTemp = output1.at(maxID);
	}
	if((maxSize1==0)&&(maxSize2==0))
	{
		maxSize=0;
		maxID=-1;
	}


	if(maxID!=-1)
	{	  
	        rcTemp.x = rcTemp.x-origin.x;
		rcTemp.y = rcTemp.y-origin.y;
		cout<<"rcTemp.x="<<rcTemp.x<<endl;
		cout<<"rcTemp.y="<<rcTemp.y<<endl;
		cout<<"rcTemp.width="<<rcTemp.width<<endl;
		cout<<"rcTemp.height="<<rcTemp.height<<endl;
		
		rcTemp.x = rcTemp.x<0?0:rcTemp.x;
		rcTemp.y = rcTemp.y<0?0:rcTemp.y;
                rectangle(image_B,rcTemp,Scalar(0,0,255));//label

		rcTemp.x = (rcTemp.x-1)<0?0:(rcTemp.x-1);
		rcTemp.y = (rcTemp.y-1)<0?0:(rcTemp.y-1);
		rcTemp.width = rcTemp.width+2;
		rcTemp.height = rcTemp.height+2;
		rectangle(image_B,rcTemp,Scalar(0,0,0));//label

                 rtRgn.fAz = presetPosInfo.fCarrierAz+arc2deg*
2*atan(H_RESOLUTION_mm/(2*f_MIN_mm*presetPosInfo.fFieldViewPP))*(rcTemp.x+rcTemp.width/2-image_A.cols/2)/H_RESOLUTION_PIXEL;
		rtRgn.fEl = presetPosInfo.fCarrierEl+arc2deg*
2*atan(V_RESOLUTION_mm/(2*f_MIN_mm*presetPosInfo.fFieldViewPP))*(rcTemp.y+rcTemp.height/2-image_A.rows/2)/V_RESOLUTION_PIXEL;
		rtRgn.fDist = presetPosInfo.fCarrierHt/(0.00001+sin(deg2arc*rtRgn.fEl)); 
		rtRgn.fP = 1;
		if(rtRgn.fAz<0)
			rtRgn.fAz=rtRgn.fAz+360;
		if(rtRgn.fAz>360)
			rtRgn.fAz=rtRgn.fAz-360;              
	 }

	else
	{
		cout<<"Two images have not changes"<<endl;
		rtRgn.fP = 0;	
	}
    // max changes finsh

        imwrite(fileName3,image_B);//save the labeled image
        return &rtRgn;

}
//参数计算
void jisuanwh(Mat img1,Mat img2,Mat &corr_img1,Mat &corr_img2,CvPoint& origin,Mat H,int &Hflag)
{   
    initModule_nonfree();//初始化模块
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF" );//创建特征检测器 
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SURF" );//特征向量生成器
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//创建特征匹配器
    if( detector.empty() || descriptor_extractor.empty() )
        cout<<"fail to create detector!";

   //特征点检测
    double t = getTickCount();//当前滴答数
    vector<KeyPoint> m_LeftKey,m_RightKey;
    vector<KeyPoint> mm_LeftKey,mm_RightKey;
    detector->detect( img1, mm_LeftKey );//检测img1中的SIFT特征点，存储到m_LeftKey中
    detector->detect( img2, mm_RightKey );
    cout<<"图像1特征点个数:"<<mm_LeftKey.size()<<endl;
    cout<<"图像2特征点个数:"<<mm_RightKey.size()<<endl;
      t = ((double)getTickCount() - t)/getTickFrequency();
    cout<<"特征点用时："<<t<<"秒"<<endl;
       
    int MAX_KEYPOINTS = 5000;
    int lkeyPointSelctStep = 0;
    int lrtMatchPoints = 0;
        lkeyPointSelctStep=mm_LeftKey.size()/MAX_KEYPOINTS;
    int j=0;
	if(lkeyPointSelctStep>=1)
	{
	  for(int i=0;i<mm_LeftKey.size();i++)
	  {
		if(j<mm_LeftKey.size())
		{
			j++;
			if((j%lkeyPointSelctStep)==0&&(lrtMatchPoints<MAX_KEYPOINTS))
			{	   			 
			  m_LeftKey.push_back(mm_LeftKey[i]);
			  lrtMatchPoints++; 
			}
		}
	  }
	}
     else
	{
	      m_LeftKey = mm_LeftKey;
	}

     
     int rkeyPointSelctStep = 0;
     int rrtMatchPoints = 0;
	 rkeyPointSelctStep=mm_LeftKey.size()/MAX_KEYPOINTS;
     int jj=0;
	if(rkeyPointSelctStep>=1)
	{
	  for(int ii=0;ii<mm_RightKey.size();ii++)
	  {
		if(jj<mm_RightKey.size())
		{
			jj++;
			if((jj%rkeyPointSelctStep)==0&&(rrtMatchPoints<MAX_KEYPOINTS))
			{
			  m_RightKey.push_back(mm_RightKey[ii]);
			  rrtMatchPoints++;
			}
		}
	  }
	}
     else
	{	
		m_RightKey = mm_RightKey;
	}
    
    double t1 = getTickCount();//当前滴答数
    //根据特征点计算特征描述子矩阵，即特征向量矩阵
    Mat descriptors1,descriptors2;
    descriptor_extractor->compute( img1, m_LeftKey, descriptors1 );
    descriptor_extractor->compute( img2, m_RightKey, descriptors2 );
    t1 = ((double)getTickCount() - t1)/getTickFrequency();
    cout<<"特征描述子矩阵用时："<<t1<<"秒"<<endl;

    cout<<"图像1特征描述矩阵大小："<<descriptors1.size()
        <<"，特征向量个数："<<descriptors1.rows<<"，维数："<<descriptors1.cols<<endl;
    cout<<"图像2特征描述矩阵大小："<<descriptors2.size()
        <<"，特征向量个数："<<descriptors2.rows<<"，维数："<<descriptors2.cols<<endl;

	//画出特征点
    Mat img_m_LeftKey,img_m_RightKey;
    drawKeypoints(img1,m_LeftKey,img_m_LeftKey,Scalar::all(-1),0);
    drawKeypoints(img2,m_RightKey,img_m_RightKey,Scalar::all(-1),0);
	// imshow("Src1",img_m_LeftKey);
	// waitKey(0);
    //imshow("Src2",img_m_RightKey);
	//waitKey(0);
    //特征匹配
    vector<DMatch> matches;//匹配结果
    descriptor_matcher->match( descriptors1, descriptors2, matches );//匹配两个图像的特征矩阵
    cout<<"Match个数："<<matches.size()<<endl;

    //计算匹配结果中距离的最大和最小值
    //距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近
    double max_dist = 0; double min_dist = 100;
    for(int i=0; i<matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    //筛选出较好的匹配点
    vector<DMatch> goodMatches1;
    for(int i=0; i<matches.size(); i++)
    {
        if(matches[i].distance < 0.2 * max_dist)
        {
            goodMatches1.push_back(matches[i]);
        }
    }
    cout<<"goodMatch1个数："<<goodMatches1.size()<<endl;
    
    //画出匹配结果
  //  Mat img_matches;
    //红色连接的是匹配的特征点对，绿色是未匹配的特征点
   // drawMatches(img1,m_LeftKey,img2,m_RightKey,goodMatches1,img_matches,
    //    Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);

  //  imshow("MatchSIFT",img_matches);
  //  waitKey(0);
    
    
    if(goodMatches1.size()==0)
    {
       Hflag=0;
       return ;
    }
    
     vector<DMatch> goodMatches;
     int MAX_GOODMATCHPOINTS = 1000;
     int matchPointSelctStep = 0;
     int rtMatchPoints = 0;
	 matchPointSelctStep=goodMatches1.size()/MAX_GOODMATCHPOINTS;
     int jjj=0;
	if(matchPointSelctStep>=1)
	{
	  for(int i=0;i<goodMatches1.size();i++)
	  {
		jjj++;
	if((jjj%matchPointSelctStep)==0&&(rtMatchPoints<MAX_GOODMATCHPOINTS)&&(jjj<goodMatches1.size()))
			{
			    goodMatches.push_back(goodMatches1[i]);
			   rtMatchPoints++;
			}
	   }
	}
	else
	{
		goodMatches = goodMatches1;
	}   
   cout<<"goodMatch个数："<<goodMatches.size()<<endl;   

    //RANSAC匹配过程
    vector<DMatch> m_Matches=goodMatches;
    int ptCount = (int)m_Matches.size();  // 分配空间
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);

    // 把Keypoint转换为Mat
    Point2f pt;
    for (int i=0; i<ptCount; i++)
    {
        pt = m_LeftKey[m_Matches[i].queryIdx].pt;
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;

        pt = m_RightKey[m_Matches[i].trainIdx].pt;
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }

    // 用RANSAC方法计算F
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;       // 这个变量用于存储RANSAC后每个点的状态
    findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

    // 计算野点个数
    int OutlinerCount = 0;
    for (int i=0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] == 0)    // 状态为0表示野点
        {
            OutlinerCount++;
        }
    }
    int InlinerCount = ptCount - OutlinerCount;   // 计算内点
    cout<<"内点数为："<<InlinerCount<<endl;

   // 这三个变量用于保存内点和匹配关系
   vector<Point2f> m_LeftInlier;
   vector<Point2f> m_RightInlier;
   vector<DMatch> m_InlierMatches;

    m_InlierMatches.resize(InlinerCount);
    m_LeftInlier.resize(InlinerCount);
    m_RightInlier.resize(InlinerCount);
    InlinerCount=0;
    float inlier_minRx=img1.cols;        //用于存储内点中右图最小横坐标，以便后续融合
    float inlier_minRy=img1.rows;

    for (int i=0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
            m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
            m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
            m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;

            if(m_RightInlier[InlinerCount].x<inlier_minRx) 
	      inlier_minRx=m_RightInlier[InlinerCount].x;   //存储内点中右图最小横坐标
	      
            if(m_RightInlier[InlinerCount].y<inlier_minRy) 
	      inlier_minRy=m_RightInlier[InlinerCount].y;   //存储内点中右图最小横坐标
	      
            InlinerCount++;
        }
    }
      cout<<"inlier_minRx= "<<inlier_minRx<<endl;
    // 把内点转换为drawMatches可以使用的格式
    vector<KeyPoint> key1(InlinerCount);
    vector<KeyPoint> key2(InlinerCount);
    KeyPoint::convert(m_LeftInlier, key1);
    KeyPoint::convert(m_RightInlier, key2);

    // 显示计算F过后的内点匹配
  //  Mat OutImage;
  //  drawMatches(img1, key1, img2, key2, m_InlierMatches, OutImage);
  //  cvNamedWindow( "Match features", 1);
    //cvShowImage("Match features", &IplImage(OutImage));
  //  imshow("Match features", OutImage);
  //  waitKey(0);
    
    
    
    cvDestroyAllWindows();

    //矩阵H用以存储RANSAC得到的单应矩阵
     H = findHomography( m_RightInlier,m_LeftInlier, RANSAC );
  

    //存储左图四角，及其变换到右图位置
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
    obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners,scene_corners, H);

    
     cout<<" scene_corners[0]="<< scene_corners[0]<<endl;
     cout<<" scene_corners[1]="<< scene_corners[1]<<endl;
     cout<<" scene_corners[2]="<< scene_corners[2]<<endl;
     cout<<" scene_corners[3]="<< scene_corners[3]<<endl;
  
     Mat imageturn=Mat::zeros(abs(MIN(scene_corners[0].x,scene_corners[3].x))+img2.cols,MIN(img1.rows,img2.rows),img2.type());
     
    warpPerspective(img2,imageturn,H,Size(abs(MIN(scene_corners[0].x,scene_corners[3].x))+img2.cols,MIN(img1.rows,img2.rows)));
    
    
    int start1 = MIN(cvRound(scene_corners[0].x),cvRound(scene_corners[3].x));
    int prowidth = img1.cols-abs(start1);
    int start2 = MIN(cvRound(scene_corners[0].y),cvRound(scene_corners[1].y));
    int proheight = imageturn.rows-abs(start2);
    
     cout<<" start1="<< start1<<endl;
     cout<<" prowidth="<< prowidth<<endl;
     cout<<" start2="<< start2<<endl;
     cout<<" proheight="<< proheight<<endl;
     
    
      Mat mat1,mat2;
      //corr_img1,corr_img2;
        mat2 = imageturn.clone();
        mat1 = img1.clone();
    
    if(start1<=0&&start2>0)
	{
	      corr_img1=mat1(Range(0,proheight),Range(start2 ,prowidth));
              corr_img2=mat2(Range(0,proheight),Range(start2 ,prowidth));
	      origin.x=0;
	     origin.y=start2;
	     imwrite("corr_img1.JPG",corr_img1);
             imwrite("corr_img2.JPG",corr_img2);

       }

      if(start1<=0&&start2<=0)
     {
	      corr_img1=mat1(Range(0,proheight),Range(0 ,prowidth));
              corr_img2=mat2(Range(0,proheight),Range(0 ,prowidth));
	      origin.x=0;
	      origin.y=0;
	      imwrite("corr_img1.JPG",corr_img1);
              imwrite("corr_img2.JPG",corr_img2);
     }

     if(start1>0&&start2<=0)
     {
              corr_img1=mat1(Range(start1,proheight),Range(0 ,prowidth));
              corr_img2=mat2(Range(start1,proheight),Range(0 ,prowidth));
	      origin.x = start1;
	      origin.y = 0;
	      imwrite("corr_img1.JPG",corr_img1);
              imwrite("corr_img2.JPG",corr_img2);
      }

    if(start1>0&&start2>0)
    {
	      corr_img1=mat1(Range(start2 ,proheight),Range(start1,prowidth));
              corr_img2=mat2(Range(start2 ,proheight),Range(start1,prowidth));
	      origin.x=start1;
	      origin.y=start2;
	      imwrite("corr_img1.JPG",corr_img1);
              imwrite("corr_img2.JPG",corr_img2);
    }

    return ; 
}

