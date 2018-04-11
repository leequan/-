
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

    	//double scale=1; //设置缩放倍数
	DETECT_PARAM params ;//= {3,7,8,400,0.01,200,1.6,3,0.3};
	params.nScale = inputParam1.nScale;//检测尺度
	params.nSzKernel = inputParam1.nSzKernel;//滤波核大小
	params.nFilters = inputParam1.nFilters;//滤波器个数
	params.nChangeRnSz = inputParam1.nChangeRnSz;//变化区域最小尺寸
	params.fSensitivity = inputParam1.fSensitivity;//检测灵敏度，分割阈值
	params.nChangeRnSzKH = inputParam1.nChangeRnSzKH;//KH变化区域最小尺寸
	params.fSensitivityKH = inputParam1.fSensitivityKH;//检测灵敏度，分割阈值
	params.whratio1 = inputParam1.whratio1;
        params.whratio2 = inputParam1.whratio2;  //矩形长宽比
        params.MergeThreshold = inputParam1.MergeThreshold;  //合并阈值
        params.MAX_KEYPOINTS = inputParam1.MAX_KEYPOINTS; //最大特征点数
	params.MAX_GOODMATCHPOINTS = inputParam1.MAX_GOODMATCHPOINTS;  //最大匹配对数
       
	PRESET_POS_INFO presetPosInfo = {0,0,0,0};
	//memcpy(&presetPosInfo,(PRESET_POS_INFO&)inputParam2,sizeof(params));

        presetPosInfo.fCarrierAz = inputParam2.fCarrierAz;// 预置位方位角
        presetPosInfo.fCarrierEl = inputParam2.fCarrierEl;//预置位俯仰角
        presetPosInfo.fCarrierHt = inputParam2.fCarrierHt;//云台高度
        presetPosInfo.fFieldViewPP = inputParam2.fFieldViewPP;//预置位视场角
        
        
        if( presetPosInfo.fCarrierAz<0) presetPosInfo.fCarrierAz =0;
	if( presetPosInfo.fCarrierAz>360) presetPosInfo.fCarrierAz =360;
	
	if(presetPosInfo.fCarrierEl<-90)  presetPosInfo.fCarrierEl=-90;
	if(presetPosInfo.fCarrierEl>90)  presetPosInfo.fCarrierEl=90;

	if(presetPosInfo.fCarrierHt<0) presetPosInfo.fCarrierHt=1;
	if(presetPosInfo.fCarrierHt>10000) presetPosInfo.fCarrierHt=1000;
	
	if(presetPosInfo.fFieldViewPP<0) presetPosInfo.fFieldViewPP=1;
	if(presetPosInfo.fFieldViewPP>22) presetPosInfo.fFieldViewPP=22;
	

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
	int maxSize3 = 0;
	int maxSize4 = 0;
	int maxID = -1;
	int maxID1 = -1;
	int maxID2 = -1;
	int maxID3 = -1;
	int maxID4 = -1;
	int leftLmt = 10;
	int rightLmt = image_B.cols-10;
	int topLmt = 150;//字符裁剪位置，垂直方向
	int bottomLmt = image_B.rows-10;
	int rightX1,bottomY1;
	int rightX2,bottomY2;
	int rightX3,bottomY3;
	int rightX4,bottomY4;
	int sz,sz1,sz2,sz3,sz4;
	CvPoint origin;//配准后交点坐标
	origin.x = 0;
	origin.y = 0;
 	int Hflag=1;

       if (flag==true)
	{ 
           Mat  H;
           Mat corr_img1,corr_img2;	   
	    jisuanwh(image_A,image_B,corr_img1,corr_img2,origin,H,Hflag,params.MAX_KEYPOINTS,params.MAX_GOODMATCHPOINTS);
 
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
	
   vector<Rect> output1;
   vector<Rect> output2;
   vector<Rect> output3;
   vector<Rect> outputtemp;
   vector<float> outputCf1,outputCf2,outputCf3;
   
   
   vector<Mat> rImgA(image_A_Color.channels());
   vector<Mat> rImgB(image_B_Color.channels());
   Mat grayImgA,grayImgB,labImgA,labImgB;
   
   
   cvtColor(image_A_Color, labImgA, CV_BGR2Lab);      
   cvtColor(image_B_Color, labImgB, CV_BGR2Lab);

   split(labImgA,rImgA);
   split(labImgB,rImgB);
   
   
    ChangeDetector(rImgA[0],rImgB[0],params,output1,outputCorrelationCf,"/home/tuxiang/liquan/Pictures/cf_bw_L.jpg");
//cout<<"save cf_bw_L ok"<<endl;
   ChangeDetector(rImgA[1],rImgB[1],params,output2,outputCorrelationCf,"/home/tuxiang/liquan/Pictures/cf_bw_a.jpg");
//cout<<"save cf_bw_a ok"<<endl;
   ChangeDetector(rImgA[2],rImgB[2],params,output3,outputCorrelationCf,"/home/tuxiang/liquan/Pictures/cf_bw_b.jpg");
   
   
     for(i = 0;i<output1.size();i++)
	 {		
		rightX1 = output1.at(i).x+output1.at(i).width;
		bottomY1= output1.at(i).y+output1.at(i).height;
		double wh1=double(output1.at(i).width)/double(output1.at(i).height);	
		if(rightX1<leftLmt||rightX1>rightLmt||bottomY1<topLmt||bottomY1>bottomLmt||wh1>params.whratio1||wh1<params.whratio2)
		{	  
			continue;
		}		
		sz1 = output1.at(i).width*output1.at(i).height;	
                if(maxSize1<sz1)
                {
                   maxSize1 = sz1;
                   maxID1 = i;
               }  
	      output.push_back(output1.at(i));
         }
        // cout<<"maxSize1="<<maxSize1<<endl;
 	 for(i = 0;i<output2.size();i++)
	 {		
		rightX2 = output2.at(i).x+output2.at(i).width;
		bottomY2= output2.at(i).y+output2.at(i).height;
		double wh1=double(output2.at(i).width)/double(output2.at(i).height);	
		if(rightX2<leftLmt||rightX2>rightLmt||bottomY2<topLmt||bottomY2>bottomLmt||wh1>params.whratio1||wh1<params.whratio2)
		{	  
			continue;
		}		
		sz1 = output2.at(i).width*output2.at(i).height;	
                if(maxSize2<sz1)
                {
                   maxSize2 = sz1;
                   maxID2 = i;
               }  
	      output.push_back(output2.at(i));
         } 
        // cout<<"maxSize2="<<maxSize2<<endl;
         for(i = 0;i<output3.size();i++)
	 {		
		rightX1 = output3.at(i).x+output3.at(i).width;
		bottomY1= output3.at(i).y+output3.at(i).height;
		double wh1=double(output3.at(i).width)/double(output3.at(i).height);	
		if(rightX1<leftLmt||rightX1>rightLmt||bottomY1<topLmt||bottomY1>bottomLmt||wh1>params.whratio1||wh1<params.whratio2)
		{	  
			continue;
		}		
		sz1 = output3.at(i).width*output3.at(i).height;	
                if(maxSize1<sz1)
                {
                   maxSize3 = sz1;
                   maxID3 = i;
               }  
	      output.push_back(output3.at(i));
         }
       //  cout<<"maxSize3="<<maxSize3<<endl;

	//  output.insert(output.end(), output1.begin(),output1.end);
  	//  output.insert(output.end(), output2.begin(),output2.end);
	//  output.insert(output.end(), output3.begin(),output3.end);
 
	//rgb convert hsv start
	Mat hsvimage;
	cvtColor(image_B, hsvimage, CV_BGR2HSV);
	//imshow("BranStarkHSV", hsvimage); 
	 //waitKey(0);
	//cout << (int)hsvimage.at<Vec3b>(0, 0).type << endl;  
       Rect temp;
       vector<Rect> outputhsv;
       int hsvNum ;
       int hsvSum ;
      // cout<<"output.size="<<output.size()<<endl;
      // cout<<"hsvimage.cols"<<hsvimage.cols<<endl;
      // cout<<"hsvimage.rows"<<hsvimage.rows<<endl;
     for (int i=0;i<output.size();i++)
     {
	    hsvNum = 0;
	    hsvSum = 0;
       temp = output.at(i);
	   // cout<<"i="<<i<<endl;
	   // cout<<"temp.x="<<temp.x<<endl;
	    /*cout<<"temp.y="<<temp.y<<endl;
	    cout<<"temp.width="<<temp.width<<endl;
	    cout<<"temp.height="<<temp.height<<endl;*/
       for (int k=temp.x;k<(temp.x+temp.width);k++)
	  {
	      for (int j=temp.y;j<(temp.y+temp.height);j++)
	      {
		// cout<<"H:="<<1*((int)hsvimage.at<Vec3b>(j,k).val[0])<<endl;
		if( (1*((int)hsvimage.at<Vec3b>(j,k).val[0]) > 26)&&    
		    (1*((int)hsvimage.at<Vec3b>(j,k).val[0]) < 99) )
			{
			    hsvNum++;	     
			}
		       hsvSum += 1*((int)hsvimage.at<Vec3b>(j,k).val[0]); 
	      }
	  }
      /*cout<<"mean="<<hsvSum/(temp.width*temp.height)<<endl;
      cout<<"hsvNum="<<hsvNum<<endl;
      cout<<"mianji="<<(temp.width*temp.height)<<endl;*/
     // cout<<"ratio="<<(hsvNum*1.0)/(temp.width*temp.height)<<endl;
	if( ((hsvNum*1.0)/(temp.width*temp.height))<0.5 )
	      {
		outputhsv.push_back(temp);
	      }
	
    }
 //cout<<"outputhsv.size="<<outputhsv.size()<<endl;
 
  	//rgb convert hsv finsh
  
     
 
	//merge start
	
	if(outputhsv.size()>0)
	{
         outputtemp= MergeAllRectangle(image_B,outputhsv,origin,params.whratio1,params.whratio2,params.MergeThreshold);
         //DrawMaxMergeRectangle(image_B,outputtemp);
         DrawMaxMergeTriangleString(image_B,outputtemp,origin,presetPosInfo.fCarrierAz, presetPosInfo.fCarrierEl,presetPosInfo.fCarrierHt); //rtRgn.fAz,rtRgn.fEl);
         
        // DrawAllMergeRectangle(image_B,outputhsv,origin);
         
         rtRgn.fAz = presetPosInfo.fCarrierAz;
	 rtRgn.fEl = presetPosInfo.fCarrierEl;
	 rtRgn.fDist = presetPosInfo.fCarrierHt;
	 rtRgn.fP = 1.00;
	}
	else
	{
	  cout<<"Two images have not changes"<<endl;
	  rtRgn.fP = 0.00;
	  rtRgn.fAz = 0.00;
	  rtRgn.fEl = 0.00;
	  rtRgn.fDist = 0.00;
	}

	//merge finsh 
	
	//max changes start 
        /*
	 for(i = 0;i<output.size();i++)
	 {		
		rightX3 = output.at(i).x+output.at(i).width;
		bottomY3 = output.at(i).y+output.at(i).height;

		double wh2=double(output.at(i).width)/double(output.at(i).height);
		if(rightX3<leftLmt||rightX3>rightLmt||bottomY3<topLmt||bottomY3>bottomLmt||wh2>params.whratio1||wh2<params.whratio2)
		{	  
			continue;
		}		
		sz3 = output.at(i).width*output.at(i).height;
              //  cout<<"Rect size"<<sz3<<endl;
     	        if(maxSize3<sz3)
                {
                   maxSize = sz3;
                   maxID = i;
                } 
                
                //all changes start
               
                 DrawAllRectangle(image_B,output,i,origin);
		//DrawAllTriangle(image_B,output,i,origin); 
                
               //all changes finsh
   
         }
      
	      
	if(maxID!=-1&&maxSize!=0)
	{		
                rcTemp = output.at(maxID); 
		rtRgn.fAz = presetPosInfo.fCarrierAz+arc2deg*
2*atan(H_RESOLUTION_mm/(2*f_MIN_mm*presetPosInfo.fFieldViewPP))*(rcTemp.x+rcTemp.width/2-image_A.cols/2)/H_RESOLUTION_PIXEL;
		rtRgn.fEl = presetPosInfo.fCarrierEl+arc2deg*
2*atan(V_RESOLUTION_mm/(2*f_MIN_mm*presetPosInfo.fFieldViewPP))*(image_A.rows/2-rcTemp.y-rcTemp.height/2)/V_RESOLUTION_PIXEL;
		rtRgn.fDist = presetPosInfo.fCarrierHt/(0.00001+sin(deg2arc*rtRgn.fEl)); 
		rtRgn.fP = 1;
		if(rtRgn.fAz<0)
			rtRgn.fAz=rtRgn.fAz+360;
		if(rtRgn.fAz>360)
			rtRgn.fAz=rtRgn.fAz-360;
		
         // DrawMaxRectangle(image_B,output,rcTemp,maxID,origin);
	  //DrawMaxTriangle(image_B,output,rcTemp,maxID,origin);
          //DrawMaxDashRectangleString(image_B,output,rcTemp,maxID,origin,rtRgn.fAz,rtRgn.fEl);
	 //DrawMaxTriangleString(image_B,output,rcTemp,maxID,origin,rtRgn.fAz,rtRgn.fEl);
	}
	else
	{
		cout<<"Two images have not changes"<<endl;
		rtRgn.fP = 0;	
	}
	*/
	//max changes finsh        
        imwrite(fileName3,image_B);//save the labeled image

        return &rtRgn;
	

}
//参数计算
void jisuanwh(Mat img1,Mat img2,Mat &corr_img1,Mat &corr_img2,CvPoint& origin,Mat H,int &Hflag,int MAX_KEYPOINTS,int MAX_GOODMATCHPOINTS )
{   
    initModule_nonfree();//初始化模块
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SURF" );//创建特征检测器 
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SURF" );//特征向量生成器
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//创建特征匹配器
    if( detector.empty() || descriptor_extractor.empty() )
        cout<<"fail to create detector!";

   //特征点检测
    vector<KeyPoint> m_LeftKey,m_RightKey;
    vector<KeyPoint> mm_LeftKey,mm_RightKey;
    detector->detect( img1, mm_LeftKey );//检测img1中的SIFT特征点，存储到m_LeftKey中
    detector->detect( img2, mm_RightKey );
       
   // int MAX_KEYPOINTS = 10000;
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
    
    //根据特征点计算特征描述子矩阵，即特征向量矩阵
    Mat descriptors1,descriptors2;
    descriptor_extractor->compute( img1, m_LeftKey, descriptors1 );
    descriptor_extractor->compute( img2, m_RightKey, descriptors2 );

    //特征匹配
    vector<DMatch> matches;//匹配结果
    descriptor_matcher->match( descriptors1, descriptors2, matches );//匹配两个图像的特征矩阵

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
    
    if(goodMatches1.size()==0)
    {
       Hflag=0;
       return ;
    }
    
     vector<DMatch> goodMatches;
    // int MAX_GOODMATCHPOINTS = 5000;
     int matchPointSelctStep = 0;
     int rtMatchPoints = 0;
	 matchPointSelctStep=goodMatches1.size()/MAX_GOODMATCHPOINTS;
     int jjj=0;
	if(matchPointSelctStep>=1)
	{
	  for(int i=0;i<goodMatches1.size();i++)
	  {
		jjj++;		if((jjj%matchPointSelctStep)==0&&(rtMatchPoints<MAX_GOODMATCHPOINTS)&&(jjj<goodMatches1.size()))
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
    // 把内点转换为drawMatches可以使用的格式
    vector<KeyPoint> key1(InlinerCount);
    vector<KeyPoint> key2(InlinerCount);
    KeyPoint::convert(m_LeftInlier, key1);
    KeyPoint::convert(m_RightInlier, key2);

    //矩阵H用以存储RANSAC得到的单应矩阵
     H = findHomography( m_RightInlier,m_LeftInlier, RANSAC );
  
    //存储左图四角，及其变换到右图位置
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0); obj_corners[1] = Point( img1.cols, 0 );
    obj_corners[2] = Point( img1.cols, img1.rows ); obj_corners[3] = Point( 0, img1.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners,scene_corners, H);
 
     Mat imageturn=Mat::zeros(abs(MIN(scene_corners[0].x,scene_corners[3].x))+img2.cols,MIN(img1.rows,img2.rows),img2.type());
     
    warpPerspective(img2,imageturn,H,Size(abs(MIN(scene_corners[0].x,scene_corners[3].x))+img2.cols,MIN(img1.rows,img2.rows)));
    
    
    int start1 = MIN(cvRound(scene_corners[0].x),cvRound(scene_corners[3].x));
    int prowidth = img1.cols-abs(start1);
    int start2 = MIN(cvRound(scene_corners[0].y),cvRound(scene_corners[1].y));
    int proheight = imageturn.rows-abs(start2);
    
    Mat mat1,mat2;
        mat2 = imageturn.clone();
        mat1 = img1.clone();
    
    if(start1<=0&&start2>0)
	{
	      corr_img1=mat1(Range(0,proheight),Range(start2 ,prowidth));
              corr_img2=mat2(Range(0,proheight),Range(start2 ,prowidth));
	      origin.x=0;
	     origin.y=start2;
       }

      if(start1<=0&&start2<=0)
     {
	      corr_img1=mat1(Range(0,proheight),Range(0 ,prowidth));
              corr_img2=mat2(Range(0,proheight),Range(0 ,prowidth));
	      origin.x=0;
	      origin.y=0;
     }

     if(start1>0&&start2<=0)
     {
              corr_img1=mat1(Range(start1,proheight),Range(0 ,prowidth));
              corr_img2=mat2(Range(start1,proheight),Range(0 ,prowidth));
	      origin.x = start1;
	      origin.y = 0;
      }

    if(start1>0&&start2>0)
    {
	      corr_img1=mat1(Range(start2 ,proheight),Range(start1,prowidth));
              corr_img2=mat2(Range(start2 ,proheight),Range(start1,prowidth));
	      origin.x=start1;
	      origin.y=start2;
    }

    return ; 
}

vector<Rect> MergeAllRectangle(Mat image_B,vector<Rect> output,CvPoint origin,float whratio1,float whratio2,int MergeThreshold)
{
    Rect rc1,rc2,MrcTemp;
    vector<Rect> outputMerge;
    vector<int>::iterator itor; 
    
    if(output.size()>1)
    {
    for(int i = 0;i<output.size();i++)
	 {	
	     rc1 = output.at(i);
	       for(int j=0;j<output.size();j++)
	       {
		    rc2 = output.at(j);
		      if (isOverlap(rc1, rc2,MergeThreshold)==true)
			  {
			    MrcTemp.x = MIN(rc1.x,rc2.x);
			    MrcTemp.y = MIN(rc1.y,rc2.y);
			    MrcTemp.width = MAX(rc1.x+rc1.width,rc2.x+rc2.width)-MrcTemp.x;
			    MrcTemp.height = MAX(rc1.y+rc1.height,rc2.y+rc2.height)-MrcTemp.y;
			    rc1 = MrcTemp;
			   // output.erase(output.begin()+j);
			   // j--; //j==0 no erase j--
			   // rectangle(image_B,rcTemp,Scalar(0,0,255));//label
			    outputMerge.push_back(MrcTemp);   			    
			  }
		      else
			  {
			    outputMerge.push_back(output.at(i));
			    continue;
			   //  itor++;
			  }
		// outputMerge.push_back(rcTemp);
	       }
	 }   
    }
    else if(output.size()==1)
    {
       for(int i = 0;i<output.size();i++)
	 {
	    outputMerge.push_back(output.at(i));
	 }
    }
        return outputMerge;
}

Mat DrawMaxMergeRectangle(Mat image_B,vector<Rect> outputtemp)
{
  Rect rcTemp; 
  int Msz;
  int MmaxID = -1;
  int MmaxSize = 0;
    
   for(int i = 0;i<outputtemp.size();i++)
	 {						
		Msz = outputtemp.at(i).width*outputtemp.at(i).height;           
     	        if(MmaxSize<Msz)
                {
                   MmaxSize = Msz;
                   MmaxID = i;
                } 
          }
                rcTemp = outputtemp.at(MmaxID);
		rcTemp.x = rcTemp.x<0?0:rcTemp.x;
		rcTemp.y = rcTemp.y<0?0:rcTemp.y;
                rectangle(image_B,rcTemp,Scalar(0,0,255));//label
		
		rcTemp.x = (rcTemp.x-1)<0?0:(rcTemp.x-1);
		rcTemp.y = (rcTemp.y-1)<0?0:(rcTemp.y-1);
		rcTemp.width = rcTemp.width+2;
		rcTemp.height = rcTemp.height+2;
		rectangle(image_B,rcTemp,Scalar(0,0,0));//label
		
         return image_B;
  
}

Mat DrawAllMergeRectangle(Mat image_B,vector<Rect> outputtemp)
{
   Rect rcTemp;    
   for(int i = 0;i<outputtemp.size();i++)
	 {						
                rcTemp = outputtemp.at(i);
		rcTemp.x = rcTemp.x<0?0:rcTemp.x;
		rcTemp.y = rcTemp.y<0?0:rcTemp.y;
                rectangle(image_B,rcTemp,Scalar(0,0,255));//label
		
		rcTemp.x = (rcTemp.x-1)<0?0:(rcTemp.x-1);
		rcTemp.y = (rcTemp.y-1)<0?0:(rcTemp.y-1);
		rcTemp.width = rcTemp.width+2;
		rcTemp.height = rcTemp.height+2;
		rectangle(image_B,rcTemp,Scalar(0,0,0));//label
	 }	
         return image_B;
  
}

bool isOverlap( Rect &rc1,  Rect &rc2 ,int MergeThreshold)
{ 
  
    if ((rc1.x + rc1.width  >= rc2.x)&&
        (rc2.x + rc2.width  >= rc1.x) &&
        (rc1.y + rc1.height >= rc2.y) &&
        (rc2.y + rc2.height >= rc1.y)
       )
        return true;
	
 else if((rc2.x-(rc1.x+rc1.width)<MergeThreshold)&&((rc2.x-(rc1.x+rc1.width)>0))&&((rc1.y+rc1.height)>(rc2.y-MergeThreshold))&&((rc1.y+rc1.height)<(rc2.y+rc2.height+rc1.height+MergeThreshold))) //left	
	 return true;  
else if(((rc1.x-(rc2.x+rc2.width))<MergeThreshold)&&((rc1.x-(rc2.x+rc2.width))>0)&&((rc1.y+rc1.height)>(rc2.y-MergeThreshold))&&((rc1.y+rc1.height)<(rc2.y+rc2.height+rc1.height+MergeThreshold))) //right
	 return true;
else if(((rc2.y-(rc1.y+rc1.height))<MergeThreshold)&&((rc2.y-(rc1.y+rc1.height))>0)&&((rc1.x+rc1.width)>(rc2.x-MergeThreshold))&&((rc1.x+rc1.width)<(rc2.x+rc2.width+rc1.width+MergeThreshold))) //top
	 return true;
else if(((rc1.y-(rc2.y+rc2.height))<MergeThreshold)&&((rc1.y-(rc2.y+rc2.height))>0)&&((rc1.x+rc1.width)>(rc2.x-MergeThreshold))&&((rc1.x+rc1.width)<(rc2.x+rc2.width+rc1.width+MergeThreshold)) ) //bottom
	 return true;
    else
       return false;
   
}

Mat DrawMaxRectangle(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin)
{
		rcTemp = output.at(maxID);
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
	return image_B;
  
}

Mat DrawMaxTriangle(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin)
{
	CvPoint triangle1,triangle2,triangle3;;
	rcTemp = output.at(maxID);
	rcTemp.x = rcTemp.x-origin.x;
	rcTemp.y = rcTemp.y-origin.y;
	rcTemp.x = rcTemp.x<0?0:rcTemp.x;
	rcTemp.y = rcTemp.y<0?0:rcTemp.y;

	triangle1.x = rcTemp.x+cvRound(0.5*rcTemp.width);
	triangle1.y = rcTemp.y;
	    
	triangle2.x = rcTemp.x;
	triangle2.y = rcTemp.y+rcTemp.height;
	    
	 triangle3.x = rcTemp.x+cvRound(rcTemp.width);
	 triangle3.y = rcTemp.y+rcTemp.height;
	    
	 line(image_B,triangle1,triangle2,Scalar(0,0,255),2);
	 line(image_B,triangle1,triangle3,Scalar(0,0,255),2);
	 line(image_B,triangle2,triangle3,Scalar(0,0,255),2);
	 /*
	//draw out start
	rcTemp.x = (rcTemp.x-2)<0?0:(rcTemp.x-2);
	rcTemp.y = (rcTemp.y-2)<0?0:(rcTemp.y-2);
	triangle1.x = rcTemp.x+cvRound(0.5*rcTemp.width)+2;
	triangle1.y = rcTemp.y;
	    
	triangle2.x = rcTemp.x;
	triangle2.y = rcTemp.y+rcTemp.height+4;
	    
	 triangle3.x = rcTemp.x+cvRound(rcTemp.width)+4;
	 triangle3.y = rcTemp.y+rcTemp.height+4;
	    
	 line(image_B,triangle1,triangle2,Scalar(0,0,0),2);
	 line(image_B,triangle1,triangle3,Scalar(0,0,0),2);
	 line(image_B,triangle2,triangle3,Scalar(0,0,0),2);
	 //draw out finsh
	 */
	return image_B;
}

Mat DrawMaxDashRectangleString(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin,float fAz,float fEl)
{
	fAz = round(fAz*100)/100.0;
	fEl = round(fEl*100)/100.0;
	rcTemp = output.at(maxID);		
        rcTemp.width = 3*(rcTemp.width);
        rcTemp.height = 4*(rcTemp.height);

	int dashlength = 5;
	int linelength = 5;
	int totallength = dashlength+linelength;
	int nCountX = rcTemp.width/totallength;
	int nCountY = rcTemp.height/totallength;
		
	CvPoint start,end;	
	start.y = rcTemp.y;
	start.x = rcTemp.x;	
	end.y = rcTemp.y;
	end.x = rcTemp.x;
	//draw dash line start	
	for (int i=0;i<nCountX;i++)
	    {
	         end.x = rcTemp.x+(i+1)*totallength-dashlength;
	         end.y = rcTemp.y;
	         start.x = rcTemp.x+i*totallength;
		 start.y = rcTemp.y;
		 line(image_B,start,end,Scalar(0,255,255),2);
	    }	
	for (int i=0;i<nCountX;i++)
	    {
		  start.x = rcTemp.x+i*totallength;
		  start.y = rcTemp.y+rcTemp.height;
		  end.x = rcTemp.x+(i+1)*totallength-dashlength;
		  end.y = rcTemp.y+rcTemp.height;
		  line(image_B,start,end,Scalar(0,255,255),2);
	    }	
	for (int i=0;i<nCountY;i++)
	    {
		  start.x = rcTemp.x;
		  start.y = rcTemp.y+i*totallength;
		  end.x = rcTemp.x;
		  end.y = rcTemp.y+(i+1)*totallength-dashlength;
		  line(image_B,start,end,Scalar(0,255,255),2);
	    }	
	for (int i=0;i<nCountY;i++)
	    {
		 start.x = rcTemp.x+rcTemp.width;
		 start.y = rcTemp.y+i*totallength;
		 end.x = rcTemp.x+rcTemp.width;
		 end.y = rcTemp.y+(i+1)*totallength-dashlength;
		 line(image_B,start,end,Scalar(0,255,255),2);
	    }
	    //draw dash line finsh
	    //draw corner line start
            start.x = rcTemp.x;
	    start.y = rcTemp.y;
	    end.x = rcTemp.x;
	    end.y = rcTemp.y+3*linelength;
	    line(image_B,start,end,Scalar(0,0,255),2);
	    
	    start.x = rcTemp.x;
	    start.y = rcTemp.y;
	    end.x = rcTemp.x+3*linelength;
	    end.y = rcTemp.y;
	    line(image_B,start,end,Scalar(0,0,255),2);  //left top
	       
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y;
	    end.x = rcTemp.x+rcTemp.width-3*linelength;
	    end.y = rcTemp.y;
	    line(image_B,start,end,Scalar(0,0,255),2);
	    
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y;
	    end.x = rcTemp.x+rcTemp.width;
	    end.y = rcTemp.y+3*linelength;
	    line(image_B,start,end,Scalar(0,0,255),2);  //right top
			
	    start.x = rcTemp.x;
	    start.y = rcTemp.y+rcTemp.height;
	    end.x = rcTemp.x;
	    end.y = rcTemp.y+rcTemp.height-3*linelength;
	    line(image_B,start,end,Scalar(0,0,255),2);
	    
	    start.x = rcTemp.x;
	    start.y = rcTemp.y+rcTemp.height;
	    end.x = rcTemp.x+3*linelength;
	    end.y = rcTemp.y+rcTemp.height;
	    line(image_B,start,end,Scalar(0,0,255),2);  //left bottom
	    
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y+rcTemp.height;
	    end.x = rcTemp.x+rcTemp.width;
	    end.y = rcTemp.y+rcTemp.height-3*linelength;
	    line(image_B,start,end,Scalar(0,0,255),2);
	    
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y+rcTemp.height;
	    end.x = rcTemp.x+rcTemp.width-3*linelength;
	    end.y = rcTemp.y+rcTemp.height;
	    line(image_B,start,end,Scalar(0,0,255),2);  //right bottom
	    //draw corner line finsh
	    
	    //write string start
	    //1.int convert string start
	     std::stringstream s1;
	     std::string str1;
	     s1<<fAz;
	     s1>>str1;  
	
	    std::stringstream s2;
	    std::string str2;
	    s2<<fEl;
	    s2>>str2;  

	    string strAz="Az:"; //"预置位方位角";
	    string strEl="El:";  //"俯仰角";
	    
	    //string str1="270.08 ";
	    //string str2="-90.12";
	      string text1 = strAz+str1;
	      string text2 = strEl+str2;
	      string text = text1+text2;
	    //1.int convert string finsh  
	      
	//2.write string start
	if(rcTemp.x<200&&rcTemp.y<200)//right bottom
	{
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y+rcTemp.height;  
	    end.x = rcTemp.x+rcTemp.width+30;
	    end.y = rcTemp.y+rcTemp.height+30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    start.x = rcTemp.x+rcTemp.width+30+2*rcTemp.width;
	    start.y = rcTemp.y+rcTemp.height+30;
	    end.x = rcTemp.x+rcTemp.width+30;
	    end.y = rcTemp.y+rcTemp.height+30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    Point pt(end.x+cvRound(0.5*rcTemp.width),end.y-7);	    
	    Scalar color = CV_RGB(0,255,255);
	    putText(image_B,text,pt,CV_FONT_HERSHEY_DUPLEX,1.0f,color); 
	}	
	
	if(rcTemp.x<200&&rcTemp.y+rcTemp.height>image_B.rows-200) //right top
	{
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y;  
	    end.x = rcTemp.x+rcTemp.width+30;
	    end.y = rcTemp.y-30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    start.x = rcTemp.x+rcTemp.width+30;
	    start.y = rcTemp.y-30;
	    end.x = rcTemp.x+rcTemp.width+30+2*rcTemp.width;
	    end.y = rcTemp.y-30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    Point pt(start.x+cvRound(0.5*rcTemp.width),start.y-7);	    
	    Scalar color = CV_RGB(0,255,255);
	    putText(image_B,text,pt,CV_FONT_HERSHEY_DUPLEX,1.0f,color);  
	}	
	//left top
	if(rcTemp.x+rcTemp.width>image_B.cols-200&&rcTemp.y+rcTemp.height>image_B.rows-200)
	{
	    start.x = rcTemp.x;
	    start.y = rcTemp.y;  
	    end.x = rcTemp.x-30;
	    end.y = rcTemp.y-30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    start.x = rcTemp.x-30;
	    start.y = rcTemp.y-30;
	    end.x = rcTemp.x-30-2*rcTemp.width;
	    end.y = rcTemp.y-30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	   Point pt(end.x+cvRound(0.5*rcTemp.width),end.y-7);	    
	    Scalar color = CV_RGB(0,255,255);
	    putText(image_B,text,pt,CV_FONT_HERSHEY_DUPLEX,1.0f,color);  
	}	
	
	if(rcTemp.y<200&&rcTemp.x+rcTemp.width>image_B.cols-200)//left bottom
	{
	    start.x = rcTemp.x;
	    start.y = rcTemp.y+rcTemp.height;  
	    end.x = rcTemp.x-30;
	    end.y = rcTemp.y+rcTemp.height+30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    start.x = rcTemp.x-30;
	    start.y = rcTemp.y+rcTemp.height+30;
	    end.x = rcTemp.x-30-2*rcTemp.width;
	    end.y = rcTemp.y+rcTemp.height+30;
	    line(image_B,start,end,Scalar(0,0,255));
	    
	    Point pt(end.x+cvRound(0.5*rcTemp.width),end.y-7);	    
	    Scalar color = CV_RGB(0,255,255);
	    putText(image_B,text,pt,CV_FONT_HERSHEY_DUPLEX,1.0f,color);  
	}
		
	else
	{  
	    start.x = rcTemp.x+rcTemp.width;
	    start.y = rcTemp.y+rcTemp.height;  
	    end.x = rcTemp.x+rcTemp.width+30;
	    end.y = rcTemp.y+rcTemp.height+30;
	    line(image_B,start,end,Scalar(0,0,255),2);
	    
	    start.x = rcTemp.x+rcTemp.width+30+rcTemp.width+110;
	    start.y = rcTemp.y+rcTemp.height+30;
	    end.x = rcTemp.x+rcTemp.width+30;
	    end.y = rcTemp.y+rcTemp.height+30;
	    line(image_B,start,end,Scalar(0,0,255),2);
	    
	    Point pt(end.x,end.y-7);	    
	    Scalar color = CV_RGB(0,0,255);
	    putText(image_B,text,pt,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); //right bottom    
	}   
	    return image_B;
  
}

Mat DrawMaxTriangleString(Mat image_B,vector<Rect> output,Rect rcTemp,int maxID,CvPoint origin,float fAz,float fEl)
{
        fAz = round(fAz*100)/100.0;
	fEl = round(fEl*100)/100.0;
	//draw triangle start
	CvPoint triangle1,triangle2,triangle3,triangle4,triangle5;
	rcTemp = output.at(maxID);
	rcTemp.x = rcTemp.x-origin.x;
	rcTemp.y = rcTemp.y-origin.y;
	rcTemp.x = rcTemp.x<0?0:rcTemp.x;
	rcTemp.y = rcTemp.y<0?0:rcTemp.y;

	triangle1.x = rcTemp.x+cvRound(0.5*rcTemp.width);
	triangle1.y = rcTemp.y;
	    
	triangle2.x = rcTemp.x;
	triangle2.y = rcTemp.y+rcTemp.height;
	    
	 triangle3.x = rcTemp.x+cvRound(rcTemp.width);
	 triangle3.y = rcTemp.y+rcTemp.height;
	    
	 line(image_B,triangle1,triangle2,Scalar(0,0,255),2);
	 line(image_B,triangle1,triangle3,Scalar(0,0,255),2);
	 line(image_B,triangle2,triangle3,Scalar(0,0,255),2);
	 //draw triangle finsh

	    //1.float convert string start
	     std::stringstream s1;
	     std::string str1;
	     s1<<fAz;
	     s1>>str1; 
	     s1.str("");
	     s1.clear();
	
	    std::stringstream s2;
	    std::string str2;
	    s2<<fEl;
	    s2>>str2; 
	    s2.str("");
	    s2.clear();
	    
	    string strAz="Az:"; //"预置位方位角";
	    string strEl="El:";  //"俯仰角";
	    string textNone = " ";
	    
	      string text1 = strAz+str1;
	      string text2 = text1+textNone;
	      string text3 = text2+strEl;
	      string text = text3+str2;
	    //1.float convert string finsh
	    //draw string start
	    	  int kh = cvRound(0.5*rcTemp.width);
	     //if(triangle1.y<330&&triangle2.x<330) //left top 
	      if((triangle1.x-kh<330)&&(triangle1.y+rcTemp.height>image_B.rows-50))//left bottom
	       {
		triangle4.x = triangle1.x+30;
		triangle4.y = triangle1.y-30;
		line(image_B,triangle1,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x+330;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2);
	      }
	    else if((triangle1.x>330+kh)&&(triangle1.x<image_B.cols-330-kh)&&(triangle1.y+rcTemp.height>image_B.rows-50))
	     {
	       
	       triangle4.x = triangle1.x+30;
		triangle4.y = triangle1.y-30;
		line(image_B,triangle1,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x+330;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2);	       
	    }
	    
	     else if((triangle1.x>image_B.cols-330-kh)&&(triangle1.y+rcTemp.height>image_B.rows-50)) //right bottom
	     {
	        triangle4.x = triangle1.x-30;
		triangle4.y = triangle1.y-30;
		line(image_B,triangle1,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x-330;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x-320,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
	     }	
	    
	    else if((triangle1.x>image_B.cols-330-kh)&&(triangle1.y>50)&&(triangle1.y+rcTemp.height<image_B.rows-50)&&(triangle1.y>50))
	    {
	      triangle4.x = triangle2.x-30;
		triangle4.y = triangle2.y+30;
		line(image_B,triangle2,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x-330;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x-320,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
		}	
	    
	    else if((triangle1.y<50)&&(triangle1.x>image_B.cols-330-kh))  //right topLmt
	     {
		triangle4.x = triangle2.x-30;
		triangle4.y = triangle2.y+30;
		line(image_B,triangle2,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x-330;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x-320,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
		}	
      
	     else
	     {
		triangle4.x = triangle3.x+30;
		triangle4.y = triangle3.y+30;
		line(image_B,triangle3,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x+330;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
	      } 
	   //draw string finsh
	   return image_B;
  
}

Mat DrawMaxMergeTriangleString(Mat image_B,vector<Rect> outputtemp,CvPoint origin,float &fAz,float &fEl,float &fDist)
{
  Rect rcTemp; 
  int Msz;
  int MmaxID = -1;
  int MmaxSize = 0;
    
   for(int i = 0;i<outputtemp.size();i++)
	 {						
		Msz = outputtemp.at(i).width*outputtemp.at(i).height;           
     	        if(MmaxSize<Msz)
                {
                   MmaxSize = Msz;
                   MmaxID = i;
                } 
          }
		rcTemp = outputtemp.at(MmaxID);
		//cout<<"rcTemp.width="<<rcTemp.width<<endl;
		//cout<<"rcTemp.height="<<rcTemp.height<<endl;		
		
		float fAz1 = fAz+ arc2deg*2*atan(H_RESOLUTION_mm/(2*f_MIN_mm*1))*(rcTemp.x+rcTemp.width/2-image_B.cols/2)/H_RESOLUTION_PIXEL;
		float fEl1 = fEl+arc2deg*2*atan(V_RESOLUTION_mm/(2*f_MIN_mm*1))*(image_B.rows/2-rcTemp.y-rcTemp.height/2)/V_RESOLUTION_PIXEL;
		float fDist1 = fDist/(0.00001+sin(deg2arc*fEl1));
	          
		
		if(fAz1<0)
			fAz1=fAz1+360;
		if(fAz1>360)
			fAz1-360;		
		
        fAz = round(fAz1*100)/100.0;
        fEl = round(fEl1*100)/100.0;
	fDist = round(fDist1*100)/100.0;
	
	//draw triangle start
	CvPoint triangle1,triangle2,triangle3,triangle4,triangle5;
	/*cout<<"rcTemp.x="<<rcTemp.x<<endl;
	cout<<"rcTemp.y="<<rcTemp.y<<endl;
	cout<<"rcTemp.width="<<rcTemp.width<<endl;
	cout<<"rcTemp.height="<<rcTemp.height<<endl;*/
	int bianjie = 300;
	
	if(rcTemp.width<bianjie&&rcTemp.height<bianjie)  
	  { 
	    rcTemp.x = rcTemp.x-((bianjie-rcTemp.width)/2);
	    rcTemp.y = rcTemp.y-((bianjie-rcTemp.height)/2);
	    rcTemp.width=bianjie; rcTemp.height=bianjie;
	  }
	if(rcTemp.width>bianjie&&rcTemp.height<bianjie)  
	  { 
	    rcTemp.x = rcTemp.x;
	    rcTemp.y = rcTemp.y-((bianjie-rcTemp.height)/2);
	    rcTemp.width=bianjie; rcTemp.height=bianjie;
	  }
	if(rcTemp.width<bianjie&&rcTemp.height>bianjie)  
	  { 
	    rcTemp.x = rcTemp.x-((bianjie-rcTemp.width)/2);
	    rcTemp.y = rcTemp.y; 
	    rcTemp.width=bianjie; rcTemp.height=bianjie;
	  }
	 else
	 {
	    rcTemp.width=bianjie; rcTemp.height=bianjie;
	    rcTemp.x = rcTemp.x;
	    rcTemp.y = rcTemp.y;
	}
	/*cout<<"rcTemp.x="<<rcTemp.x<<endl;
	cout<<"rcTemp.y="<<rcTemp.y<<endl;
	cout<<"rcTemp.width="<<rcTemp.width<<endl;
	cout<<"rcTemp.height="<<rcTemp.height<<endl; */   
	
	
	rcTemp.x = rcTemp.x-origin.x;
	rcTemp.y = rcTemp.y-origin.y;
	rcTemp.x = rcTemp.x<0?0:rcTemp.x;
	rcTemp.y = rcTemp.y<0?0:rcTemp.y;

	triangle1.x = rcTemp.x+cvRound(0.5*rcTemp.width);
	triangle1.y = rcTemp.y;
	triangle1.x = triangle1.x-3<0?0:triangle1.x-3;
	triangle1.y = triangle1.y-3<0?0:triangle1.y-3;
	triangle1.x = triangle1.x-3>image_B.cols?image_B.cols-3:triangle1.x-3;
	triangle1.y = triangle1.y-3>image_B.rows?image_B.rows-3:triangle1.y-3;
	    
	triangle2.x = rcTemp.x;
	triangle2.y = rcTemp.y+rcTemp.height;
	triangle2.x = triangle2.x-3<0?0:triangle2.x-3;
	triangle2.y = triangle2.y-3<0?0:triangle2.y-3;
	triangle2.x = triangle2.x-3>image_B.cols?image_B.cols-3:triangle2.x-3;
	triangle2.y = triangle2.y-3>image_B.rows?image_B.rows-3:triangle2.y-3;  
	
	 triangle3.x = rcTemp.x+cvRound(rcTemp.width);
	 triangle3.y = rcTemp.y+rcTemp.height;
	 triangle3.x = triangle3.x-3<0?0:triangle3.x-3;
	 triangle3.y = triangle3.y-3<0?0:triangle3.y-3;
	 triangle3.x = triangle3.x-3>image_B.cols?image_B.cols-3:triangle3.x-3;
	 triangle3.y = triangle3.y-3>image_B.rows?image_B.rows-3:triangle3.y-3; 
	 
	 line(image_B,triangle1,triangle2,Scalar(0,0,255),2);
	 line(image_B,triangle1,triangle3,Scalar(0,0,255),2);
	 line(image_B,triangle2,triangle3,Scalar(0,0,255),2);
	 //draw triangle finsh

	    //1.float convert string start
	     std::stringstream s1;
	     std::string str1;
	     s1<<fAz;
	     s1>>str1; 
	     s1.str("");
	     s1.clear();
	
	    std::stringstream s2;
	    std::string str2;
	    s2<<fEl;
	    s2>>str2; 
	    s2.str("");
	    s2.clear();
	    
	    string strAz="Az:"; //"预置位方位角";
	    string strEl="El:";  //"俯仰角";
	    string textNone = " ";
	    
	      string text1 = strAz+str1;
	      string text2 = text1+textNone;
	      string text3 = text2+strEl;
	      string text = text3+str2;
	    //1.float convert string finsh
	    //draw string start
	      int kh = cvRound(0.5*rcTemp.width);
	      int StringLength = 330;
	      int BrokenlineLength = 30;
	      int BoundaryThreshold = 50;
	     //if(triangle1.y<330&&triangle2.x<330) //left top 
	      if((triangle1.x-kh<StringLength)&&(triangle1.y+rcTemp.height>image_B.rows-BoundaryThreshold))//left bottom
	       {
		triangle4.x = triangle1.x+BrokenlineLength;
		triangle4.y = triangle1.y-BrokenlineLength;
		line(image_B,triangle1,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x+StringLength;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2);
	      }
	    else if((triangle1.x>StringLength+kh)&&(triangle1.x<image_B.cols-StringLength-kh)&&(triangle1.y+rcTemp.height>image_B.rows-BoundaryThreshold))
	     {
	       
	       triangle4.x = triangle1.x+BrokenlineLength;
		triangle4.y = triangle1.y-BrokenlineLength;
		line(image_B,triangle1,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x+StringLength;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2);	       
	    }
	    
	     else if((triangle1.x>image_B.cols-StringLength-kh)&&(triangle1.y+rcTemp.height>image_B.rows-BoundaryThreshold)) //right bottom
	     {
	        triangle4.x = triangle1.x-BrokenlineLength;
		triangle4.y = triangle1.y-BrokenlineLength;
		line(image_B,triangle1,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x-StringLength;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x-320,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
	     }	
	    
	    else if((triangle1.x>image_B.cols-StringLength-kh)&&(triangle1.y>BoundaryThreshold)&&(triangle1.y+rcTemp.height<image_B.rows-BoundaryThreshold)&&(triangle1.y>BoundaryThreshold))
	    {
	      triangle4.x = triangle2.x-BrokenlineLength;
		triangle4.y = triangle2.y+BrokenlineLength;
		line(image_B,triangle2,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x-StringLength;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x-320,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
		}	
	    
	    else if((triangle1.y<BoundaryThreshold)&&(triangle1.x>image_B.cols-StringLength-kh))  //right topLmt
	     {
		triangle4.x = triangle2.x-BrokenlineLength;
		triangle4.y = triangle2.y+BrokenlineLength;
		line(image_B,triangle2,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x-StringLength;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x-320,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
		}	
      
	     else
	     {
		triangle4.x = triangle3.x+BrokenlineLength;
		triangle4.y = triangle3.y+BrokenlineLength;
		line(image_B,triangle3,triangle4,Scalar(0,0,255),2);
		
		triangle5.x = triangle4.x+StringLength;
		triangle5.y = triangle4.y;
		line(image_B,triangle4,triangle5,Scalar(0,0,255),2);
		
		Point pt3(triangle4.x,triangle4.y-7);
		Scalar color = CV_RGB(0,0,255);
		putText(image_B,text,pt3,CV_FONT_HERSHEY_DUPLEX,1.0f,color,2); 
	      } 
	   //draw string finsh
	   return image_B;
  
  
}

Mat DrawAllRectangle(Mat image_B,vector<Rect> output,int i,CvPoint origin)
{
		Rect rcTemp;
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
	return image_B;
  
}

Mat DrawAllTriangle(Mat image_B,vector<Rect> output,int i,CvPoint origin)
{
	Rect rcTemp;
        CvPoint triangle1,triangle2,triangle3;;
	rcTemp = output.at(i);
	rcTemp.x = rcTemp.x-origin.x;
	rcTemp.y = rcTemp.y-origin.y;
	rcTemp.x = rcTemp.x<0?0:rcTemp.x;
	rcTemp.y = rcTemp.y<0?0:rcTemp.y;

	triangle1.x = rcTemp.x+cvRound(0.5*rcTemp.width);
	triangle1.y = rcTemp.y;
	    
	triangle2.x = rcTemp.x;
	triangle2.y = rcTemp.y+rcTemp.height;
	    
	 triangle3.x = rcTemp.x+cvRound(rcTemp.width);
	 triangle3.y = rcTemp.y+rcTemp.height;
	    
	 line(image_B,triangle1,triangle2,Scalar(0,0,255),2);
	 line(image_B,triangle1,triangle3,Scalar(0,0,255),2);
	 line(image_B,triangle2,triangle3,Scalar(0,0,255),2);
	 /*
	//draw out start
	rcTemp.x = (rcTemp.x-2)<0?0:(rcTemp.x-2);
	rcTemp.y = (rcTemp.y-2)<0?0:(rcTemp.y-2);
	triangle1.x = rcTemp.x+cvRound(0.5*rcTemp.width)+2;
	triangle1.y = rcTemp.y;
	    
	triangle2.x = rcTemp.x;
	triangle2.y = rcTemp.y+rcTemp.height+4;
	    
	 triangle3.x = rcTemp.x+cvRound(rcTemp.width)+4;
	 triangle3.y = rcTemp.y+rcTemp.height+4;
	    
	 line(image_B,triangle1,triangle2,Scalar(0,0,0),2);
	 line(image_B,triangle1,triangle3,Scalar(0,0,0),2);
	 line(image_B,triangle2,triangle3,Scalar(0,0,0),2);
	 //draw out finsh
	 */
	return image_B;
}

Mat DrawAllMergeRectangle(Mat image_B,vector<Rect> output,CvPoint origin)
{
    
  for(int i = 0;i<output.size();i++)
	 {		 
                 DrawAllRectangle(image_B,output,i,origin);		   
         }
  return image_B;
}
