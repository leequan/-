
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "changerIO.hpp"
#include "change.hpp"


CHANGE_RGN* SceneChangeDetector(char* inputPath1,char* inputPath2,char* outputPath ,int thr,float thr2,PRESET_POS_INFO inputParam2)

{
       
	char fileName1[256];
	char fileName2[256];
	char fileName3[256];

    strcpy(fileName1,inputPath1);
	strcpy(fileName2,inputPath2);
	strcpy(fileName3,outputPath);

	PRESET_POS_INFO presetPosInfo = {0,0,0,0};

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
	

        //int i;
        Mat image_A,image_B; 
        Mat image_A1,image_B1;
        CHANGE_RGN rtRgn;// = NULL;

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


   int f,x,y;


   resize(image_A,image_A1,cvSize(1920,1024));
   resize(image_B,image_B1,cvSize(1920,1024));

   f=change(image_A1.data,image_B1.data,thr,thr2);


   if(f!=0)
   {
       x=f%(PWNUM)*SHIG;  //变化区域X坐标
       y=f/(PWNUM)*SHIG;  //变化区域Y坐标

     //  rectangle(image_B1,cvPoint(f%(PWNUM)*SHIG,f/(PWNUM)*SHIG),cvPoint(f%(PWNUM)*SHIG+PWID,f/(PWNUM)*SHIG+PWID),CV_RGB(255,0,0),2,8,0);
     DrawMaxMergeTriangleString(image_B1,x,y,presetPosInfo.fCarrierAz, presetPosInfo.fCarrierEl,presetPosInfo.fCarrierHt); //rtRgn.fAz,rtRgn.fEl);
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

        imwrite(fileName3,image_B1);//save the labeled image
       // imshow("change", image_B1);
       // waitKey();

        return &rtRgn;
}



Mat DrawMaxMergeTriangleString(Mat image_B,int x, int y,float &fAz,float &fEl,float &fDist)
{
  Rect rcTemp; 

    
  rcTemp.x = x;
  rcTemp.y = y;
  rcTemp.width = PHIG;
  rcTemp.height = PWID;
		
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
