
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <string>
#include <iostream>
using namespace std;

extern "C"
{
#include "mymosaic.h"
}

int main( int argc, char** argv )
{
    //变量声明
    string name1,name2;
    IplImage *img1=NULL, *img2=NULL ,*resultimg;
    IplImage *result_img1, *result_img2;
 
    //加载图片
    name1 = "A.jpg";name2="B.jpg";
    img1 = cvLoadImage( name1.c_str());
    img2 = cvLoadImage( name2.c_str());
   
    correct(img1,img2,result_img1,result_img2);
   // result_img1=mymosaic(img1,img2);

    return 0;
}


