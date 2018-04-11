#include "changerIO.hpp"


int main( int argc, char** argv)  
{    
  
	DETECT_PARAM params = {3,7,8,400,0.01,200,1.6,5,0.2};
	PRESET_POS_INFO presetPosInfo = {0,0,0,0};
   	bool correctflag=true;
        CHANGE_RGN* rtRgn = NULL;

	clock_t start,finish;  
    	start=clock();  

	Mat image_A,image_B;
	string fileName1, fileName2, fileName3;
	fileName1 = argv[1];
	fileName2 = argv[2];
	fileName3 = argv[3];
cout<<fileName1<<endl;
cout<<fileName2<<endl;
cout<<fileName3<<endl;
        //fileName1="/home/guangdian/zhangtao/ftp/2016-07-11/5/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-07-11/5/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-07-11/5/AB.jpg"; 
        
	//fileName1="/home/guangdian/zhangtao/ftp/2016-08-18/5/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-08-18/5/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-08-18/5/AB.jpg"; 
  
        //fileName1="/home/guangdian/zhangtao/ftp/2016-08-20/8/A.jpg";  
        //fileName2="/home/guangdian/zhangtao/ftp/2016-08-20/8/B.jpg"; 
        //fileName3="/home/guangdian/zhangtao/ftp/2016-08-20/8/AB.jpg"; 

        //fileName1="/home/guangdian/zhangtao/ftp/2016-08-21/6/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-08-21/6/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-08-21/6/AB.jpg"; 
		
        //fileName1="/home/guangdian/zhangtao/ftp/2016-08-22/4/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-08-22/4/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-08-22/4/AB.jpg"; 
	
	//fileName1="/home/guangdian/zhangtao/ftp/2016-08-26/5/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-08-26/5/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-08-26/5/AB.jpg"; 

	//fileName1="/home/guangdian/zhangtao/ftp/2016-08-27/1/1504.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-08-27/1/1518.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-08-31/orign/yuan/02_0909_1.bmp"; 
	
	//fileName1="/home/guangdian/zhangtao/ftp/2016-08-31/09/0.bmp";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-08-31/09/9.bmp"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-08-31/orign/yuan/02_0909_1.bmp"; 

	//fileName1="/home/guangdian/zhangtao/ftp/2016-09-01/4/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-09-01/4/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-09-01/4/AB.jpg"; 

        //fileName1="/home/guangdian/zhangtao/ftp/2016-09-09/1/A.jpg";  
	//fileName2="/home/guangdian/zhangtao/ftp/2016-09-09/1/B.jpg"; 
	//fileName3="/home/guangdian/zhangtao/ftp/2016-09-09/1/AB.jpg"; 



	char* img1;
	int fileName1_L=fileName1.length();
	img1=(char *)malloc((fileName1_L+1)*sizeof(char));
	fileName1.copy(img1,fileName1_L,0);

	char *img2;
	int fileName2_L=fileName2.length();
	img2=(char *)malloc((fileName2_L+1)*sizeof(char));
	fileName2.copy(img2,fileName2_L,0);

	char *img3;
	int fileName3_L=fileName3.length();
	img3=(char *)malloc((fileName3_L+1)*sizeof(char));
	fileName3.copy(img3,fileName3_L,0);


        rtRgn = SceneChangeDetector(img1,img2,img3,params,presetPosInfo,correctflag);
	//if(rtRgn!=NULL)
	//delete []rtRgn;	
	free(img1);
	free(img2);
	free(img3);

	finish=clock(); 
	cout<<"All time is "<<(finish-start)/1000000<<"s"<<endl;
        return 0;  
}  
