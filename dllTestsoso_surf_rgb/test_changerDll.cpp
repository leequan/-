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
	//fileName1="/home/tuxiang/liquan/Pictures/2016-08-21/H/A.jpg";  
	//fileName2="/home/tuxiang/liquan/Pictures/2016-08-21/H/B.jpg"; 
	//fileName3="/home/tuxiang/liquan/Pictures/2016-08-21/H/AB_surf_0919.jpg"; 

	fileName1="/home/tuxiang/liquan/Pictures/0818/2/A.jpg";  
	fileName2="/home/tuxiang/liquan/Pictures/0818/2/B.jpg"; 
	fileName3="/home/tuxiang/liquan/Pictures/0818/2/AB_surf_0919.jpg";
	
	//fileName1="/home/tuxiang/liquan/Pictures/2016-07-11/2/A.jpg";  
	//fileName2="/home/tuxiang/liquan/Pictures/2016-07-11/2/B.jpg"; 
	//fileName3="/home/tuxiang/liquan/Pictures/2016-07-11/2/AB_surf_0919.jpg"; 
	
	//fileName1="/home/tuxiang/liquan/Pictures/2016-09-09/th0010_01/11/a.jpg";  
	//fileName2="/home/tuxiang/liquan/Pictures/2016-09-09/th0010_01/11/b.jpg"; 
	//fileName3="/home/tuxiang/liquan/Pictures/2016-09-09/th0010_01/11/ab_surf_0919_all0000.jpg";
	
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
