#include "changerIO.hpp"


int main( int argc, char** argv)  
{    
  
	DETECT_PARAM params = {3,7,8,400,0.01,100,1.6,5,0.2,100,100000,50000};
	PRESET_POS_INFO presetPosInfo = {0,0,1,1};
   	bool correctflag=true;
        CHANGE_RGN* rtRgn = NULL;

        Mat image_A,image_B;
	string fileName1, fileName2, fileName3;
	
	fileName1 = argv[1];
	fileName2 = argv[2];
	fileName3 = argv[3];
	cout<<fileName1<<endl;
	cout<<fileName2<<endl;
	cout<<fileName3<<endl;
	
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

        clock_t start,finish;
   	start=clock();

        rtRgn = SceneChangeDetector(img1,img2,img3,params,presetPosInfo,correctflag);
	//if(rtRgn!=NULL)
	//delete []rtRgn;	
	free(img1);
	free(img2);
	free(img3);

	finish=clock();
	cout<<"run time is "<<(finish-start)/1000000<<"s"<<endl;
        return 0;  
}  
