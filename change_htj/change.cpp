#include "change.h"
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int change(u_char *img1,u_char *img2,int *wart);

int main()
{
	int wart[6];
	int i,f,x,y;
	Mat image1;
	Mat image2;
//	Mat chang=cvCreateImage( cvSize( 1920, 1080 ), IPL_DEPTH_8U, 1 );

	cvNamedWindow( "����1", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "����2", CV_WINDOW_AUTOSIZE );
    image1=imread("/home/jalywangtuxiang/QT/picture/change/A/0.jpg",1);
    image2=imread("/home/jalywangtuxiang/QT/picture/change/A/1.jpg",1);
	if((image1.data==NULL)||(image2.data==NULL))
	{
		printf( "error1 \n" );
        return 1;
	}
	f=change(image1.data,image2.data,wart);
    i=0;
    while(i<16)
	{
        f=wart[i];
        i++;
		x=f%(PWNUM)*SHIG+PWID/2;  //�仯��������X����
		y-f/(PWNUM)*SHIG+PHIG/2;  //�仯��������Y����

        rectangle(image2,cvPoint(f%(PWNUM)*SHIG,f/(PWNUM)*SHIG),cvPoint(f%(PWNUM)*SHIG+PWID,f/(PWNUM)*SHIG+PWID),CV_RGB(255,0,0),2,8,0);
	}
	imshow("����1", image1);
	imshow("����2", image2);

	waitKey();

	while(1);
}


//�ֿ�
void getpatch(u_char* a,int no,u_char *b)
{
	int i,j,w,h;
	if(no<PNUM)
	{
		h=no/(PWNUM);
		w=no%(PWNUM);
		for(i=0;i<PHIG;i++)
		{
			for(j=0;j<PWID*3;j++)
			{
				b[i*(PWID)+j]=a[(h*(SHIG)+i)*(IWID)*3+w*(SWID)*3+j];
			}
		}
	}
}


//�ֿ�
void getgraypatch(u_char* a,int no,complex *b)
{
	int i,j,w,h;
	if(no<PNUM)
	{
		h=no/(PWNUM);
		w=no%(PWNUM);
		for(i=0;i<PHIG;i++)
		{
			for(j=0;j<PWID;j++)
			{
				b[i*(PWID)+j].real=(float)(a[((h*(SHIG)+i)*(IWID)+w*(SWID)+j)*3]*30+a[((h*(SHIG)+i)*(IWID)+w*(SWID)+j)*3+1]*59+a[((h*(SHIG)+i)*(IWID)+w*(SWID)+j)*3+2]*11+50)/100;
				b[i*(PWID)+j].imag=0;
			}
		}
	}
}

int change(u_char *img1,u_char *img2,int *wart)
{
	// ������ͼ����зֿ飬��ߴ�Ϊ [120 120]
	// ���ƶ��Ĳ���������֮����ص�����
	int num_change=0;				// ͳ������6:30������18:30ʱ����ڱ仯��ͼ����
	int i,x,y,p,cnt1,cnt2,cntmax;
	//double k;
	int flag;

	complex *peak_val=(complex*)malloc(PSIZE*sizeof(complex));
	u_char *candi1=(u_char*)malloc(PSIZE*3*sizeof(u_char));
	u_char *candi2=(u_char*)malloc(PSIZE*3*sizeof(u_char));
	complex *patches1=(complex*)malloc(PSIZE*sizeof(complex));
	complex *patches2=(complex*)malloc(PSIZE*sizeof(complex));
	complex *pp=(complex*)malloc(PSIZE*sizeof(complex));
	complex *h=(complex*)malloc(PSIZE*sizeof(complex));
	complex *ps=(complex*)malloc(PSIZE*sizeof(complex));
	float *h1=(float*)malloc(PSIZE*sizeof(float));
	float *h2=(float*)malloc(PSIZE*sizeof(float));

	float val;
	
	flag=0;
	//��������ͼ��
	//������ͼ��ֿ�		
		
	for(i=0;i<PNUM;i++)//��ÿ����д���
	{
		//ȡͼ�����patches
		getgraypatch(img1,i,patches1);
		getgraypatch(img2,i,patches2);
		//matcprintf(patches1,4,4);
		//matcprintf(patches2,4,4);

		//���θ���Ҷ�任
		fft2(patches1,PWID,PHIG,0);
		fft2(patches2,PWID,PHIG,0);
		//matcprintf(patches1,4,4);
		//matcprintf(patches2,4,4);

		//ƽ��
		fftshift(patches1,PWID,PHIG,0);
		fftshift(patches2,PWID,PHIG,0);

		// �󽻲湦����h
		vec_conj(patches2,PSIZE);
		vec_mul(patches1,patches2,pp,PSIZE);

		vec_abs(patches1,PSIZE);
		vec_abs(patches2,PSIZE);
		vec_mul(patches1,patches2,ps,PSIZE);
		vec_addc(ps,jeps,PSIZE);
		vec_div(pp,ps,h,PSIZE);

		fft2(h,PWID,PHIG,1);
		fftshift(h,PWID,PHIG,1);
													//�����湦���׽��и���Ҷ��任������h
			val= vec_max(h,PSIZE).real;			
													
			peak_val[i].real = val;					//��ÿһ��õ������ֵ�洢��peak_val��
		}
		//matcprintf(peak_val,19,11);
		vec_sort(peak_val,PNUM,1);    //��peak_val�е�ֵ�Ӵ�С��������ps�鲿��¼��Ӧ���������
		cntmax=99999;
        for(i=0;i<20;i++)
		{
            printf("[%f,%d]\n",peak_val[PNUM-i-1].real,(int)peak_val[PNUM-i-1].imag);
			 //���ݶ�Ӧ��������Ž��ÿ�ӳ�䵽ԭʼͼ���У�[x y]��ʾ�ÿ����Ͻ���ԭʼͼ���е�λ��				
			 getpatch(img1,peak_val[PNUM-i-1].imag,candi1);
			 getpatch(img2,peak_val[PNUM-i-1].imag,candi2);
			 rgb2hsv(PSIZE,candi1,h1);
			 rgb2hsv(PSIZE,candi2,h2);
			 cnt1=0;cnt2=0;
			 for(p=0;p<PSIZE;p++)
			 {
				if((h1[p]>0.1)&&(h1[p]<0.4))   cnt1++;
				if((h2[p]>0.1)&&(h2[p]<0.4))   cnt2++;
			 }
            // if((cnt1+cnt2)<cntmax)
            // {
             cntmax=cnt1+cnt2;
            // flag=peak_val[PNUM-i-1].imag;
            // }
              if(cntmax<5500)
             wart[flag++]=peak_val[PNUM-i-1].imag;
            // rectangle(img2,cvPoint((int)(peak_val[PNUM-i-1].imag)%(PWNUM)*SHIG,(int)(peak_val[PNUM-i-1].imag)/(PWNUM)*SHIG),cvPoint((int)(peak_val[PNUM-i-1].imag)%(PWNUM)*SHIG+PWID,(int)(peak_val[PNUM-i-1].imag)/(PWNUM)*SHIG+PWID),CV_RGB(255,0,0),2,8,0);
		}		

    //if(cntmax>4000)
    //	flag=0;
	//�龯ȥ��
	
	free(peak_val);
	free(candi1);
	free(candi2);
	free(patches1);
	free(patches2);
	free(pp);
	free(h);
	free(h1);
	free(h2);
	free(ps);
	return flag;
	
	//���������ޱ仯
}
