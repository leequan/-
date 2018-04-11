#include "change.h"
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include <opencv2/opencv.hpp>

using namespace cv;

int change(u_char *img1,u_char *img2,int *wart);

void main()
{
    int wart[6];
    int i,f,x,y;
    Mat image1;
    Mat image2;
//	Mat chang=cvCreateImage( cvSize( 1920, 1080 ), IPL_DEPTH_8U, 1 );

    cvNamedWindow( "场景1", CV_WINDOW_AUTOSIZE );
    cvNamedWindow( "场景2", CV_WINDOW_AUTOSIZE );
    image1=imread("C:\\Users\\htj.THTW\\Desktop\\to htj\\标记差异素材\\4组\\0.jpg",1);
    image2=imread("C:\\Users\\htj.THTW\\Desktop\\to htj\\标记差异素材\\4组\\1.jpg",1);
    if((image1.data==NULL)||(image2.data==NULL))
    {
        printf( "error1 \n" );
        return ;
    }
    f=change(image1.data,image2.data,wart);
    if(f!=0)
    {
        x=f%(PWNUM)*SHIG+PWID/2;  //变化区域中心X坐标
        y-f/(PWNUM)*SHIG+PHIG/2;  //变化区域中心Y坐标

        rectangle(image2,cvPoint(f%(PWNUM)*SHIG,f/(PWNUM)*SHIG),cvPoint(f%(PWNUM)*SHIG+PWID,f/(PWNUM)*SHIG+PWID),CV_RGB(255,0,0),2,8,0);
    }
    imshow("场景1", image1);
    imshow("场景2", image2);

    waitKey();

    while(1);
}





//分块
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


//分块
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
    // 将整幅图象进行分块，块尺寸为 [120 120]
    // 块移动的步长，即块之间的重叠区域
    int num_change=0;				// 统计早上6:30到晚上18:30时间段内变化的图像数
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
    //读入两幅图像
    //对两幅图像分块

    for(i=0;i<PNUM;i++)//对每块进行处理
    {
        //取图块放入patches
        getgraypatch(img1,i,patches1);
        getgraypatch(img2,i,patches2);
        //matcprintf(patches1,4,4);
        //matcprintf(patches2,4,4);

        //二次傅立叶变换
        fft2(patches1,PWID,PHIG,0);
        fft2(patches2,PWID,PHIG,0);
        //matcprintf(patches1,4,4);
        //matcprintf(patches2,4,4);

        //平移
        fftshift(patches1,PWID,PHIG,0);
        fftshift(patches2,PWID,PHIG,0);

        // 求交叉功率谱h
        vec_conj(patches2,PSIZE);
        vec_mul(patches1,patches2,pp,PSIZE);

        vec_abs(patches1,PSIZE);
        vec_abs(patches2,PSIZE);
        vec_mul(patches1,patches2,ps,PSIZE);
        vec_addc(ps,jeps,PSIZE);
        vec_div(pp,ps,h,PSIZE);

        fft2(h,PWID,PHIG,1);
        fftshift(h,PWID,PHIG,1);
                                                    //将交叉功率谱进行傅立叶逆变换到空域h
            val= vec_max(h,PSIZE).real;

            peak_val[i].real = val;					//将每一块得到的最大值存储到peak_val中
        }
        //matcprintf(peak_val,19,11);
        vec_sort(peak_val,PNUM,1);    //将peak_val中的值从大到小进行排序，ps虚部记录对应块的索引号
        cntmax=99999;
        for(i=0;i<6;i++)
        {
             //根据对应块的索引号将该块映射到原始图像中，[x y]表示该块左上角在原始图像中的位置
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
             if((cnt1+cnt2)<cntmax)
             {
                 cntmax=cnt1+cnt2;
                 flag=peak_val[PNUM-i-1].imag;
             }
            //  if((cnt1<2000)||(cnt2<2000))
            //	 wart[flag++]=peak_val[PNUM-i-1].imag;
        }

    if(cntmax>4000)
        flag=0;
    //虚警去除

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

    //场景基本无变化
}
