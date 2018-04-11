#include "change.hpp"
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include <opencv2/opencv.hpp>

using namespace cv;


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
            for(j=0;j<PWID;j++)
			{
                b[i*(PWID)*3+j*3+2]=a[((h*(SHIG)+i)*(IWID)+w*(SWID)+j)*3];
                b[i*(PWID)*3+j*3+1]=a[((h*(SHIG)+i)*(IWID)+w*(SWID)+j)*3+1];
                b[i*(PWID)*3+j*3]=a[((h*(SHIG)+i)*(IWID)+w*(SWID)+j)*3+2];
			}
		}
	}
}


//�ֿ�
void getgraypatch(u_char* a,int no,complex_1 *b)
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


int change(u_char *img1,u_char *img2,int thr,float thr2)
{
    // ������ͼ����зֿ飬��ߴ�Ϊ [120 120]
    // ���ƶ��Ĳ���������֮����ص�����
    int num_change=0;				// ͳ������6:30������18:30ʱ����ڱ仯��ͼ����
    int i,x,y,p,cnt1,cnt2,cvt1,cvt2,cntmax;
    //double k;
    int flag;

    complex_1 *peak_val=(complex_1*)malloc(PSIZE*sizeof(complex_1));
    u_char *candi1=(u_char*)malloc(PSIZE*3*sizeof(u_char));
    u_char *candi2=(u_char*)malloc(PSIZE*3*sizeof(u_char));
    complex_1 *patches1=(complex_1*)malloc(PSIZE*sizeof(complex_1));
    complex_1 *patches2=(complex_1*)malloc(PSIZE*sizeof(complex_1));
    complex_1 *pp=(complex_1*)malloc(PSIZE*sizeof(complex_1));
    complex_1 *h=(complex_1*)malloc(PSIZE*sizeof(complex_1));
    complex_1 *ps=(complex_1*)malloc(PSIZE*sizeof(complex_1));
    float *h1=(float*)malloc(PSIZE*sizeof(float));
    float *h2=(float*)malloc(PSIZE*sizeof(float));
    float *v1=(float*)malloc(PSIZE*sizeof(float));
    float *v2=(float*)malloc(PSIZE*sizeof(float));

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
        val=peak_val[PNUM-1].real;

     //   if(val>thr2)
     //       return 0;

        for(i=0;i<20;i++)
        {
             //���ݶ�Ӧ��������Ž��ÿ�ӳ�䵽ԭʼͼ���У�[x y]��ʾ�ÿ����Ͻ���ԭʼͼ���е�λ��
            if(peak_val[PNUM-i-1].imag/PWNUM>1)
            {
             getpatch(img1,peak_val[PNUM-i-1].imag,candi1);
             getpatch(img2,peak_val[PNUM-i-1].imag,candi2);
             rgb2hsv(PSIZE,candi1,h1,v1);
             rgb2hsv(PSIZE,candi2,h2,v2);
             cnt1=0;cnt2=0;
             cvt1=0;cvt2=0;
             for(p=0;p<PSIZE;p++)
             {
                if((h1[p]>0.1)&&(h1[p]<0.4))   cnt1++;
                if((h2[p]>0.1)&&(h2[p]<0.4))   cnt2++;
                if(v1[p]<0.4) cvt1++;
                if(v2[p]<0.4) cvt2++;
             }


             if(((cnt1+cnt2)<10000)&&((cnt1+cnt2)>10)&&(cvt1+cvt2<20000))
             {
                 flag=peak_val[PNUM-i-1].imag;
             }
            }
            //  if((cnt1<2000)||(cnt2<2000))
            //	 wart[flag++]=peak_val[PNUM-i-1].imag;
        }

   // if(cntmax>thr)
    //    flag=0;


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
    free(v1);
    free(v2);
    free(ps);
    return flag;

    //���������ޱ仯
}
