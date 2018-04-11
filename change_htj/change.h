#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define		u_char   unsigned char

#define		PI  	3.1415926535		//圆周率
#define		jeps     2.2204e-16

//图片
#define IWID	1920		//图片宽
#define IHIG	1024 		//图片高
#define ISIZE	(IWID*IHIG)	//图片总长

//分块 
#define PWID	128			//块宽
#define PHIG	128			//块高
#define PSIZE	(PWID*PHIG)	//块总长

#define SWID    (PWID/2)		//横向步长
#define SHIG	(PHIG/2)		//纵向步长

#define PWNUM   (IWID/SWID-1)   //横向块数
#define PHNUM   (IHIG/SHIG-1)   //纵向块数
#define PNUM    (PWNUM*PHNUM)   //总块数

//灰度共生矩阵
#define GRAYG	1			//灰度压缩级数
#define GSIZE	(256/GRAYG)*(256/GRAYG)	


//实数
typedef struct
{
	float real;  //实部
	float imag;  //虚部 （当为实数时代表位置索引）
}complex;


//区域
typedef struct
{
	int a;
}area;








 void fft(complex *a,int size,char iopt);
 void fft2(complex *a,int wid,int hig,char iopt);
 void fftshift(complex *a,int wid,int hig,char iopt);

 void vec_sort(complex *a,int size,char iopt);
 void vec_addc(complex *a,float b,int size);
 void vec_mul(complex *a,complex *b,complex *c,int size);
 void vec_div(complex *a,complex *b,complex *c,int size);
 void vec_abs(complex *a,int size);
 void vec_conj(complex *a,int size);
 complex vec_max(complex *a,int size);
 void rgb2hsv(int size,u_char *im,float *h);