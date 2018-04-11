#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define		u_char   unsigned char

#define		PI  	3.1415926535		//Բ����
#define		jeps     2.2204e-16

//ͼƬ
#define IWID	1920		//ͼƬ��
#define IHIG	1024 		//ͼƬ��
#define ISIZE	(IWID*IHIG)	//ͼƬ�ܳ�

//�ֿ� 
#define PWID	128			//���
#define PHIG	128			//���
#define PSIZE	(PWID*PHIG)	//���ܳ�

#define SWID    (PWID/2)		//���򲽳�
#define SHIG	(PHIG/2)		//���򲽳�

#define PWNUM   (IWID/SWID-1)   //�������
#define PHNUM   (IHIG/SHIG-1)   //�������
#define PNUM    (PWNUM*PHNUM)   //�ܿ���

//�Ҷȹ�������
#define GRAYG	1			//�Ҷ�ѹ������
#define GSIZE	(256/GRAYG)*(256/GRAYG)	


//ʵ��
typedef struct
{
	float real;  //ʵ��
	float imag;  //�鲿 ����Ϊʵ��ʱ����λ��������
}complex;


//����
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