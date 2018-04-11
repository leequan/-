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
}complex_1;


//����
typedef struct
{
	int a;
}area;




void fft(complex_1 *a,int size,char iopt);
void fft2(complex_1 *a,int wid,int hig,char iopt);
void fftshift(complex_1 *a,int wid,int hig,char iopt);

void vec_sort(complex_1 *a,int size,char iopt);
void vec_addc(complex_1 *a,float b,int size);
void vec_mul(complex_1 *a,complex_1 *b,complex_1 *c,int size);
void vec_div(complex_1 *a,complex_1 *b,complex_1 *c,int size);
void vec_abs(complex_1 *a,int size);
void vec_conj(complex_1 *a,int size);
complex_1 vec_max(complex_1 *a,int size);
void rgb2hsv(int size,u_char *im,float *h);

extern int change(u_char *img1,u_char *img2,int thr,float thr2);




