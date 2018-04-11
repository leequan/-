#include "change.hpp"

//�������
void vec_mul(complex_1 *a,complex_1 *b,complex_1 *c,int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		c[i].real=a[i].real*b[i].real-a[i].imag*b[i].imag;
		c[i].imag=a[i].real*b[i].imag+a[i].imag*b[i].real;
	}
}


//�������
void vec_div(complex_1 *a,complex_1 *b,complex_1 *c,int size)
{
	int i; float d;
	for(i=0;i<size;i++)
	{
		d=b[i].real*b[i].real+b[i].imag*b[i].imag;
		c[i].real=(a[i].real*b[i].real+a[i].imag*b[i].imag)/d;
		c[i].imag=(a[i].imag*b[i].real-a[i].real*b[i].imag)/d;
	}
}

//�����ӳ���
void vec_addc(complex_1 *a,float b,int size)
{
	int i; 
	for(i=0;i<size;i++)
	{
		a[i].real=a[i].real+b;
	}
}

//����ʵ���鲿��ƽ���͵�ƽ����
void vec_abs(complex_1 *a,int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		a[i].real=sqrt(a[i].real*a[i].real+a[i].imag*a[i].imag);
		a[i].imag=0;
	}
}

//������������ֵ
void vec_conj(complex_1 *a,int size)
{
	int i;
	for(i=0;i<size;i++)
	{
		a[i].real=a[i].real;
		a[i].imag=0-a[i].imag;
	}
}

//���������ֵ���鲿Ϊ������
complex_1 vec_max(complex_1 *a,int size)
{
	int i;
    complex_1 c;
	c.real=0;
	for(i=0;i<size;i++)
	{
		if(c.real<a[i].real)
		{
			c.real=a[i].real;
			c.imag=i;
		}
	}
	return c;
}

//��������Сֵ���鲿Ϊ������
complex_1 vec_min(complex_1 *a,int size)
{
	int i;
    complex_1 c;
	for(i=0;i<size;i++)
	{
		if(c.real>a[i].real)
		{
			c.real=a[i].real;	
			c.imag=i;
		}
	}
	return c;
}

//�����Ӵ�С�����鲿Ϊ������
void vec_sort(complex_1 *a,int size,char iopt)
{
	int i,j;
    complex_1 b;
	for(i=0;i<size;i++)
	{
		a[i].imag=i;
	}
	if(iopt)
	{
		for(i=0;i<size-1;i++)
		{
			for(j=i+1;j<size;j++)
			{
				if(a[i].real<a[j].real)
				{
					b=a[i];
					a[i]=a[j];
					a[j]=b;
				}
			}
		}
	}
	else
	{
		for(i=0;i<size-1;i++)
		{
			for(j=i+1;j<size;j++)
			{
				if(a[i].real>a[j].real)
				{
					b=a[i];
					a[i]=a[j];
					a[j]=b;
				}
			}
		}
	}
}

float trf_min(float a,float b,float c)
{
	return (a>=b)?((b>=c)?c:b):((a>=c)?c:a);
}

float trf_max(float a,float b,float c)
{
	return (a<=b)?((b<=c)?c:b):((a<=c)?c:a);
}

//RGBתHSV��ʽ
void rgb2hsv(int size,u_char *im,float *h,float *v)
{
	int i,k;


	float *fr=(float*)malloc(size*sizeof(float));
	float *fg=(float*)malloc(size*sizeof(float));
	float *fb=(float*)malloc(size*sizeof(float));
    //float *v=(float*)malloc(size*sizeof(float));
	float *s=(float*)malloc(size*sizeof(float));
	float *z=(float*)malloc(size*sizeof(float));

	for(i=0;i<size;i++)
	{
		fr[i]=(float)im[i*3]/255;
		fg[i]=(float)im[i*3+1]/255;
		fb[i]=(float)im[i*3+2]/255;
		v[i]=trf_max(fr[i],fg[i],fb[i]);
		h[i]=0;
		s[i]=v[i]-trf_min(fr[i],fg[i],fb[i]);
	}

	for(i=0;i<size;i++)
	{
		 z[i]=!s[i];
		 s[i]=s[i]+z[i];
	}
	for(i=0;i<size;i++)
	{
		if(fr[i]==v[i])
			h[i]=(fg[i]-fb[i])/s[i];
	}
	for(i=0;i<size;i++)
	{
		if(fg[i]==v[i])
			h[i]=2+(fb[i]-fr[i])/s[i];
	}
	for(i=0;i<size;i++)
	{
		if(fb[i]==v[i])
			h[i]=4+(fr[i]-fg[i])/s[i];
	}
	for(i=0;i<size;i++)
	{
		h[i]=h[i]/6;
		if(h[i]<0)
			h[i]=h[i]+1;
		h[i]=(!z[i])*h[i];
		if(v[i]!=0)
			s[i]=(!z[i])*s[i]/v[i];
		else 
			s[i]=0;
	}
	free(s);
    //free(v);
	free(z);
	free(fr);
	free(fg);
	free(fb);
}
