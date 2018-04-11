#include "change.hpp"




void fillfft(complex_1 *a,complex_1 *b)
{
	int i,j;
	for(i=0;i<128;i++)
		for(j=0;j<128;j++)
		{
			b[i*128+j].real=0;
			b[i*128+j].imag=0;
		}
	for(i=0;i<120;i++)
		for(j=0;j<120;j++)
		{
			b[i*128+j].real=a[i*120+j].real;
		}
}


//���ٸ���Ҷ�任��aΪ��������ָ�룬sizeΪ�������鳤�ȣ�ioptΪ0ʱ��������Ҷ�任��1ʱ���渵��Ҷ�任��
void fft(complex_1 *a,int size,char iopt)
{
	int nv2,nm1,le,le1,ip;
	float pile1,tmp;
    complex_1 u,w,t,*p;
	int i,j,l,f,k;

	if(iopt)
	{
		for(i=0;i<size;i++)
		{
			a[i].real/=size;
			a[i].imag/=-size;
		}
	}
	nv2=size/2;
	nm1=size-1;
	j=0;
	for(i=0;i<nm1;i++)
	{
		if(i<j)
		{
			t=a[j];
			a[j]=a[i];
			a[i]=t;
		}
		l=nv2;
		while(l<=j)
		{
			j=j-l;
			l=l/2;
		}
		j=j+l;
	}
	f=size;
	for(k=1;(f=f/2)!=1;k++);
	for(l=1;l<=k;l++)
	{
		le=2<<(l-1);
		le1=le/2;
		u.real=1.0;
		u.imag=0.0;
		pile1=PI/le1;
		w.real=cos(pile1);w.imag=-sin(pile1);
		for(j=0;j<le1;j++)
		{
			for(i=j;i<size;i+=le)
			{
				ip=i+le1;
				t.real=a[ip].real*u.real-a[ip].imag*u.imag;
				t.imag=a[ip].real*u.imag+a[ip].imag*u.real;
				a[ip].real=a[i].real-t.real;
				a[ip].imag=a[i].imag-t.imag;
				a[i].real+=t.real;
				a[i].imag+=t.imag;
			}
			tmp=u.real;
			u.real=u.real*w.real-u.imag*w.imag;
			u.imag=tmp*w.imag+u.imag*w.real;
		}
	}
	if(iopt)
	{
		for(i=0;i<size;i++)
		{
			a[i].imag=-a[i].imag;
		}
	}
}

//���ٶ�ά����Ҷ�任��aΪ������ά����ָ�룬widΪ������ά����������higΪ������ά����������ioptΪ0ʱ��������Ҷ�任��1ʱ���渵��Ҷ�任��
void fft2(complex_1 *a,int wid,int hig,char iopt)
{
	int i,j;
    complex_1 *b;
    b=(complex_1*)malloc(wid*hig*sizeof(complex_1));
	//���н��и���Ҷ�任
	for(i=0;i<wid;i++)
	{
		for(j=0;j<hig;j++)
			b[j+i*hig]=a[j*wid+i];
		fft(b+i*hig,hig,iopt);
	}
	//���н��и���Ҷ�任
	for(i=0;i<hig;i++)
	{
		for(j=0;j<wid;j++)
			a[j+i*hig]=b[j*wid+i];
		fft(a+i*wid,wid,iopt);
	}
	free(b);
}

//����Ҷ�任ƽ�ƣ�aΪ������ά����ָ�룬widΪ������ά����������higΪ������ά����������ioptΪ0ʱ��������Ҷ�任��1ʱ���渵��Ҷ�任��
void fftshift(complex_1 *a,int wid,int hig,char iopt)
{
	int i,j;
    complex_1 *b;
    b=(complex_1*)malloc(wid*hig*sizeof(complex_1));
	if(!iopt)
	{
		for(i=0;i<hig;i++)
		{
			for(j=0;j<wid;j++)
			{
				if((i<hig/2)&&(j<wid/2))
					b[j+i*wid]=a[j+wid/2+(i+hig/2)*wid];
				else if((i>=hig/2)&&(j<wid/2))
					b[j+i*wid]=a[j+wid/2+(i-hig/2)*wid];
				else if((i<hig/2)&&(j>=wid/2))
					b[j+i*wid]=a[j-wid/2+(i+hig/2)*wid];
				else
					b[j+i*wid]=a[j-wid/2+(i-hig/2)*wid];
			}
		}
	}
	else
	{
		for(i=0;i<hig;i++)
		{
			for(j=0;j<wid;j++)
			{
				if((i<hig/2)&&(j<wid/2))
					b[j+i*wid]=a[j+wid/2+(i+hig/2)*wid];
				else if((i>=hig/2)&&(j<wid/2))
					b[j+i*wid]=a[j+wid/2+(i-hig/2)*wid];
				else if((i<hig/2)&&(j>=wid/2))
					b[j+i*wid]=a[j-wid/2+(i+hig/2)*wid];
				else
					b[j+i*wid]=a[j-wid/2+(i-hig/2)*wid];
			}
		}
	}
    for(i=0;i<hig*wid;i++)
	{
		a[i]=b[i];
	}
	free(b);
}
