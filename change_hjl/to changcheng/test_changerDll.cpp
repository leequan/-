#include "changerIO.hpp"

int main( int argc, char** argv)  
{    

    PRESET_POS_INFO presetPosInfo = {0,0,1,1};//Ԥ��λ��λ�ǡ�Ԥ��λ�����ǡ���̨�߶ȡ�Ԥ��λ�г���
    int thr = 20000;//ɫ����ֵ��12000-24000������ʾ����Ҷ��ɫ���жȣ�ԽС����ɫԽ���У�Խ���׼�⵽����
    float thr2=0.3; //�Ա仯�����жȣ���-1����ԽС�Ա仯Խ�����У�Խ�����׼�⵽�仯
    CHANGE_RGN* rtRgn = NULL;

    char *img1 = "/home/jalywangtuxiang/QT/picture/change/4/0.jpg";
    char *img2 = "/home/jalywangtuxiang/QT/picture/change/4/1.jpg";
    char *img3 = "/home/jalywangtuxiang/QT/picture/change/4/01.jpg";

    rtRgn = SceneChangeDetector(img1,img2,img3,thr,thr2,presetPosInfo);
        return 0;  
}  
