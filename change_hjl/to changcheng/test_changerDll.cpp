#include "changerIO.hpp"

int main( int argc, char** argv)  
{    

    PRESET_POS_INFO presetPosInfo = {0,0,1,1};//预置位方位角、预置位俯仰角、云台高度、预置位市场角
    int thr = 20000;//色度阈值（12000-24000），表示对树叶颜色敏感度，越小对绿色越敏感，越容易检测到树动
    float thr2=0.3; //对变化的敏感度（０-1），越小对变化越不敏感，越不容易检测到变化
    CHANGE_RGN* rtRgn = NULL;

    char *img1 = "/home/jalywangtuxiang/QT/picture/change/4/0.jpg";
    char *img2 = "/home/jalywangtuxiang/QT/picture/change/4/1.jpg";
    char *img3 = "/home/jalywangtuxiang/QT/picture/change/4/01.jpg";

    rtRgn = SceneChangeDetector(img1,img2,img3,thr,thr2,presetPosInfo);
        return 0;  
}  
