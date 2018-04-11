#ifndef _CHANGERDLL_HPP
#define _CHANGERDLL_HPP


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
using namespace cv;
using namespace std;

const double PI = 3.14159265;
#define H_RESOLUTION_PIXEL (1920)
#define V_RESOLUTION_PIXEL (1080)
#define H_RESOLUTION_mm (7.34)
#define V_RESOLUTION_mm (4.13)
#define f_MIN_mm (6.5)
#define arc2deg (57.29577957)
#define deg2arc (0.017453292)
#define MAXGROWS (160000)
typedef struct _RGN{
    int sz;
    int cx;
    int cy;
    int left;
    int top;
    int right;
    int bottom;

}RGN;

#endif
