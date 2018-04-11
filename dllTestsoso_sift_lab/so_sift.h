//#ifndef _SO_SIFT_H
//#define _SO_SIFT_H

#include "stdio.h"
#include "opencv/cxcore.h"
#include <stdlib.h>


//在k-d树上进行BBF搜索的最大次数
/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

//目标点与最近邻和次近邻的距离的比值的阈值，若大于此阈值，则剔除此匹配点对
//通常此值取0.6，值越小找到的匹配点对越精确，但匹配数目越少
/* threshold on squared ratio of distances between NN and 2nd NN */
//#define NN_SQ_DIST_RATIO_THR 0.49
#define NN_SQ_DIST_RATIO_THR 0.5

//窗口名字符串
#define IMG1 "图1"
#define IMG2 "图2"
#define IMG1_FEAT "图1特征点"
#define IMG2_FEAT "图2特征点"
#define IMG_MATCH1 "距离比值筛选后的匹配结果"
#define IMG_MATCH2 "RANSAC筛选后的匹配结果"
#define IMG_MOSAIC_TEMP "临时拼接图像"
#define IMG_MOSAIC_SIMPLE "简易拼接图"
#define IMG_MOSAIC_BEFORE_FUSION "重叠区域融合前"
#define IMG_MOSAIC_PROC12 "拼接图1-2"

//extern double processWidth;
//extern int processHeight;
  
  //void CalcFourCorner(CvMat* *H,CvPoint* leftTop,CvPoint* leftBottom, CvPoint* rightTop,CvPoint* rightBottom,IplImage* img2);
  //int detectionFeature(IplImage* img,struct feature** feat);
  //IplImage* spliceImage(IplImage* img1,IplImage* img2);
  //IplImage* correct(IplImage *img1,IplImage *img2,IplImage *result_img1,IplImage *result_img2);




/**@file
Functions and structures for dealing with image features

Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>

@version 1.1.2-20100521
*/

/*
  此文件中定义了存储特征点的结构体feature，以及几个函数原型的声明：
1、特征点的导入导出
2、特征点绘制
*/






/*特征点的类型：
FEATURE_OXFD表示是牛津大学VGG提供的源码中的特征点格式，
FEATURE_LOWE表示是David.Lowe提供的源码中的特征点格式
*/
/** FEATURE_OXFD <BR> FEATURE_LOWE */
enum feature_type
{
	FEATURE_OXFD,
	FEATURE_LOWE,
};

/*特征点匹配类型：
FEATURE_FWD_MATCH：表明feature结构中的fwd_match域是对应的匹配点
FEATURE_BCK_MATCH：表明feature结构中的bck_match域是对应的匹配点
FEATURE_MDL_MATCH：表明feature结构中的mdl_match域是对应的匹配点
*/
/** FEATURE_FWD_MATCH <BR> FEATURE_BCK_MATCH <BR> FEATURE_MDL_MATCH */
enum feature_match_type
{
	FEATURE_FWD_MATCH,
	FEATURE_BCK_MATCH,
	FEATURE_MDL_MATCH,
};

/*画出的特征点的颜色*/
/* colors in which to display different feature types */
#define FEATURE_OXFD_COLOR CV_RGB(255,255,0)
#define FEATURE_LOWE_COLOR CV_RGB(255,0,255)

/*最大特征描述子长度，定为128*/
/** max feature descriptor length */
#define FEATURE_MAX_D 128

/*特征点结构体
此结构体可存储2中类型的特征点：
FEATURE_OXFD表示是牛津大学VGG提供的源码中的特征点格式，
FEATURE_LOWE表示是David.Lowe提供的源码中的特征点格式。
如果是OXFD类型的特征点，结构体中的a,b,c成员描述了特征点周围的仿射区域(椭圆的参数)，即邻域。
如果是LOWE类型的特征点，结构体中的scl和ori成员描述了特征点的大小和方向。
fwd_match，bck_match，mdl_match一般同时只有一个起作用，用来指明此特征点对应的匹配点
*/
/**
Structure to represent an affine invariant image feature.  The fields
x, y, a, b, c represent the affine region around the feature:
a(x-u)(x-u) + 2b(x-u)(y-v) + c(y-v)(y-v) = 1
*/
struct feature
{
    double x;                      /**< x coord */ //特征点的x坐标
    double y;                      /**< y coord */ //特征点的y坐标
    double a;                      /**< Oxford-type affine region parameter */ //OXFD特征点中椭圆的参数
    double b;                      /**< Oxford-type affine region parameter */ //OXFD特征点中椭圆的参数
    double c;                      /**< Oxford-type affine region parameter */ //OXFD特征点中椭圆的参数
    double scl;                    /**< scale of a Lowe-style feature *///LOWE特征点的尺度
    double ori;                    /**< orientation of a Lowe-style feature */ //LOWE特征点的方向
    int d;                         /**< descriptor length */ //特征描述子的长度，即维数，一般是128
    double descr[FEATURE_MAX_D];   /**< descriptor */ //128维的特征描述子，即一个double数组
    int type;                      /**< feature type, OXFD or LOWE */ //特征点类型
	int category;                  /**< all-purpose feature category */
    struct feature* fwd_match;     /**< matching feature from forward image */   //指明此特征点对应的匹配点
    struct feature* bck_match;     /**< matching feature from backmward image */ //指明此特征点对应的匹配点
    struct feature* mdl_match;     /**< matching feature from model */           //指明此特征点对应的匹配点
    CvPoint2D64f img_pt;           /**< location in image */ //特征点的坐标,等于(x,y)
    CvPoint2D64f mdl_pt;           /**< location in model */ //当匹配类型是mdl_match时用到
    void* feature_data;            /**< user-definable data */ //用户定义的数据:
                                                               //在SIFT极值点检测中，是detection_data结构的指针
                                                               //在k-d树搜索中，是bbf_data结构的指针
                                                               //在RANSAC算法中，是ransac_data结构的指针
};


/*从文件中读入图像特征
文件中的特征点格式必须是FEATURE_OXFD或FEATURE_LOWE格式
参数：
filename：文件名
type：特征点类型
feat：用来存储特征点的feature数组的指针
返回值：导入的特征点个数
*/
/**
Reads image features from file.  The file should be formatted as from
the code provided by the Visual Geometry Group at Oxford or from the
code provided by David Lowe.
@param filename location of a file containing image features
@param type determines how features are input.  If \a type is FEATURE_OXFD,
	the input file is treated as if it is from the code provided by the VGG
	at Oxford: http://www.robots.ox.ac.uk:5000/~vgg/research/affine/index.html
	<BR><BR>
	If \a type is FEATURE_LOWE, the input file is treated as if it is from
	David Lowe's SIFT code: http://www.cs.ubc.ca/~lowe/keypoints  
@param feat pointer to an array in which to store imported features; memory for
    this array is allocated by this function and must be freed by the caller using free(*feat)
@return Returns the number of features imported from filename or -1 on error
*/
extern int import_features( char* filename, int type, struct feature** feat );


/*导出feature数组到文件
参数：
filename：文件名
feat：特征数组
n：特征点个数
返回值：0：成功；1：失败
*/
/**
Exports a feature set to a file formatted depending on the type of
features, as specified in the feature struct's type field.
@param filename name of file to which to export features
@param feat feature array
@param n number of features 
@return Returns 0 on success or 1 on error
*/
extern int export_features( char* filename, struct feature* feat, int n );


/*在图片上画出特征点
参数：
img：图像
feat：特征点数组
n：特征点个数
*/
/**
Displays a set of features on an image
@param img image on which to display features
@param feat array of Oxford-type features
@param n number of features
*/
extern void draw_features( IplImage* img, struct feature* feat, int n );


/*计算两个特征描述子间的欧氏距离的平方
参数：
f1:第一个特征点
f2:第二个特征点
返回值：欧氏距离的平方
*/
/**
Calculates the squared Euclidian distance between two feature descriptors.
@param f1 first feature
@param f2 second feature
@return Returns the squared Euclidian distance between the descriptors of
\a f1 and \a f2.
*/
extern double descr_dist_sq( struct feature* f1, struct feature* f2 );





struct feature;

/*K-D树中的结点结构*/
/** a node in a k-d tree */
struct kd_node
{
    int ki;                      /**< partition key index */ //分割位置(枢轴)的维数索引(哪一维是分割位置)，取值为1-128
    double kv;                   /**< partition key value */  //枢轴的值(所有特征向量在枢轴索引维数上的分量的中值)
    int leaf;                    /**< 1 if node is a leaf, 0 otherwise */ //是否叶子结点的标志
    struct feature* features;    /**< features at this node */  //此结点对应的特征点集合(数组)
    int n;                       /**< number of features */ //特征点的个数
    struct kd_node* kd_left;     /**< left child */  //左子树
    struct kd_node* kd_right;    /**< right child */  //右子树
};


/*************************** Function Prototypes *****************************/
/*根据给定的特征点集合建立k-d树
参数：
features：特征点数组，注意：此函数将会改变features数组中元素的排列顺序
n：特征点个数
返回值：建立好的k-d树的树根指针
*/
/**
A function to build a k-d tree database from keypoints in an array.

@param features an array of features; <EM>this function rearranges the order
	of the features in this array, so you should take appropriate measures
	if you are relying on the order of the features (e.g. call this function
	before order is important)</EM>
@param n the number of features in \a features
@return Returns the root of a kd tree built from \a features.
*/
extern struct kd_node* kdtree_build( struct feature* features, int n );


/*用BBF算法在k-d树中查找指定特征点的k个最近邻特征点
参数：
kd_root：图像特征的k-d树的树根
feat：目标特征点
k：近邻个数
nbrs：k个近邻特征点的指针数组，按到目标特征点的距离升序排列
      此数组的内存将在本函数中被分配，使用完后必须在调用出释放：free(*nbrs)
max_nn_chks：搜索的最大次数，超过此值不再搜索
返回值：存储在nbrs中的近邻个数，返回-1表示失败
*/
/**
Finds an image feature's approximate k nearest neighbors in a kd tree using
Best Bin First search.

@param kd_root root of an image feature kd tree
@param feat image feature for whose neighbors to search
@param k number of neighbors to find
@param nbrs pointer to an array in which to store pointers to neighbors
	in order of increasing descriptor distance; memory for this array is
	allocated by this function and must be freed by the caller using
	free(*nbrs)
@param max_nn_chks search is cut off after examining this many tree entries

@return Returns the number of neighbors found and stored in \a nbrs, or
	-1 on error.
*/
extern int kdtree_bbf_knn( struct kd_node* kd_root, struct feature* feat,
						  int k, struct feature*** nbrs, int max_nn_chks );


/**
Finds an image feature's approximate k nearest neighbors within a specified
spatial region in a kd tree using Best Bin First search.

@param kd_root root of an image feature kd tree
@param feat image feature for whose neighbors to search
@param k number of neighbors to find
@param nbrs pointer to an array in which to store pointers to neighbors
	in order of increasing descriptor distance; memory for this array is
	allocated by this function and must be freed by the caller using
	free(*nbrs)
@param max_nn_chks search is cut off after examining this many tree entries
@param rect rectangular region in which to search for neighbors
@param model if true, spatial search is based on kdtree features' model
	locations; otherwise it is based on their image locations

@return Returns the number of neighbors found and stored in \a nbrs
	(in case \a k neighbors could not be found before examining
	\a max_nn_checks keypoint entries).
*/
extern int kdtree_bbf_spatial_knn( struct kd_node* kd_root,
								struct feature* feat, int k,
								struct feature*** nbrs, int max_nn_chks,
								CvRect rect, int model );


/*释放k-d树占用的存储空间
*/
/**
De-allocates memory held by a kd tree

@param kd_root pointer to the root of a kd tree
*/
extern void kdtree_release( struct kd_node* kd_root );









/******************************* Defs and macros *****************************/

/* initial # of priority queue elements for which to allocate space */
#define MINPQ_INIT_NALLOCD 512  //初始分配空间个数

/********************************** Structures *******************************/
/*结点结构*/
/** an element in a minimizing priority queue */
struct pq_node
{
	void* data;
	int key;
};

/*最小优先队列结构*/
/** a minimizing priority queue */
struct min_pq
{
    struct pq_node* pq_array;    /* array containing priority queue */ //结点指针
    int nallocd;                 /* number of elements allocated */ //分配的空间个数
    int n;                       /**< number of elements in pq */ //元素个数
};


/*************************** Function Prototypes *****************************/
/*初始化最小优先级队列
*/
/**
Creates a new minimizing priority queue.
*/
extern struct min_pq* minpq_init();

/*插入元素到优先队列
参数：
min_pq：优先队列
data:要插入的数据
key:与data关联的键值
返回值：0：成功，1：失败
*/
/**
Inserts an element into a minimizing priority queue.

@param min_pq a minimizing priority queue
@param data the data to be inserted
@param key the key to be associated with \a data

@return Returns 0 on success or 1 on failure.
*/
extern int minpq_insert( struct min_pq* min_pq, void* data, int key );


/*返回优先队列中键值最小的元素，但并不删除它
参数：min_pq：优先队列
返回值：最小元素的指针
*/
/**
Returns the element of a minimizing priority queue with the smallest key
without removing it from the queue.
@param min_pq a minimizing priority queue
@return Returns the element of \a min_pq with the smallest key or NULL if \a min_pq is empty
*/
extern void* minpq_get_min( struct min_pq* min_pq );


/*返回并移除具有最小键值的元素
参数：min_pq：优先级队列
返回值：最小元素的指针
*/
/**
Removes and returns the element of a minimizing priority queue with the smallest key.
@param min_pq a minimizing priority queue
@return Returns the element of \a min_pq with the smallest key of NULL if \a min_pq is empty
*/
extern void* minpq_extract_min( struct min_pq* min_pq );

/*释放优先队列
*/
/**
De-allocates the memory held by a minimizing priorioty queue
@param min_pq pointer to a minimizing priority queue
*/
extern void minpq_release( struct min_pq** min_pq );







/******************************** Structures *********************************/

//极值点检测中用到的结构
//在SIFT特征提取过程中，此类型数据会被赋值给feature结构的feature_data成员
/** holds feature data relevant to detection */
struct detection_data
{
    int r;      //特征点所在的行
    int c;      //特征点所在的列
    int octv;   //高斯差分金字塔中，特征点所在的组
    int intvl;  //高斯差分金字塔中，特征点所在的组中的层
    double subintvl;  //特征点在层方向(σ方向,intvl方向)上的亚像素偏移量
    double scl_octv;  //特征点所在的组的尺度
};

struct feature;


/******************************* 一些默认参数 *****************************/
/******************************* Defs and macros *****************************/

//高斯金字塔每组内的层数
/** default number of sampled intervals per octave */
#define SIFT_INTVLS 3

//第0层的初始尺度，即第0层高斯模糊所使用的参数
/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA 1.6

//对比度阈值，针对归一化后的图像，用来去除不稳定特征
/** default threshold on keypoint contrast |D(x)| */
#define SIFT_CONTR_THR 0.04

//主曲率比值的阈值，用来去除边缘特征
/** default threshold on keypoint ratio of principle curvatures */
#define SIFT_CURV_THR 10

//是否将图像放大为之前的两倍
/** double image size before pyramid construction? */
#define SIFT_IMG_DBL 1

//输入图像的尺度为0.5
/* assumed gaussian blur for input image */
#define SIFT_INIT_SIGMA 0.5

//边界的像素宽度，检测过程中将忽略边界线中的极值点，即只检测边界线以内是否存在极值点
/* width of border in which to ignore keypoints */
#define SIFT_IMG_BORDER 5

//通过插值进行极值点精确定位时，最大差值次数，即关键点修正次数
/* maximum steps of keypoint interpolation before failure */
#define SIFT_MAX_INTERP_STEPS 5

//特征点方向赋值过程中，梯度方向直方图中柱子(bin)的个数
/* default number of bins in histogram for orientation assignment */
#define SIFT_ORI_HIST_BINS 36

//特征点方向赋值过程中，搜索邻域的半径为：3 * 1.5 * σ
/* determines gaussian sigma for orientation assignment */
#define SIFT_ORI_SIG_FCTR 1.5

//特征点方向赋值过程中，搜索邻域的半径为：3 * 1.5 * σ
/* determines the radius of the region used in orientation assignment */
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

//特征点方向赋值过程中，梯度方向直方图的平滑次数，计算出梯度直方图后还要进行高斯平滑
/* number of passes of orientation histogram smoothing */
#define SIFT_ORI_SMOOTH_PASSES 2

//特征点方向赋值过程中，梯度幅值达到最大值的80%则分裂为两个特征点
/* orientation magnitude relative to max that results in new feature */
#define SIFT_ORI_PEAK_RATIO 0.8

//计算特征描述子过程中，计算方向直方图时，将特征点附近划分为d*d个区域，每个区域生成一个直方图，SIFT_DESCR_WIDTH即d的默认值
/** default width of descriptor histogram array */
#define SIFT_DESCR_WIDTH 4

//计算特征描述子过程中，每个方向直方图的bin个数
/** default number of bins per histogram in descriptor array */
#define SIFT_DESCR_HIST_BINS 8

//计算特征描述子过程中，特征点周围的d*d个区域中，每个区域的宽度为m*σ个像素，SIFT_DESCR_SCL_FCTR即m的默认值，σ为特征点的尺度
/* determines the size of a single descriptor orientation histogram */
#define SIFT_DESCR_SCL_FCTR 3.0

//计算特征描述子过程中，特征描述子向量中元素的阈值(最大值，并且是针对归一化后的特征描述子)，超过此阈值的元素被强行赋值为此阈值
/* threshold on magnitude of elements of descriptor vector */
#define SIFT_DESCR_MAG_THR 0.2

//计算特征描述子过程中，将浮点型的特征描述子变为整型时乘以的系数
/* factor used to convert floating-point descriptor to unsigned char */
#define SIFT_INT_DESCR_FCTR 512.0

//定义了一个带参数的函数宏，用来提取参数f中的feature_data成员并转换为detection_data格式的指针
/* returns a feature's detection data */
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )


/*************************** Function Prototypes *****************************/

/*使用默认参数在图像中提取SIFT特征点
参数：
img：图像指针
feat：用来存储特征点的feature数组的指针
      此数组的内存将在本函数中被分配，使用完后必须在调用出释放：free(*feat)
返回值：提取的特征点个数，若返回-1表明提取失败
*/
/**
Finds SIFT features in an image using default parameter values.  All
detected features are stored in the array pointed to by \a feat.
@param img the image in which to detect features
@param feat a pointer to an array in which to store detected features; memory
	for this array is allocated by this function and must be freed by the caller
	using free(*feat)
@return Returns the number of features stored in \a feat or -1 on failure
@see _sift_features()
*/
extern int sift_features( IplImage* img, struct feature** feat );


/*使用用户指定的参数在图像中提取SIFT特征点
参数：
img：输入图像
feat：存储特征点的数组的指针
      此数组的内存将在本函数中被分配，使用完后必须在调用出释放：free(*feat)
intvls：每组的层数
sigma：初始高斯平滑参数σ
contr_thr：对比度阈值，针对归一化后的图像，用来去除不稳定特征
curv_thr：去除边缘的特征的主曲率阈值
img_dbl：是否将图像放大为之前的两倍
descr_width：特征描述过程中，计算方向直方图时，将特征点附近划分为descr_width*descr_width个区域，每个区域生成一个直方图
descr_hist_bins：特征描述过程中，每个直方图中bin的个数
返回值：提取的特征点个数，若返回-1表明提取失败
*/
/**
Find a SIFT features in an image using user-specified parameter values.  All
detected features are stored in the array pointed to by \a feat.

@param img the image in which to detect features
@param feat a pointer to an array in which to store detected features; memory
	for this array is allocated by this function and must be freed by the caller
	using free(*feat)
@param intvls the number of intervals sampled per octave of scale space
@param sigma the amount of Gaussian smoothing applied to each image level
	before building the scale space representation for an octave
@param contr_thr a threshold on the value of the scale space function
	\f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
	feature location and scale, used to reject unstable features;  assumes
pixel values in the range [0, 1]
@param curv_thr threshold on a feature's ratio of principle curvatures
	used to reject features that are too edge-like
@param img_dbl should be 1 if image doubling prior to scale space
	construction is desired or 0 if not
@param descr_width the width, \f$n\f$, of the \f$n \times n\f$ array of
	orientation histograms used to compute a feature's descriptor
@param descr_hist_bins the number of orientations in each of the
	histograms in the array used to compute a feature's descriptor

@return Returns the number of keypoints stored in \a feat or -1 on failure
@see sift_features()
*/
extern int _sift_features( IplImage* img, struct feature** feat, int intvls,
						  double sigma, double contr_thr, int curv_thr,
						  int img_dbl, int descr_width, int descr_hist_bins );



//求x的绝对值
/* absolute value */

//内联函数，读取和设置各种类型图像的像素值
/***************************** Inline Functions ******************************/


/**
A function to get a pixel value from an 8-bit unsigned image.

@param img an image
@param r row
@param c column
@return Returns the value of the pixel at (\a r, \a c) in \a img
*/
static __inline int pixval8( IplImage* img, int r, int c )
{
	return (int)( ( (uchar*)(img->imageData + img->widthStep*r) )[c] );
}


/**
A function to set a pixel value in an 8-bit unsigned image.

@param img an image
@param r row
@param c column
@param val pixel value
*/
static __inline void setpix8( IplImage* img, int r, int c, uchar val)
{
	( (uchar*)(img->imageData + img->widthStep*r) )[c] = val;
}


/*从32位浮点型单通道图像中获取指定坐标的像素值，内联函数
参数：
img：输入图像
r：行坐标
c：列坐标
返回值：坐标(c,r)处(r行c列)的像素值
*/
/**
A function to get a pixel value from a 32-bit floating-point image.
@param img an image
@param r row
@param c column
@return Returns the value of the pixel at (\a r, \a c) in \a img
*/
static __inline float pixval32f( IplImage* img, int r, int c )
{
	return ( (float*)(img->imageData + img->widthStep*r) )[c];
}


/**
A function to set a pixel value in a 32-bit floating-point image.

@param img an image
@param r row
@param c column
@param val pixel value
*/
static __inline void setpix32f( IplImage* img, int r, int c, float val )
{
	( (float*)(img->imageData + img->widthStep*r) )[c] = val;
}


/**
A function to get a pixel value from a 64-bit floating-point image.

@param img an image
@param r row
@param c column
@return Returns the value of the pixel at (\a r, \a c) in \a img
*/
static __inline double pixval64f( IplImage* img, int r, int c )
{
	return (double)( ( (double*)(img->imageData + img->widthStep*r) )[c] );
}


/**
A function to set a pixel value in a 64-bit floating-point image.

@param img an image
@param r row
@param c column
@param val pixel value
*/
static __inline void setpix64f( IplImage* img, int r, int c, double val )
{
	( (double*)(img->imageData + img->widthStep*r) )[c] = val;
}


/**************************** Function Prototypes ****************************/

//错误处理
/**
Prints an error message and aborts the program.  The error message is
of the form "Error: ...", where the ... is specified by the \a format
argument

@param format an error message format string (as with \c printf(3)).
*/
extern void fatal_error( char* format, ... );


//获取一个文件全名，将文件名和扩展名连接到一起
/**
Replaces a file's extension, which is assumed to be everything after the
last dot ('.') character.
@param file the name of a file
@param extn a new extension for \a file; should not include a dot (i.e.
    \c "jpg", not \c ".jpg") unless the new file extension should contain two dots.
@return Returns a new string formed as described above.  If \a file does
	not have an extension, this function simply adds one.
*/
extern char* replace_extension( const char* file, const char* extn );

//文件名中去掉路径c:\\traffic.jpg => traffic.jpg
/**
A function that removes the path from a filename.  Similar to the Unix
basename command.
@param pathname a (full) path name
@return Returns the basename of \a pathname.
*/
//extern char* basename( const char* pathname );

//显示程序进度
/**
Displays progress in the console with a spinning pinwheel.  Every time this
function is called, the state of the pinwheel is incremented.  The pinwheel
has four states that loop indefinitely: '|', '/', '-', '\'.

@param done if 0, this function simply increments the state of the pinwheel;
	otherwise it prints "done"
*/
extern void progress( int done );


/**
Erases a specified number of characters from a stream.
@param stream the stream from which to erase characters
@param n the number of characters to erase
*/
extern void erase_from_stream( FILE* stream, int n );

//数组长度加倍
/**
Doubles the size of an array with error checking
@param array pointer to an array whose size is to be doubled
@param n number of elements allocated for \a array
@param size size in bytes of elements in \a array
@return Returns the new number of elements allocated for \a array.  If no
	memory is available, returns 0 and frees array.
*/
extern int array_double( void** array, int n, int size );

//计算两点的对角线距离
/**
Calculates the squared distance between two points.
@param p1 a point
@param p2 another point
*/
extern double dist_sq_2D( CvPoint2D64f p1, CvPoint2D64f p2 );

//在点pt画个叉，本质就是在那个点画四条线
/**
Draws an x on an image.
@param img an image
@param pt the center point of the x
@param r the x's radius
@param w the x's line weight
@param color the color of the x
*/
extern void draw_x( IplImage* img, CvPoint pt, int r, int w, CvScalar color );


/*将两张图像合成为一张，垂直排放,高是二者之和，宽是二者的较大者
参数：img1：位于上方的图像的指针，img2：位于下方的图像的指针
返回值：合成图像
*/
/**
Combines two images by scacking one on top of the other
@param img1 top image
@param img2 bottom image
@return Returns the image resulting from stacking \a img1 on top if \a img2
*/
extern IplImage* stack_imgs( IplImage* img1, IplImage* img2 );

/*将两张图像合成为一张，水平排放
参数：img1：位于左边的图像的指针，img2：位于右边的图像的指针
返回值：合成图像
*/
extern IplImage* stack_imgs_horizontal( IplImage* img1, IplImage* img2 );


/**
Allows user to view an array of images as a video.  Keyboard controls
are as follows:

<ul>
<li>Space - start and pause playback</li>
<li>Page Up - skip forward 10 frames</li>
<li>Page Down - jump back 10 frames</li>
<li>Right Arrow - skip forward 1 frame</li>
<li>Left Arrow - jump back 1 frame</li>
<li>Backspace - jump back to beginning</li>
<li>Esc - exit playback</li>
<li>Closing the window also exits playback</li>
</ul>

@param imgs an array of images
@param n number of images in \a imgs
@param win_name name of window in which images are displayed
*/
extern void vid_view( IplImage** imgs, int n, char* win_name );

//查看某个窗口是否已经关闭
/**
Checks if a HighGUI window is still open or not
@param name the name of the window we're checking
@return Returns 1 if the window named \a name has been closed or 0 otherwise
*/
extern int win_closed( char* name );





/********************************** Structures *******************************/

struct feature;

//RANSAC算法中用到的结构
//在RANSAC算法过程中，此类型数据会被赋值给feature结构的feature_data成员
/** holds feature data relevant to ransac */
struct ransac_data
{
    void* orig_feat_data; //保存此特征点的feature_data域的以前的值
    int sampled; //标识位，值为1标识此特征点是否被选为样本
};

//一些宏定义
/******************************* Defs and macros *****************************/

/*RANSAC算法的容错度
对于匹配点对<pt,mpt>，以及变换矩阵H，
如果pt经H变换后的点和mpt之间的距离的平方小于RANSAC_ERR_TOL，则可将其加入当前一致集
*/
/* RANSAC error tolerance in pixels */
#define RANSAC_ERR_TOL 3

//内点数目占样本总数目的百分比的最小值
/** pessimistic estimate of fraction of inliers for RANSAC */
#define RANSAC_INLIER_FRAC_EST 0.25

//一个匹配点对支持错误模型的概率（不知道是干什么用的）
/** estimate of the probability that a correspondence supports a bad model */
#define RANSAC_PROB_BAD_SUPP 0.10

//定义了一个带参数的函数宏，用来提取参数feat中的feature_data成员并转换为ransac_data格式的指针
/* extracts a feature's RANSAC data */
#define feat_ransac_data( feat ) ( (struct ransac_data*) (feat)->feature_data )


/*定义了一个函数指针类型ransac_xform_fn，其返回值是CvMat*类型，有三个参数
之后可以用ransac_xform_fn来定义函数指针
此类型的函数指针被用在ransac_form()函数的参数中
此类型的函数会根据匹配点对集合计算出一个变换矩阵作为返回值
参数：
pts：点的集合
mpts：点的集合，pts[i]与mpts[i]是互相对应的匹配点
n：pts和mpts数组中点的个数，pts和mpts中点的个数必须相同
返回值：一个变换矩阵，将pts中的每一个点转换为mpts中的对应点，返回值为空表示失败
*/
/**
Prototype for transformation functions passed to ransac_xform().  Functions
of this type should compute a transformation matrix given a set of point
correspondences.
@param pts array of points
@param mpts array of corresponding points; each \a pts[\a i], \a i=0..\a n-1,
	corresponds to \a mpts[\a i]
@param n number of points in both \a pts and \a mpts
@return Should return a transformation matrix that transforms each point in
	\a pts to the corresponding point in \a mpts or NULL on failure.
*/
typedef CvMat* (*ransac_xform_fn)( CvPoint2D64f* pts, CvPoint2D64f* mpts,int n );


/*定义了一个函数指针类型ransac_err_fn，其返回值是double类型，有三个参数
之后可以用ransac_err_fn来定义函数指针
此类型的函数指针被用在ransac_form()函数的参数中
此类型的函数会根据匹配点对(pt,mpt)和变换矩阵M计算出一个double类型的错误度量值作为返回值
此错误度量值用来评判"点mpt"和"点pt经M矩阵变换后的点"之间是否相一致
参数：
pt：一个点
mpt：点pt的对应匹配点
M：变换矩阵
返回值："点mpt"和"点pt经M矩阵变换后的点"之间的错误度量值
*/
/**
Prototype for error functions passed to ransac_xform().  For a given
point, its correspondence, and a transform, functions of this type should
compute a measure of error between the correspondence and the point after
the point has been transformed by the transform.
@param pt a point
@param mpt \a pt's correspondence
@param T a transform
@return Should return a measure of error between \a mpt and \a pt after
	\a pt has been transformed by the transform \a T.
*/
typedef double (*ransac_err_fn)( CvPoint2D64f pt, CvPoint2D64f mpt, CvMat* M );


/***************************** Function Prototypes ***************************/

/*利用RANSAC算法进行特征点筛选，计算出最佳匹配的变换矩阵
参数：
features：特征点数组，只有当mtype类型的匹配点存在时才被用来进行单应性计算
n：特征点个数
mtype：决定使用每个特征点的哪个匹配域进行变换矩阵的计算，应该是FEATURE_MDL_MATCH，
    FEATURE_BCK_MATCH，FEATURE_MDL_MATCH中的一个。若是FEATURE_MDL_MATCH，
    对应的匹配点对坐标是每个特征点的img_pt域和其匹配点的mdl_pt域，
    否则，对应的匹配点对是每个特征点的img_pt域和其匹配点的img_pt域。
xform_fn：函数指针，指向根据输入的点对进行变换矩阵计算的函数，一般传入lsq_homog()函数
m：在函数xform_fn中计算变换矩阵需要的最小特征点对个数
p_badxform：允许的错误概率，即允许RANSAC算法计算出的变换矩阵错误的概率，当前计算出的模型的错误概率小于p_badxform时迭代停止
err_fn：函数指针，对于给定的变换矩阵，计算推定的匹配点对之间的变换误差，一般传入homog_xfer_err()函数
err_tol：容错度，对于给定的变换矩阵，在此范围内的点对被认为是内点
inliers：输出参数，指针数组，指向计算出的最终的内点集合，若为空，表示没计算出符合要求的一致集。
        此数组的内存将在本函数中被分配，使用完后必须在调用出释放：free(*inliers)
n_in：输出参数，最终计算出的内点的数目
返回值：RANSAC算法计算出的变换矩阵，若为空，表示出错或无法计算出可接受矩阵
*/
/**
Calculates a best-fit image transform from image feature correspondences using RANSAC.

For more information refer to:
Fischler, M. A. and Bolles, R. C.  Random sample consensus: a paradigm for
model fitting with applications to image analysis and automated cartography.
<EM>Communications of the ACM, 24</EM>, 6 (1981), pp. 381--395.

@param features an array of features; only features with a non-NULL match
	of type \a mtype are used in homography computation
@param n number of features in \a feat
@param mtype determines which of each feature's match fields to use
	for transform computation; should be one of FEATURE_FWD_MATCH,
	FEATURE_BCK_MATCH, or FEATURE_MDL_MATCH; if this is FEATURE_MDL_MATCH,
	correspondences are assumed to be between a feature's img_pt field
	and its match's mdl_pt field, otherwise correspondences are assumed to
	be between the the feature's img_pt field and its match's img_pt field
@param xform_fn pointer to the function used to compute the desired
	transformation from feature correspondences
@param m minimum number of correspondences necessary to instantiate the
	transform computed by \a xform_fn
@param p_badxform desired probability that the final transformation
	returned by RANSAC is corrupted by outliers (i.e. the probability that
	no samples of all inliers were drawn)
@param err_fn pointer to the function used to compute a measure of error
	between putative correspondences for a given transform
@param err_tol correspondences within this distance of each other are
	considered as inliers for a given transform
@param inliers if not NULL, output as an array of pointers to the final
	set of inliers; memory for this array is allocated by this function and
	must be freed by the caller using free(*inliers)
@param n_in if not NULL, output as the final number of inliers

@return Returns a transformation matrix computed using RANSAC or NULL
	on error or if an acceptable transform could not be computed.
*/
extern CvMat* ransac_xform( struct feature* features, int n, int mtype,
						   ransac_xform_fn xform_fn, int m,
						   double p_badxform, ransac_err_fn err_fn,
						   double err_tol, struct feature*** inliers,
						   int* n_in ,int *Hflag);


/*运用线性变换，进行点匹配计算平面单应性
参数：
pts：点的集合
mpts：点的集合，pts[i]与mpts[i]是互相对应的匹配点
n：pts和mpts数组中点的个数，pts和mpts中点的个数必须相同，并且点对个数至少为4
返回值：变换矩阵，可将pts中的点变换为mpts中的点，若点个数少于4则返回空
*/
/**
Calculates a planar homography from point correspondeces using the direct
linear transform.  Intended for use as a ransac_xform_fn.
@param pts array of points
@param mpts array of corresponding points; each \a pts[\a i], \a i=0..\a
	n-1, corresponds to \a mpts[\a i]
@param n number of points in both \a pts and \a mpts; must be at least 4
@return Returns the \f$3 \times 3\f$ planar homography matrix that
	transforms points in \a pts to their corresponding points in \a mpts
	or NULL if fewer than 4 correspondences were provided
*/
extern CvMat* dlt_homog( CvPoint2D64f* pts, CvPoint2D64f* mpts, int n );


/*根据4对坐标点计算最小二乘平面单应性变换矩阵
参数：
pts：点的集合
mpts：点的集合，pts[i]与mpts[i]是互相对应的匹配点
n：pts和mpts数组中点的个数，pts和mpts中点的个数必须相同，并且点对个数至少为4
返回值：变换矩阵，可将pts中的点变换为mpts中的点，若点个数少于4则返回空
*/
/**
Calculates a least-squares planar homography from point correspondeces.
Intended for use as a ransac_xform_fn.
@param pts array of points
@param mpts array of corresponding points; each \a pts[\a i], \a i=0..\a n-1,
	corresponds to \a mpts[\a i]
@param n number of points in both \a pts and \a mpts; must be at least 4
@return Returns the \f$3 \times 3\f$ least-squares planar homography
	matrix that transforms points in \a pts to their corresponding points
	in \a mpts or NULL if fewer than 4 correspondences were provided
*/
extern CvMat* lsq_homog( CvPoint2D64f* pts, CvPoint2D64f* mpts, int n );


/*对于给定的单应性矩阵H，计算输入点pt精H变换后的点与其匹配点mpt之间的误差
例如：给定点x，其对应点x'，单应性矩阵H，则计算x'与Hx之间的距离的平方，d(x', Hx)^2
参数：
pt：一个点
mpt：pt的对应匹配点
H：单应性矩阵
返回值：转换误差
*/
/**
Calculates the transfer error between a point and its correspondence for
a given homography, i.e. for a point \f$x\f$, it's correspondence \f$x'\f$,
and homography \f$H\f$, computes \f$d(x', Hx)^2\f$.  Intended for use as a
ransac_err_fn.
@param pt a point
@param mpt \a pt's correspondence
@param H a homography matrix
@return Returns the transfer error between \a pt and \a mpt given \a H
*/
extern double homog_xfer_err( CvPoint2D64f pt, CvPoint2D64f mpt, CvMat* H );


/*计算点pt经透视变换后的点，即给定一点pt和透视变换矩阵T，计算变换后的点
给定点(x,y)，变换矩阵M，计算[x',y',w']^T = M * [x,y,1]^T(^T表示转置)，
则变换后的点是(u,v) = (x'/w', y'/w')
注意：仿射变换是透视变换的特例
参数：
pt：一个二维点
T：透视变换矩阵
返回值：pt经透视变换后的点
*/
/**
Performs a perspective transformation on a single point.  That is, for a
point \f$(x, y)\f$ and a \f$3 \times 3\f$ matrix \f$T\f$ this function
returns the point \f$(u, v)\f$, where<BR>
\f$[x' \ y' \ w']^T = T \times [x \ y \ 1]^T\f$,<BR>
and<BR>
\f$(u, v) = (x'/w', y'/w')\f$.
Note that affine transforms are a subset of perspective transforms.
@param pt a 2D point
@param T a perspective transformation matrix
@return Returns the point \f$(u, v)\f$ as above.
*/
extern CvPoint2D64f persp_xform_pt( CvPoint2D64f pt, CvMat* T );


//#endif

