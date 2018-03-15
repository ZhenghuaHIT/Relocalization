#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <cmath>
#include <boost/make_shared.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

using namespace cv;

/**
 * @brief 对输入图像进行细化
 * @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
 * @param maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
 * @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
 */
cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1)
{
    assert(src.type() == CV_8UC1);
    cv::Mat dst;
    int width  = src.cols;
    int height = src.rows;
    src.copyTo(dst);
    int count = 0;  //记录迭代次数
    while (true)
    {
        count++;
        if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达
            break;
        std::vector<uchar *> mFlag; //用于标记需要删除的点
        //对点标记
        for (int i = 0; i < height ;++i)
        {
            uchar * p = dst.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            {
                //如果满足四个条件，进行标记
                //  p9 p2 p3
                //  p8 p1 p4
                //  p7 p6 p5
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
                if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                {
                    int ap = 0;
                    if (p2 == 0 && p3 == 1) ++ap;
                    if (p3 == 0 && p4 == 1) ++ap;
                    if (p4 == 0 && p5 == 1) ++ap;
                    if (p5 == 0 && p6 == 1) ++ap;
                    if (p6 == 0 && p7 == 1) ++ap;
                    if (p7 == 0 && p8 == 1) ++ap;
                    if (p8 == 0 && p9 == 1) ++ap;
                    if (p9 == 0 && p2 == 1) ++ap;

                    if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
                    {
                        //标记
                        mFlag.push_back(p+j);
                    }
                }
            }
        }

        //将标记的点删除
        for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
        {
            **i = 0;
        }

        //直到没有点满足，算法结束
        if (mFlag.empty())
        {
            break;
        }
        else
        {
            mFlag.clear();//将mFlag清空
        }

        //对点标记
        for (int i = 0; i < height; ++i)
        {
            uchar * p = dst.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            {
                //如果满足四个条件，进行标记
                //  p9 p2 p3
                //  p8 p1 p4
                //  p7 p6 p5
                uchar p1 = p[j];
                if (p1 != 1) continue;
                uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
                uchar p8 = (j == 0) ? 0 : *(p + j - 1);
                uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
                uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
                uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
                uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
                uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
                uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

                if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
                {
                    int ap = 0;
                    if (p2 == 0 && p3 == 1) ++ap;
                    if (p3 == 0 && p4 == 1) ++ap;
                    if (p4 == 0 && p5 == 1) ++ap;
                    if (p5 == 0 && p6 == 1) ++ap;
                    if (p6 == 0 && p7 == 1) ++ap;
                    if (p7 == 0 && p8 == 1) ++ap;
                    if (p8 == 0 && p9 == 1) ++ap;
                    if (p9 == 0 && p2 == 1) ++ap;

                    if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
                    {
                        //标记
                        mFlag.push_back(p+j);
                    }
                }
            }
        }

        //将标记的点删除
        for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
        {
            **i = 0;
        }

        //直到没有点满足，算法结束
        if (mFlag.empty())
        {
            break;
        }
        else
        {
            mFlag.clear();//将mFlag清空
        }
    }
    return dst;
}

/**
 * @brief 对输入的点云区域逐层求解复杂度
 * @param cloud_complexity为输入点云区域
 * @param z_min为点云最小高度值
 * @param z_max为点云最大高度值
 * @return 为输入点云区域中点云层复杂度最高的值，以及该点云层的Z值大小
 */
Point3f compute_complexity( const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_complexity ,const float y_min ,const float y_max )
{
    // 点云分层厚度处理
    float y_value = y_max-y_min;
    int floor_num = y_value/0.1; // 以0.1点云厚度分层，floor_num为层数
    float floor_num2f = y_value/floor_num; // 以floor_num2f近似0.1的厚度进行点云分层
    vector<Point2f> floor_values;
    for ( int i = 0;i < floor_num; i++  )
    {
        Point2f floor_value;
        floor_value.x = y_max-(i+1)*floor_num2f;
        floor_value.y = y_max-i*floor_num2f;
        floor_values.push_back(floor_value);
    }

    std::vector<Point3f> complexitys; // 保存复杂度与点云上下边界
    std::vector<float> complexitys_f; // 单独保存复杂度
    for ( int n = 0; n < floor_num; n++) // 逐层处理
    {
    // 单独抽一层处理
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal_pass (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_complexity);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (floor_values[n].x,floor_values[n].y);
    pass.filter (*cloud_voxel_removal_pass);
/**
                *****绘制并保存分层点云*****
    pcl::io::savePCDFile("map_0.15h.pcd",*cloud_voxel_removal_pass,true);
    std::cerr<<"Saved "<<cloud_voxel_removal_pass->points.size()<<" data points to map_0.15h."<<std::endl;
*/

    // 获取点云长宽值
    std::vector<int> size_x; // 点云X坐标集合
    std::vector<int> size_z; // 点云Y坐标集合
    for (size_t i = 0; i < cloud_voxel_removal_pass->points.size (); ++i)
      {
        float xx = cloud_voxel_removal_pass->points[i].x;
        int xxx = xx*100;
        size_x.push_back(xxx);
        float zz = cloud_voxel_removal_pass->points[i].z;
        int zzz = zz*100;
        size_z.push_back(zzz);
      }
    std::vector<int>::iterator max_x = std::max_element(size_x.begin(),size_x.end());
    std::vector<int>::iterator min_x = std::min_element(size_x.begin(),size_x.end());
    std::vector<int>::iterator max_z = std::max_element(size_z.begin(),size_z.end());
    std::vector<int>::iterator min_z = std::min_element(size_z.begin(),size_z.end());
    int x_value=*max_x-*min_x;
    int z_value=*max_z-*min_z;

   // 创建投影图像
    Mat projectionImage(x_value+100,z_value+100,CV_8UC1,Scalar(255));
    vector<Point> point_edges;
    for (size_t i = 0; i < size_x.size (); ++i)
      {
        // 每个投影点为5*5的像素方块
        for(int j=1;j<6;j++)
        {
            for(int q=1;q<6;q++)
            {
                 projectionImage.at<uchar>(size_x[i]+47-*min_x+j,size_z[i]+47-*min_z+q)=0;
            }
        }
        // 保存为point，凸包使用
        Point point_edge;
        point_edge.y=size_x[i]+47-*min_x;
        point_edge.x=size_z[i]+47-*min_z;
        point_edges.push_back(point_edge);
      }
/**
               *****绘制并保存点云投影图像*****

    namedWindow("点云投影图像");
    imshow("点云投影图像",projectionImage);
    imwrite("点云投影图像.jpg",projectionImage);
    std::cerr<<"Saved points cloud map projection to jpg."<<std::endl;
*/

    // 投影图像闭运算处理
    Mat projectionImage_open(projectionImage.rows,projectionImage.cols,CV_8UC1,Scalar(255));
    Mat element=getStructuringElement(MORPH_RECT,Size(15,15));
    morphologyEx(projectionImage,projectionImage_open,MORPH_OPEN,element); // 开运算处理函数接口
/**
                   *****绘制并保存点云投影图像-开运算*****

    namedWindow("点云投影图像-开运算");
    imshow("点云投影图像-开运算",projectionImage_open);
    imwrite("点云投影图像-开运算.jpg",projectionImage_open);
    std::cerr<<"Saved points cloud map projection & open to jpg."<<std::endl;
*/

    // 图像细化算法
    threshold(projectionImage_open, projectionImage_open, 128, 1, cv::THRESH_BINARY_INV); //将原图像转换为二值图像
    Mat projectionImage_open_thin = thinImage(projectionImage_open);
    projectionImage_open_thin = projectionImage_open_thin * 255;
    int edge_num = 0;
    for(int u = 0; u < projectionImage_open_thin.rows; u++)
    {
        for(int v = 0; v < projectionImage_open_thin.cols; v++)
        {
            if (projectionImage_open_thin.at<uchar>(u, v) == 255)
            {
                projectionImage_open_thin.at<uchar>(u, v) = 0;
                edge_num=edge_num+1;
            }
            else
            {
                projectionImage_open_thin.at<uchar>(u, v) = 255;
            }
        }
    }

/**
           *****绘制并保存点云投影图像-开运算-细化*****

    namedWindow("点云投影图像-开运算-细化", CV_WINDOW_AUTOSIZE);
    imshow("点云投影图像-开运算-细化", projectionImage_open_thin);
    imwrite("点云投影图像-开运算-细化.jpg",projectionImage_open_thin);
    std::cerr<<"Saved points cloud map projection & open & thining to jpg."<<std::endl;
    std::cerr<<"The points cloud map projection & open & thining edge length :"<<edge_num<<std::endl;
 */

    // 计算绘制点云图像凸包，计算凸包周长
    vector<int> hull;
    vector<Point> point_length;
    convexHull(Mat(point_edges),hull,true);
    int hullcount = (int)hull.size(); // 凸包的边数
    Point point_edge0 = point_edges[hull[hullcount-1]]; // 连接凸包边的坐标点，第一个起点初始化
    for (int n = 0;n < hullcount ; n++)   // 绘制凸包的边
    {
        Point point_edge1 = point_edges[hull[n]];
        line(projectionImage_open_thin,point_edge0,point_edge1,Scalar(0),1,CV_AA);
        point_edge0=point_edge1;
        point_length.push_back(point_edge0);
    }
    int length;
    length = arcLength(point_length,true); // 计算凸包周长
/**
                     *****绘制并保存点云投影图像-开运算-细化-凸包*****
    namedWindow("点云投影图像-开运算-细化-凸包", CV_WINDOW_AUTOSIZE);
    imshow("点云投影图像-开运算-细化-凸包", projectionImage_open_thin);
    imwrite("点云投影图像-开运算-细化-凸包.jpg",projectionImage_open_thin);
    std::cerr<<"Saved points cloud map projection & open & thining & hull to jpg."<<std::endl;
    std::cerr<<"The points cloud map projection & open & thining & hull edge length :"<<length<<std::endl;
*/

    // 计算点云层复杂度
    float complexity =(float)edge_num / (float)length;
    Point3f complexity_3f(floor_values[n].x,floor_values[n].y,complexity);
    complexitys.push_back(complexity_3f);
    complexitys_f.push_back(complexity);
/**
                     *****绘制并保存点云复杂度*****
    std::cerr<<"The points cloud map projection complexity :"<<complexity<<std::endl;
    waitKey(0);
*/
    }

    // 计算点云层复杂度最大值
    std::vector<float>::iterator complexitys_max = std::max_element(complexitys_f.begin(),complexitys_f.end());
    int order = complexitys_max - complexitys_f.begin();
    Point3f complexity_max = complexitys[order];

    return(complexity_max);
}

/**
 * 对 compute_complexity 函数产生的复杂度最大的点云层单独处理
 * 保存复杂度最大点云层的：
 *     点云投影图像
 *     点云投影图像-开运算
 *     点云投影图像-开运算-细化
 *     点云投影图像-开运算-细化-凸包
 *     复杂度
 *  函数最后的数字为保存图片的序号
 */
int compute_complexity_max(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud_complexity, const float y_min, const float y_max, const string str)
{

    // 单独抽一层处理
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal_pass (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_complexity);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (y_min,y_max);
    pass.filter (*cloud_voxel_removal_pass);
    pcl::io::savePCDFile("map_0.1h"+str+".pcd",*cloud_voxel_removal_pass,true);
    std::cerr<<"Saved "<<cloud_voxel_removal_pass->points.size()<<" data points to map_0.1h."<<std::endl;

    std::vector<int> size_x;
    std::vector<int> size_z;

    // 获取点云长宽值
    for (size_t i = 0; i < cloud_voxel_removal_pass->points.size (); ++i)
      {
        float xx = cloud_voxel_removal_pass->points[i].x;
        int xxx = xx*100;
        size_x.push_back(xxx);
        float zz = cloud_voxel_removal_pass->points[i].z;
        int zzz = zz*100;
        size_z.push_back(zzz);
      }
    std::vector<int>::iterator max_x = std::max_element(size_x.begin(),size_x.end());
    std::vector<int>::iterator min_x = std::min_element(size_x.begin(),size_x.end());
    std::vector<int>::iterator max_z = std::max_element(size_z.begin(),size_z.end());
    std::vector<int>::iterator min_z = std::min_element(size_z.begin(),size_z.end());
    int x_value=*max_x-*min_x;
    int z_value=*max_z-*min_z;

   // 创建投影图像
    Mat projectionImage(x_value+100,z_value+100,CV_8UC1,Scalar(255));
    vector<Point> point_edges;
    for (size_t i = 0; i < size_x.size (); ++i)
      {
        // 每个投影点为5*5的像素方块
        for(int j=1;j<6;j++)
        {
            for(int q=1;q<6;q++)
            {
                 projectionImage.at<uchar>(size_x[i]+47-*min_x+j,size_z[i]+47-*min_z+q)=0;
            }
        }
        // 保存为point，凸包使用
        Point point_edge;
        point_edge.y=size_x[i]+47-*min_x;
        point_edge.x=size_z[i]+47-*min_z;
        point_edges.push_back(point_edge);
      }
//    namedWindow("点云投影图像"+str+".jpg");
//    imshow("点云投影图像"+str+".jpg",projectionImage);
    imwrite("点云投影图像"+str+".jpg",projectionImage);
    std::cerr<<"Saved points cloud map projection to jpg."<<std::endl;


    // 投影图像闭运算处理
    Mat projectionImage_open(projectionImage.rows,projectionImage.cols,CV_8UC1,Scalar(255));
    Mat element=getStructuringElement(MORPH_RECT,Size(15,15));
    morphologyEx(projectionImage,projectionImage_open,MORPH_OPEN,element);
//    namedWindow("点云投影图像-开运算"+str+".jpg");
//    imshow("点云投影图像-开运算"+str+".jpg",projectionImage_open);
    imwrite("点云投影图像-开运算"+str+".jpg",projectionImage_open);
    std::cerr<<"Saved points cloud map projection & open to jpg."<<std::endl;

    // 图像细化算法
    threshold(projectionImage_open, projectionImage_open, 128, 1, cv::THRESH_BINARY_INV); //将原图像转换为二值图像
    Mat projectionImage_open_thin = thinImage(projectionImage_open);
    projectionImage_open_thin = projectionImage_open_thin * 255;
    int edge_num = 0;
    for(int u = 0; u < projectionImage_open_thin.rows; u++)
    {
        for(int v = 0; v < projectionImage_open_thin.cols; v++)
        {
            if (projectionImage_open_thin.at<uchar>(u, v) == 255)
            {
                projectionImage_open_thin.at<uchar>(u, v) = 0;
                edge_num=edge_num+1;
            }
            else
            {
                projectionImage_open_thin.at<uchar>(u, v) = 255;
            }
        }
    }

//    namedWindow("点云投影图像-开运算-细化"+str+".jpg", CV_WINDOW_AUTOSIZE);
//    imshow("点云投影图像-开运算-细化"+str+".jpg", projectionImage_open_thin);
    imwrite("点云投影图像-开运算-细化"+str+".jpg",projectionImage_open_thin);
    std::cerr<<"Saved points cloud map projection & open & thining to jpg."<<std::endl;
    std::cerr<<"The points cloud map projection & open & thining edge length :"<<edge_num<<std::endl;

    // 计算绘制点云图像凸包，计算凸包周长
    vector<int> hull;
    vector<Point> point_length;
    convexHull(Mat(point_edges),hull,true);
    int hullcount = (int)hull.size(); // 凸包的边数
    Point point_edge0 = point_edges[hull[hullcount-1]]; // 连接凸包边的坐标点，第一个起点初始化
    for (int n = 0;n < hullcount ; n++)   // 绘制凸包的边
    {
        Point point_edge1 = point_edges[hull[n]];
        line(projectionImage_open_thin,point_edge0,point_edge1,Scalar(0),1,CV_AA);
        point_edge0=point_edge1;
        point_length.push_back(point_edge0);
    }
    int length;
    length = arcLength(point_length,true); // 计算凸包周长
//    namedWindow("点云投影图像-开运算-细化-凸包"+str+".jpg", CV_WINDOW_AUTOSIZE);
//    imshow("点云投影图像-开运算-细化-凸包"+str+".jpg", projectionImage_open_thin);
    imwrite("点云投影图像-开运算-细化-凸包"+str+".jpg",projectionImage_open_thin);
    std::cerr<<"Saved points cloud map projection & open & thining & hull to jpg."<<std::endl;
    std::cerr<<"The points cloud map projection & open & thining & hull edge length :"<<length<<std::endl;

    // 计算点云层复杂度
    float complexity =(float)edge_num / (float)length ;
    std::cerr<<"The points cloud map projection complexity max :"<<complexity<<std::endl;
    std::cerr<<"The points cloud map projection complexity max position :("<<y_min<<","<<y_max<<")."<<std::endl;

    return(0);
}

/**
 * @brief 对输入的点云区域逐层进行NDT变换配准
 * @param 点云区域，‘1’上层，‘2’中层，‘3’下层
 * @return 变换矩阵Eigen4*4类型
 */
Eigen::VectorXd ndt_transform(const string str, const Eigen::VectorXd Initial_transform)
{
    // 读取点云进行匹配
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_layer (new pcl::PointCloud<pcl::PointXYZ>); // 复杂度最大的层点云 目标点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_layer (new pcl::PointCloud<pcl::PointXYZ>); // 对应的局部点云层 输入点云
    pcl::io::loadPCDFile<pcl::PointXYZ>("map_0.1h"+str+".pcd",*cloud_layer);
    std::cout<<"Loaded data points from map_0.1h" << str <<".pcd as the cloud_layer. "<<std::endl;
    pcl::io::loadPCDFile<pcl::PointXYZ>("map_0.1hpart"+str+".pcd",*part_cloud_layer);
    std::cout<<"Loaded data points from map_0.1hpart" << str <<".pcd as the part_cloud_layer. "<<std::endl;

    std::vector<int> indices1,indices2; //保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_layer,*cloud_layer, indices1); //去除点云中的NaN点
    pcl::removeNaNFromPointCloud(*part_cloud_layer,*part_cloud_layer, indices2); //去除点云中的NaN点

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt; //初始化正态分布变换（NDT）

    ndt.setTransformationEpsilon (0.01); //设置依赖尺度NDT参数，设置依赖尺度NDT参数
    ndt.setStepSize (0.1); // 为More-Thuente线搜索设置最大步长
    ndt.setResolution (1.0); // 设置NDT网格结构的分辨率（VoxelGridCovariance）
    ndt.setMaximumIterations (1000); // 设置匹配迭代的最大次数
    ndt.setInputSource(part_cloud_layer); // 设置要配准的点云 局部点云
    ndt.setInputTarget (cloud_layer);  // 设置点云配准目标 全局点云

    // 初值：2.484552145 -0.060922295 -2.703711987 -0.001523848 0.998953700 -0.034495004 -0.029987484
    // 设置矩阵变换矩阵初值
    Eigen::Quaterniond init_q_cloud (Initial_transform(6), Initial_transform(3), Initial_transform(4), Initial_transform(5)); // XYZW -> WXYZ
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate ( init_q_cloud );
    T.pretranslate( Eigen::Vector3d( Initial_transform(0), Initial_transform(1),Initial_transform(2)));
    Eigen::Matrix4d init_guessd (T.matrix());
    Eigen::Matrix4f init_guess = init_guessd.cast<float>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_layer_output (new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align (*part_cloud_layer_output, init_guess); //计算需要的刚体变换以便将输入的点云匹配到目标点云

    // 保存变换矩阵与欧式适合度评分
    Eigen::Matrix4f Matrix_cloud =  ndt.getFinalTransformation();
    double FitnessScore = 1.0/ndt.getFitnessScore (); // 评分的倒数

    // 转换输出为平移+四元数
    Eigen::Matrix4d Matrix_cloudd = Matrix_cloud.cast<double>();
    Eigen::Matrix3d Matrix_cloud_rotationd ;
    Eigen::Vector3d Matrix_cloud_translationd;
    Matrix_cloud_rotationd << Matrix_cloudd(0,0),Matrix_cloudd(0,1),Matrix_cloudd(0,2),
                              Matrix_cloudd(1,0),Matrix_cloudd(1,1),Matrix_cloudd(1,2),
                              Matrix_cloudd(2,0),Matrix_cloudd(2,1),Matrix_cloudd(2,2);
    Matrix_cloud_translationd << Matrix_cloudd(0,3),Matrix_cloudd(1,3),Matrix_cloudd(2,3);
    Eigen::Quaterniond q_cloud(Matrix_cloud_rotationd); // 旋转矩阵转四元数 输入必须为Double类型
    Eigen::AngleAxisd v_cloud;
    v_cloud.fromRotationMatrix(Matrix_cloud_rotationd);
    Eigen::VectorXd transform(8);
    transform << Matrix_cloudd(0,3),Matrix_cloudd(1,3),Matrix_cloudd(2,3),q_cloud.x(),q_cloud.y(),q_cloud.z(),q_cloud.w(),FitnessScore;

    // 转换输出为平移+欧拉角
//    Eigen::Vector3d euler_angles = Matrix_cloud_rotationd.eulerAngles ( 1,2,0 ); // YZX顺序，即pitch roll yaw顺序

    // 输出变换矩阵与欧式适合度评分  评分越低越好
    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
              << " score: " << FitnessScore << std::endl;
    std::cout << "Normal Distributions Transform Matrix of part_cloud_layer is :" << std::endl
              << Matrix_cloud << std::endl;
    std::cout <<"Normal Distributions Transform Quaternion of part_cloud_layer is :"<< std::endl
              << Matrix_cloud_translationd.transpose()<<" "
              << q_cloud.x()<<" "<<q_cloud.y()<<" "<<q_cloud.z()<<" "<<q_cloud.w()<< std::endl;

    // 输出欧拉角 旋转角
//    std::cout <<"Normal Distributions Transform pitch(Y) roll(Z) yaw(X) of part_cloud_layer is :"  << std::endl
//              << Matrix_cloud_translationd.transpose() << " " << euler_angles.transpose() << std::endl;
    std::cout <<"Normal Distributions Transform vvv part_cloud_layer is :"<< std::endl
              <<v_cloud.angle()<< std::endl;

/**
                    *****保存转换的局部点云*****
    pcl::io::savePCDFile("part_cloud_layer_output.pcd",*part_cloud_layer_output,true); // 'true'保存为二进制
    std::cerr<<"Saved part_cloud_layer after transform to part_cloud_layer_output.pcd."<<std::endl;
*/

    // 初始化点云可视化界面
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
    viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer_final->setBackgroundColor (255, 255, 255);
    //对目标点云着色（红色）并可视化
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    target_color (cloud_layer, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZ> (cloud_layer, target_color, "cloud_layer");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1, "cloud_layer");
    //对转换后的目标点云着色（绿色）并可视化
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    output_color (part_cloud_layer_output, 255, 0, 0);
    viewer_final->addPointCloud<pcl::PointXYZ> (part_cloud_layer_output, output_color, "part_cloud_layer_output");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1, "part_cloud_layer_output");
    // 启动可视化
    viewer_final->addCoordinateSystem (1.0);
    viewer_final->initCameraParameters ();
    //等待直到可视化窗口关闭。
    while (!viewer_final->wasStopped ())
    {
      viewer_final->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return transform;
}

/**
 * @brief 已知三角形三边边长a,b,c;
 * @param 求解三角形外接圆半径
 * @param 求解三角形内接圆半径
 * @return 求解重心离三顶点的距离
 */
class trangle_param
{
public:
    trangle_param(std::vector<float>&);
    ~trangle_param();
public:
    void cal_out_r();
    void cal_in_r();
    void cal_dist();
public:
    float out_r, in_r;
    Point gravity;
    std::vector<float> dist;
    std::vector<float> param;
};
trangle_param::trangle_param(std::vector<float>& param)
{
    this->param = param;
    dist.resize(3);
}
trangle_param::~trangle_param()
{

}
void trangle_param::cal_dist()
{
    int n = 0, sum_square = 0;
    std::vector<float> theta;
    theta.resize(3);
    theta[0] = float(pow(param[1], 2) + pow(param[2], 2)- pow(param[0], 2)) / float(param[1] * param[2]) / 2.0;
    theta[1] = float(pow(param[0], 2) + pow(param[2], 2)- pow(param[1], 2)) / float(param[2] * param[0]) / 2.0;
    theta[2] = float(pow(param[1], 2) + pow(param[0], 2)- pow(param[2], 2)) / float(param[1] * param[0]) / 2.0;
    this->dist[0] = sqrt(float(pow(param[1], 2) + pow(param[0] / 2.0, 2)) - float(param[1] * param[0] * theta[2]))*2.0 / 3.0; // 到0边所对的顶点
    this->dist[1] = sqrt(float(pow(param[2], 2) + pow(param[1] / 2.0, 2)) - float(param[2] * param[1] * theta[0]))*2.0 / 3.0; // 到1边所对的顶点
    this->dist[2] = sqrt(float(pow(param[0], 2) + pow(param[2] / 2.0, 2)) - float(param[0] * param[2] * theta[1]))*2.0 / 3.0; // 到2边所对的顶点
}
void trangle_param::cal_out_r()
{
    int n = 3;
    float multiply = 1, sum = 0;
    while (n--)
    {
        multiply = multiply*param[n];
        sum = sum + param[n];
    }
    (this->out_r) = multiply / sqrt(sum*(sum - 2.0*param[0]) * (sum - 2.0*param[1]) * (sum - 2.0*param[2]));
}
void trangle_param::cal_in_r()
{
    int n = 3;
    float sum = 0;
    while (n--)
    {
        sum = sum + param[n];
    }
    (this->in_r) = sqrt((sum - 2.0*param[0]) * (sum - 2.0*param[1]) * (sum - 2.0*param[2]) / sum) / 2.0;
}

/**
 * @brief 求解点云的XYZ坐标的极值;
 * @param 输入XYZ类型点云
 * @return 返回XZY坐标的最大值与最小值
 */
class cloud_size
{
public:
    cloud_size(pcl::PointCloud<pcl::PointXYZ>&);
    ~cloud_size();
public:
    void cal_cloud_size();
public:
    std::vector<Point2d> coordinate;
    pcl::PointCloud<pcl::PointXYZ> input_cloud ;
};
cloud_size::cloud_size(pcl::PointCloud<pcl::PointXYZ>& input_cloud)
{
    this->input_cloud = input_cloud;
    coordinate.resize(3);
}
cloud_size::~cloud_size()
{

}
void cloud_size::cal_cloud_size()
{
    // 获取点云坐标值
    std::vector<float> size_x,size_z,size_y;
    for (size_t i = 0; i < input_cloud.points.size (); ++i)
      {
        float xx = input_cloud.points[i].x;
        size_x.push_back(xx);
        float zz = input_cloud.points[i].z;
        size_z.push_back(zz);
        float yy = input_cloud.points[i].y;
        size_y.push_back(yy);
      }
    std::vector<float>::iterator max_x = std::max_element(size_x.begin(),size_x.end());
    std::vector<float>::iterator min_x = std::min_element(size_x.begin(),size_x.end());
    std::vector<float>::iterator max_z = std::max_element(size_z.begin(),size_z.end());
    std::vector<float>::iterator min_z = std::min_element(size_z.begin(),size_z.end());
    std::vector<float>::iterator max_y = std::max_element(size_y.begin(),size_y.end());
    std::vector<float>::iterator min_y = std::min_element(size_y.begin(),size_y.end());
    Point2d coordinate_x (*max_x, *min_x);
    Point2d coordinate_z (*max_z, *min_z);
    Point2d coordinate_y (*max_y, *min_y);
    this->coordinate[0] = coordinate_x;
    this->coordinate[1] = coordinate_z;
    this->coordinate[2] = coordinate_y;
}

/**
 * @brief 对输入的点云区域进行ICP变换配准
 * @param 输入的融合位姿
 * @return 平移向量+四元数
 */
Eigen::VectorXd icp_transform(const Eigen::VectorXd Transform_cloud_last)
{
    // 不经过函数传递，直接加载需要的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZ>); // 加载没有NDT初始变换的局部点云
    pcl::io::loadPCDFile<pcl::PointXYZ>("part_map_voxel_removal.pcd",*cloud_src);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal (new pcl::PointCloud<pcl::PointXYZ>); // 加载全局点云
    pcl::io::loadPCDFile<pcl::PointXYZ>("map_voxel_removal.pcd",*cloud_voxel_removal);

    // 利用NDT初值，对全局点云进行分割
    pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_after_tf (new pcl::PointCloud<pcl::PointXYZ>); // 变换后的点云
    Eigen::Quaterniond q_cloud_last (Transform_cloud_last(6), Transform_cloud_last(3), Transform_cloud_last(4), Transform_cloud_last(5));
    Eigen::Isometry3d T_last=Eigen::Isometry3d::Identity();
    T_last.rotate ( q_cloud_last );
    T_last.pretranslate ( Eigen::Vector3d ( Transform_cloud_last(0),Transform_cloud_last(1),Transform_cloud_last(2)));
    Eigen::Matrix4d Matrix_cloud_lastd (T_last.matrix());
    Eigen::Matrix4f Matrix_cloud_last = Matrix_cloud_lastd.cast<float>();
    pcl::transformPointCloud (*cloud_src, *part_cloud_after_tf, Matrix_cloud_last); // 对初始局部点云进行变换
    cloud_size part_cloud_size(*part_cloud_after_tf);
    part_cloud_size.cal_cloud_size();
    pcl::io::savePCDFile("part_map_after_tf.pcd",*part_cloud_after_tf,true); // 'true'保存为二进制
    std::cerr<<"Saved part pointcloud after transforem to part_map_after_tf.pcd."<<std::endl;

    // 直通滤波器进行区域分割
    pcl::PassThrough<pcl::PointXYZ> passx,passz,passy;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passx (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passxz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passxzy (new pcl::PointCloud<pcl::PointXYZ>);
    passx.setInputCloud (cloud_voxel_removal);
    passx.setFilterFieldName ("x");
    passx.setFilterLimits (part_cloud_size.coordinate[0].y,part_cloud_size.coordinate[0].x);
    passx.filter (*cloud_passx);
    passz.setInputCloud (cloud_passx);
    passz.setFilterFieldName ("z");
    passz.setFilterLimits (part_cloud_size.coordinate[1].y,part_cloud_size.coordinate[1].x);
    passz.filter (*cloud_passxz);
    passy.setInputCloud (cloud_passxz);
    passy.setFilterFieldName ("y");
    passy.setFilterLimits (part_cloud_size.coordinate[2].y,part_cloud_size.coordinate[2].x);
    passy.filter (*cloud_passxzy);

    pcl::io::savePCDFile("map_passxzy.pcd",*cloud_passxzy,true); // 保存全局地图分割后的点云地图
    std::cerr<<"Saved pointcloud after pass xzy to cloud_passxzy.pcd."<<std::endl;

    // 1.去除NaN点
    std::vector<int> indices_src,indices_tgt; // 保存去除的点的索引
    pcl::removeNaNFromPointCloud(*cloud_src, *cloud_src, indices_src); //去除点云中的NaN点
    pcl::removeNaNFromPointCloud(*cloud_passxzy, *cloud_passxzy, indices_tgt); //去除点云中的NaN点

    // 2.计算曲面法线和曲率
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt (new pcl::PointCloud<pcl::PointNormal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> norm_est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    norm_est.setSearchMethod (tree);
    norm_est.setKSearch (30);
    norm_est.setInputCloud (cloud_src);
    norm_est.compute (*points_with_normals_src);
    pcl::copyPointCloud (*cloud_src, *points_with_normals_src);
    norm_est.setInputCloud (cloud_passxzy);
    norm_est.compute (*points_with_normals_tgt);
    pcl::copyPointCloud (*cloud_passxzy, *points_with_normals_tgt);

    // 3.设置ICP算法参数
    pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> icp;
    icp.setTransformationEpsilon (1e-15);
    icp.setMaxCorrespondenceDistance(10);
    icp.setEuclideanFitnessEpsilon(0.0001);
    icp.setMaximumIterations(10000);
    icp.setInputSource (points_with_normals_src);
    icp.setInputTarget (points_with_normals_tgt);
    pcl::PointCloud<pcl::PointNormal>::Ptr part_cloud_final_normal (new pcl::PointCloud<pcl::PointNormal>);
    icp.align(*part_cloud_final_normal,Matrix_cloud_last); // 设置NDT变换初值，函数传入

    // 4.保存变换矩阵与欧式适合度评分
    Eigen::Matrix4f Matrix_cloud = icp.getFinalTransformation();
    double FitnessScore_icp = 1.0/icp.getFitnessScore (); // 评分的倒数

    // 5.转换输出为平移+四元数
    Eigen::Matrix4d Matrix_cloudd = Matrix_cloud.cast<double>();
    Eigen::Matrix3d Matrix_cloud_rotationd ;
    Eigen::Vector3d Matrix_cloud_translationd;
    Matrix_cloud_rotationd << Matrix_cloudd(0,0),Matrix_cloudd(0,1),Matrix_cloudd(0,2),
                              Matrix_cloudd(1,0),Matrix_cloudd(1,1),Matrix_cloudd(1,2),
                              Matrix_cloudd(2,0),Matrix_cloudd(2,1),Matrix_cloudd(2,2);
    Matrix_cloud_translationd << Matrix_cloudd(0,3),Matrix_cloudd(1,3),Matrix_cloudd(2,3);
    Eigen::Quaterniond q_cloud(Matrix_cloud_rotationd); // 旋转矩阵转四元数 输入必须为Double类型

    std::cout << "Iterative Closest Point transform has converged:" << icp.hasConverged ()
              << " score: " << FitnessScore_icp << std::endl;

    Eigen::VectorXd transform(8);
    transform << Matrix_cloudd(0,3),Matrix_cloudd(1,3),Matrix_cloudd(2,3),q_cloud.x(),q_cloud.y(),q_cloud.z(),q_cloud.w(),FitnessScore_icp;

    // 转换点云格式 XYZNormal->XYZ
    pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_final (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*part_cloud_final_normal, *part_cloud_final);

    pcl::io::savePCDFile("part_map_final.pcd",*part_cloud_final,true); // 'true'保存为二进制
    std::cerr<<"Saved part_map_final after icp final transform to part_map_final.pcd."<<std::endl;

    // 初始化点云可视化界面
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
    viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer_final->setBackgroundColor (255, 255, 255);
    //对目标点云着色（红色）并可视化
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
//    target_color (cloud_voxel_removal, 255, 0, 0);
//    viewer_final->addPointCloud<pcl::PointXYZ> (cloud_voxel_removal, target_color, "cloud_layer");
    target_color (cloud_passxzy, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZ> (cloud_passxzy, target_color, "cloud_layer");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1, "cloud_layer");
    //对转换后的目标点云着色（绿色）并可视化
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    output_color (part_cloud_final, 255, 0, 0);
    viewer_final->addPointCloud<pcl::PointXYZ> (part_cloud_final, output_color, "part_cloud_layer_output");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,1, "part_cloud_layer_output");
    // 启动可视化
    viewer_final->addCoordinateSystem (1.0);
    viewer_final->initCameraParameters ();
    //等待直到可视化窗口关闭。
    while (!viewer_final->wasStopped ())
    {
      viewer_final->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return transform;
}

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_xyzrgba (new pcl::PointCloud<pcl::PointXYZRGBA>); // 创建输入点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>); // 创建去除颜色信息的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel (new pcl::PointCloud<pcl::PointXYZ>); // 创建体素滤波后的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal (new pcl::PointCloud<pcl::PointXYZ>); // 创建去除外点后的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal_pass1 (new pcl::PointCloud<pcl::PointXYZ>); // 创建区域分割点云上层
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal_pass2 (new pcl::PointCloud<pcl::PointXYZ>); // 创建区域分割点云中层
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel_removal_pass3 (new pcl::PointCloud<pcl::PointXYZ>); // 创建区域分割点云下层

  // 打开点云文件
  if(pcl::io::loadPCDFile<pcl::PointXYZRGBA>("map.pcd",*cloud_xyzrgba)==-1)
  {
  PCL_ERROR("Couldn't read file map.pcd\n");
  return(-1);
  }
  std::cout<<"Loaded "<<cloud_xyzrgba->width*cloud_xyzrgba->height <<" data points from map.pcd with the following fields: "<<std::endl;

  // 转换点云格式 XYZRGB->XYZ
  pcl::copyPointCloud(*cloud_xyzrgba, *cloud_xyz);

  // 进行体素滤波
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud_xyz);
  sor.setLeafSize (0.04f, 0.04f, 0.04f);
  sor.filter (*cloud_voxel);
  std::cerr << "PointCloud after voxel filtering: " << cloud_voxel->width * cloud_voxel->height
       << " data points (" << pcl::getFieldsList (*cloud_voxel) << ")."<<std::endl;

  // 统计滤波器移除外点
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> rem;
  rem.setInputCloud (cloud_voxel);
  rem.setMeanK (100); // 设置在进行统计时考虑查询点的邻近点数
  rem.setStddevMulThresh (1.0); // 设置判断是否为离群点的阈值
  rem.filter (*cloud_voxel_removal);
  std::cerr << "PointCloud after Statistical Outlier Removal filtering: " << cloud_voxel_removal->width * cloud_voxel_removal->height
       << " data points (" << pcl::getFieldsList (*cloud_voxel_removal) << ")."<<std::endl;

  // 保存点云
  pcl::io::savePCDFile("map_voxel_removal.pcd",*cloud_voxel_removal,true); // 'true'保存为二进制
  std::cerr<<"Saved "<<cloud_voxel_removal->points.size()<<" data points to map_xyz_voxel_removal.pcd."<<std::endl;

  /**
   * 计算点云y字段范围
   * 以下程序适用于ORBSLAM，垂直方向为Y轴,上为Y轴负方向，平面为X-Z平面，相机朝向为Z轴正方向
   * RGBDSLAM 垂直方向为Z轴正方向，同时需要修改投影平面为X-Y
   */
  std::vector<float> size_y;
  for (size_t i = 0; i < cloud_voxel_removal->points.size (); ++i)
    {
      float yy =cloud_voxel_removal->points[i].y;
      size_y.push_back(yy);
    }
  std::vector<float>::iterator max = std::max_element(size_y.begin(),size_y.end());
  std::cout << "Max element is " << *max << std::endl;
  std::vector<float>::iterator min = std::min_element(size_y.begin(),size_y.end());
  std::cout << "Min element is " << *min << std::endl;

  float y_value = *max-*min; // 点云高度值
  float y1 = *max-0.8*y_value;
  float y2 = *max-0.5*y_value;
  float y3 = *max-0.3*y_value;
  float y4 = *max-0.15*y_value;

  // 直通滤波器进行区域分割
  pcl::PassThrough<pcl::PointXYZ> pass1,pass2,pass3;
  pass1.setInputCloud (cloud_voxel_removal);
  pass2.setInputCloud (cloud_voxel_removal);
  pass3.setInputCloud (cloud_voxel_removal);
  pass1.setFilterFieldName ("y");
  pass2.setFilterFieldName ("y");
  pass3.setFilterFieldName ("y");
  pass1.setFilterLimits (y1,y2); // 设置过滤字段上的范围 50%-80%
  pass2.setFilterLimits (y2,y3); // 设置过滤字段上的范围 30%-50%
  pass3.setFilterLimits (y3,y4); // 设置过滤字段上的范围 15%-30%
  //pass.setFilterLimitsNegative (true);
  pass1.filter (*cloud_voxel_removal_pass1);
  pass2.filter (*cloud_voxel_removal_pass2);
  pass3.filter (*cloud_voxel_removal_pass3);

  // 保存区域分割点云
  pcl::io::savePCDFile("map_xyz_voxel_removal_pass1.pcd",*cloud_voxel_removal_pass1,true); // 'true'保存为二进制
  std::cerr<<"Saved "<<cloud_voxel_removal_pass1->points.size()<<" data points to map_xyz_voxel_removal_pass1.pcd."<<std::endl;
  pcl::io::savePCDFile("map_xyz_voxel_removal_pass2.pcd",*cloud_voxel_removal_pass2,true); // 'true'保存为二进制
  std::cerr<<"Saved "<<cloud_voxel_removal_pass2->points.size()<<" data points to map_xyz_voxel_removal_pass2.pcd."<<std::endl;
  pcl::io::savePCDFile("map_xyz_voxel_removal_pass3.pcd",*cloud_voxel_removal_pass3,true); // 'true'保存为二进制
  std::cerr<<"Saved "<<cloud_voxel_removal_pass3->points.size()<<" data points to map_xyz_voxel_removal_pass3.pcd."<<std::endl;

  // 处理三区域的点云，求解最大复杂度
  Point3f complexity_max1,complexity_max2,complexity_max3;
  complexity_max1 = compute_complexity(cloud_voxel_removal_pass1,y1,y2);
  complexity_max2 = compute_complexity(cloud_voxel_removal_pass2,y2,y3);
  complexity_max3 = compute_complexity(cloud_voxel_removal_pass3,y3,y4);

  // 保存复杂度最大值对应的点云相关图像
  compute_complexity_max(cloud_voxel_removal_pass1,complexity_max1.x,complexity_max1.y,"1");
  compute_complexity_max(cloud_voxel_removal_pass2,complexity_max2.x,complexity_max2.y,"2");
  compute_complexity_max(cloud_voxel_removal_pass3,complexity_max3.x,complexity_max3.y,"3");

  // 获取部分点云进行二维匹配
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr part_cloud_xyzrgba (new pcl::PointCloud<pcl::PointXYZRGBA>); // 创建输入点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>); // 创建去除颜色信息的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_voxel (new pcl::PointCloud<pcl::PointXYZ>); // 创建体素滤波后的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr part_cloud_voxel_removal (new pcl::PointCloud<pcl::PointXYZ>); // 创建去除外点后的点云

  // 打开点云文件
  if(pcl::io::loadPCDFile<pcl::PointXYZRGBA>("part_map.pcd",*part_cloud_xyzrgba)==-1)
  {
  PCL_ERROR("Couldn't read file part_map.pcd\n");
  return(-1);
  }
  std::cout<<"Loaded "<<part_cloud_xyzrgba->width*part_cloud_xyzrgba->height <<" data points from part_map.pcd with the following fields: "<<std::endl;

  // 转换点云格式 XYZRGB->XYZ
  pcl::copyPointCloud(*part_cloud_xyzrgba, *part_cloud_xyz);

  // 进行体素滤波
  pcl::VoxelGrid<pcl::PointXYZ> part_sor;
  part_sor.setInputCloud (part_cloud_xyz);
  part_sor.setLeafSize (0.04f, 0.04f, 0.04f);
  part_sor.filter (*part_cloud_voxel);
  std::cerr << "Part PointCloud after voxel filtering: " << part_cloud_voxel->width * part_cloud_voxel->height
       << " data points (" << pcl::getFieldsList (*part_cloud_voxel) << ")."<<std::endl;

  // 统计滤波器移除外点
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> part_rem;
  part_rem.setInputCloud (part_cloud_voxel);
  part_rem.setMeanK (100); // 设置在进行统计时考虑查询点的邻近点数
  part_rem.setStddevMulThresh (5.0); // 设置判断是否为离群点的阈值
  part_rem.filter (*part_cloud_voxel_removal);
  std::cerr << "Part PointCloud after Statistical Outlier Removal filtering: " << part_cloud_voxel_removal->width * part_cloud_voxel_removal->height
       << " data points (" << pcl::getFieldsList (*part_cloud_voxel_removal) << ")."<<std::endl;

  // 保存点云
  pcl::io::savePCDFile("part_map_voxel_removal.pcd",*part_cloud_voxel_removal,true); // 'true'保存为二进制
  std::cerr<<"Saved "<<part_cloud_voxel_removal->points.size()<<" data points to part_map_xyz_voxel_removal.pcd."<<std::endl;

  // 根据之前复杂度最大值分割部分点云
  compute_complexity_max(part_cloud_voxel_removal,complexity_max1.x,complexity_max1.y,"part1");
  compute_complexity_max(part_cloud_voxel_removal,complexity_max2.x,complexity_max2.y,"part2");
  compute_complexity_max(part_cloud_voxel_removal,complexity_max3.x,complexity_max3.y,"part3");

  /**
   * 使用NDT进行二维点云配准
   */
  Eigen::VectorXd Initial_transform(7); // 设置初值
  Initial_transform <<3.0, -0.05, -3.0, -0.002, 1.0, -0.034,-0.030;

  Eigen::VectorXd Transform_cloud1(8);
  Transform_cloud1 = ndt_transform("1",Initial_transform); // 对上层点云进行NDT变换

  Eigen::VectorXd Transform_cloud2(8);
  Transform_cloud2 = ndt_transform("2",Initial_transform); // 对上层点云进行NDT变换

  Eigen::VectorXd Transform_cloud3(8);
  Transform_cloud3 = ndt_transform("3",Initial_transform); // 对上层点云进行NDT变换

  // 一致性检测
  double  sum_d12=0, sum_d13=0, sum_d23=0;
  for (int s=0 ;s<3;s++)
  {
      sum_d12=sum_d12+(Transform_cloud1[s]-Transform_cloud2[s])*(Transform_cloud1[s]-Transform_cloud2[s]);
      sum_d13=sum_d13+(Transform_cloud1[s]-Transform_cloud3[s])*(Transform_cloud1[s]-Transform_cloud3[s]);
      sum_d23=sum_d23+(Transform_cloud2[s]-Transform_cloud3[s])*(Transform_cloud2[s]-Transform_cloud3[s]);
  }
  double Angle_d1 = 2*acos(Transform_cloud1[6]); // 取四元数W值
  double Angle_d2 = 2*acos(Transform_cloud2[6]);
  double Angle_d3 = 2*acos(Transform_cloud3[6]);
  double Eur_d12 = sqrt(sum_d12+(Angle_d1-Angle_d2)*(Angle_d1-Angle_d2));
  double Eur_d13 = sqrt(sum_d13+(Angle_d1-Angle_d3)*(Angle_d1-Angle_d3));
  double Eur_d23 = sqrt(sum_d23+(Angle_d2-Angle_d3)*(Angle_d2-Angle_d3));
  std::vector<float> Side_length;
  Side_length.resize(3);
  Side_length[0] = Eur_d23;// Eur_d23对应Transform_cloud1
  Side_length[1] = Eur_d13;// Eur_d13对应Transform_cloud2
  Side_length[2] = Eur_d12;// Eur_d12对应Transform_cloud3
  std::vector<float> FitnessScore;
  FitnessScore.resize(3);
  FitnessScore[0] = Transform_cloud1[7];
  FitnessScore[1] = Transform_cloud2[7];
  FitnessScore[2] = Transform_cloud3[7];

  // 计算三角形边长大小序列号
  std::vector<float>::iterator Side_length_max = std::max_element(std::begin(Side_length), std::end(Side_length));
  int num_max = std::distance(std::begin(Side_length), Side_length_max);
  std::vector<float>::iterator Side_length_min = std::min_element(std::begin(Side_length), std::end(Side_length));
  int num_min = std::distance(std::begin(Side_length), Side_length_min);
  int num_mid = 3 - num_max - num_min;

  // [1]两边之和小于第三边，重定位失败
  if (Side_length[num_min]+Side_length[num_mid] < Side_length[num_max])
  {
      std::cout << "   ———— [1]THE RELOCALIZATION FAILED！ ————   " << std::endl;
      exit(0); // 退出主程序[1]
  }

  trangle_param t_param(Side_length); // 计算内外接圆半径，重心距离等
  t_param.cal_out_r();
  t_param.cal_in_r();
  t_param.cal_dist();

  // [2]2D重定位整体误差过大：外接圆半径大于阈值
  if (t_param.out_r > 2.5)
  {
      std::cout << "   ———— [2]THE RELOCALIZATION FAILED！ ————   " << std::endl;
      exit(0); // 退出主程序[2]
  }

  // [3]2D重定位离散值过大：外接圆半径与内接圆半径之比大于阈值
  std::vector<float>::iterator FitnessScore_max = std::max_element(std::begin(FitnessScore), std::end(FitnessScore));
  int num_max_score = std::distance(std::begin(FitnessScore), FitnessScore_max);
  if (t_param.out_r/t_param.in_r > 5.0)
  {
      if (num_min == num_max_score ) // 最短边对应的为离散点，离散点为打分最高的点
      {
          std::cout << "   ———— [3]THE RELOCALIZATION FAILED！ ————   " << std::endl;
          exit(0); // 退出主程序[2]
      }
      else // 去掉离散点
      {
          std::cout << "   ———— [4]THE RELOCALIZATION USE 2 POSES！ ————   " << std::endl;
          t_param.dist[num_mid] = 1;// Side_length[num_min]/2.0;
          t_param.dist[num_max] = 1;// Side_length[num_min]/2.0;
          t_param.dist[num_min] = 0;// t_param.dist[num_min] 丢弃
      }
  }

  // 归一化
  std::vector<float> complexity_max;
  complexity_max.resize(3);
  complexity_max[0] = complexity_max1.z;
  complexity_max[1] = complexity_max2.z;
  complexity_max[2] = complexity_max3.z;
  std::vector<float>::iterator complexity_max_max = std::max_element(std::begin(complexity_max), std::end(complexity_max));
  int num_max_complexity = std::distance(std::begin(complexity_max), complexity_max_max);
  float max_complexity = complexity_max[num_max_complexity];
  complexity_max[0] = complexity_max[0]/max_complexity;
  complexity_max[1] = complexity_max[1]/max_complexity;
  complexity_max[2] = complexity_max[2]/max_complexity;
  float max_score = FitnessScore[num_max_score];
  FitnessScore[0] = FitnessScore[0]/max_score;
  FitnessScore[1] = FitnessScore[1]/max_score;
  FitnessScore[2] = FitnessScore[2]/max_score;
  if (t_param.dist[num_min] != 0)
  {
      t_param.dist[0] = 1.0/t_param.dist[0];
      t_param.dist[1] = 1.0/t_param.dist[1];
      t_param.dist[2] = 1.0/t_param.dist[2];
  }
  std::vector<float>::iterator dist_max = std::max_element(std::begin(t_param.dist), std::end(t_param.dist));
  int num_max_dist = std::distance(std::begin(t_param.dist), dist_max);
  float max_dist = t_param.dist[num_max_dist];
  if (t_param.dist[num_min] != 0)
  {
      t_param.dist[0] = t_param.dist[0]/max_dist;
      t_param.dist[1] = t_param.dist[1]/max_dist;
      t_param.dist[2] = t_param.dist[2]/max_dist;
  }

  // 位姿融合
  std::vector<float> weight;
  weight.resize(3);
  if (t_param.dist[num_min] == 0)
  {
      weight[0] = 0.3*complexity_max[num_mid] + 0.4*FitnessScore[num_mid] + 0.3*t_param.dist[num_mid];
      weight[1] = 0.3*complexity_max[num_max] + 0.4*FitnessScore[num_max] + 0.3*t_param.dist[num_max];
      std::cerr<<"Weight: "<<complexity_max[num_mid]<<"+"<<FitnessScore[num_mid]<<"+"<<t_param.dist[num_mid]<<"="<<weight[0]<<std::endl;
      std::cerr<<"Weight: "<<complexity_max[num_max]<<"+"<<FitnessScore[num_max]<<"+"<<t_param.dist[num_max]<<"="<<weight[1]<<std::endl;
  }
  else
  {
      weight[0] = 0.3*complexity_max[0] + 0.4*FitnessScore[0] + 0.3*t_param.dist[0];
      weight[1] = 0.3*complexity_max[1] + 0.4*FitnessScore[1] + 0.3*t_param.dist[1];
      weight[2] = 0.3*complexity_max[2] + 0.4*FitnessScore[2] + 0.3*t_param.dist[2];
      std::cerr<<"Weight: "<<complexity_max[0]<<"+"<<FitnessScore[0]<<"+"<<t_param.dist[0]<<"="<<weight[0]<<std::endl;
      std::cerr<<"Weight: "<<complexity_max[1]<<"+"<<FitnessScore[1]<<"+"<<t_param.dist[1]<<"="<<weight[1]<<std::endl;
      std::cerr<<"Weight: "<<complexity_max[2]<<"+"<<FitnessScore[2]<<"+"<<t_param.dist[2]<<"="<<weight[2]<<std::endl;
  }
  Eigen::VectorXd Transform_cloud(7);
  for( int k=0; k<7;k++ )
  {
      Transform_cloud1[k] = Transform_cloud1[k]*weight[0]/(weight[0]+weight[1]+weight[2]);
      Transform_cloud2[k] = Transform_cloud2[k]*weight[1]/(weight[0]+weight[1]+weight[2]);
      Transform_cloud3[k] = Transform_cloud3[k]*weight[2]/(weight[0]+weight[1]+weight[2]);
      Transform_cloud[k] =  Transform_cloud1[k]+Transform_cloud2[k]+Transform_cloud3[k];
  }
  std::cerr<<" ———TRANSFORM FUSION SUCCESS AND THE TRANSFORM IS : "<<std::endl;
  for( int e=0; e<7; e++ ){ std::cerr<<Transform_cloud[e]<<" "; }
  std::cerr<<std::endl;

  // 带入评分Weight最大的点云层，将位姿作为初值再次NDT优化
  std::vector<float>::iterator weight_max = std::max_element(std::begin(weight), std::end(weight));
  int num_max_weight = std::distance(std::begin(weight), weight_max);

  Eigen::VectorXd Transform_cloud_last(8);
  string last_name = boost::lexical_cast<string>(num_max_weight+1);
  Transform_cloud_last = ndt_transform(last_name,Transform_cloud);
  std::cerr<<" ———RELOCALIZATION SUCCESS AND THE NDT TRANSFORM IS : "<<std::endl;
  for( int w=0; w<7; w++ ){ std::cerr<<Transform_cloud_last[w]<<" "; }
  std::cerr<<std::endl;

  /**
   * 使用ICP进行三维点云配准
   */
  Eigen::VectorXd Transform_final(8);
  Transform_final = icp_transform(Transform_cloud_last); // 对点云进行ICP变换

  std::cerr<<" ———RELOCALIZATION SUCCESS AND THE ICP FINAL TRANSFORM IS : "<<std::endl;
  for( int t=0; t<7; t++ ){ std::cerr<<Transform_final[t]<<" "; }
  std::cerr<<std::endl;

  return (0);
}


