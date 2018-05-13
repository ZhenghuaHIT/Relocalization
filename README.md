# Point_cloud_relocalization.
**Authors**: Zhenghua.Hou HIT 16S108281

**2017.11.9**: Finish pointcloud processing work.

**2017.11.18**: Finish point cloud and part pointcloud processing work.

# 1. License
Only Myself and My junior of laboratory.

# 2. Prerequisites
## C++11 or C++0x Compiler
I use the new thread and chrono functionalities of C++11.
## OpenCV
I use [OpenCV](http://opencv.org).
## PCL
I use [PCL](http://pointclouds.org).
## Eigen
I use [Eigen](http://eigen.tuxfamily.org).

# 3. Building 
```
cd hzh
mkdir build
cd build 
cmake ..
make 
```
# 4. Usage

```
./point_cloud_relocalization
```
We must change input gobal pointcloud name to **map.pcd** & Must change input current pointcloud name to **part_map.pcd**  
We must provide the initial value of registration  
We have provided a template for **.pcd**  

# 4. Result
**Actual Situation:** HIT Robot Institute 107 Room.  
 
![](https://github.com/ZhenghuaHIT/Relocalization/raw/master/images/p1.jpg)  

**Public Dataset:** [TUM RGBD Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset) Category: Robot SLAM.  
 
![](https://github.com/ZhenghuaHIT/Relocalization/raw/master/images/p2.jpg)  

