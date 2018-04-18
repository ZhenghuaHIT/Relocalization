# point_cloud_relocalization.
**Authors**: Zhenghua.Hou HIT 16S108281

**2017.11.9**: Finish pointcloud processing work.

**2017.11.18**: Finish pointcloud and part pointcloud processing work.

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
We must change input pointcloud name to **map.pcd** & Must change input part pointcloud name to **part_map.pcd**  
We must provide the initial value of registration  
We have provided a template for **.pcd**  

# 4. Result
**Actual Situation:** HIT Robot Institute 107 Room.  
![](http://i2.bvimg.com/641465/8d0523b8f7a28e87.jpg)  

**Public Dataset:** [TUM RGBD Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset) Category: Robot SLAM. 
![](http://i2.bvimg.com/641465/bece0d44f8ac0f51.png)  

