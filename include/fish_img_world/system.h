//system.h
#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <string.h>
#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <math.h>


namespace fisheye{
  
class System
{
public:
     System();
    ~System();
    
    void SolverFrame(const cv::Mat &img, cv::Point2d targetPoint);
    void undistortImage(cv::Mat &distorted, cv::Mat &undistorted,
        cv::Mat &K, cv::Mat &D,  cv::Mat &Knew, const cv::Size& new_size, cv::Point2d distorted_point);
    
    void undistortPoints(cv::Mat &distorted, cv::Mat &undistorted, 
			 cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P);
    void  initUndistortRectifyMap( cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P,
    const cv::Size& size, int m1type, cv::Mat &map1, cv::Mat &map2 );
    
    void calculateFromDistort( cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P,
    const cv::Size& size, int m1type, const cv::Point2d distorted_point, cv::Point2d &undistorted_point);
    
    inline void showImg(const cv::Mat &img, std::string windowName);
    inline void createShowImgWindows(std::string windowName);

    cv::Point2d point; 
    cv::Point2d distorted, undistorted_point;
    
public:
    double fx, fy, u0, v0;
    double k1, k2, k3, k4;
    cv::Mat new_cam_k, cam_k, cam_distort;
    
private:
    cv::Mat src_img, undistort_img;
    cv::Size img_size;
    
};

}
#endif
