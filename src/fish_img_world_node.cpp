#include "system.h"
#include <fish_img_world/pose.h>

namespace fisheye{
class ImageGrabber
{
public:
    fisheye::System* mpFish;
    
    ImageGrabber(fisheye::System* pFish);
    ~ImageGrabber();

    void image_rect_callback(const sensor_msgs::ImageConstPtr& msg);

};

ImageGrabber::ImageGrabber(fisheye::System* pFish):mpFish(pFish){}

ImageGrabber::~ImageGrabber()
{
    delete mpFish;
    ROS_INFO("Destroying ImageGrabber......");
}

void ImageGrabber::image_rect_callback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat img;
    cv::Point2d target_point;
    
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
       cv_ptr=cv_bridge::toCvShare(msg, "bgr8");
       img = cv_ptr->image;
    } 
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    target_point.x = 20;
    target_point.y = 20;
    mpFish->SolverFrame(img, target_point);
}
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "fish_img_world_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it_(nh);
    image_transport::Subscriber image_rect_sub;
    ros::Publisher global_irobot_pose_pub;
    fisheye::System myfisheye;
    fisheye::ImageGrabber svo_node(&myfisheye);
    
    image_rect_sub = it_.subscribe("/usb_cam/image_raw", 1, &fisheye::ImageGrabber::image_rect_callback, &svo_node);
   // global_irobot_pose_pub = nh.advertise<fish_img_world::pose>("/global_irobot_detect/pose", 1);
    ros::Rate loop_rate(50);
    
    while (ros::ok())
    {
        ros::spinOnce();
        //fish_img_world::pose gloab_irobot;
	//global_irobot_pose_pub.publish(gloab_irobot);
	loop_rate.sleep();
    }
    printf("SVO Stop!!!\n");
    return 0;
}
