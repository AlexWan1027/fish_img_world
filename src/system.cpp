#include "system.h"

namespace fisheye{

System::System():
fx(358.586), fy(358.886), u0(926.382), v0(555.110),
k1(-0.0602), k2(-0.0429), k3(0.0671), k4(-0.0394)

{
   
    createShowImgWindows("distort_img");
    createShowImgWindows("undistort_img");
    
    
}

System::~System()
{
    ROS_INFO("Destroying svo_node......");
}

void System::SolverFrame(const cv::Mat &img, cv::Point2d targetPoint)
{
    src_img = img.clone();
    img_size = src_img.size();
    
    
    cv::FileStorage fs("/home/alex/ros_ws/fisheye_ws/fisheye.yml", cv::FileStorage::READ);
    fs["cameraMatrix"] >> cam_k;
    fs[ "distCoeffs"] >> cam_distort;
    new_cam_k = cam_k.clone();

    
    new_cam_k.at<double>(0, 0) *= 1.2;
    new_cam_k.at<double>(1, 1) *= 1.2;   
    new_cam_k.at<double>(0, 2) += 800;
    new_cam_k.at<double>(1, 2) += 600;
    
    undistortImage(src_img, undistort_img, cam_k, cam_distort, new_cam_k, img_size, targetPoint);
    
    showImg(undistort_img, "undistort_img");
    return;
    
}

void System::undistortImage(cv::Mat &distorted, cv::Mat &undistorted,
        cv::Mat &K, cv::Mat &D,  cv::Mat &Knew, const cv::Size& new_size, cv::Point2d distorted_point)
{
    cv::Size size = new_size.area() != 0 ? new_size : distorted.size();

    cv::Mat map1, map2;
    cv::Mat PP1 = cv::Mat::eye(3,3,CV_32F);
    cv::Mat P;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, new_size,
							    PP1, P, 1);
/*    
    std::cout << "fx" << P.at<double>(0,0) << std::endl;
    std::cout << "fy" << P.at<double>(1,1) << std::endl;
    std::cout << "u0" << P.at<double>(0,2) << std::endl;
    std::cout << "v0" << P.at<double>(1,2) << std::endl;
    */
    cv::Mat org_points(1, 1, CV_64FC2);
    cv::Mat points(1, 4, CV_64FC2);
    cv::Vec2d* pptr = points.ptr<cv::Vec2d>();
    pptr[0] = cv::Vec2d(new_size.width/2, 0);
    pptr[1] = cv::Vec2d(new_size.width, new_size.height/2);
    pptr[2] = cv::Vec2d(new_size.width/2, new_size.height);
    pptr[3] = cv::Vec2d(0, new_size.height/2);
//     cv::Vec2d* pptr = org_points.ptr<cv::Vec2d>();
//     pptr[0] = cv::Vec2d(new_size.width/2.0, new_size.height/2.0);
// //     org_points.at<int>(0,0) = new_size.width/2.0;
// //     org_points.at<int>(0,1) = new_size.height/2.0;
    
    System::undistortPoints(points, points, K, D, PP1, P);
    
    
//     points.x = org_points.at<int>(0,0);
//     points.y = org_points.at<int>(0,1);
    point.x = points.at<double>(0,0);
    point.y = points.at<double>(0,1);
    
    cv::circle(distorted, point, 10, cv::Scalar(0, 0, 255), 2, 8, 0);
    showImg(distorted, "distort_img");
    
    cv::Mat PP2 = cv::Mat::eye(3,3,CV_32F);
    System::initUndistortRectifyMap(K, D, PP2, P, size, CV_32FC1, map1, map2 );
    cv::remap(distorted, undistorted, map1, map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
    System::calculateFromDistort(K, D, PP2, P, size, CV_16SC2, distorted_point, undistorted_point);
//     std::cout << "undistorted_point u:" << undistorted_point.x << std::endl;
//     std::cout << "undistorted_point v:" << undistorted_point.y << std::endl;
}

void System::calculateFromDistort( cv::Mat &distorted, cv::Mat &undistorted, cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P)
{
    // will support only 2-channel data now for points
    CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());

    CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));
    CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == cv::Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        cv::Matx33f camMat = K;
        f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        cv::Matx33d camMat = K;
        f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
    }

    cv::Vec4d k = D.depth() == CV_32F ? (cv::Vec4d)*D.ptr<cv::Vec4f>(): *D.ptr<cv::Vec4d>();

    cv::Matx33d RR = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == cv::Size(3, 3))
        R.convertTo(RR, CV_64F);

    if(!P.empty())
    {
        cv::Matx33d PP;
        P.colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }

    // start undistorting
    const cv::Vec2f* srcf = distorted.ptr<cv::Vec2f>();
    const cv::Vec2d* srcd = distorted.ptr<cv::Vec2d>();
    cv::Vec2f* dstf = undistorted.ptr<cv::Vec2f>();
    cv::Vec2d* dstd = undistorted.ptr<cv::Vec2d>();

    size_t n = distorted.total();
    int sdepth = distorted.depth();

    for(size_t i = 0; i < n; i++ )
    {
        cv::Vec2d pi = sdepth == CV_32F ? (cv::Vec2d)srcf[i] : srcd[i];  // image point
        cv::Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

        double scale = 1.0;

        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

        // the current camera model is only valid up to 180 FOV
        // for larger FOV the loop below does not converge
        // clip values so we still get plausible results for super fisheye images > 180 grad
        theta_d = std::min(std::max(-CV_PI/2., theta_d), CV_PI/2.);

        if (theta_d > 1e-8)
        {
            // compensate distortion iteratively
            double theta = theta_d;
            for(int j = 0; j < 10; j++ )
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                theta = theta_d / (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);
            }

            scale = std::tan(theta) / theta_d;
        }

        cv::Vec2d pu = pw * scale; //undistorted point

        // reproject
        cv::Vec3d pr = RR * cv::Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        cv::Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
}

void System::initUndistortRectifyMap( cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P,
    const cv::Size& size, int m1type, cv::Mat &map1, cv::Mat &map2 )
{
    CV_Assert( m1type == CV_16SC2 || m1type == CV_32F || m1type <=0 );
    map1.create( size, m1type <= 0 ? CV_16SC2 : m1type );
    map2.create( size, map1.type() == CV_16SC2 ? CV_16UC1 : CV_32F );

    CV_Assert((K.depth() == CV_32F || K.depth() == CV_64F) && (D.depth() == CV_32F || D.depth() == CV_64F));
    CV_Assert((P.depth() == CV_32F || P.depth() == CV_64F) && (R.depth() == CV_32F || R.depth() == CV_64F));
    CV_Assert(K.size() == cv::Size(3, 3) && (D.empty() || D.total() == 4));
    CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        cv::Matx33f camMat = K;
        f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        cv::Matx33d camMat = K;
        f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
    }

    cv::Vec4d k = cv::Vec4d::all(0);
    if (!D.empty())
        k = D.depth() == CV_32F ? (cv::Vec4d)*D.ptr<cv::Vec4f>(): *D.ptr<cv::Vec4d>();

    cv::Matx33d RR  = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == cv::Size(3, 3))
        R.convertTo(RR, CV_64F);

    cv::Matx33d PP = cv::Matx33d::eye();
    if (!P.empty())
        P.colRange(0, 3).convertTo(PP, CV_64F);

    cv::Matx33d iR = (PP * RR).inv(cv::DECOMP_SVD);

    for( int i = 0; i < size.height; ++i)
    {
        float* m1f = map1.ptr<float>(i);
        float* m2f = map2.ptr<float>(i);
        short*  m1 = (short*)m1f;
        ushort* m2 = (ushort*)m2f;

        double _x = i*iR(0, 1) + iR(0, 2),
               _y = i*iR(1, 1) + iR(1, 2),
               _w = i*iR(2, 1) + iR(2, 2);

        for( int j = 0; j < size.width; ++j)
        {
            double x = _x/_w, y = _y/_w;

            double r = sqrt(x*x + y*y);
            double theta = atan(r);

            double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta4*theta4;
            double theta_d = theta * (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8);

            double scale = (r == 0) ? 1.0 : theta_d / r;
            double u = f[0]*x*scale + c[0];
            double v = f[1]*y*scale + c[1];

            if( m1type == CV_16SC2 )
            {
                int iu = cv::saturate_cast<int>(u*cv::INTER_TAB_SIZE);
                int iv = cv::saturate_cast<int>(v*cv::INTER_TAB_SIZE);
                m1[j*2+0] = (short)(iu >> cv::INTER_BITS);
                m1[j*2+1] = (short)(iv >> cv::INTER_BITS);
                m2[j] = (ushort)((iv & (cv::INTER_TAB_SIZE-1))*cv::INTER_TAB_SIZE + (iu & (cv::INTER_TAB_SIZE-1)));
            }
            else if( m1type == CV_32FC1 )
            {
                m1f[j] = (float)u;
                m2f[j] = (float)v;
            }

            _x += iR(0, 0);
            _y += iR(1, 0);
            _w += iR(2, 0);
        }
    }
}
//输入： distorted points、 undistorted points、
void System::undistortPoints(cv::Mat &distorted, cv::Mat &undistorted, cv::Mat &K, cv::Mat &D, cv::Mat &R, cv::Mat &P)
{

    // will support only 2-channel data now for points
    CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());

    CV_Assert(P.empty() || P.size() == cv::Size(3, 3) || P.size() == cv::Size(4, 3));
    CV_Assert(R.empty() || R.size() == cv::Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == cv::Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        cv::Matx33f camMat = K;
        f = cv::Vec2f(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        cv::Matx33d camMat = K;
        f = cv::Vec2d(camMat(0, 0), camMat(1, 1));
        c = cv::Vec2d(camMat(0, 2), camMat(1, 2));
    }

    cv::Vec4d k = D.depth() == CV_32F ? (cv::Vec4d)*D.ptr<cv::Vec4f>(): *D.ptr<cv::Vec4d>();

    cv::Matx33d RR = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == cv::Size(3, 3))
        R.convertTo(RR, CV_64F);

    if(!P.empty())
    {
        cv::Matx33d PP;
        P.colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }

    // start undistorting
    const cv::Vec2f* srcf = distorted.ptr<cv::Vec2f>();
    const cv::Vec2d* srcd = distorted.ptr<cv::Vec2d>();
    cv::Vec2f* dstf = undistorted.ptr<cv::Vec2f>();
    cv::Vec2d* dstd = undistorted.ptr<cv::Vec2d>();

    size_t n = distorted.total();
    int sdepth = distorted.depth();

    for(size_t i = 0; i < n; i++ )
    {
        cv::Vec2d pi = sdepth == CV_32F ? (cv::Vec2d)srcf[i] : srcd[i];  // image point
        cv::Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

        double scale = 1.0;

        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

        // the current camera model is only valid up to 180 FOV
        // for larger FOV the loop below does not converge
        // clip values so we still get plausible results for super fisheye images > 180 grad
        theta_d = std::min(std::max(-CV_PI/2., theta_d), CV_PI/2.);

        if (theta_d > 1e-8)
        {
            // compensate distortion iteratively
            double theta = theta_d;
            for(int j = 0; j < 10; j++ )
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                theta = theta_d / (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);
            }

            scale = std::tan(theta) / theta_d;
        }

        cv::Vec2d pu = pw * scale; //undistorted point

        // reproject
        cv::Vec3d pr = RR * cv::Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        cv::Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
}


inline void System::showImg(const cv::Mat &img, std::string windowName)
{
    cv::imshow(windowName, img);
    cv::waitKey(1);
    return;
}

inline void System::createShowImgWindows(std::string windowName)
{
    cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    return;

}

}