#include <ros/ros.h>
#include <string>
#include <opencv2/core.hpp>

// ORB-SLAM3
#include "System.h"
#include "LocalMapping.h"
#include "KeyFrame.h"


#include <geometry_msgs/Pose.h>
#include <orb3_frontend_bridge/KeyFrame.h>
#include <orb3_frontend_bridge/KeyPoint2D.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

// Global (or capture in a struct)
static int   g_img_cols = 0, g_img_rows = 0;
static float g_k1 = 0.f, g_k2 = 0.f, g_p1 = 0.f, g_p2 = 0.f, g_k3 = 0.f;

static bool FileExists(const std::string& p){ std::ifstream f(p); return f.good(); }

static bool LoadCameraParamsFromYaml(const std::string& settings_path) {

if (settings_path.empty() || !FileExists(settings_path)) {
    ROS_ERROR_STREAM("Settings YAML missing or unreadable: '" << settings_path << "'");
    return false;
  }
  try {
    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
      ROS_ERROR_STREAM("cv::FileStorage could not open: " << settings_path);
      return false;
    }

  // These keys match ORB_SLAM3 example YAMLs. Adjust if your YAML differs.
  fs["Camera.width"]  >> g_img_cols;
  fs["Camera.height"] >> g_img_rows;

  // Distortion is optional; if missing, stay at zero.
  fs["Camera.k1"] >> g_k1; fs["Camera.k2"] >> g_k2;
  fs["Camera.p1"] >> g_p1; fs["Camera.p2"] >> g_p2;
  fs["Camera.k3"] >> g_k3;

  // If keys are missing, cv::FileStorage leaves them unchanged; make sure ints are sane.
  if(g_img_cols <= 0 || g_img_rows <= 0) {
    ROS_WARN("Camera.width/height not found; publishing 0 for image size.");
    g_img_cols = g_img_rows = 0;
  }
  return true;
 } 
 catch (const cv::Exception& e) {
    ROS_ERROR_STREAM("OpenCV exception reading settings: " << e.what());
    return false;
  }
}




static geometry_msgs::Pose ToPoseMsgTwc(const Sophus::SE3f& Tcw) {
  const Sophus::SE3f Twc = Tcw.inverse();
  Eigen::Quaternionf q = Twc.unit_quaternion();
  const Eigen::Vector3f t = Twc.translation();

  geometry_msgs::Pose p;
  p.position.x = t.x(); p.position.y = t.y(); p.position.z = t.z();
  p.orientation.x = q.x(); p.orientation.y = q.y();
  p.orientation.z = q.z(); p.orientation.w = q.w();
  return p;
}

static void FillIntrinsics(ORB_SLAM3::KeyFrame* kf,
                           float& fx, float& fy, float& cx, float& cy,
                           float& k1, float& k2, float& p1, float& p2, float& k3,
                           int& cols, int& rows)
{
  // from KeyFrame (these exist)
  fx = kf->fx; fy = kf->fy; cx = kf->cx; cy = kf->cy;

  // from settings YAML (cached); if not present they stay zero (undistorted case)
  k1 = g_k1; k2 = g_k2; p1 = g_p1; p2 = g_p2; k3 = g_k3;

  // image size from YAML (KeyFrame doesn't store it)
  cols = g_img_cols; rows = g_img_rows;
}


static void ExportKeypointsAndDescriptors(ORB_SLAM3::KeyFrame* kf,
    std::vector<orb3_frontend_bridge::KeyPoint2D>& out_kpts,
    std::vector<uint8_t>& out_desc,
    int& desc_stride)
{
  using KptMsg = orb3_frontend_bridge::KeyPoint2D;

  // Keypoints
  std::vector<cv::KeyPoint> ks;
  kf->GetUndistortedKeypoints(ks);
  out_kpts.resize(ks.size());
  for (size_t i = 0; i < ks.size(); ++i) {
    const cv::KeyPoint& kp = ks[i];
    KptMsg m; m.u = kp.pt.x; m.v = kp.pt.y; m.octave = kp.octave; m.angle = kp.angle; m.response = kp.response;
    out_kpts[i] = std::move(m);
  }

  // Descriptors
  cv::Mat D = kf->GetDescriptorsCopy();       // CV_8U, cols ~ 32
  desc_stride = D.cols > 0 ? D.cols : 32;      // default 32 if unknown
  out_desc.resize(static_cast<size_t>(D.rows * D.cols));
  if (D.data) {
    if (D.isContinuous()) {
      std::memcpy(out_desc.data(), D.ptr(), out_desc.size());
    } else {
      for (int r = 0; r < D.rows; ++r)
        std::memcpy(&out_desc[static_cast<size_t>(r) * D.cols], D.ptr(r), D.cols);
    }
  }
}



int main(int argc, char** argv) {
  ros::init(argc, argv, "bridge_node");
  ros::NodeHandle nh("~");

  // --- 1) Read params ---
  std::string vocab, settings;
  int sensor_mode = 0;
  nh.param<std::string>("vocab",    vocab,    std::string());
  nh.param<std::string>("settings", settings, std::string());
  nh.param<int>("sensor_mode", sensor_mode, 0);

  //nh.param<std::string>("vocab",    vocab,    std::string(getenv("HOME")) + "/dev/ORB_SLAM3/Vocabulary/ORBvoc.txt");
  //nh.param<std::string>("settings", settings, std::string(getenv("HOME")) + "/dev/ORB_SLAM3/Examples/Monocular/TUM1.yaml");

  if (vocab.empty())   { ROS_FATAL("Param '~vocab' is empty");   return 1; }
  if (settings.empty()){ ROS_FATAL("Param '~settings' is empty"); return 1; }
  if (!FileExists(vocab))    { ROS_FATAL_STREAM("Vocab not found: " << vocab); return 1; }
  if (!LoadCameraParamsFromYaml(settings)) { ROS_FATAL("Failed to load settings"); return 1; }

  // --- 2) Choose sensor enum ---
  ORB_SLAM3::System::eSensor eS = ORB_SLAM3::System::MONOCULAR;
  if (sensor_mode == 1) eS = ORB_SLAM3::System::STEREO;
  if (sensor_mode == 2) eS = ORB_SLAM3::System::RGBD;

  // --- 3) Construct SLAM ONCE ---
  ORB_SLAM3::System slam(vocab, settings, eS, /*useViewer=*/false);

  // --- 4) Publisher BEFORE lambda ---
  ros::Publisher pub_kf =
      nh.advertise<orb3_frontend_bridge::KeyFrame>("keyframe_out", 10, false);

  // --- 5) Get LocalMapping and register callback ---
  ORB_SLAM3::LocalMapping* lm = slam.GetLocalMapping();
  if (!lm) { ROS_FATAL("GetLocalMapping() returned null"); return 1; }

  lm->onKeyFrameInserted = [pub_kf](ORB_SLAM3::KeyFrame* kf) {
    if (!kf) return;

    orb3_frontend_bridge::KeyFrame msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "map";
    msg.map_id = kf->GetMap()->GetId();
    msg.kf_id  = kf->mnId;

    // Pose (Twc)
    const Sophus::SE3f Tcw = kf->GetPose();   // thread-safe getter
    msg.pose = ToPoseMsgTwc(Tcw);

    // Intrinsics + image size from YAML cache
    FillIntrinsics(kf, msg.fx, msg.fy, msg.cx, msg.cy,
                      msg.k1, msg.k2, msg.p1, msg.p2, msg.k3,
                      msg.img_cols, msg.img_rows);

    // Keypoints + descriptors via safe getters you added to KeyFrame
    int stride = 0;
    ExportKeypointsAndDescriptors(kf, msg.keypoints, msg.descriptors, stride);
    msg.desc_stride = stride;

    pub_kf.publish(msg);
  };

  // --- 6) Spin (or add your image subscribers / dataset loop) ---
  ros::spin();
  slam.Shutdown();
  return 0;
}



