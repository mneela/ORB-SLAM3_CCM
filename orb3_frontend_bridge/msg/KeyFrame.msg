std_msgs/Header header            # stamp=KF time, frame_id="map"
int32           map_id
int32           kf_id
geometry_msgs/Pose pose           # Twc

# Camera model (pinhole)
float32 fx
float32 fy
float32 cx
float32 cy

# Distortion (leave zero if unknown / none)
float32 k1
float32 k2
float32 p1
float32 p2
float32 k3

int32 img_cols
int32 img_rows

# Keypoints and descriptors
KeyPoint2D[] keypoints
uint8[]      descriptors          # row-major, 32 bytes per kp
int32        desc_stride          # bytes per descriptor (32)

