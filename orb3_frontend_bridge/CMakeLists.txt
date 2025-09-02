cmake_minimum_required(VERSION 3.10)
project(orb3_frontend_bridge)

find_package(catkin REQUIRED COMPONENTS
  roscpp std_msgs geometry_msgs message_generation
)

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs std_msgs)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)


add_message_files(
  FILES
  KeyPoint2D.msg
  KeyFrame.msg
)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)
catkin_package(CATKIN_DEPENDS roscpp std_msgs geometry_msgs message_runtime)

# Path to your ORB_SLAM3 checkout
set(ORB3_ROOT "$ENV{HOME}/dev/ORB_SLAM3")

# Headers
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ORB3_ROOT}
  ${ORB3_ROOT}/include
  ${ORB3_ROOT}/include/CameraModels
  ${ORB3_ROOT}/Thirdparty
  ${ORB3_ROOT}/Thirdparty/DBoW2
  ${ORB3_ROOT}/Thirdparty/Sophus
  ${ORB3_ROOT}/Thirdparty/g2o
  /usr/include/eigen3
  ${Pangolin_INCLUDE_DIRS} 
)

# >>> Point the linker at the correct lib directory <<<
link_directories(${ORB3_ROOT}/lib)

# (Even better: find the exact .so)
find_library(ORB3_LIB NAMES ORB_SLAM3
  PATHS ${ORB3_ROOT}/lib ${ORB3_ROOT}/build
  NO_DEFAULT_PATH)
if(NOT ORB3_LIB)
  message(FATAL_ERROR "Could not find libORB_SLAM3.so under ${ORB3_ROOT}/lib or build/")
endif()

catkin_package()

add_executable(bridge_node src/bridge_node.cpp)
target_compile_features(bridge_node PUBLIC cxx_std_17)
target_compile_options(bridge_node PRIVATE -Wno-error=deprecated-copy -Wno-deprecated-copy)

# Link with the found library
target_link_libraries(bridge_node ${Pangolin_LIBRARIES} ${ORB3_LIB} ${OPENGL_gl_LIBRARY} ${OPENGL_egl_LIBRARY}  ${catkin_LIBRARIES} ${OpenCV_LIBS} )

# Make runtime loader find the .so without env vars
set_target_properties(bridge_node PROPERTIES
  BUILD_RPATH "${ORB3_ROOT}/lib"
  INSTALL_RPATH "${ORB3_ROOT}/lib")
