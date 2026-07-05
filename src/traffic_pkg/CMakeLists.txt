cmake_minimum_required(VERSION 3.0.2)
project(traffic_pkg)

find_package(catkin REQUIRED COMPONENTS
  autohyu_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS autohyu_msgs rospy sensor_msgs std_msgs
)

# 파이썬 노드/리소스 설치
install(PROGRAMS
  script/inference_traffic.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  script/traffic_light_constants.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 런타임에서 쓰는 모델만 설치
install(FILES
  models/0608_stage1.pt
  models/0705_stage2.pt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
