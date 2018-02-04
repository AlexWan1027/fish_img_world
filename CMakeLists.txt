cmake_minimum_required(VERSION 2.8.3)
project(fish_img_world)

add_compile_options(-std=c++11)

SET(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
  image_transport
  message_generation
)

add_message_files(
   FILES
   pose.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
   CATKIN_DEPENDS message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${fish_img_world_SOURCE_DIR}/include/fish_img_world
)

add_executable(${PROJECT_NAME}_node src/system.cpp src/fish_img_world_node.cpp)

add_dependencies(${PROJECT_NAME}_node fish_img_world_generate_messages_cpp)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
)