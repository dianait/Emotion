cmake_minimum_required(VERSION 2.8.3)
project(jinko_games)

find_package(catkin REQUIRED COMPONENTS
  rospy
  jinko_games_message
)

catkin_package(
 CATKIN_DEPENDS rospy
 jinko_games_message

)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
