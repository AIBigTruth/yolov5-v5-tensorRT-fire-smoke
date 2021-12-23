cmake_minimum_required(VERSION 2.6)

project(yolov5-google-video)
set(OpenCV_DIR "D:\\opencv345\\build") #2
set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}\\include) #3
set(OpenCV_LIB_DIRS ${OpenCV_DIR}\\x64\\vc15\\lib) #4
set(OpenCV_Debug_LIBS "opencv_world345d.lib") #5
set(OpenCV_Release_LIBS "opencv_world345.lib") #6
set(TRT_DIR "D:\\TensorRT7.2.1.6-cuda11.1-cudnn8.0") #3
set(TRT_INCLUDE_DIRS ${TRT_DIR}\\include) #8
set(TRT_LIB_DIRS ${TRT_DIR}\\lib) #9
set(Dirent_INCLUDE_DIRS "F:\\github\\tensorrtx-v5\\include") #10

add_definitions(-DAPI_EXPORTS)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

enable_language(CUDA) # add this line, then no need to setup cuda path in vs

include_directories(${PROJECT_SOURCE_DIR}\\include) #11
include_directories(${TRT_INCLUDE_DIRS}) #12
link_directories(${TRT_LIB_DIRS}) #13
include_directories(${OpenCV_INCLUDE_DIRS}) #14
link_directories(${OpenCV_LIB_DIRS}) #15
include_directories(${Dirent_INCLUDE_DIRS}) #16

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")


find_package(OpenCV QUIET
NO_MODULE
NO_DEFAULT_PATH
NO_CMAKE_PATH
NO_CMAKE_ENVIRONMENT_PATH
NO_SYSTEM_ENVIRONMENT_PATH
NO_CMAKE_PACKAGE_REGISTRY
NO_CMAKE_BUILDS_PATH
NO_CMAKE_SYSTEM_PATH
NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " lib path: ${OpenCV_LIB_DIRS}")
message(STATUS " Debug libraries: ${OpenCV_Debug_LIBS}")
message(STATUS " Release libraries: ${OpenCV_Release_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp ${PROJECT_SOURCE_DIR}/common.hpp ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/yololayer.h) #17

target_link_libraries(yolov5 "nvinfer" "nvinfer_plugin") #18
target_link_libraries(yolov5 debug ${OpenCV_Debug_LIBS}) #19
target_link_libraries(yolov5 optimized ${OpenCV_Release_LIBS}) #20
target_link_libraries(yolov5 ${CUDA_LIBRARIES}) #21
target_link_libraries(yolov5 Threads::Threads)