
cmake_minimum_required (VERSION 2.8)
project(MSSD)
add_definitions(-std=c++11)
set( INSTALL_DIR ${TENGINE_DIR}/install/)
set( TENGINE_LIBS tengine hclcpu)

if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
endif()
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
endif()

if( BLAS_DIR)
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set( BLAS_LIB ${BLAS_DIR}/arm32/lib/libopenblas.so)
    endif()
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         set( BLAS_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
    endif()
endif()

if ( NOT (ARM OR ANDROID))
set (BLAS_LIB openblas)
endif()

set( CODE_SRCS mssd.cpp ../common/common.cpp)
set( CAPTRUE_SRCS src/mssd_cvCaptrue.cpp src/config.cpp ../common/common.cpp)
set( V4L2_SRCS src/mssd_V4L2.cpp src/config.cpp src/v4l2/v4l2.cpp ../common/common.cpp src/screen/screen.cpp)
set( DATA_SRCS src/captrue_data.cpp src/config.cpp src/v4l2/v4l2.cpp )
set( V4L2_KNN_SRCS src/mssd_V4L2_knn.cpp src/config.cpp src/v4l2/v4l2.cpp src/knn/knn.cpp ../common/common.cpp src/screen/screen.cpp)
set( IMG_SRCS src/mssd_img.cpp src/config.cpp ../common/common.cpp)
set( VIDEO_KNN_SRCS src/mssd_video_knn.cpp src/config.cpp src/knn/knn.cpp ../common/common.cpp)
set( VIDEO_SRCS src/mssd_video.cpp src/knn/knn.cpp src/config.cpp ../common/common.cpp)
set( CAP_VIDEO_SRCS src/captrue_video.cpp src/config.cpp src/v4l2/v4l2.cpp )
set( SCREEN_SHOW_SRCS src/screen_show.cpp src/config.cpp src/v4l2/v4l2.cpp src/screen/screen.cpp )
#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common )

#lib
link_directories( ${INSTALL_DIR}/lib)

#exe
add_executable(MSSD ${CODE_SRCS})
add_executable(CAPTRUE ${CAPTRUE_SRCS})
add_executable(V4L2 ${V4L2_SRCS})
add_executable(CAP_IMG ${DATA_SRCS})
add_executable(V4L2_KNN ${V4L2_KNN_SRCS})
add_executable(IMG ${IMG_SRCS})
add_executable(VIDEO_KNN ${VIDEO_KNN_SRCS})
add_executable(VIDEO ${VIDEO_SRCS})
add_executable(CAP_VIDEO ${CAP_VIDEO_SRCS})
add_executable(SCREEN_SHOW ${SCREEN_SHOW_SRCS})
if( ANDROID)
   target_link_libraries(MSSD ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(MSSD ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(CAPTRUE ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(V4L2 ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB} pthread)
   target_link_libraries(CAP_IMG ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB} pthread)
   target_link_libraries(V4L2_KNN ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB} pthread)
   target_link_libraries(IMG ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(VIDEO_KNN ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(VIDEO ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(CAP_VIDEO ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB} pthread)
   target_link_libraries(SCREEN_SHOW ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB}  pthread)
endif()


