CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(FaceProject)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(dlib REQUIRED)

find_package(cpptorch REQUIRED)


#ADD_DEFINITIONS(-mavx)
ADD_DEFINITIONS(-msse4)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")


#add_executable(face_detection processor.cpp face_detection.cpp)
#target_link_libraries(face_detection ${OpenCV_LIBS} ${dlib_LIBRARIES} ${CPPTORCH_LIBRARIES})

add_executable(analyze_train_mov processor.cpp analyze_train_mov.cpp)
target_link_libraries(analyze_train_mov ${OpenCV_LIBS} ${dlib_LIBRARIES} ${CPPTORCH_LIBRARIES})

