#!/bin/sh


yum install -y cmake unzip gcc gcc-c++

############# ffmpeg #############
yum install -y nasm
curl --output /tmp/ffmpeg.tar.gz -L https://github.com/FFmpeg/FFmpeg/archive/n3.1.3.tar.gz
tar zxvf /tmp/ffmpeg.tar.gz -C /tmp/
cd /tmp/FFmpeg-*
./configure --prefix=/usr/local --enable-shared
make
make install


############# opencv #############
curl --output /tmp/opencv.zip -L https://github.com/Itseez/opencv/archive/2.4.13.zip
unzip /tmp/opencv.zip -d /tmp/
rm -f /tmp/opencv.zip
cd /tmp/opencv-*
mkdir build
cd build

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/
cmake -D CMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
make install


############# openblas ###############
curl --output /tmp/openblas.tar.gz -L https://github.com/xianyi/OpenBLAS/archive/v0.2.18.tar.gz
tar zxvf /tmp/openblas.tar.gz -C /tmp/
cd /tmp/OpenBLAS-*
make USE_OPENMP=1 USE_THREADS=0
make install PREFIX=/usr/local


############# dlib ###############
curl --output /tmp/dlib.tar.gz -L https://github.com/davisking/dlib/archive/v19.0.tar.gz
tar zxvf /tmp/dlib.tar.gz -C /tmp/
cd /tmp/dlib-*
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
make install

