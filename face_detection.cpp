#include "processor.h"

#include <unistd.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>


bool working = false;
cv::Mat current_frame_image;


void *process_thread(void *ptr)
{
	cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
	Processor p("nn4.small2.v1.t7", "shape_predictor_68_face_landmarks.dat");
	p.setDebugWindowName("debug");

	for (;;)
	{
		if (working)
		{
			auto begin = std::chrono::high_resolution_clock::now();

			cv::Mat small;
			cv::resize(current_frame_image, small, cv::Size(), 0.5, 0.5);

			p.processImage(small);

			auto end = std::chrono::high_resolution_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

			working = false;
		}
		else
		{
			usleep(1);
		}
	}

	return 0;
}


int main(int argc, char** argv)
{
	pthread_t worker;
	if (pthread_create(&worker, NULL, process_thread, NULL))
	{
		std::cout << "Cannot create thread" << std::endl;
		return -1;
	}

//	cv::VideoCapture cap("../train.mov");
	cv::VideoCapture cap("rtsp://admin:dreamer123@192.168.1.111:554/ISAPI/streaming/channels/101");

	if (!cap.isOpened())
	{
		std::cout << "Cannot open the video file" << std::endl;
		return -1;
	}

	while (cap.isOpened())
	{
		bool ret = cap.grab();
		if (!ret)
		{
			break;
		}

		if (!working)
		{
			if (cap.retrieve(current_frame_image))
			{
				working = true;
			}
		}

		cv::waitKey(1);
	}

	return 0;
}
