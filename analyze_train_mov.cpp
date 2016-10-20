#include "processor.h"

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>


int main(int argc, char** argv)
{
	if (argc != 4)
	{
		std::cout << "Usage: gen_train_data [movie file] [vector data file] [sampling gap]" << std::endl;
		return 1;
	}

	cv::VideoCapture cap(argv[1]);

	if (!cap.isOpened())
	{
		std::cerr << "Cannot open the video file" << std::endl;
		return 2;
	}

	std::ofstream outfile(argv[2], std::ofstream::binary);
	if (!outfile.good())
	{
		std::cerr << "Cannot open the output file" << std::endl;
		return 2;
	}

	cv::namedWindow("debug", cv::WINDOW_AUTOSIZE);
	Processor p("nn4.small2.v1.t7", "shape_predictor_68_face_landmarks.dat");
	p.setDebugWindowName("debug");

	cv::Mat current_frame_image;
	int each = (int)ceil(cap.get(CV_CAP_PROP_FRAME_COUNT) / 20);
	int i = 0;
	while (cap.isOpened())
	{
		bool ret = cap.read(current_frame_image);
		if (!ret)
		{
			break;
		}

		if (i++ % each == 1)
		{
			//cv::transpose(current_frame_image, current_frame_image);  
			//flip(current_frame_image, current_frame_image, 1);
			cv::imshow("debug", current_frame_image);

			std::vector<cpptorch::Tensor<float>> ret = p.processImage(current_frame_image);
			for (auto &item : ret)
			{
				outfile.write((char*)item.data(), sizeof(float) * 128);
			}

			std::cout << "======================== " << i << std::endl;
		}
		cv::waitKey(1);
	}
	outfile.close();

	return 0;
}
