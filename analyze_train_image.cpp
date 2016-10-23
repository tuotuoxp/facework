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
		std::cout << "Usage: gen_train_data [image file] [vector data file]" << std::endl;
		return 1;
	}

	Processor p("nn4.small2.v1.t7", "shape_predictor_68_face_landmarks.dat");
	p.setDebugWindowName("debug");

	cv::Mat image = imread(argv[1]);

	std::ofstream outfile(argv[2], std::ofstream::binary);
	if (!outfile.good())
	{
		std::cerr << "Cannot open the output file" << std::endl;
		return 2;
	}
	std::vector<cpptorch::Tensor<float>> ret = p.processImage(current_frame_image);
	for (auto &item : ret)
	{
		outfile.write((char*)item.data(), sizeof(float) * 128);
	}
	outfile.close();

	return 0;
}
