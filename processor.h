#pragma once
#include <cpptorch/cpptorch.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <string>


class Processor
{
public:

	Processor(const std::string &net_file, const std::string &landmark_file);

	void setDebugWindowName(const std::string &name)
	{
		debug_window_ = name;
	}

	std::vector<cpptorch::Tensor<float>> processImage(const cv::Mat &frame_image);


protected:
	cpptorch::Tensor<float> runNN(const cv::Mat &image);

	dlib::frontal_face_detector detector_;
	dlib::shape_predictor sp_;
	std::shared_ptr<cpptorch::nn::Layer<float>> net_;
	cpptorch::Tensor<float> input_;

	std::string debug_window_;
};
