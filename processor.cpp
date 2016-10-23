#include "processor.h"
#include <dlib/image_io.h>
#include <dlib/opencv.h>


float align_template[68][2] =
{
	{ 0.0792396913815f, 0.339223741112f }, { 0.0829219487236f, 0.456955367943f },
	{ 0.0967927109165f, 0.575648016728f }, { 0.122141515615f,  0.691921601066f },
	{ 0.168687863544f,  0.800341263616f }, { 0.239789390707f,  0.895732504778f },
	{ 0.325662452515f,  0.977068762493f }, { 0.422318282013f,  1.04329000149f  },
	{ 0.531777802068f,  1.06080371126f  }, { 0.641296298053f,  1.03981924107f  },
	{ 0.738105872266f,  0.972268833998f }, { 0.824444363295f,  0.889624082279f },
	{ 0.894792677532f,  0.792494155836f }, { 0.939395486253f,  0.681546643421f },
	{ 0.96111933829f,   0.562238253072f }, { 0.970579841181f,  0.441758925744f },
	{ 0.971193274221f,  0.322118743967f }, { 0.163846223133f,  0.249151738053f },
	{ 0.21780354657f,   0.204255863861f }, { 0.291299351124f,  0.192367318323f },
	{ 0.367460241458f,  0.203582210627f }, { 0.4392945113f,    0.233135599851f },
	{ 0.586445962425f,  0.228141644834f }, { 0.660152671635f,  0.195923841854f },
	{ 0.737466449096f,  0.182360984545f }, { 0.813236546239f,  0.192828009114f },
	{ 0.8707571886f,    0.235293377042f }, { 0.51534533827f,   0.31863546193f  },
	{ 0.516221448289f,  0.396200446263f }, { 0.517118861835f,  0.473797687758f },
	{ 0.51816430343f,   0.553157797772f }, { 0.433701156035f,  0.604054457668f },
	{ 0.475501237769f,  0.62076344024f  }, { 0.520712933176f,  0.634268222208f },
	{ 0.565874114041f,  0.618796581487f }, { 0.607054002672f,  0.60157671656f  },
	{ 0.252418718401f,  0.331052263829f }, { 0.298663015648f,  0.302646354002f },
	{ 0.355749724218f,  0.303020650651f }, { 0.403718978315f,  0.33867711083f  },
	{ 0.352507175597f,  0.349987615384f }, { 0.296791759886f,  0.350478978225f },
	{ 0.631326076346f,  0.334136672344f }, { 0.679073381078f,  0.29645404267f  },
	{ 0.73597236153f,   0.294721285802f }, { 0.782865376271f,  0.321305281656f },
	{ 0.740312274764f,  0.341849376713f }, { 0.68499850091f,   0.343734332172f },
	{ 0.353167761422f,  0.746189164237f }, { 0.414587777921f,  0.719053835073f },
	{ 0.477677654595f,  0.706835892494f }, { 0.522732900812f,  0.717092275768f },
	{ 0.569832064287f,  0.705414478982f }, { 0.635195811927f,  0.71565572516f  },
	{ 0.69951672331f,   0.739419187253f }, { 0.639447159575f,  0.805236879972f },
	{ 0.576410514055f,  0.835436670169f }, { 0.525398405766f,  0.841706377792f },
	{ 0.47641545769f,   0.837505914975f }, { 0.41379548902f,   0.810045601727f },
	{ 0.380084785646f,  0.749979603086f }, { 0.477955996282f,  0.74513234612f  },
	{ 0.523389793327f,  0.748924302636f }, { 0.571057789237f,  0.74332894691f  },
	{ 0.672409137852f,  0.744177032192f }, { 0.572539621444f,  0.776609286626f },
	{ 0.5240106503f,    0.783370783245f }, { 0.477561227414f,  0.778476346951f }
};

const int outer_eyes_and_nose_index[] = { 36, 45, 33 };
const int torch_img_dim = 96;

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))


static void normalize_template()
{
	auto minmax0 = std::minmax_element(align_template, align_template + COUNT_OF(align_template),
		[] (const float (&i)[2], const float (&j)[2]) -> bool { return i[0] < j[0]; });
	auto minmax1 = std::minmax_element(align_template, align_template + COUNT_OF(align_template),
		[] (const float (&i)[2], const float (&j)[2]) -> bool { return i[1] < j[1]; });

	float min[] = { (*minmax0.first)[0], (*minmax1.first)[1] };
	float base[] = { (*minmax0.second)[0] - (*minmax0.first)[0], (*minmax1.second)[1] - (*minmax1.first)[1] };
	std::for_each(align_template, align_template + COUNT_OF(align_template),
		[&min, &base] (float (&i)[2])
		{
			i[0] = (i[0] - min[0]) / base[0];
			i[1] = (i[1] - min[1]) / base[1];
		});
}


Processor::Processor(const std::string &net_file, const std::string &landmark_file)
{
	input_.create();
	input_.resize({ 1, 3, torch_img_dim, torch_img_dim });

	std::ifstream fs_net(net_file, std::ios::binary);
	assert(fs_net.good());
	auto obj_t = cpptorch::load(fs_net);
	net_ = cpptorch::read_net<float>(obj_t.get());

	detector_ = dlib::get_frontal_face_detector();
	dlib::deserialize(landmark_file) >> sp_;
	normalize_template();
}

cpptorch::Tensor<float> Processor::runNN(const cv::Mat &image)
{
	if (image.channels() != 3 || image.rows != torch_img_dim || image.cols != torch_img_dim)
	{
		std::cerr << "invalid size" << image.channels() << " " << image.rows << " " << image.cols << " " << std::endl;
		return cpptorch::Tensor<float>();
	}
	const unsigned char *img = image.ptr(0);

	float *ten = input_.data();
	for (size_t c = 0; c < 3; c++)
	{
		for (size_t p = 0; p < torch_img_dim * torch_img_dim; p++)
		{
			ten[c * torch_img_dim * torch_img_dim + p] = (float)img[p * 3 + 2 - c] / 255;
		}
	}

	return net_->forward(input_);
}

std::vector<cpptorch::Tensor<float>> Processor::processImage(const cv::Mat &frame_image)
{
	std::vector<cpptorch::Tensor<float>> ret;
	cv::Mat debug;
	if (!debug_window_.empty())
	{
		debug = frame_image;
	}

	// face detection
	dlib::cv_image<dlib::bgr_pixel> img_for_dlib(frame_image);

	// for each face
	std::vector<dlib::rectangle> dets = detector_(img_for_dlib);
	for (size_t i = 0; i < dets.size(); i++)
	{
		// face landmark detection
		dlib::full_object_detection shape = sp_(img_for_dlib, dets[i]);
		if (shape.num_parts() != COUNT_OF(align_template))
		{
			continue;
		}
		for (int i = 0; i < COUNT_OF(align_template); i++)
		{
			dlib::point pt = shape.part(i);
			std::cout << pt.x() << " " << pt.y() << " ";
		}
		std::cout << std::endl;

		cv::Point2f input_src[COUNT_OF(outer_eyes_and_nose_index)];
		cv::Point2f input_dst[COUNT_OF(outer_eyes_and_nose_index)];

		for (size_t j = 0; j < COUNT_OF(outer_eyes_and_nose_index); j++)
		{
			int idx = outer_eyes_and_nose_index[j];
			dlib::point pt = shape.part(idx);
			// prepare align data
			input_src[j] = cv::Point2f(pt.x(), pt.y());
			input_dst[j] = cv::Point2f(torch_img_dim * align_template[idx][0], torch_img_dim * align_template[idx][1]);
		}

		// align
		cv::Mat thumbnail;
		cv::Mat affine_trans = cv::getAffineTransform(input_src, input_dst);
		cv::warpAffine(frame_image, thumbnail, affine_trans, cv::Size(torch_img_dim, torch_img_dim));
		cpptorch::Tensor<float> output = runNN(thumbnail);
		if (output.valid())
		{
			ret.push_back(output);
		}

		// draw
		if (!debug_window_.empty())
		{
			dlib::rectangle &rect = dets[i];
			cv::rectangle(debug, cv::Rect(rect.left(), rect.top(), rect.width(), rect.height()), cv::Scalar(255, 0, 0), 3);

			for (size_t j = 0; j < COUNT_OF(outer_eyes_and_nose_index); j++)
			{
				cv::circle(debug, input_src[j], 3, cv::Scalar(0, 0, 255), -1);
			}
			thumbnail.copyTo(debug.rowRange(0, torch_img_dim).colRange(i * torch_img_dim, (i + 1) * torch_img_dim));
		}
	}

	if (!debug_window_.empty())
	{
		cv::imshow(debug_window_.c_str(), debug);
	}
	return ret;
}
