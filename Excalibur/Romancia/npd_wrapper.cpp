#include "npd_wrapper.hpp"


namespace glasssix
{
	npd_wrapper::npd_wrapper(int device)
	{
		device_ = device;
		npd_ = new nplogic(device_);
		npd_->load();
	}

	npd_wrapper::~npd_wrapper()
	{
		delete npd_;
	}

	int npd_wrapper::facedetect_npd(unsigned char* image_data, int width, int height, int min_size)
	{
		return npd_->detect(image_data, width, height, min_size);
	}

	std::vector<int> npd_wrapper::get_x()
	{
		return npd_->getXs();
	}

	std::vector<int> npd_wrapper::get_y()
	{
		return npd_->getYs();
	}

	std::vector<int> npd_wrapper::get_size()
	{
		return npd_->getSs();
	}

	std::vector<float> npd_wrapper::get_score()
	{
		return npd_->getScores();
	}

}