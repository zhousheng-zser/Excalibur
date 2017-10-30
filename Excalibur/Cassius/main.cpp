#include <iostream>
#include "../Excalibur/io.hpp"
#include "unicorn_net.hpp"

#include <filesystem>

using namespace excalibur;
using namespace glasssix;

int main()
{
	cv::Mat image = cv::imread("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\0.jpg");
	std::shared_ptr<tensor> tensor_data = nullptr;
	io::images2tensor(std::vector<cv::Mat>{image}, tensor_data);
	unicorn_net unicorn = unicorn_net(-1);
	unicorn.Forward(tensor_data);
	std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
	for (size_t i = 0; i < 10; i++)
	{
		unicorn.Forward(tensor_data);
	}
	std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
	std::cout << "total forward time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;

	auto a = unicorn.get_pool5();
	const float* output = a->cpu_data();
	for (int i = 0; i<a->count()/1 ; i++)
	{
		std::cout << output[i] << " ";
	}

	return 0;
}