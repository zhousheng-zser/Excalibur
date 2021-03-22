#include "../../include/Excalibur/pipeline.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "opencv2\opencv.hpp"
#include "../../include/Primitives/profiler.hpp"

using namespace glasssix;

template <typename Dtype>
static void mat2tensor_cpu(const cv::Mat &srcu, std::shared_ptr<memory::tensor<Dtype>>& dst, memory::orderType order = memory::NCHW, bool bgr2rgb = false)
{
	if (srcu.data == NULL)
	{
		LOG(ERROR) << "No data.";
		return;
	}

	int channels = srcu.channels();
	int width = srcu.cols;
	int height = srcu.rows;
	cv::Mat src;
	srcu.convertTo(src, CV_32F);
	/*int type_id = src.type() % 8;
	auto type_name = std::string(typeid(Dtype).name());*/

	if (order == memory::NCHW)
	{
		dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, -1, memory::NCHW));
		Dtype* dst_data = dst->mutable_cpu_data();
		int dst_offset = width * height;
		int* c_dst_offset = new int[channels];

		for (int c = 0; c < channels; c++)
		{
			if (bgr2rgb)
			{
				c_dst_offset[c] = (channels - 1 - c) * dst_offset;
			}
			else
			{
				c_dst_offset[c] = c * dst_offset;
			}
		}

		for (int c = 0; c < channels; c++)
		{
			for (int h = 0; h < height; h++)
			{
				const Dtype* src_data = src.ptr<Dtype>(h);
				int dst_sub_offset = h * width;

				for (int w = 0; w < width; w++)
				{
					dst_data[c_dst_offset[c] + dst_sub_offset + w] =
						src_data[w * channels + c];
				}
			}
		}

		delete[] c_dst_offset;
	}
	else if (order == memory::NHWC)
	{
		dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, -1, memory::NHWC));
		Dtype* dst_data = dst->mutable_cpu_data();
		memcpy(dst_data, src.data, height * width * channels * sizeof(Dtype));
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}

int main()
{
	auto all = new memory::pool_allocator<float>();
	profiler* pro = profiler::get();
	excalibur::pipeline<float> p = excalibur::pipeline<float>("..\\..\\models\\unicorn.phai",
		"..\\..\\models\\unicorn.racy");
	//excalibur::pipeline<float> p = excalibur::pipeline<float>("D:\\Excalibur\\models\\unicorn_int8.phai",
	//"D:\\Excalibur\\models\\unicorn_int8.racy", -1);


	/*excalibur::pipeline<float> p = excalibur::pipeline<float>("D:\\Excalibur\\models\\mobile_unicorn.phai",
		"D:\\Excalibur\\models\\mobile_unicorn.racy", -1);*/


	p.enable_profiler(); 
	cv::Mat img = cv::imread("..\\..\\images\\yswinfread.jpg");
	//cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\480p2.jpg");
	std::shared_ptr<memory::tensor<float>> temp;
	//cv::resize(img, img, cv::Size(320, 320));
	mat2tensor_cpu(img, temp, memory::NCHW, false);
	temp->set_allocator(all);
	auto res_vec = p.forward_cpu(temp);
	auto res = p.get_featmap("conv5_dw");
	for (size_t i = 0; i < 20; i++)
	{
		p.forward_cpu(temp);
	}
	const float* res_data = res->cpu_data();
	int count = res->count() >= 100 ? 100 : res->count();
	for (size_t i = res->count() - 100; i < res->count(); i++)
	{
		if (i % 25 == 0)
		{
			std::cout << std::endl;
		}
		std::cout << res_data[i] << " ";
	}
	std::cout << std::endl;
	//pro->turn_off();
	pro->DumpProfile("D:\\3.json");
	//pro->turn_off();


	return 0;
}
