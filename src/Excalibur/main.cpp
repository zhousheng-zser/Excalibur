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
	/*std::shared_ptr<memory::tensor<float>> t1, t2;
	t1.reset(new memory::tensor<float>(100, -1, memory::NCHW, all));
	t2.reset(new memory::tensor<float>(2, -1, memory::NCHW, all));

	auto t1d = t1->mutable_cpu_data();
	auto t2d = t2->mutable_cpu_data();*/

	profiler* pro = profiler::get();
	excalibur::pipeline<float> p = excalibur::pipeline<float>("D:\\Research\\Excalibur\\models\\retina.phai",
		"D:\\Research\\Excalibur\\models\\retina.racy");
	p.enable_profiler();
	//cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\yswvisible.jpg");
	cv::Mat img = cv::imread("C:\\Users\\Glasssix-Admin\\Desktop\\480p2.jpg");
	std::shared_ptr<memory::tensor<float>> temp;
	//temp.reset(new memory::tensor<float>(std::vector<int>{1,3,12,12 }));
	//cv::resize(img, img, cv::Size(320, 320));
	mat2tensor_cpu(img, temp, memory::NCHW, false);
	temp->set_allocator(all);
	/*for (size_t i = 0; i < temp->count(); i++)
	{
		std::cout << temp->cpu_data()[i] << " ";
	}*/
	auto res_vec = p.forward_cpu(temp);
	auto res = p.get_featmap("fc5");
	for (size_t i = 0; i < 10; i++)
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
		/*if (res_data[i] != 0)
		{
			std::cout << i << std::endl;
		}*/
		std::cout << res_data[i] << " ";
	}
	std::cout << std::endl;
	pro->turn_off();
	pro->dump_profile("D:\\retina_face.json");
	//pro->turn_off();
	return 0;
}
