#include "ImageReader.hpp"

namespace excalibur
{
	ImageReader::ImageReader()
	{
	}


	ImageReader::~ImageReader()
	{
	}

#ifdef USE_OPENCV
	void ImageReader::image2tensor(const cv::Mat image, std::shared_ptr<ImageTensor<unsigned char>>& tensor_data)
	{
		int channel = 1;
		int width = image.cols;
		int height = image.rows;
		if ((image.data == NULL) || (image.type() != CV_8UC1)) {
			std::cout << "image's type is wrong!!Please set CV_8UC1" << std::endl;
			return;
		}
		tensor_data.reset(new ImageTensor<unsigned char>(std::vector<int>{1, channel, height, width}, -1));
		unsigned char* transformed_data = (tensor_data)->mutable_cpu_data();
		//transformed_data = image.data;
		for (int rowI = 0; rowI < height; rowI++) {
			for (int colK = 0; colK < width; colK++) {
				*transformed_data = image.at<unsigned char>(rowI, colK);
				transformed_data++;
			}
		}
		return;
	}

	void ImageReader::images2tensor(const std::vector<cv::Mat> images, std::shared_ptr<ImageTensor<unsigned char>>& tensor_data)
	{
		int num = images.size();
		int channel = 1;
		int width = images[0].cols;
		int height = images[0].rows;
		tensor_data.reset(new ImageTensor<unsigned char>(std::vector<int>{num, channel, height, width}, -1));
		unsigned char* transformed_data = (tensor_data)->mutable_cpu_data();
		int batch_offset = channel*width*height;
		for (int i = 0; i < images.size(); i++)
		{
			CHECK_EQ(images[0].size, images[i].size);
			if ((images[i].data == NULL) || (images[i].type() != CV_8UC1))
			{
				std::cout << "image's type is wrong!!Please set CV_8UC1" << std::endl;
				return;
			}
		}
		for (int rowI = 0; rowI < height; rowI++) {
			for (int colK = 0; colK < width; colK++) {
				for (int i = 0; i < images.size(); i++)
				{
					*(transformed_data + i*batch_offset) = images[i].at<unsigned char>(rowI, colK);
				}
				transformed_data++;
			}
		}
		return;
	}

	void ImageReader::tensor2image(const std::shared_ptr<ImageTensor<unsigned char>> tensor_data, cv::Mat& image)
	{
		CHECK_EQ(tensor_data->channels(), 1);
		image = cv::Mat(tensor_data->height(), tensor_data->width(), CV_8UC1, (unsigned char *)tensor_data->cpu_data());
		return;
	}

	void ImageReader::tensor2images(const std::shared_ptr<ImageTensor<unsigned char>> tensor_data, std::vector<cv::Mat>& images)
	{
		CHECK_EQ(tensor_data->channels(), 1);
		images = std::vector<cv::Mat>(tensor_data->num());
		int width = tensor_data->width();
		int height = tensor_data->height();
		int offset = width * height;
		for (int i = 0; i < images.size(); i++)
		{
			images[i] = cv::Mat(height, width, CV_8UC1, (unsigned char *)(tensor_data->cpu_data() + offset * i));
		}
		return;
	}


#endif
}


