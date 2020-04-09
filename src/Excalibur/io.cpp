#include "io.hpp"

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		io::io()
		{
		}


		io::~io()
		{
		}

		void io::bytes2tensor(const unsigned char* bytes, int num, int channel, int height, int width,
			std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean, float scale)
		{
			tensor_data.reset(new tensor<float>(std::vector<int>{num, channel, height, width}, -1));
			float* float_data = tensor_data->mutable_cpu_data();
			int n_offset = channel * height * width;
			int c_offset = height * width;
			float mean[] = { 0.0f, 0.0f, 0.0f };
			if (minus_mean)
			{
				if (channel == 3)
				{
					mean[0] = 104.0f;
					mean[1] = 117.0f;
					mean[2] = 124.0f;
				}
				if (channel == 1)
				{
					mean[0] = 127.5f;
				}
			}
			// Lazily traverse, optimization required.
			for (int n = 0; n < num; n++)
			{
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < channel; c++)
						{
							float_data[n*n_offset + c*c_offset + h*width + w] =
								(static_cast<float>(bytes[n*n_offset + c*c_offset + h*width + w]) - mean[c]) * scale;
						}
					}
				}
			}
		}


		void io::bytes2tensor(const char* bytes, int num, int channel, int height, int width,
			std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean, float scale)
		{
			tensor_data.reset(new tensor<float>(std::vector<int>{num, channel, height, width}, -1));
			float* float_data = tensor_data->mutable_cpu_data();
			int n_offset = channel * height * width;
			int c_offset = height * width;
			float mean[] = { 0.0f, 0.0f, 0.0f };
			if (minus_mean)
			{
				if (channel == 3)
				{
					mean[0] = 104.0f;
					mean[1] = 117.0f;
					mean[2] = 124.0f;
				}
				if (channel == 1)
				{
					mean[0] = 127.5f;
				}
			}
			// Lazily traverse, optimization required.
			for (int n = 0; n < num; n++)
			{
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						for (int c = 0; c < channel; c++)
						{
							float_data[n*n_offset + c*c_offset + h*width + w] =
								(static_cast<float>(static_cast<unsigned char>(bytes[n*n_offset + c*c_offset + h*width + w])) - mean[c]) * scale;
						}
					}
				}
			}
		}


#ifdef USE_OPENCV
		void io::images2tensor(const std::vector<cv::Mat> images, std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean, float scale)
		{
			int num = images.size();
			int channel = 3;
			int width = images[0].cols;
			int height = images[0].rows;
			tensor_data.reset(new tensor<float>(std::vector<int>{num, channel, height, width}, -1));
			float* transformed_data = (tensor_data)->mutable_cpu_data();
			int batch_offset = channel*width*height;
			float mean[3];
			if (minus_mean)
			{
				mean[0] = 104.0f;
				mean[1] = 117.0f;
				mean[2] = 124.0f;
			}
			else
			{
				mean[0] = 0.0f;
				mean[1] = 0.0f;
				mean[2] = 0.0f;
			}
			for (int i = 0; i < images.size(); i++)
			{
				CHECK_EQ(images[0].size, images[i].size);
				if ((images[i].data == NULL) || (images[i].type() != CV_8UC3))
				{
					std::cout << "image's type is wrong!!Please set CV_8UC3" << std::endl;
					return;
				}
			}
			for (int rowI = 0; rowI < height; rowI++) {
				for (int colK = 0; colK < width; colK++) {
					for (int i = 0; i < images.size(); i++)
					{
						*(transformed_data + i*batch_offset) = (images[i].at<cv::Vec3b>(rowI, colK)[0] - mean[0])*scale;
						*(transformed_data + i*batch_offset + height*width) = (images[i].at<cv::Vec3b>(rowI, colK)[1] - mean[1])*scale;
						*(transformed_data + i*batch_offset + 2 * height*width) = (images[i].at<cv::Vec3b>(rowI, colK)[2] - mean[2])*scale;
					}
					transformed_data++;
				}
			}
			return;
		}

		void io::image2tensor(const cv::Mat image, std::shared_ptr<tensor<float>>& tensor_data, bool minus_mean, float scale)
		{
			int channel = 3;
			int width = image.cols;
			int height = image.rows;
			if ((image.data == NULL) || (image.type() != CV_8UC3)) {
				std::cout << "image's type is wrong!!Please set CV_8UC3" << std::endl;
				return;
			}
			/*if (*tensor_data!=nullptr)
			{
			delete *tensor_data;
			}*/
			tensor_data.reset(new tensor<float>(std::vector<int>{1, channel, height, width}, -1));
			float* transformed_data = (tensor_data)->mutable_cpu_data();
			float mean[3];
			if (minus_mean)
			{
				mean[0] = 104.0f;
				mean[1] = 117.0f;
				mean[2] = 124.0f;
			}
			else
			{
				mean[0] = 0.0f;
				mean[1] = 0.0f;
				mean[2] = 0.0f;
			}
			for (int rowI = 0; rowI < height; rowI++) {
				for (int colK = 0; colK < width; colK++) {
					*transformed_data = (image.at<cv::Vec3b>(rowI, colK)[0] - mean[0])*scale;
					*(transformed_data + height*width) = (image.at<cv::Vec3b>(rowI, colK)[1] - mean[1])*scale;
					*(transformed_data + 2 * height*width) = (image.at<cv::Vec3b>(rowI, colK)[2] - mean[2])*scale;
					transformed_data++;
				}
			}
			return;
		}
#endif

#ifdef CAFFEMODEL_SOPPORT
		void io::WriteProtoToTextFile(const Message& proto, const char* filename)
		{
			int fd = _open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
			FileOutputStream* output = new FileOutputStream(fd);
			google::protobuf::TextFormat::Print(proto, output);
			delete output;
			_close(fd);
		}

		bool io::ReadProtoFromBinaryFile(const char* file, Message* net)
		{
			int fd = _open(file, O_RDONLY | O_BINARY);
			if (fd == -1) return false;

			ZeroCopyInputStream* raw_input = new FileInputStream(fd);
			CodedInputStream* coded_input = new CodedInputStream(raw_input);
			bool success = net->ParseFromCodedStream(coded_input);
			delete coded_input;
			delete raw_input;
			_close(fd);
			return success;
		}

		bool io::readcaffemodel(const std::string modelpath, NetParameter& net)
		{
			bool success = ReadProtoFromBinaryFile(modelpath.c_str(), &net);
			if (!success) {
				printf("error:%s\n", modelpath.c_str());
			}
			return success;
		}

		std::vector<float*> io::readdataformcaffemodel(NetParameter net1, int id)
		{
			LayerParameter& param = *net1.mutable_layer(id);
			int n = net1.mutable_layer(id)->mutable_blobs()->size();
			std::vector<float*> output;
			if (n)
			{
				output = std::vector<float*>(n);
				const BlobProto& blob = param.blobs(0);
				output[0] = new float[blob.data_size()];
				memcpy(output[0], blob.data().data(), blob.data_size() * sizeof(float));
				//length.push_back(blob.data_size());
				if (n>1)
				{
					const BlobProto& bias = param.blobs(1);
					output[1] = new float[bias.data_size()];
					memcpy(output[1], bias.data().data(), bias.data_size() * sizeof(float));
					//length.push_back(bias.data_size());
				}
			}
			return output;
		}
#endif
	}
}
