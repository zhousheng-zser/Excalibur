#include "./include/Romancia.hpp"
#include "./include/landmarkNet.hpp"



namespace glasssix
{
	namespace longinus
	{
		Romancia::Romancia(int device)
		{
			baseNet_ = new LandmarkNet(device);
		}

		Romancia::~Romancia()
		{
			delete baseNet_;
		}

		void Romancia::Forward(const float* input_data, unsigned num)
		{
			baseNet_->Forward(input_data, num);
		}

		void Romancia::Forward(const unsigned char* input_data, unsigned num)
		{
			baseNet_->Forward(input_data, num);
		}

		void Romancia::getParam(std::vector<std::vector<float> > &keypointParam, unsigned num)
		{
			baseNet_->getParam(keypointParam, num);
		}

		std::vector<unsigned char> Romancia::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks)
		{
			return baseNet_->alignFace(ori_image, n, channels, height, width, bbox, landmarks);
		}
	}
}