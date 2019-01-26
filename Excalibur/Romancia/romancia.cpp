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
	}
}