#include "romancia.hpp"
#include "banshee.hpp"

namespace glasssix
{
	namespace longinus
	{
		Romancia::Romancia(int device)
		{
			bansheelia_ = new Banshee(device);
		}

		Romancia::~Romancia()
		{
			delete bansheelia_;
		}

		void Romancia::Forward(const float* input_data, unsigned num, int order)
		{
			bansheelia_->Forward(input_data, num, order);
		}

		void Romancia::Forward(const unsigned char* input_data, unsigned num, int order)
		{
			bansheelia_->Forward(input_data, num, order);
		}

		void Romancia::getParam(std::vector<std::vector<float> > &keypointParam, unsigned num)
		{
			bansheelia_->getParam(keypointParam, num);
		}

		std::vector<unsigned char> Romancia::alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks)
		{
			return bansheelia_->alignFace(ori_image, n, channels, height, width, bbox, landmarks);
		}

	}
}