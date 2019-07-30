#include "damocles.hpp"
#include "mtcnn.hpp"


namespace glasssix
{
	namespace longinus
	{
		Damocles::Damocles(int device)
		{
			diodorus_ = new MTCNN(device);
		}

		Damocles::~Damocles()
		{
			delete diodorus_;
		};

		std::vector<FaceInfomation> Damocles::Detect(const unsigned char* img, const int channels, const int height, const int width,
			const int min_size, const float* threshold, const float factor, const int stage, int order) const
		{
			return diodorus_->Detect(img, channels, height, width, min_size, threshold, factor, stage, order);
		}
	}
}
