#include "Damocles.hpp"
#include "mtcnn.hpp"


namespace glasssix
{
	namespace longinus
	{
		Damocles::Damocles(int device)
		{
			baseCNN_ = new MTCNN(device);
		}

		Damocles::~Damocles()
		{
			delete baseCNN_;
		};

		std::vector<FaceInfoX> Damocles::Detect(const unsigned char* img, const int channels, const int height, const int width,
			const int min_size, const float* threshold, const float factor, const int stage)
		{
			return baseCNN_->Detect(img, channels, height, width, min_size, threshold, factor, stage);
		}
	}
}