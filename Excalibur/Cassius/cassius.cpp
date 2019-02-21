#include "cassius.hpp"
#include "unicornNet.hpp"


namespace glasssix
{
	namespace cassius
	{
		Cassius::Cassius(int device)
		{
			baseNet_ = new unicorn_net(device);
		}

		Cassius::~Cassius()
		{
			delete baseNet_;
		}

		std::vector<std::vector<float> > Cassius::Forward(const float* input_data, unsigned num, int order)
		{
			return baseNet_->Forward(input_data, num, order);
		}

		std::vector<std::vector<float> > Cassius::Forward(const unsigned char* input_data, unsigned num, int order)
		{
			return baseNet_->Forward(input_data, num, order);
		}
	}
}