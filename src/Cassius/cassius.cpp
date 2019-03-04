#include "cassius.hpp"
#include "Unicorn.hpp"


namespace glasssix
{
	namespace cassius
	{
		Cassius::Cassius(int device)
		{
			unicornia_ = new Unicorn(device);
		}

		Cassius::~Cassius()
		{
			delete unicornia_;
		}

		std::vector<std::vector<float> > Cassius::Forward(const float* input_data, unsigned num, int order)
		{
			return unicornia_->Forward(input_data, num, order);
		}

		std::vector<std::vector<float> > Cassius::Forward(const unsigned char* input_data, unsigned num, int order)
		{
			return unicornia_->Forward(input_data, num, order);
		}
	}
}