#include "GaiusFeature.hpp"
#include "unicorn_mobile.hpp"


namespace glasssix
{
	namespace gaius
	{
		GaiusFeature::GaiusFeature(int device)
		{
			mobile_unicornia_ = new Unicorn_mobile(device);
		}

		GaiusFeature::~GaiusFeature()
		{
			delete mobile_unicornia_;
		}

		std::vector<std::vector<float> > GaiusFeature::Forward(const float* input_data, unsigned num, int order) const
		{
			return mobile_unicornia_->Forward(input_data, num, order);
		}

		std::vector<std::vector<float> > GaiusFeature::Forward(const unsigned char* input_data, unsigned num, int order) const
		{
			return mobile_unicornia_->Forward(input_data, num, order);
		}
	}
}