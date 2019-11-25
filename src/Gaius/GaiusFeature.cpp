#include "GaiusFeature.hpp"
#include "unicorn_mobile.hpp"


namespace glasssix
{
	namespace gaius
	{
		GaiusFeature::GaiusFeature(int device)
		{
			mobile_unicornia_.reset(new Unicorn_mobile(device));
		}

		GaiusFeature::~GaiusFeature()
		{
		}

		std::vector<std::vector<float> > GaiusFeature::Forward(const float* input_data, unsigned num, int order) const
		{
			return mobile_unicornia_->Forward(input_data, num, order);
		}

		std::vector<std::vector<float> > GaiusFeature::Forward(const unsigned char* input_data, unsigned num, int order) const
		{
			return mobile_unicornia_->Forward(input_data, num, order);
		}

		std::string GaiusFeature::getVersion()
		{
#ifdef TRIAL
			return std::string("Glasssix Trial FaceSDK");
#else
			return std::string("Glasssix");
#endif // TRIAL	
		}
	}
}