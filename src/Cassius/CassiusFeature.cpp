#include "CassiusFeature.hpp"
#include "unicorn.hpp"

namespace glasssix
{
	namespace cassius
	{
		CassiusFeature::CassiusFeature(int device)
		{
			unicornia_ = new Unicorn(device);
		}

		CassiusFeature::~CassiusFeature()
		{
			delete unicornia_;
		}

		std::vector<std::vector<float> > CassiusFeature::Forward(const float* input_data, unsigned num, int order) const
		{
			return unicornia_->Forward(input_data, num, order);
		}

		std::vector<std::vector<float> > CassiusFeature::Forward(const unsigned char* input_data, unsigned num, int order) const
		{
			return unicornia_->Forward(input_data, num, order);
		}

		std::string CassiusFeature::getVersion()
		{
#ifdef TRIAL
			return std::string("Glasssix Trial FaceSDK");
#else
			return std::string("Glasssix");
#endif // TRIAL	
		}
	}
}