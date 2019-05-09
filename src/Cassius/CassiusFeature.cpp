#include "CassiusFeature.hpp"
#include "unicorn.hpp"
#include "unicorn_mobile.hpp"


namespace glasssix
{
	namespace cassius
	{
		CassiusFeature::CassiusFeature(bool is_mobile, int device)
		{
			if (is_mobile)
			{
				unicornia_ = new Unicorn_mobile(device);
			}
			else
			{
				unicornia_ = new Unicorn(device);
			}			
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
	}
}