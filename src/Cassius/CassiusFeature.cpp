#include "CassiusFeature.hpp"
#include "unicorn.hpp"

#include <memory>

namespace glasssix
{
	namespace cassius
	{
		class CassiusFeature::impl
		{

		public:

			impl() :impl{ -1 }
			{
			}

			impl(int device) : unicornia_{ std::make_shared<Unicorn>(device) }
			{
			}

			virtual ~impl() = default;

			std::vector<std::vector<float>> Forward(const std::uint8_t* input_data, int num, int order = 0) const
			{
				return unicornia_->Forward(input_data, num, order);
			}

			static const char* getVersion()
			{
#ifdef TRIAL
				return "Glasssix Trial FaceSDK";
#else
				return "Glasssix";
#endif
			}
		private:
			std::vector<std::vector<float>> Forward(const float* input_data, int num, int order = 0) const
			{
				return unicornia_->Forward(input_data, num, order);
			}

			std::shared_ptr<Unicorn> unicornia_;
		};

		CassiusFeature::CassiusFeature() : impl_{ new impl }
		{
		}

		CassiusFeature::CassiusFeature(int device) : impl_{ new impl{device} }
		{
		}

		CassiusFeature::~CassiusFeature()
		{
			if (impl_ != nullptr)
			{
				delete  impl_;
				impl_ = nullptr;
			}
		}

		std::vector<std::vector<float> > CassiusFeature::Forward(const unsigned char* input_data, int num, int order) const
		{
			return impl_->Forward(input_data, num, order);
		}

		const char* CassiusFeature::getVersion()
		{
			return impl::getVersion();
		}
	}
}
