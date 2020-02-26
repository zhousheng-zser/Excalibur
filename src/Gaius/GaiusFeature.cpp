#include "GaiusFeature.hpp"
#include "unicorn_mobile.hpp"

#include <memory>

namespace glasssix
{
	namespace gaius
	{
		class GaiusFeature::impl
		{
		public:
			impl() : impl{ -1 }
			{
			}

			impl(int device) : mobile_unicornia_{ std::make_shared<Unicorn_mobile>(device) }
			{
			}

			virtual ~impl() = default;

			std::vector<std::vector<float>> Forward(const std::uint8_t* input_data, unsigned num, int order = 0) const
			{
				return mobile_unicornia_->Forward(input_data, num, order);
			}

			static const char* getVersion()
			{
#ifdef TRIAL
				return "Glasssix Trial FaceSDK";
#else
				return "Glasssix";
#endif // TRIAL
			}
		private:
			std::vector<std::vector<float>> Forward(const float* input_data, unsigned num, int order = 0) const
			{
				return mobile_unicornia_->Forward(input_data, num, order);
			}

			std::shared_ptr<Unicorn_mobile> mobile_unicornia_;
		};

		GaiusFeature::GaiusFeature() : impl_{ new impl }
		{
		}

		GaiusFeature::GaiusFeature(int device) : impl_{ new impl{ device } }
		{
		}

		GaiusFeature::~GaiusFeature()
		{
			if (impl_ != nullptr)
			{
				delete impl_;
				impl_ = nullptr;
			}
		}

		std::vector<std::vector<float>> GaiusFeature::Forward(const std::uint8_t* input_data, unsigned num, int order) const
		{
			return impl_->Forward(input_data, num, order);
		}

		const char* GaiusFeature::getVersion()
		{
			return impl::getVersion();
		}
	}
}
