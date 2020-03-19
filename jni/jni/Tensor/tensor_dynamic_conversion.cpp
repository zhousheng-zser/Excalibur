#include "tensor_dynamic_conversion.hpp"

#include <cstdint>
#include <type_traits>
#include <unordered_map>

#include <tensor_conversions.hpp>

using namespace glasssix::excalibur;

namespace glasssix::utils
{
	namespace
	{
		template<typename Functor>
		auto detect_tensor_real_type(const tensor_base& base, Functor&& handler)
		{
			if (auto real = dynamic_cast<const tensor<std::int8_t>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<int>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<short>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<long>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<std::int64_t>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<std::uint8_t>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<std::uint16_t>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<std::uint32_t>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<std::uint64_t>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<float>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			if (auto real = dynamic_cast<const tensor<double>&>(base))
			{
				return std::forward<Functor>(handler)(real);
			}

			return std::forward<Functor>(handler)();
		}
	}

	tensor_base* convert_tensor_image_layout(const tensor_base& source, tensor_layout layout)
	{
		return detect_tensor_real_type(source, [&](auto&& real)
			{
				switch (layout)
				{
				case tensor_layout::rgb:
					return std::forward<decltype(real)>(real) | tensor_convert_layout_to<tensor_layout::rgb>;
				case tensor_layout::rgba:
					return std::forward<decltype(real)>(real) | tensor_convert_layout_to<tensor_layout::rgb>;
				case tensor_layout::grayscale:
					break;
				case tensor_layout::grayscale_3:
					break;
				default:
					break;
				}
			});
	}
}
