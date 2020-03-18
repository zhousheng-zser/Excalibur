#include "tensor_instantiation.hpp"
#include "jvm_runtime_info.hpp"
#include "tensor_cache_key.hpp"
#include "cache_key.hpp"

#include <tuple>
#include <cstdint>
#include <unordered_map>

using namespace glasssix::excalibur;

namespace glasssix::jni::utils
{
	namespace
	{
		std::unordered_map<jclass, tensor_instantiation> instantiation_cache;
	}

	/// <summary>
	/// Generates tensor constructors for T.
	/// </summary>
	template<typename T>
	struct tensor_constructors
	{
		static tensor_base* create_1(orderType layout)
		{
			return new tensor<T>{ layout };
		}

		static tensor_base* create_2(const int shape, int device, orderType layout)
		{
			return new tensor<T>{ shape, device, layout };
		}

		static tensor_base* create_3(const std::vector<int>& shape, int device, orderType layout)
		{
			return new tensor<T>{ shape, device, layout };
		}
	};

	/// <summary>
	/// Stores the constructors of tensor at compile-time.
	/// </summary>
	template<int Key, typename T>
	struct tensor_instantiation_traits
	{
		static constexpr int key = Key;
		static constexpr tensor_instantiation data
		{
			&tensor_constructors<T>::create_1,
			&tensor_constructors<T>::create_2,
			&tensor_constructors<T>::create_3
		};
	};

	void tensor_instantiation::initialize()
	{
		constexpr std::tuple<
			tensor_instantiation_traits<arg_enum_v<tensor_class_key::byte>, std::uint8_t>,
			tensor_instantiation_traits<arg_enum_v<tensor_class_key::integer>, jint>,
			tensor_instantiation_traits<arg_enum_v<tensor_class_key::short_integer>, jshort>,
			tensor_instantiation_traits<arg_enum_v<tensor_class_key::long_integer>, jlong>,
			tensor_instantiation_traits<arg_enum_v<tensor_class_key::single_float>, float>,
			tensor_instantiation_traits<arg_enum_v<tensor_class_key::double_float>, double>
		> metadata;

		std::apply([](auto&&... traits)
			{
				([&]
					{
						if (auto clazz = jvm_runtime_info::instance().get_class_cache(std::decay_t<decltype(traits)>::key))
						{
							instantiation_cache.emplace(clazz, std::decay_t<decltype(traits)>::data);
						}
					}(), ...);
			}, metadata);
	}

	std::optional<tensor_instantiation> tensor_instantiation::get_by_primitive(jclass type)
	{
		auto iter = instantiation_cache.find(type);

		return iter != instantiation_cache.end() ? std::make_optional(iter->second) : std::nullopt;
	}
}
