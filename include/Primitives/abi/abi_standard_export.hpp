#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"
#include "param_string.hpp"
#include "class_factory.hpp"

#include <tuple>
#include <cstdint>
#include <utility>
#include <functional>
#include <type_traits>
#include <string_view>
#include <unordered_map>

namespace glasssix::exposing
{
	namespace details
	{
		/// <summary>
		/// A scoped static initializer.
		/// </summary>
		struct static_initializer
		{
			template<typename Callable, typename... Args>
			static_initializer(Callable&& handler)
			{
				std::forward<Callable>(handler)(std::forward<Args>(args)...);
			}
		};
		
		template<typename Tuple, typename = void>
		struct make_standard_export_functions_impl;

		template<typename... ComponentImpls>
		struct make_standard_export_functions_impl<std::tuple<ComponentImpls...>, std::enable_if_t<std::conjunction_v<impl::has_external_qualified_name<ComponentImpls>...>>>
		{
			struct class_factory_impl : implements<class_factory_impl, class_factory>
			{
				inline static const std::unordered_map<std::string_view, std::function<unknown_object()>> map;
				inline static const static_initializer initializer{ [&]
					{
						((map[impl::get_external_qualified_name_v<ComponentImpls>] = []() { return impl::to_abi<impl::first_interface_t<ComponentImpls>>(*new ComponentImpls); }), ...);
					}
				};

				unknown_object create_instance(const param_string& qualified_name)
				{
					auto iter = map.find(qualified_name);

					return iter != map.end() ? iter->second() : nullptr; 
				}
			};

			static std::int32_t G6_ABI_CALL dll_create_factory_impl(impl::abi_out_t<class_factory> factory) noexcept
			{
				if (factory == nullptr)
				{
					return error_null_pointer;
				}

				return (*factory = detach_abi(make<class_factory_impl>()), error_success);
			}

			static bool G6_ABI_CALL dll_can_unload_now() noexcept
			{

			}
		};
	}

	template<typename... ComponentImpls>
	struct make_standard_export_functions : details::make_standard_export_functions_impl<std::tuple<ComponentImpls...>>
	{
	};
}
