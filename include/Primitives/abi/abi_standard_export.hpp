#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"
#include "param_string.hpp"
#include "class_factory.hpp"
#include "g6_attributes.hpp"
#include "fundamental_semantics.hpp"

#include <tuple>
#include <atomic>
#include <cstdint>
#include <utility>
#include <functional>
#include <type_traits>
#include <string_view>
#include <unordered_map>

#ifdef _WIN32
#define EXPORT_DIRECTIVE_FOR_MAKE_ABI_STANDARD_EXPORT_FUNCTIONS __declspec(dllexport)
#else
#define EXPORT_DIRECTIVE_FOR_MAKE_ABI_STANDARD_EXPORT_FUNCTIONS
#endif

#define MAKE_ABI_STANDARD_EXPORT_FUNCTIONS(...) \
	extern "C" template EXPORT_DIRECTIVE_FOR_MAKE_ABI_STANDARD_EXPORT_FUNCTIONS std::int32_t G6_ABI_CALL glasssix::exposing::dll_create_factory<__VA_ARGS__>(glasssix::exposing::impl::abi_out_t<glasssix::exposing::class_factory> factory) noexcept; \
	extern "C" EXPORT_DIRECTIVE_FOR_MAKE_ABI_STANDARD_EXPORT_FUNCTIONS bool dll_can_unload_now() noexcept { return glasssix::exposing::get_module_ref_count() == 0; };

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
		struct make_standard_export_functions_impl<std::tuple<ComponentImpls...>, std::enable_if_t<std::conjunction_v<std::is_default_constructible<ComponentImpls>..., impl::has_external_qualified_name<ComponentImpls>...>>>
		{
			/// <summary>
			/// Implements a corresponding class factory.
			/// </summary>
			struct class_factory_impl : implements<class_factory_impl, class_factory>
			{
				inline static const std::unordered_map<std::string_view, std::function<unknown_object()>> map;
				inline static const static_initializer initializer{ [&]
					{
						((map[impl::get_external_qualified_name_v<ComponentImpls>] = []() { return impl::to_abi<impl::first_interface_t<ComponentImpls>>(*new ComponentImpls); }), ...);
					}
				};

				/// <summary>
				/// Creates an instance by a qualified name.
				/// </summary>
				/// <param name="qualified_name">The qualified name</param>
				/// <returns>The instance</returns>
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
		};
	}

	/// <summary>
	/// Makes DLL standard export functions for a couple of components.
	/// </summary>
	template<typename... ComponentImpls>
	struct make_standard_export_functions : details::make_standard_export_functions_impl<std::tuple<ComponentImpls...>>
	{
	};

	template<typename... ComponentImpls>
	std::int32_t G6_ABI_CALL dll_create_factory(impl::abi_out_t<class_factory> factory) noexcept
	{
		using impl_type = make_standard_export_functions<ComponentImpls...>;

		return impl_type::dll_create_factory_impl(factory);
	}
}
