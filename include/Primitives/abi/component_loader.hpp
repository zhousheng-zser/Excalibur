#pragma once

#include "singleton.hpp"
#include "dllexport.hpp"
#include "exceptions.hpp"
#include "filesystem.hpp"
#include "g6_attributes.hpp"
#include "class_factory.hpp"
#include "pure_c_handle_utils.h"
#include "platform_encoding.hpp"

#include <mutex>
#include <vector>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <unordered_map>

namespace glasssix::exposing::dll
{
	DEFINE_PURE_C_HANDLE(dll);

	extern "C" EXPORT_EXCALIBUR_PRIMITIVES dll_handle G6_ABI_CALL load_library(const utf8_char * path) noexcept;
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL free_library(dll_handle handle) noexcept;
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES void* G6_ABI_CALL get_symbol_address(dll_handle handle, const utf8_char * name) noexcept;

	using dll_handle_ptr = std::shared_ptr<std::remove_pointer_t<dll_handle>>;
}

namespace glasssix::exposing::dll_routines
{
	using dll_can_unload_now_handler_type = bool(G6_ABI_CALL*)() noexcept;
	using dll_create_factory_handler_type = std::int32_t(G6_ABI_CALL*)(void** factory) noexcept;

	inline constexpr utf8_string_view dll_can_unload_now_handler_name{ u8"dll_can_unload_now" };
	inline constexpr utf8_string_view dll_create_factory_handler_name{ u8"dll_create_factory" };
}

namespace glasssix::exposing
{
	/// <summary>
	/// A manager for in-process components (that is a component from within a DLL).
	/// </summary>
	class component_loader final : public singleton<component_loader>
	{
	public:
		friend singleton<component_loader>;

		/// <summary>
		/// Adds a module.
		/// </summary>
		/// <param name="path">The path</param>
		/// <returns>True if the opeartion succeeds; otherwise false</returns>
		bool add_module(utf8_string_view path) noexcept
		{
			if (dll::dll_handle_ptr handle{ dll::load_library(path.data()), &dll::free_library })
			{
				if (auto dll_create_factory = static_cast<dll_routines::dll_create_factory_handler_type>(dll::get_symbol_address(handle.get(), dll_routines::dll_create_factory_handler_name.data())))
				{
					if (class_factory factory{ nullptr }; dll_create_factory(put_abi(factory)) == error_success)
					{
						auto names = factory.get_qualified_names();
						{
							std::lock_guard<std::mutex> guard{ lock_ };

							modules_.emplace_back(handle);
							factories_.insert_or_assign(factory.get_component_name(), factory);
						}

						return true;
					}

					return false;
				}

				return false;
			}

			return false;
		}

		/// <summary>
		/// Adds a few modules.
		/// </summary>
		/// <param name="paths">The paths</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::size_t add_modules(std::initializer_list<utf8_string_view> paths) noexcept
		{
			return std::count_if(paths.begin(), paths.end(), [&](utf8_string_view path) { return add_module(path); });
		}

		/// <summary>
		/// Finds modules in a directory
		/// </summary>
		/// <param name="directory">The directory</param>
		/// <param name="recursive">Indicates whether to find modules recursively</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::size_t add_modules_in_directory(const fs::path& directory, bool recursive = false) noexcept
		{
			if (recursive)
			{
				return add_modules_in_directory_impl<true>(directory);
			}
			else
			{
				return add_modules_in_directory_impl<false>(directory);
			}
		}

		/// <summary>
		/// Lookups a class factory by qualified name.
		/// </summary>
		/// <param name="qualified_name">The qualified name</param>
		/// <returns>The class factory</returns>
		class_factory lookup(utf8_string_view qualified_name) const noexcept
		{
			auto iter = factories_.find(qualified_name);

			return iter != factories_.end() ? iter->second : nullptr;
		}
	private:
		template<bool recursive>
		std::size_t add_modules_in_directory_impl(const fs::path& directory) noexcept
		{
			using iterator_type = std::conditional_t<recursive, fs::recursive_directory_iterator, fs::directory_iterator>;

			std::error_code code;
			iterator_type iter_end;
			iterator_type iter_begin{ directory, fs::directory_options::skip_permission_denied, code };
			
			return std::count_if(iter_begin, iter_end, [&](const fs::directory_entry& item) { return item.path().has_extension() && item.path().extension() == ".dll" ? add_module(item.path().u8string()) : false; });
		}

		std::mutex lock_;
		std::vector<dll::dll_handle_ptr> modules_;
		std::unordered_map<param_string, class_factory> factories_;
	};
}
