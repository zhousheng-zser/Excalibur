#pragma once

#include "singleton.hpp"
#include "dllexport.hpp"
#include "exceptions.hpp"
#include "g6_attributes.hpp"
#include "class_factory.hpp"
#include "pure_c_handle_utils.h"
#include "platform_encoding.hpp"

#include <mutex>
#include <vector>
#include <memory>
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
	class in_process_component_loader final : public singleton<in_process_component_loader>
	{
	public:
		friend singleton;

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
						factory.get_qualified_names();

						std::lock_guard<std::mutex> guard{ lock_ };
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
		/// <returns>True if all the opeartions succeed; otherwise false</returns>
		bool add_module(std::initializer_list<utf8_string_view> paths) noexcept
		{
			bool success = false;

			for (auto& item : paths)
			{
				success = add_module(item);
			}

			return success;
		}
	private:
		std::mutex lock_;
		std::vector<std::shared_ptr<dll::dll_handle>> modules_;
		std::unordered_map<utf8_string, class_factory> factories_;
	};
}
