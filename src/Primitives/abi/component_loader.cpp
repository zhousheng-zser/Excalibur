#include "abi/component_loader.hpp"
#include "abi/platform_encoding.hpp"
#include "filesystem.hpp"

#include <list>
#include <regex>
#include <mutex>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#else
#error "Unspported platform."
#endif

namespace glasssix::exposing::dll
{
	using dll_handle_ptr = std::shared_ptr<std::remove_pointer_t<dll_handle>>;

	EXPORT_EXCALIBUR_PRIMITIVES dll_handle G6_ABI_CALL load_library(const utf8_char* path) noexcept
	{
#ifdef _WIN32
		return path ? pure_c::to_handle<dll_handle>(LoadLibraryA(platform_encoding::utf8_to_narrow(path).c_str())) : nullptr;
#else
		return path ? pure_c::to_handle<dll_handle>(dlopen(platform_encoding::utf8_to_narrow(path).c_str(), RTLD_NOW)) : nullptr;
#endif
	}

	EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL free_library(dll_handle handle) noexcept
	{
		if (handle)
		{
#ifdef _WIN32
			FreeLibrary(pure_c::from_handle<std::remove_pointer_t<HMODULE>>(handle));
#else
			dlclose(pure_c::from_handle<void>(handle));
#endif
		}
	}

	EXPORT_EXCALIBUR_PRIMITIVES symbol_func_ptr G6_ABI_CALL get_symbol_address(dll_handle handle, const utf8_char* name) noexcept
	{
		if (handle == nullptr || name == nullptr)
		{
			return nullptr;
		}

#ifdef _WIN32
		return reinterpret_cast<symbol_func_ptr>(GetProcAddress(pure_c::from_handle<std::remove_pointer_t<HMODULE>>(handle), platform_encoding::utf8_to_narrow(name).c_str()));
#else
		return reinterpret_cast<symbol_func_ptr>(dlsym(pure_c::from_handle<void>(handle), platform_encoding::utf8_to_narrow(name).c_str()));
#endif
	}
}

namespace glasssix::exposing
{
	namespace
	{
#ifdef _WIN32
		constexpr utf8_string_view dll_extension{ u8".dll" };
#elif defined(__linux__)
		constexpr utf8_string_view dll_extension{ u8".so" };
#else
#error "Unsupported Platform."
#endif

		param_string force_unix_conventional_library_file_name(utf8_string_view name)
		{
			static constexpr utf8_string_view file_name_prefix{ u8"lib" };

#ifdef _WIN32
			static constexpr auto regex_flags = std::regex_constants::ECMAScript | std::regex_constants::icase;
#else
			static constexpr auto regex_flags = std::regex_constants::ECMAScript;
#endif
			thread_local std::basic_regex<utf8_char> pattern{ format(FMT_STRING(u8R"(({})?(.+?)(\.{})?)"), file_name_prefix, dll_extension), regex_flags };

			if (std::cmatch matches; std::regex_match(name.data(), name.data() + name.size(), matches, pattern))
			{
				return format(
					u8"{}{}{}",
					matches[1].matched ? param_string{} : param_string{ file_name_prefix },
					matches[2].str(),
					matches[3].matched ? param_string{} : param_string{ dll_extension });
			}

			return name;
		}
	}
}

namespace glasssix::exposing::impl
{
	/// <summary>
	/// A manager for in-process components (that is a component from within a DLL).
	/// </summary>
	class component_loader_impl final : public implements<component_loader_impl, component_loader>
	{
	public:
		/// <summary>
		/// Adds a module.
		/// </summary>
		/// <param name="path">The path</param>
		/// <returns>True if the opeartion succeeds; otherwise false</returns>
		bool add_module(const param_string& path)
		{
			return add_module_with_factory(path);
		}

		/// <summary>
		/// Adds a module and returns the class factory.
		/// </summary>
		/// <param name="path">The path</param>
		/// <returns>The class factory</returns>
		class_factory add_module_with_factory(const param_string& path)
		{
			std::error_code code;

			// Checks whether the DLL has been already loaded.
			if (auto factory = try_get_existing_factory(path))
			{
				return factory;
			}

			if (dll::dll_handle_ptr handle{ dll::load_library(path.data()), &dll::free_library })
			{
				if (auto dll_create_factory = reinterpret_cast<dll_routines::dll_create_factory_handler_type>(dll::get_symbol_address(handle.get(), dll_routines::dll_create_factory_handler_name.data())))
				{
					if (class_factory factory{ nullptr }; dll_create_factory(put_abi(factory)) == error_success)
					{
						return (parse_metadata(path, handle, factory), factory);
					}

					return nullptr;
				}

				return nullptr;
			}

			return nullptr;
		}

		/// <summary>
		/// Adds a few modules.
		/// </summary>
		/// <param name="paths">The paths</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::uint64_t add_modules(param_span<param_string> paths)
		{
			return std::count_if(paths.begin(), paths.end(), [&](utf8_string_view inner) { return add_module(inner); });
		}

		/// <summary>
		/// Adds a few modules.
		/// </summary>
		/// <param name="paths">The paths</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::uint64_t add_modules(const param_vector<param_string>& paths)
		{
			return std::count_if(begin(paths), end(paths), [&](utf8_string_view inner) { return add_module(inner); });
		}

		/// <summary>
		/// Adds a few modules and returns the available class factories.
		/// </summary>
		/// <param name="paths">The paths</param>
		/// <returns>The class factories of the successfully loaded modules</returns>
		param_hash_map<param_string, class_factory> add_modules_with_factories(param_span<param_string> paths)
		{
			return add_modules_with_factories_impl(paths, std::bind(&component_loader_impl::add_module_with_factory, this, std::placeholders::_1));
		}

		/// <summary>
		/// Adds a few modules and returns the available class factories.
		/// </summary>
		/// <param name="paths">The paths</param>
		/// <returns>The class factories of the successfully loaded modules</returns>
		param_hash_map<param_string, class_factory> add_modules_with_factories(const param_vector<param_string>& paths)
		{
			return add_modules_with_factories_impl(paths, std::bind(&component_loader_impl::add_module_with_factory, this, std::placeholders::_1));
		}

		/// <summary>
		/// Finds modules in a directory and adds them.
		/// </summary>
		/// <param name="directory">The directory</param>
		/// <param name="recursive">Indicates whether to find modules recursively</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::uint64_t add_modules_in_directory(const param_string& directory, bool recursive)
		{
			auto handler = [this](std::uint64_t& result, const fs::path& item) { if (add_module(to_param_string(item.string()))) { result++; } };

			if (recursive)
			{
				return for_each_dll_files<true>(directory, handler, std::uint64_t{});
			}
			else
			{
				return for_each_dll_files<false>(directory, handler, std::uint64_t{});
			}
		}

		/// <summary>
		/// Finds modules in a directory, adds them and returns the available class factories.
		/// </summary>
		/// <param name="directory">The directory</param>
		/// <param name="recursive">Indicates whether to find modules recursively</param>
		/// <returns>The class factories of the successfully loaded modules</returns>
		param_hash_map<param_string, class_factory> add_modules_with_factories_in_directory(const param_string& directory, bool recursive)
		{
			auto handler = [this](param_hash_map<param_string, class_factory>& result, const fs::path& item) { if (auto factory = add_module_with_factory(to_param_string(item.string()))) { result.add_or_update(factory.library_name(), factory); } };

			if (recursive)
			{
				return for_each_dll_files<true>(directory, handler, make_param_hash_map<param_string, class_factory>());
			}
			else
			{
				return for_each_dll_files<false>(directory, handler, make_param_hash_map<param_string, class_factory>());
			}
		}

		/// <summary>
		/// Adds a module by name.
		/// </summary>
		/// <param name="path">The conventional library name</param>
		/// <returns>True if the opeartion succeeds; otherwise false</returns>
		bool add_module_by_name(const param_string& name)
		{
			return add_module(force_unix_conventional_library_file_name(name));
		}

		/// <summary>
		/// Adds a module by name and returns the class factory.
		/// </summary>
		/// <param name="name">The conventional library name</param>
		/// <returns>The class factory</returns>
		class_factory add_module_by_name_with_factory(const param_string& name)
		{
			return add_module_with_factory(force_unix_conventional_library_file_name(name));
		}

		/// <summary>
		/// Adds a few modules by name.
		/// </summary>
		/// <param name="names">The conventional names</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::uint64_t add_modules_by_name(param_span<param_string> names)
		{
			return std::count_if(names.begin(), names.end(), [&](utf8_string_view inner) { return add_module_by_name(inner); });
		}

		/// <summary>
		/// Adds a few modules by name.
		/// </summary>
		/// <param name="names">The conventional names</param>
		/// <returns>The count of successfully loaded modules</returns>
		std::uint64_t add_modules_by_name(const param_vector<param_string>& names)
		{
			return std::count_if(begin(names), end(names), [&](utf8_string_view inner) { return add_module_by_name(inner); });
		}

		/// <summary>
		/// Adds a few modules by name and returns the available class factories.
		/// </summary>
		/// <param name="names">The conventional names</param>
		/// <returns>The class factories of the successfully loaded modules</returns>
		param_hash_map<param_string, class_factory> add_modules_with_factories_by_name(param_span<param_string> names)
		{
			return add_modules_with_factories_impl(names, std::bind(&component_loader_impl::add_module_by_name_with_factory, this, std::placeholders::_1));
		}

		/// <summary>
		/// Adds a few modules by name and returns the available class factories.
		/// </summary>
		/// <param name="names">The conventional names</param>
		/// <returns>The class factories of the successfully loaded modules</returns>
		param_hash_map<param_string, class_factory> add_modules_with_factories_by_name(const param_vector<param_string>& names)
		{
			return add_modules_with_factories_impl(names, std::bind(&component_loader_impl::add_module_by_name_with_factory, this, std::placeholders::_1));
		}

		/// <summary>
		/// Lookups a class factory by library name.
		/// </summary>
		/// <param name="library_name">The library name</param>
		/// <returns>The class factory</returns>
		class_factory lookup_factory(const param_string& library_name)
		{
			std::scoped_lock lock{ mutex_ };
			auto iter = name_factory_map_.find(library_name);

			return iter != name_factory_map_.end() ? iter->second : nullptr;
		}

		/// <summary>
		/// Retrieves the loaded library names.
		/// </summary>
		/// <returns>The loaded library names</returns>
		param_vector<param_string> library_names()
		{
			auto result = make_param_vector<param_string>();
			{
				std::scoped_lock lock{ mutex_ };

				for (const auto& item : name_factory_map_)
				{
					result.push_back(item.first);
				}

				return result;
			}
		}

		/// <summary>
		/// Retrieves the loaded factories.
		/// </summary>
		/// <returns>The factories</returns>
		param_hash_map<param_string, class_factory> factories()
		{
			auto result = make_param_hash_map<param_string, class_factory>();
			{
				std::scoped_lock lock{ mutex_ };

				for (const auto& [key, value] : name_factory_map_)
				{
					result.add_or_update(key, value);
				}

				return result;
			}
		}

		/// <summary>
		/// Checks whether an external qualified name exists.
		/// </summary>
		/// <param name="qualified_name">The qualified name</param>
		/// <returns>True if it exists; otherwise false</returns>
		bool contains_qualified_name(const param_string& qualified_name)
		{
			std::scoped_lock lock{ mutex_ };

			return qualified_name_activator_map_.find(qualified_name) != qualified_name_activator_map_.end();
		}

		/// <summary>
		/// Checks whether an interface ID exists.
		/// </summary>
		/// <param name="qualified_name">The interface ID</param>
		/// <returns>True if it exists; otherwise false</returns>
		bool contains_interface_id(const guid& interface_id)
		{
			std::scoped_lock lock{ mutex_ };

			return interface_id_activator_map_.find(interface_id) != interface_id_activator_map_.end();
		}

		/// <summary>
		/// Creates an instance by external qualified name.
		/// </summary>
		/// <param name="qualified_name">The qualified name</param>
		/// <returns>The object</returns>
		unknown_object create_by_name(const param_string& qualified_name)
		{
			auto handler = [&]
			{
				std::scoped_lock lock{ mutex_ };
				auto iter = qualified_name_activator_map_.find(qualified_name);

				return iter != qualified_name_activator_map_.end() ? iter->second : std::function<unknown_object()>{};
			}();

			return handler ? handler() : throw abi_no_interface{ format(u8"Failed to create an instance by qualified name: {}.", to_param_string(qualified_name)) };
		}

		/// <summary>
		/// Creates an instance by first interface ID.
		/// </summary>
		/// <param name="qualified_name">The first interface ID</param>
		/// <returns>The object</returns>
		unknown_object create_by_interface_id(const guid& interface_id)
		{
			auto handler = [&]
			{
				std::scoped_lock lock{ mutex_ };
				auto iter = interface_id_activator_map_.find(interface_id);

				return iter != interface_id_activator_map_.end() ? iter->second : std::function<unknown_object()>{};
			}();

			return handler ? handler() : throw abi_no_interface{ format(u8"Failed to create an instance by interface ID: {}.", to_param_string(interface_id)) };
		}
	private:
		template<template<typename> typename Container, typename Callable>
		param_hash_map<param_string, class_factory> add_modules_with_factories_impl(const Container<param_string>& paths, Callable&& callable)
		{
			auto result = make_param_hash_map<param_string, class_factory>();

			for (const auto& item : paths)
			{
				if (auto factory = callable(item))
				{
					result.add_or_update(factory.library_name(), factory);
				}
			}

			return result;
		}

		template<bool Recursive, typename Result, typename Callable>
		Result for_each_dll_files(utf8_string_view directory, Callable&& handler, Result&& initial_value)
		{
			using iterator_type = std::conditional_t<Recursive, fs::recursive_directory_iterator, fs::directory_iterator>;

			std::error_code code;
			Result result{ std::forward<Result>(initial_value) };

			for (auto& item : iterator_type{ to_narrow_string(directory), fs::directory_options::skip_permission_denied, code })
			{
				if (item.path().has_extension() && case_insensitive_path_comare(item.path().extension(), dll_extension))
				{
					std::forward<Callable>(handler)(result, item.path());
				}
			}

			return result;
		}

		class_factory try_get_existing_factory(utf8_string_view path)
		{
			std::error_code code;
			std::scoped_lock lock{ mutex_ };

			if (auto iter_pair = std::find_if(modules_.begin(), modules_.end(), [&](const std::pair<fs::path, dll::dll_handle_ptr>& inner) { return fs::equivalent(inner.first, to_narrow_string(path), code); }); iter_pair != modules_.end())
			{
				if (auto iter = handle_factory_map_.find(iter_pair->second); iter != handle_factory_map_.end())
				{
					return iter->second;
				}

				modules_.erase(iter_pair);
			}

			return nullptr;
		}

		void parse_metadata(utf8_string_view path, const dll::dll_handle_ptr& handle, const class_factory& factory)
		{
			auto names = factory.qualified_names();
			auto library_name = factory.library_name();
			auto interface_ids = factory.interface_ids();
			{
				std::scoped_lock lock{ mutex_ };

				modules_.emplace_back(to_narrow_string(path), handle);
				name_factory_map_.insert_or_assign(library_name, factory);
				std::for_each(exposing::begin(names), exposing::end(names), [&](const param_string& inner) { qualified_name_activator_map_.insert_or_assign(inner, [=] { return factory.create_by_name(inner); }); });
				std::for_each(exposing::begin(interface_ids), exposing::end(interface_ids), [&](const guid& inner) { interface_id_activator_map_.insert_or_assign(inner, [=] { return factory.create_by_interface_id(inner); }); });
			}
		}

		std::mutex mutex_;
		std::list<std::pair<fs::path, dll::dll_handle_ptr>> modules_;
		std::unordered_map<param_string, class_factory> name_factory_map_;
		std::unordered_map<dll::dll_handle_ptr, class_factory> handle_factory_map_;
		std::unordered_map<guid, std::function<unknown_object()>> interface_id_activator_map_;
		std::unordered_map<param_string, std::function<unknown_object()>> qualified_name_activator_map_;
	};
}

namespace glasssix::exposing
{
	EXPORT_EXCALIBUR_PRIMITIVES void* component_loader_add_ref_get_singleton_abi()
	{
		static component_loader singleton{ make_as_first<impl::component_loader_impl>() };
		auto abi = get_abi(singleton);

		return (static_cast<impl::abi_unknown_object*>(abi)->add_ref(), abi);
	}
}
