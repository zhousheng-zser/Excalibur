#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"
#include "dllexport.hpp"
#include "g6_attributes.hpp"
#include "platform_encoding.hpp"
#include "pure_c_handle_utils.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <iterator>
#include <type_traits>
#include <string_view>

namespace glasssix::exposing
{
	class param_string;
}

namespace glasssix::exposing::allocations
{
	DEFINE_PURE_C_HANDLE(param_string);

	extern "C" param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL create_param_string(const utf8_char* str, std::size_t size) noexcept;
	extern "C" param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL create_param_string_from_narrow(const char* narrow_str, std::size_t size) noexcept;
	extern "C" param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL duplicate_param_string(param_string_handle str) noexcept;
	extern "C" param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL concat_param_string(param_string_handle left, param_string_handle right) noexcept;
	extern "C" void EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL free_param_string(param_string_handle str) noexcept;
	extern "C" const utf8_char* EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL get_param_string_data(param_string_handle str) noexcept;
	extern "C" std::size_t EXPORT_EXCALIBUR_PRIMITIVES get_param_string_size(param_string_handle str) noexcept;
}

namespace glasssix::exposing::impl
{
	/// <summary>
	/// The ABI of a param_string.
	/// </summary>
	template<> struct abi<param_string>
	{
		using identity_type = type_identity_primitive;
		using type = void*;

		static constexpr guid id{ "47534958-7061-7261-0605-737472696E67" };
	};
}

namespace glasssix::exposing
{
	/// <summary>
	/// An ABI-safe string for parameters.
	/// </summary>
	class param_string
	{
	public:
		using value_type = utf8_char;
		using view_type = utf8_string_view;
		using const_iterator = const value_type*;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		/// <summary>
		/// Creates an instance.
		/// </summary>
		param_string() noexcept : handle_{}
		{
		}

		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="str">The string</param>
		param_string(const value_type* str) noexcept : param_string{ view_type{ str } }
		{
		}

		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="str">The string</param>
		param_string(view_type str) noexcept : param_string{ str.data(), str.size() }
		{
		}

		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="str">The string</param>
		/// <param name="size">The size</param>
		param_string(const value_type* str, std::size_t size) noexcept : handle_{ allocations::create_param_string(str, size) }
		{
		}

		param_string(const param_string& other) noexcept : handle_{ allocations::duplicate_param_string(other.handle_) }
		{
		}

		param_string(param_string&& other) noexcept : handle_{ std::exchange(other.handle_, nullptr) }
		{
		}

		~param_string() noexcept
		{
			clear();
		}

		param_string& operator=(const param_string& right) noexcept
		{
			clear();
			handle_ = allocations::duplicate_param_string(right.handle_);

			return *this;
		}

		param_string& operator=(param_string&& right) noexcept
		{
			clear();
			handle_ = std::exchange(right.handle_, nullptr);

			return *this;
		}

		/// <summary>
		/// Provides access to certain element.
		/// </summary>
		/// <param name="index">The index</param>
		/// <returns>A const reference to the element</returns>
		const value_type& operator[](std::size_t index) const noexcept
		{
			return *(data() + index);
		}

		/// <summary>
		/// Supports casting to std::basic_string_view.
		/// </summary>
		/// <returns>The result</returns>
		operator view_type() const noexcept
		{
			return view_type{ data(), size() };
		}

		/// <summary>
		/// Clears the string.
		/// </summary>
		void clear() noexcept
		{
			if (handle_)
			{
				allocations::free_param_string(handle_);
				handle_ = nullptr;
			}
		}

		/// <summary>
		/// Gets the data of the string.
		/// </summary>
		/// <returns>The data</returns>
		const value_type* data() const noexcept
		{
			return allocations::get_param_string_data(handle_);
		}

		/// <summary>
		/// Gets the size of the string.
		/// </summary>
		/// <returns>The size</returns>
		std::size_t size() const noexcept
		{
			return allocations::get_param_string_size(handle_);
		}

		/// <summary>
		/// Gets an iterator at the beginning.
		/// </summary>
		/// <returns>The iterator</returns>
		const_iterator begin() const noexcept
		{
			return data();
		}

		/// <summary>
		/// Gets a reverse iterator at the beginning.
		/// </summary>
		/// <returns>The iterator</returns>
		const_reverse_iterator rbegin() const noexcept
		{
			return const_reverse_iterator{ begin() };
		}

		/// <summary>
		/// Gets a const iterator at the beginning.
		/// </summary>
		/// <returns>The iterator</returns>
		const_iterator cbegin() const noexcept
		{
			return begin();
		}

		/// <summary>
		/// Gets a const-reverse iterator at the beginning.
		/// </summary>
		/// <returns>The iterator</returns>
		const_reverse_iterator crbegin() const noexcept
		{
			return rbegin();
		}

		/// <summary>
		/// Gets an iterator at the end.
		/// </summary>
		/// <returns>The iterator</returns>
		const_iterator end() const noexcept
		{
			return data() + size();
		}

		/// <summary>
		/// Gets a reverse iterator at the end.
		/// </summary>
		/// <returns>The iterator</returns>
		const_reverse_iterator rend() const noexcept
		{
			return const_reverse_iterator{ end() };
		}

		/// <summary>
		/// Gets an iterator at the end.
		/// </summary>
		/// <returns>The iterator</returns>
		const_iterator cend() const noexcept
		{
			return cend();
		}

		/// <summary>
		/// Gets a const-reverse iterator at the end.
		/// </summary>
		/// <returns>The iterator</returns>
		const_reverse_iterator crend() const noexcept
		{
			return rend();
		}
	private:
		allocations::param_string_handle handle_;
	};

	/// <summary>
	/// Converts a string to a platform-dependent narrow string.
	/// </summary>
	/// <param name="str">The string</param>
	/// <returns>The narrow string</returns>
	inline std::string to_narrow(const param_string& str) noexcept
	{
		platform_encoding::utf8_to_narrow(str);
	}

	/// <summary>
	/// Gets the ABI of a string with type information erased.
	/// </summary>
	/// <param name="str">The string</param>
	/// <returns>The ABI</returns>
	inline void* get_abi(const param_string& str) noexcept
	{
		return meta::get_standard_layout_first_member<allocations::param_string_handle>(str);
	}

	/// <summary>
	/// Gets a pointer to the ABI of a string with type information erased.
	/// </summary>
	/// <param name="str">The string</param>
	/// <returns>The pointer to the ABI</returns>
	inline void** put_abi(param_string& str) noexcept
	{
		str.clear();
		
		return reinterpret_cast<void**>(&meta::get_standard_layout_first_member<allocations::param_string_handle>(str));
	}

	/// <summary>
	/// Attaches an ABI to a string.
	/// </summary>
	/// <param name="str">The string</param>
	/// <param name="abi">The ABI</param>
	inline void attach_abi(param_string& str, void* abi) noexcept
	{
		*put_abi(str) = abi;
	}
	
	/// <summary>
	/// Detaches the ABI from a string.
	/// </summary>
	/// <param name="str">The string</param>
	/// <returns>The ABI detached from the string</returns>
	template<typename ParamString, typename = std::enable_if_t<meta::is_non_const_reference_v<ParamString>>>
	void* detach_abi(ParamString&& str) noexcept
	{
		// When the string is a named rvalue reference, we just pass it to the put_abi function as a lvalue reference (without perfect forwarding).
		// Thus, the ABI of the string is capable of being exchanged.
		return std::exchange(*put_abi(str), nullptr);
	}

	/// <summary>
	/// Duplicates an ABI and assignes it to a new string.
	/// </summary>
	/// <param name="str">The string</param>
	/// <param name="abi">The ABI</param>
	inline void copy_from_abi(param_string& str, void* abi) noexcept
	{
		*put_abi(str) = allocations::duplicate_param_string(static_cast<allocations::param_string_handle>(abi));
	}

	/// <summary>
	/// Copy the ABI of a string to another ABI..
	/// </summary>
	/// <param name="str">The string</param>
	/// <param name="abi">The ABI</param>
	inline void copy_to_abi(const param_string& str, void*& abi) noexcept
	{
		abi = get_abi(str);
	}
}
