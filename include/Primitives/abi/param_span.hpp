#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "fundamental_semantics.hpp"

#include <cstdint>
#include <cstddef>
#include <utility>
#include <type_traits>

namespace glasssix::exposing
{
	/// <summary>
	/// Contains a span of elements.
	/// </summary>
	/// <typeparam name="T">The element type</typeparam>
	template<typename T>
	class param_span
	{
	public:
		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="data">The data</param>
		/// <param name="size">The size</param>
		param_span(T* data, std::size_t size) noexcept : data_{ data }, size_{ size }
		{
		}

		/// <summary>
		/// Creates an instance from a random access container.
		/// </summary>
		/// <typeparam name="Container">The container type</typeparam>
		/// <param name="container">The container</param>
		template<typename Container, typename = std::enable_if_t<meta::is_iterator_category_same_v<std::decay_t<Container>, std::random_access_iterator_tag>>>
		param_span(Container&& container) noexcept : data_{ &*std::forward<Container>(container).begin() }, size_{ std::forward<Container>(container).end() - std::forward<Container>(container).begin() }
		{
		}

		/// <summary>
		/// Create an instance with an ABI from which ownership is taken.
		/// </summary>
		/// <param name="abi">The ABI</param>
		param_span(take_over_abi_from_void_ptr abi)
		{
			*this = *abi.to<param_span*>();
		}

		T* data() const noexcept
		{
			return data_;
		}

		std::size_t size() const noexcept
		{
			return size_;
		}
	private:
		T* data_;
		std::size_t size_;
	};
}

namespace glasssix::exposing::impl
{
	template<typename T>
	struct abi<param_span<T>>
	{
		using identity_type = type_identity_primitive;
		using type = void*;

		static constexpr guid id{ "4BBC2561-97C4-4C12-A413-7636DBCD70F9" };
	};
}
