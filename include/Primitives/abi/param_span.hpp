#pragma once

#include "base.hpp"
#include "base_abi.hpp"

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
		template<typename Container, typename = std::enable_if_t<meta::is_iterator_category_same_v<Container, std::random_access_iterator_tag>>>
		param_span(Container&& container) noexcept
		{

		}
	private:
		T* data_;
		std::size_t size_;
	};
}
