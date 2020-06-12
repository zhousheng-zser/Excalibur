#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"
#include "exceptions.hpp"

#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <iterator>
#include <type_traits>

namespace glasssix::exposing
{
	template<typename T>
	struct param_vector;
}

namespace glasssix::exposing::impl
{
	/// <summary>
	/// The ABI of a param_vector.
	/// </summary>
	template<typename T>
	struct abi<param_vector<T>>
	{
		using identity_type = type_identity_generic_interface;
		static constexpr guid id{ "DCB2A5A5-1D17-4E0A-83C2-640912AECD25" };

		struct type : abi_unknown_object
		{
			virtual std::int32_t G6_ABI_CALL size(abi_out_t<std::uint64_t> result) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL at(std::uint64_t index, abi_out_t<T> result) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL set_at(std::uint64_t index, abi_in_t<T> item) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL push_back(abi_in_t<T> item) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL remove_at(std::uint64_t index) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL insert_at(std::uint64_t index, abi_in_t<T> item) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL clear() noexcept = 0;
		};
	};

	/// <summary>
	/// The vtable of a param_vector.
	/// </summary>
	template<typename Derived, typename T>
	struct interface_vtable<Derived, param_vector<T>> : interface_vtable_base<Derived, param_vector<T>>
	{
		virtual std::int32_t G6_ABI_CALL size(abi_out_t<std::uint64_t> result) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().size()); });
		}

		virtual std::int32_t G6_ABI_CALL at(std::uint64_t index, abi_out_t<T> result) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().at(index)); });
		}

		virtual std::int32_t G6_ABI_CALL set_at(std::uint64_t index, abi_in_t<T> item) noexcept override
		{
			return abi_safe_call([&] { this->self().set_at(index, create_from_abi<T>(item)); });
		}
		
		virtual std::int32_t G6_ABI_CALL push_back(abi_in_t<T> item) noexcept override
		{
			return abi_safe_call([&] { this->self().push_back(create_from_abi<T>(item)); });
		}

		virtual std::int32_t G6_ABI_CALL remove_at(std::uint64_t index) noexcept override
		{
			return abi_safe_call([&] { this->self().remove_at(index); });
		}
		
		virtual std::int32_t G6_ABI_CALL insert_at(std::uint64_t index, abi_in_t<T> item) noexcept override
		{
			return abi_safe_call([&] { this->self().insert_at(index, create_from_abi<T>(item)); });
		}
		
		virtual std::int32_t G6_ABI_CALL clear() noexcept override
		{
			return abi_safe_call([&] { this->self().clear(); });
		}
	};

	/// <summary>
	/// The ABI adapter of a param_vector.
	/// </summary>
	template<typename T>
	struct abi_adapter<param_vector<T>>
	{
		template<typename Derived>
		struct type : enable_self_abi_awareness<param_vector<T>>
		{
			std::uint64_t size()
			{
				std::uint64_t result = 0;
				
				return (check_abi_result(this->self_abi().size(put_abi(result))), result);
			}

			T at(std::uint64_t index)
			{
				T result{};

				return (check_abi_result(this->self_abi().at(index, put_abi(result))), result);
			}

			void set_at(std::uint64_t index, const T& item)
			{
				check_abi_result(this->self_abi().set_at(index, get_abi(item)));
			}

			void push_back(const T& item)
			{
				check_abi_result(this->self_abi().push_back(get_abi(item)));
			}

			void remove_at(std::uint64_t index)
			{
				check_abi_result(this->self_abi().remove_at(get_abi(index)));
			}

			void insert_at(std::uint64_t index, const T& element)
			{
				check_abi_result(this->self_abi().insert_at(get_abi(index), get_abi(element)));
			}

			void clear()
			{
				check_abi_result(this->self_abi().clear());
			}
		};
	};
}

namespace glasssix::exposing
{
	/// <summary>
	/// A mutable vector that is capable of being parameters.
	/// </summary>
	template<typename T>
	struct param_vector : inherits<param_vector<T>>
	{
		using inherits<param_vector<T>>::inherits;
	};
}

namespace glasssix::exposing::impl
{
	template<typename T>
	class param_vector_impl : public implements<param_vector_impl<T>, param_vector<T>>
	{
	public:
		param_vector_impl()
		{
		}

		template<typename... Args, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<Args, T>...>>>
		param_vector_impl(Args&&... args) : buffer_{ std::forward<Args>(args)... }
		{
		}

		std::uint64_t size()
		{
			return buffer_.size();
		}

		T at(std::uint64_t index)
		{
			return buffer_[index];
		}

		void set_at(std::uint64_t index, const T& item)
		{
			buffer_[index] = item;
		}

		void push_back(const T& item)
		{
			buffer_.emplace_back(item);
		}

		void remove_at(std::uint64_t index)
		{
			buffer_.erase(buffer_.begin() + index);
		}

		void insert_at(std::uint64_t index, const T& item)
		{
			buffer_.insert(buffer_.begin() + index, item);
		}

		void clear()
		{
			buffer_.clear();
		}
	private:
		std::vector<T> buffer_;
	};
}

namespace glasssix::exposing
{
	/// <summary>
	/// Creates a N-dimensional param_vector.
	/// </summary>
	/// <typeparam name="T">The element type</typeparam>
	/// <returns>The result</returns>
	template<typename T, std::size_t Dimension = 1, typename = std::enable_if_t<std::conjunction_v<impl::has_abi_type<T>>>>
	auto make_param_vector()
	{
		using element_type = meta::make_multidimensional_container_t<param_vector, T, Dimension - 1>;

		return make_as_first<impl::param_vector_impl<element_type>>();
	}

	/// <summary>
	/// Creates a one-dimensional param_vector.
	/// </summary>
	/// <typeparam name="T">The element type</typeparam>
	/// <typeparam name="...Args">The types of the initializer</typeparam>
	/// <param name="...args">The initializer</param>
	/// <returns>The result</returns>
	template<typename T, typename... Args, typename = std::enable_if_t<std::conjunction_v<impl::has_abi_type<T>, std::is_convertible<Args, T>...>>>
	param_vector<T> make_param_vector(Args&&... args)
	{
		return make_as_first<impl::param_vector_impl<T>>(std::forward<Args>(args)...);
	}
}
