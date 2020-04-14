#pragma once

#include "jni_utils.hpp"

#include <utility>
#include <variant>
#include <type_traits>

#include <jni.h>

namespace glasssix::jni
{
	/// <summary>
	/// A uniform accessor for Java fields.
	/// </summary>
	template<typename T, typename Category = jobject, typename = std::enable_if_t<std::conjunction_v<std::is_default_constructible<T>>>>
	class jvm_field_accessor
	{
	public:
		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="field">The field ID</param>
		jvm_field_accessor(jfieldID field) noexcept : field_{ field }
		{
		}

		virtual ~jvm_field_accessor() = default;

		/// <summary>
		/// Indicates whether the object is valid.
		/// </summary>
		/// <returns>True if the object is valid; otherwise false</returns>
		operator bool() const noexcept
		{
			return field_;
		}

		/// <summary>
		/// Sets a new value.
		/// </summary>
		/// <typeparam name="U">The value type</typeparam>
		/// <param name="value">The value</param>
		template<typename U>
		auto set(jobject obj, U&& value) -> std::enable_if_t<std::is_convertible_v<U, T>>
		{
			if (*this)
			{
				utils::set_field_value<T>(obj, field_, std::forward<U>(value));
			}
		}

		/// <summary>
		/// Gets the current value.
		/// </summary>
		/// <typeparam name="Source">The source type</typeparam>
		/// <returns>The current value</returns>
		auto get(jobject obj) const
		{
			using return_type = decltype(utils::get_field_value<T>(obj, field_));

			return *this ? utils::get_field_value<T>(obj, field_) : return_type{};
		}
	private:
		jfieldID field_;
	};

	/// <summary>
	/// A uniform accessor for Java fields.
	/// </summary>
	template<typename T>
	class jvm_field_accessor<T, jclass>
	{
	public:
		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="clazz">The class</param>
		/// <param name="field">The field ID</param>
		jvm_field_accessor(jclass clazz, jfieldID field) noexcept : clazz_{ clazz }, field_{ field }
		{
		}

		virtual ~jvm_field_accessor() = default;

		/// <summary>
		/// Indicates whether the object is valid.
		/// </summary>
		/// <returns>True if the object is valid; otherwise false</returns>
		operator bool() const noexcept
		{
			return clazz_ && field_;
		}

		/// <summary>
		/// Sets a new value.
		/// </summary>
		/// <typeparam name="U">The value type</typeparam>
		/// <param name="value">The value</param>
		template<typename U>
		auto set(U&& value) -> std::enable_if_t<std::is_convertible_v<U, T>>
		{
			if (*this)
			{
				utils::set_field_value(clazz_, field_, std::forward<U>(value));
			}
		}

		/// <summary>
		/// Gets the current value.
		/// </summary>
		/// <returns>The current value</returns>
		auto get() const
		{
			using return_type = decltype(utils::get_field_value<T>(clazz_, field_));

			return *this ? utils::get_field_value<T>(clazz_, field_) : return_type{};
		}
	private:
		jclass clazz_;
		jfieldID field_;
	};
}
