#pragma once

#include "jni_utils.hpp"

#include <variant>
#include <type_traits>

#include <jni.h>

namespace glasssix::jni
{
	/// <summary>
	/// A uniform accessor for Java fields.
	/// </summary>
	template<typename T, typename = std::enable_if_t<std::is_default_constructible_v<T>>>
	class jvm_field_accessor
	{
	public:
		/// <summary>
		/// Creates an instance.
		/// </summary>
		/// <param name="env">The JVM environment</param>
		/// <param name="source">The source which may be jobject or jclass</param>
		/// <param name="field">The field ID</param>
		jvm_field_accessor(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field) : env_{ env }, source_{ source }, field_{ field }
		{
		}

		virtual ~jvm_field_accessor() = default;

		/// <summary>
		/// Indicates whether the object is valid.
		/// </summary>
		/// <returns>True if the object is valid; otherwise false</returns>
		operator bool() const noexcept
		{
			return env_ != nullptr && field_ != nullptr && !source_.valueless_by_exception() && std::visit([](auto&& value) { return std::forward<decltype(value)>(value) != nullptr; }, source_);
		}

		/// <summary>
		/// Sets a new value.
		/// </summary>
		/// <typeparam name="U">The value type</typeparam>
		/// <param name="value">The value</param>
		/// <returns>The instance</returns>
		template<typename U>
		auto operator=(U&& value) -> std::enable_if_t<std::is_convertible_v<U, T>, jvm_field_accessor&>
		{
			if (*this)
			{
				utils::set_field_value(env_, source_, field_, std::forward<U>(value);
			}

			return *this;
		}

		/// <summary>
		/// Gets the current value.
		/// </summary>
		/// <returns>The current value</returns>
		T get() const
		{
			return *this ? utils::get_field_value(env_, source_, field_) : T{};
		}
	private:
		JNIEnv* env_;
		jfieldID field_;
		std::variant<jobject, jclass> source_;
	};
}
