#pragma once

#include "jni_utils.hpp"

#include <utility>
#include <type_traits>

#include <jni.h>

namespace glasssix::jni
{
	/// <summary>
	/// Represents the global reference for a java object.
	/// </summary>
	class jvm_local_ref
	{
	public:
		jvm_local_ref() noexcept;
		jvm_local_ref(std::nullptr_t) noexcept;
		jvm_local_ref(JNIEnv* env, jobject obj);
		jvm_local_ref(JNIEnv* env, jobject obj, bool takeOverOnly);
		jvm_local_ref(const jvm_local_ref& other);
		jvm_local_ref(jvm_local_ref&& other) noexcept;
		virtual ~jvm_local_ref();
		operator bool() const noexcept;
		jvm_local_ref& operator=(const jvm_local_ref& right);
		jvm_local_ref& operator=(jvm_local_ref&& right) noexcept;
		jobject get() const noexcept;
	private:
		JNIEnv* env_;
		jobject ref_;
	};

	template<typename JObject, typename>
	class jvm_local_ref_ex : public jvm_local_ref
	{
	public:
		jvm_local_ref_ex() noexcept = default;
		jvm_local_ref_ex(std::nullptr_t) noexcept : jvm_local_ref{ nullptr }
		{
		}

		jvm_local_ref_ex(JNIEnv* env, JObject obj) : jvm_local_ref{ env, obj }
		{
		}

		jvm_local_ref_ex(JNIEnv* env, JObject obj, bool takeOverOnly) : jvm_local_ref{ env, obj, takeOverOnly }
		{
		}

		jvm_local_ref_ex(const jvm_local_ref_ex& other) : jvm_local_ref{ other }
		{
		}

		jvm_local_ref_ex(jvm_local_ref_ex&& other) noexcept : jvm_local_ref{ other }
		{
		}

		virtual ~jvm_local_ref_ex() = default;

		jvm_local_ref_ex& operator=(const jvm_local_ref_ex& right)
		{
			static_cast<jvm_local_ref&>(*this) = right;

			return *this;
		}

		jvm_local_ref_ex& operator=(jvm_local_ref_ex&& right)
		{
			static_cast<jvm_local_ref&>(*this) = std::move(right);

			return *this;
		}

		JObject get() const noexcept
		{
			return utils::jobject_as<JObject>(static_cast<const jvm_local_ref&>(*this).get());
		}
	};
}
