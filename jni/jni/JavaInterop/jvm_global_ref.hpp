#pragma once

#include "jni_utils.hpp"

#include <type_traits>

#include <jni.h>

namespace glasssix::jni
{
	/// <summary>
	/// Represents the global reference for a java object.
	/// </summary>
	class jvm_global_ref
	{
	public:
		jvm_global_ref() noexcept;
		jvm_global_ref(std::nullptr_t) noexcept;
		jvm_global_ref(JNIEnv* env, jobject obj);
		jvm_global_ref(JNIEnv* env, jobject obj, bool takeOverOnly);
		jvm_global_ref(const jvm_global_ref& other);
		jvm_global_ref(jvm_global_ref&& other) noexcept;
		virtual ~jvm_global_ref();
		operator bool() const noexcept;
		jvm_global_ref& operator=(const jvm_global_ref& right);
		jvm_global_ref& operator=(jvm_global_ref&& right) noexcept;

		template<typename JObject, typename = std::enable_if_t<!std::is_same_v<JObject, bool>>>
		operator JObject() noexcept
		{
			return utils::jobject_as<JObject>(ref_);
		}
	private:
		JNIEnv* env_;
		jobject ref_;
	};
}
