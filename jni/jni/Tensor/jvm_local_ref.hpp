#pragma once

#include "jni_utils.hpp"

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
