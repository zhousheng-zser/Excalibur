#pragma once

#include <jni.h>

#include <glasssix/singleton.hpp>

namespace glasssix::jni
{
	class jvm_thread_env : public singleton<jvm_thread_env>
	{
	public:
		friend singleton;

		jvm_thread_env() noexcept;
		virtual ~jvm_thread_env() = default;
		void initialize(JavaVM* jvm, int version) noexcept;
		JNIEnv* value() const;
	private:
		JavaVM* jvm_;
		int version_;
	};
}
