#include "jvm_global_ref.hpp"

#include <utility>

namespace glasssix::jni
{
	jvm_global_ref::jvm_global_ref() noexcept : jvm_global_ref{ nullptr, nullptr }
	{
	}

	jvm_global_ref::jvm_global_ref(std::nullptr_t) noexcept : jvm_global_ref{}
	{
	}

	jvm_global_ref::jvm_global_ref(JNIEnv* env, jobject obj) : jvm_global_ref{ env, obj, false }
	{
	}

	jvm_global_ref::jvm_global_ref(JNIEnv* env, jobject obj, bool takeOverOnly) : env_{ env }, ref_{ !takeOverOnly && env != nullptr && obj != nullptr ? env->NewGlobalRef(obj) : obj }
	{
	}

	jvm_global_ref::jvm_global_ref(const jvm_global_ref& other) : env_{ other.env_ }, ref_{ env_ != nullptr && other.ref_ != nullptr ? env_ ->NewGlobalRef(other.ref_) : nullptr }
	{
	}

	jvm_global_ref::jvm_global_ref(jvm_global_ref&& other) noexcept : env_{ std::exchange(other.env_, nullptr) }, ref_{ std::exchange(other.ref_, nullptr) }
	{
	}

	jvm_global_ref::~jvm_global_ref()
	{
		if (env_ != nullptr)
		{
			if (ref_ != nullptr)
			{
				env_->DeleteGlobalRef(ref_);
				ref_ = nullptr;
			}

			env_ = nullptr;
		}
	}

	jvm_global_ref::operator bool() const noexcept
	{
		return env_ != nullptr && ref_ != nullptr;
	}

	jvm_global_ref& jvm_global_ref::operator=(const jvm_global_ref& right)
	{
		env_ = right.env_;
		ref_ = env_ != nullptr && right.ref_ != nullptr ? env_->NewGlobalRef(right.ref_) : nullptr;

		return *this;
	}

	jvm_global_ref& jvm_global_ref::operator=(jvm_global_ref&& right) noexcept
	{
		env_ = std::exchange(right.env_, nullptr);
		ref_ = std::exchange(right.ref_, nullptr);

		return *this;
	}
}
