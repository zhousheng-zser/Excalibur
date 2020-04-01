#include "jvm_local_ref.hpp"

#include <utility>

namespace glasssix::jni
{
	jvm_local_ref::jvm_local_ref() noexcept : jvm_local_ref{ nullptr, nullptr }
	{
	}

	jvm_local_ref::jvm_local_ref(std::nullptr_t) noexcept : jvm_local_ref{}
	{
	}

	jvm_local_ref::jvm_local_ref(JNIEnv* env, jobject obj) : jvm_local_ref{ env, obj, false }
	{
	}

	jvm_local_ref::jvm_local_ref(JNIEnv* env, jobject obj, bool takeOverOnly) : env_{ env }, ref_{ !takeOverOnly && env && obj ? env->NewLocalRef(obj) : obj }
	{
	}

	jvm_local_ref::jvm_local_ref(const jvm_local_ref& other) : env_{ other.env_ }, ref_{ env_ && other.ref_ ? env_ ->NewLocalRef(other.ref_) : nullptr }
	{
	}

	jvm_local_ref::jvm_local_ref(jvm_local_ref&& other) noexcept : env_{ std::exchange(other.env_, nullptr) }, ref_{ std::exchange(other.ref_, nullptr) }
	{
	}

	jvm_local_ref::~jvm_local_ref()
	{
		if (env_)
		{
			if (ref_)
			{
				env_->DeleteLocalRef(ref_);
				ref_ = nullptr;
			}

			env_ = nullptr;
		}
	}

	jvm_local_ref::operator bool() const noexcept
	{
		return env_ && ref_;
	}

	jvm_local_ref& jvm_local_ref::operator=(const jvm_local_ref& right)
	{
		env_ = right.env_;
		ref_ = env_ && right.ref_ ? env_->NewLocalRef(right.ref_) : nullptr;

		return *this;
	}

	jvm_local_ref& jvm_local_ref::operator=(jvm_local_ref&& right) noexcept
	{
		env_ = std::exchange(right.env_, nullptr);
		ref_ = std::exchange(right.ref_, nullptr);

		return *this;
	}

	jobject jvm_local_ref::get() const noexcept
	{
		return ref_;
	}
}
