#include "jvm_local_ref.hpp"
#include "jvm_thread_env.hpp"

#include <utility>

namespace glasssix::jni
{
	jvm_local_ref::jvm_local_ref() noexcept : jvm_local_ref{ nullptr }
	{
	}

	jvm_local_ref::jvm_local_ref(jobject obj) : jvm_local_ref{ obj, false }
	{
	}

	jvm_local_ref::jvm_local_ref(jobject obj, bool takeOverOnly) : ref_{}
	{
		if (auto env = jvm_thread_env::instance().value(); env && obj)
		{
			ref_ = takeOverOnly ? obj : env->NewLocalRef(obj);
		}
	}

	jvm_local_ref::jvm_local_ref(const jvm_local_ref& other) : jvm_local_ref{ other.ref_ }
	{
	}

	jvm_local_ref::jvm_local_ref(jvm_local_ref&& other) noexcept : ref_{ std::exchange(other.ref_, nullptr) }
	{
	}

	jvm_local_ref::~jvm_local_ref()
	{
		if (auto env = jvm_thread_env::instance().value(); env && ref_)
		{
			env->DeleteLocalRef(ref_);
			ref_ = nullptr;
		}
	}

	jvm_local_ref::operator bool() const noexcept
	{
		return ref_;
	}

	jvm_local_ref& jvm_local_ref::operator=(const jvm_local_ref& right)
	{
		auto env = jvm_thread_env::instance().value();

		ref_ = env && right.ref_ ? env->NewLocalRef(right.ref_) : nullptr;

		return *this;
	}

	jvm_local_ref& jvm_local_ref::operator=(jvm_local_ref&& right) noexcept
	{
		ref_ = std::exchange(right.ref_, nullptr);

		return *this;
	}

	jobject jvm_local_ref::get() const noexcept
	{
		return ref_;
	}
}
