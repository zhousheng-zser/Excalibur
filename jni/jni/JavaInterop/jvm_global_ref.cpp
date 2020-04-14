#include "jvm_global_ref.hpp"
#include "jvm_thread_env.hpp"

#include <utility>

namespace glasssix::jni
{
	jvm_global_ref::jvm_global_ref() noexcept : jvm_global_ref{ nullptr }
	{
	}

	jvm_global_ref::jvm_global_ref(jobject obj) : jvm_global_ref{ obj, false }
	{
	}

	jvm_global_ref::jvm_global_ref(jobject obj, bool takeOverOnly) : ref_{}
	{
		if (auto env = jvm_thread_env::instance().value(); env && obj)
		{
			ref_ = takeOverOnly ? obj : env->NewGlobalRef(obj);
		}
	}

	jvm_global_ref::jvm_global_ref(const jvm_global_ref& other) : jvm_global_ref{ other.ref_ }
	{
	}

	jvm_global_ref::jvm_global_ref(jvm_global_ref&& other) noexcept : ref_{ std::exchange(other.ref_, nullptr) }
	{
	}

	jvm_global_ref::~jvm_global_ref()
	{
		if (auto env = jvm_thread_env::instance().value(); env && ref_)
		{
			env->DeleteGlobalRef(ref_);
			ref_ = nullptr;
		}
	}

	jvm_global_ref::operator bool() const noexcept
	{
		return ref_;
	}

	jvm_global_ref& jvm_global_ref::operator=(const jvm_global_ref& right)
	{
		auto env = jvm_thread_env::instance().value();

		ref_ = env && right.ref_ ? env->NewGlobalRef(right.ref_) : nullptr;

		return *this;
	}

	jvm_global_ref& jvm_global_ref::operator=(jvm_global_ref&& right) noexcept
	{
		ref_ = std::exchange(right.ref_, nullptr);

		return *this;
	}

	jobject jvm_global_ref::get() const noexcept
	{
		return ref_;
	}
}
