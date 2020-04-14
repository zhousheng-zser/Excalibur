#pragma once

#include "jni_utils.hpp"

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
		jvm_global_ref(jobject obj);
		jvm_global_ref(jobject obj, bool takeOverOnly);
		jvm_global_ref(const jvm_global_ref& other);
		jvm_global_ref(jvm_global_ref&& other) noexcept;
		virtual ~jvm_global_ref();
		operator bool() const noexcept;
		jvm_global_ref& operator=(const jvm_global_ref& right);
		jvm_global_ref& operator=(jvm_global_ref&& right) noexcept;
		jobject get() const noexcept;
	private:
		jobject ref_;
	};

	template<typename JObject, typename = std::enable_if_t<utils::is_derived_from_jobject_v<JObject>, JObject>>
	class jvm_global_ref_ex : public jvm_global_ref
	{
	public:
		jvm_global_ref_ex() noexcept = default;

		jvm_global_ref_ex(JObject obj) : jvm_global_ref{ obj }
		{
		}

		jvm_global_ref_ex(JObject obj, bool takeOverOnly) : jvm_global_ref{ obj, takeOverOnly }
		{
		}

		jvm_global_ref_ex(const jvm_global_ref_ex& other) : jvm_global_ref{ other }
		{
		}

		jvm_global_ref_ex(jvm_global_ref_ex&& other) noexcept : jvm_global_ref{ std::move(other) }
		{
		}

		virtual ~jvm_global_ref_ex() = default;

		jvm_global_ref_ex& operator=(const jvm_global_ref_ex& right)
		{
			static_cast<jvm_global_ref&>(*this) = right;

			return *this;
		}

		jvm_global_ref_ex& operator=(jvm_global_ref_ex&& right) noexcept
		{
			static_cast<jvm_global_ref&>(*this) = std::move(right);

			return *this;
		}

		JObject get() const noexcept
		{
			return utils::jobject_as<JObject>(static_cast<const jvm_global_ref&>(*this).get());
		}
	};
}
