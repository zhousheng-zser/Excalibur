#pragma once

#include <tuple>
#include <memory>
#include <functional>
#include <string_view>
#include <initializer_list>

#include <jni.h>
#include <glasssix/singleton.hpp>

namespace glasssix::jni
{
	/// <summary>
	/// JVM runtime information.
	/// </summary>
	class jvm_runtime_info : public singleton<jvm_runtime_info>
	{
	public:
		class impl;
		friend class singleton;

		virtual ~jvm_runtime_info();
		int version() const noexcept;
		JNIEnv* env() const noexcept;
		bool env(const std::function<int(void**, int)>& handler);
		jclass get_class_cache(int key) const;
		jfieldID get_field_cache(int key) const;
		jmethodID get_method_cache(int key) const;
		void add_class_cache(int key, std::string_view name);
		void add_field_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> fields);
		void add_method_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> methods);
	private:
		jvm_runtime_info();
		impl* impl_;
	};
}
