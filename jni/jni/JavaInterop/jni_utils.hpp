#pragma once

#include "jvm_thread_env.hpp"

#include <string>
#include <variant>
#include <utility>
#include <type_traits>
#include <string_view>
#include <unordered_map>

#include <jni.h>

namespace glasssix::jni::utils
{
	template<typename JObject>
	inline constexpr bool is_derived_from_jobject_v = std::conjunction_v<std::is_pointer<JObject>, std::is_base_of<std::remove_pointer_t<jobject>, std::remove_pointer_t<JObject>>>;
}

namespace glasssix::jni
{
	template<typename JObject, typename = std::enable_if_t<utils::is_derived_from_jobject_v<JObject>>>
	class jvm_local_ref_ex;
}

namespace glasssix::jni::utils
{
	namespace details
	{
		template<typename... Functor>
		struct functor_package : Functor...
		{
			using Functor::operator()...;
		};

		template<typename... Functor>
		functor_package(Functor&&...)->functor_package<Functor...>;

		template<typename T, typename = void>
		struct jvm_field_operator;

		/// <summary>
		/// jobject
		/// </summary>
		template<typename JObject> struct jvm_field_operator<JObject, std::enable_if_t<is_derived_from_jobject_v<JObject>>>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return jvm_local_ref_ex<JObject>{ static_cast<JObject>(env->GetObjectField(obj, field)), true }; },
						[&](jclass clazz) { return jvm_local_ref_ex<JObject>{ static_cast<JObject>(env->GetStaticObjectField(clazz, field)), true }; }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jobject value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetObjectField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticObjectField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jboolean
		/// </summary>
		template<> struct jvm_field_operator<jboolean>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetBooleanField(obj, field); },
						[&](jclass clazz) { return env->GetStaticBooleanField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jboolean value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetBooleanField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticBooleanField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jbyte
		/// </summary>
		template<> struct jvm_field_operator<jbyte>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetByteField(obj, field); },
						[&](jclass clazz) { return env->GetStaticByteField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jbyte value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetByteField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticByteField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jint
		/// </summary>
		template<> struct jvm_field_operator<jint>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetIntField(obj, field); },
						[&](jclass clazz) { return env->GetStaticIntField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jint value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetIntField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticIntField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jshort
		/// </summary>
		template<> struct jvm_field_operator<jshort>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetShortField(obj, field); },
						[&](jclass clazz) { return env->GetStaticShortField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jshort value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetShortField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticShortField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jlong
		/// </summary>
		template<> struct jvm_field_operator<jlong>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetLongField(obj, field); },
						[&](jclass clazz) { return env->GetStaticLongField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jlong value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetLongField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticLongField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jfloat
		/// </summary>
		template<> struct jvm_field_operator<jfloat>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetFloatField(obj, field); },
						[&](jclass clazz) { return env->GetStaticFloatField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jfloat value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetFloatField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticFloatField(clazz, field, value); }
					}, source);
			}
		};

		/// <summary>
		/// jdouble
		/// </summary>
		template<> struct jvm_field_operator<jdouble>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetDoubleField(obj, field); },
						[&](jclass clazz) { return env->GetStaticDoubleField(clazz, field); }
					}, source);
			}

			static void set(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, jdouble value)
			{
				std::visit(functor_package
					{
						[&](jobject obj) { env->SetDoubleField(obj, field, value); },
						[&](jclass clazz) { env->SetStaticDoubleField(clazz, field, value); }
					}, source);
			}
		};
	}

	template<typename JObject>
	constexpr auto jobject_as(jobject obj) noexcept -> std::enable_if_t<is_derived_from_jobject_v<JObject>, JObject>
	{
		return obj ? static_cast<JObject>(obj) : nullptr;
	}

	template<typename T = void, typename... Args>
	auto throw_new_exception(jclass clazz, std::string_view message, Args&&... args) -> std::enable_if_t<std::is_void_v<T> || std::is_constructible_v<T, Args...>, T>
	{
		if (auto env = jvm_thread_env::instance().value())
		{
			env->ThrowNew(clazz, message.data());
		}

		// Throwing a java exception does not imply omitting the return value in native code.
		// Thus, we just return the default value here.
		if constexpr (!std::is_void_v<T>)
		{
			return T{ std::forward<Args>(args)... };
		}
	}

	std::string to_string(jstring str);
	jstring to_jstring(std::string_view str);

	constexpr jboolean to_jboolean(bool value) noexcept
	{
		return value ? JNI_TRUE : JNI_FALSE;
	}

	template<typename T>
	auto get_field_value(const std::variant<jobject, jclass>& source, jfieldID field)
	{
		auto env = jvm_thread_env::instance().value();
		using result_type = decltype(details::jvm_field_operator<T>::get(env, source, field));

		return env ? details::jvm_field_operator<T>::get(env, source, field) : result_type{};
	}

	template<typename T>
	void set_field_value(const std::variant<jobject, jclass>& source, jfieldID field, T&& value)
	{
		if (auto env = jvm_thread_env::instance().value())
		{
			details::jvm_field_operator<T>::set(env, source, field, std::forward<T>(value));
		}
	}
}
