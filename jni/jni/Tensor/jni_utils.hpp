#pragma once

#include <variant>
#include <type_traits>

#include <jni.h>

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

		template<typename T>
		struct jvm_field_operator;

		/// <summary>
		/// jboolean
		/// </summary>
		template<typename JObject> struct jvm_field_operator<std::enable_if_t<std::conjunction_v<std::is_pointer<JObject>, std::is_base_of<std::remove_pointer_t<jobject>, std::remove_pointer_t<JObject>>>, JObject>>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetObjectField(obj, field); },
						[&](jclass source) { return env->GetStaticObjectField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jobject value)
			{
				return env->SetObjectField(obj, field, value);
			}
		};

		/// <summary>
		/// jboolean
		/// </summary>
		template<> struct jvm_field_operator<jboolean>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetBooleanField(obj, field); },
						[&](jclass source) { return env->GetStaticBooleanField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jboolean value)
			{
				return env->SetBooleanField(obj, field, value);
			}
		};

		/// <summary>
		/// jbyte
		/// </summary>
		template<> struct jvm_field_operator<jbyte>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetByteField(obj, field); },
						[&](jclass source) { return env->GetStaticByteField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jbyte value)
			{
				return env->SetByteField(obj, field, value);
			}
		};

		/// <summary>
		/// jint
		/// </summary>
		template<> struct jvm_field_operator<jint>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetIntField(obj, field); },
						[&](jclass source) { return env->GetStaticIntField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jint value)
			{
				return env->SetIntField(obj, field, value);
			}
		};

		/// <summary>
		/// jshort
		/// </summary>
		template<> struct jvm_field_operator<jshort>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetShortField(obj, field); },
						[&](jclass source) { return env->GetStaticShortField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jshort value)
			{
				return env->SetShortField(obj, field, value);
			}
		};

		/// <summary>
		/// jlong
		/// </summary>
		template<> struct jvm_field_operator<jlong>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetLongField(obj, field); },
						[&](jclass source) { return env->GetStaticLongField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jlong value)
			{
				return env->SetLongField(obj, field, value);
			}
		};

		/// <summary>
		/// jfloat
		/// </summary>
		template<> struct jvm_field_operator<jfloat>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetFloatField(obj, field); },
						[&](jclass source) { return env->GetStaticFloatField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jfloat value)
			{
				return env->SetFloatField(obj, field, value);
			}
		};

		/// <summary>
		/// jdouble
		/// </summary>
		template<> struct jvm_field_operator<jdouble>
		{
			static auto get(JNIEnv* env, const std::variant<jobject, jclass>& obj, jfieldID field)
			{
				return std::visit(functor_package
					{
						[&](jobject obj) { return env->GetDoubleField(obj, field); },
						[&](jclass source) { return env->GetStaticDoubleField(source, field); }
					}, obj);
			}

			static void set(JNIEnv* env, jobject obj, jfieldID field, jdouble value)
			{
				return env->SetDoubleField(obj, field, value);
			}
		};
	}

	template<typename JObject>
	constexpr auto jobject_as(jobject obj) noexcept -> std::enable_if_t<std::conjunction_v<std::is_pointer<JObject>, std::is_base_of<std::remove_pointer_t<jobject>, std::remove_pointer_t<JObject>>>, JObject>
	{
		return obj != nullptr ? static_cast<JObject>(obj) : nullptr;
	}

	constexpr jboolean to_jboolean(bool value) noexcept
	{
		return value ? JNI_TRUE : JNI_FALSE;
	}

	template<typename T>
	auto get_field_value(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field)
	{
		return details::jvm_field_operator<T>::get(env, source, field);
	}
	
	template<typename T>
	auto set_field_value(JNIEnv* env, const std::variant<jobject, jclass>& source, jfieldID field, T&& value)
	{
		details::jvm_field_operator<T>::set(env, source, field, std::forward<T>(value));
	}
}
