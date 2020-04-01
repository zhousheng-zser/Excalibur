#include "com_glasssix_common_Tensor.h"
#include "cache_key.hpp"
#include "jni_utils.hpp"
#include "android_logger.hpp"
#include "tensor_cache_key.hpp"
#include "jvm_runtime_info.hpp"
#include "jvm_field_accessor.hpp"
#include "tensor_instantiation.hpp"

#include <algorithm>
#include <unordered_map>

#include <glasssix/tensor.hpp>
#include <tensor_conversions.hpp>

using namespace glasssix::jni;
using glasssix::excalibur::tensor;
using glasssix::excalibur::tensor_base;
using utils::tensor_instantiation;
using utils::arg_enum_v;

DEFINE_ANDROID_LOGGER(Tensor)

namespace
{
	jclass clazz_null_pointer_exception;
	jclass clazz_illegal_argument_exception;
	jclass clazz_index_out_of_bounds_exception;

	jmethodID method_tensor_constructor;
	jmethodID method_clazz_get_type_parameters;

	std::shared_ptr<jvm_field_accessor<jlong>> field_tensor_m_impl;

	template<typename T = void>
	auto throw_null_pointer_exception(JNIEnv* env)
	{
		return utils::throw_new_exception<T>(env, clazz_null_pointer_exception, "The underlying implementation is null.");
	}

	auto get_native_tensor_builder(JNIEnv* env, jclass clazz)
	{
		auto parameters = static_cast<jobjectArray>(env->CallStaticObjectMethod(clazz, method_clazz_get_type_parameters));
		auto underlying_type = static_cast<jclass>(env->GetObjectArrayElement(parameters, 0));

		return tensor_instantiation::get_by_primitive(underlying_type);
	}
}

JNIEXPORT void JNICALL Java_com_glasssix_common_Tensor_close(JNIEnv* env, jobject obj)
{
	if (auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj)))
	{
		delete impl;
		field_tensor_m_impl->set(obj, 0);
	}
}

JNIEXPORT jboolean JNICALL Java_com_glasssix_common_Tensor_empty(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? utils::to_jboolean(impl->empty()) : throw_null_pointer_exception<jboolean>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_num(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->num() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_channels(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->channels() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_width(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->width() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_height(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->height() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_count__II(JNIEnv* env, jobject obj, jint start_axis, jint end_axis)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->count(start_axis, end_axis) : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_count__(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->count() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_device(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->device() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_offset(JNIEnv* env, jobject obj, jint n, jint c, jint h, jint w)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->offset(n, c, h, w) : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jintArray JNICALL Java_com_glasssix_common_Tensor_dataShape(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception<jintArray>(env);
	}

	auto shape = impl->data_shape();
	auto size = static_cast<jsize>(shape.size());
	auto result = env->NewIntArray(size);

	env->SetIntArrayRegion(result, 0, size, shape.data());

	return result;
}

JNIEXPORT void JNICALL Java_com_glasssix_common_Tensor_copyFrom(JNIEnv* env, jobject obj, jbyteArray buffer, jint index, jint size)
{
	if (buffer == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The buffer cannot be null.");
	}

	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	jsize full_size = env->GetArrayLength(buffer);

	// Checks the bounds.
	if (index < 0 || index >= full_size)
	{
		return utils::throw_new_exception(env, clazz_index_out_of_bounds_exception, "The index is out of bounds.");
	}

	// Gets the correct size.
	jsize real_size = std::min(env->GetArrayLength(buffer) - index, size);
	
	if (real_size > impl->bytes())
	{
		return utils::throw_new_exception(env, clazz_illegal_argument_exception, "The size of the buffer cannot be larger than that of the tensor.");
	}

	auto native_buffer = env->GetByteArrayElements(buffer, nullptr);

	std::memcpy(impl->cpu_data_any(), native_buffer, real_size);
	env->ReleaseByteArrayElements(buffer, native_buffer, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_glasssix_common_Tensor_convertToInternal(JNIEnv* env, jobject obj, jbyteArray buffer, jint layout_ordinal)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}
}

JNIEXPORT void JNICALL Java_com_glasssix_common_Tensor_initialize(JNIEnv* env, jobject obj)
{
	// Gets the implementation pointer.
	clazz_null_pointer_exception = jvm_runtime_info::instance().get_class_cache(arg_enum_v<tensor_class_key::null_pointer_exception>);
	clazz_illegal_argument_exception = jvm_runtime_info::instance().get_class_cache(arg_enum_v<tensor_class_key::illegal_argument_exception>);
	clazz_index_out_of_bounds_exception = jvm_runtime_info::instance().get_class_cache(arg_enum_v<tensor_class_key::index_out_of_bounds_exception>);
	method_tensor_constructor = jvm_runtime_info::instance().get_method_cache(arg_enum_v<tensor_method_key::tensor_constructor>);
	method_clazz_get_type_parameters = jvm_runtime_info::instance().get_method_cache(arg_enum_v<tensor_method_key::clazz_get_type_parameters>);
	field_tensor_m_impl = std::make_shared<jvm_field_accessor<jlong>>(env, jvm_runtime_info::instance().get_field_cache(arg_enum_v<tensor_field_key::tensor_m_impl>));
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_layoutInternal(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_base*>(field_tensor_m_impl->get(obj));

	return impl ? impl->order() : throw_null_pointer_exception<jint>(env);
}

JNIEXPORT jobject JNICALL Java_com_glasssix_common_Tensor_createInternal__II(JNIEnv* env, jclass clazz, jint device, jint layout_ordinal)
{
	if (auto builder = get_native_tensor_builder(env, clazz))
	{
		auto result = reinterpret_cast<jlong>(builder->create_1(static_cast<glasssix::excalibur::orderType>(layout_ordinal)));

		return env->NewObject(clazz, method_tensor_constructor, result);
	}

	return nullptr;
}

JNIEXPORT jobject JNICALL Java_com_glasssix_common_Tensor_createInternal__III(JNIEnv* env, jclass clazz, jint shape, jint device, jint layout_ordinal)
{
	if (auto builder = get_native_tensor_builder(env, clazz))
	{
		auto result = reinterpret_cast<jlong>(builder->create_2(shape, device, static_cast<glasssix::excalibur::orderType>(layout_ordinal)));

		return env->NewObject(clazz, method_tensor_constructor, result);
	}

	return nullptr;
}

JNIEXPORT jobject JNICALL Java_com_glasssix_common_Tensor_createInternal___3III(JNIEnv* env, jclass clazz, jintArray shape, jint device, jint layout_ordinal)
{
	if (auto builder = get_native_tensor_builder(env, clazz))
	{
		auto shape_ptr = env->GetIntArrayElements(shape, nullptr);
		auto size = env->GetArrayLength(shape);
		auto result = reinterpret_cast<jlong>(builder->create_3(std::vector<int>(shape_ptr, shape_ptr + size), device, static_cast<glasssix::excalibur::orderType>(layout_ordinal)));

		env->ReleaseIntArrayElements(shape, shape_ptr, JNI_ABORT);

		return env->NewObject(clazz, method_tensor_constructor, result);
	}

	return nullptr;
}
