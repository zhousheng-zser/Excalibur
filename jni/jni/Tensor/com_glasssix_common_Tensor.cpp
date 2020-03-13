#include "com_glasssix_common_Tensor.h"
#include "cache_key.hpp"
#include "jni_utils.hpp"
#include "android_logger.hpp"
#include "tensor_cache_key.hpp"
#include "jvm_runtime_info.hpp"
#include "jvm_field_accessor.hpp"

#include <unordered_map>

#include <glasssix/tensor.hpp>

using namespace glasssix::jni;
using glasssix::excalibur::tensor;
using glasssix::excalibur::tensor_;
using utils::arg_enum_v;

DEFINE_ANDROID_LOGGER(Tensor)

namespace
{
	std::shared_ptr<jvm_field_accessor<jlong>> field_m_impl;
}

JNIEXPORT void JNICALL Java_com_glasssix_common_Tensor_close(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());
	
	if (impl != nullptr)
	{
		delete impl;
	}
}

JNIEXPORT jboolean JNICALL Java_com_glasssix_common_Tensor_empty(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? utils::to_jboolean(impl->empty()) : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_num(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->num() : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_channels(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->channels() : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_width(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->width() : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_height(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->height() : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_count__II(JNIEnv* env, jobject obj, jint start_axis, jint end_axis)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->count(start_axis, end_axis) : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_count__(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->count() : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_device(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->device() : 0;
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_offset(JNIEnv* env, jobject obj, jint n, jint c, jint h, jint w)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());

	return impl != nullptr ? impl->offset(n, c, h, w) : 0;
}

JNIEXPORT jintArray JNICALL Java_com_glasssix_common_Tensor_dataShape(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());
}

JNIEXPORT jint JNICALL Java_com_glasssix_common_Tensor_layoutInternal(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<tensor_*>(field_m_impl->get());
}

JNIEXPORT void JNICALL Java_com_glasssix_common_Tensor_initialize(JNIEnv* env, jobject obj)
{
	field_m_impl = std::make_shared<jvm_field_accessor<jlong>>(jvm_runtime_info::instance().env(), obj, jvm_runtime_info::instance().get_field_cache(arg_enum_v<tensor_field_key::tensor_m_impl>));
}

JNIEXPORT jobject JNICALL Java_com_glasssix_common_Tensor_createInternal__II(JNIEnv* env, jclass clazz, jint device, jint layout_ordinal)
{

}

JNIEXPORT jobject JNICALL Java_com_glasssix_common_Tensor_createInternal__III(JNIEnv* env, jclass clazz, jint shape, jint device, jint layout_ordinal)
{
}

JNIEXPORT jobject JNICALL Java_com_glasssix_common_Tensor_createInternal___3III(JNIEnv* env, jclass clazz, jintArray shape, jint device, jint layout_ordinal)
{
}
