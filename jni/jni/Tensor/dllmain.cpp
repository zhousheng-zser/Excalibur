#include "android_logger.hpp"
#include "jvm_runtime_info.hpp"
#include "tensor_cache_key.hpp"
#include "cache_key.hpp"

#include <functional>

#include <jni.h>

using namespace glasssix::jni;
using utils::arg_enum_v;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
	if (!jvm_runtime_info::instance().env(std::bind(&JavaVM::GetEnv, vm, std::placeholders::_1, std::placeholders::_2)))
	{
		return JNI_ERR;
	}

	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::number>, "Ljava/lang/Number;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::byte>, "Ljava/lang/Byte;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::integer>, "Ljava/lang/Integer;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::short_integer>, "Ljava/lang/Short;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::long_integer>, "Ljava/lang/Long;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::single_float>, "Ljava/lang/Float;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::double_float>, "Ljava/lang/Double;");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::tensor>, "com/glasssix/common/Tensor");
	jvm_runtime_info::instance().add_field_caches(arg_enum_v<tensor_class_key::tensor>,
	{
		{ arg_enum_v<tensor_field_key::tensor_m_impl>, "mImpl", "J" }
	});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<tensor_class_key::number>,
	{
		{ arg_enum_v<tensor_method_key::number_byte_value>, "byteValue", "()"},
		{ arg_enum_v<tensor_method_key::number_int_value>, "intValue", "()"},
		{ arg_enum_v<tensor_method_key::number_short_value>, "shortValue", "()"},
		{ arg_enum_v<tensor_method_key::number_long_value>, "longValue", "()"},
		{ arg_enum_v<tensor_method_key::number_float_value>, "floatValue", "()"},
		{ arg_enum_v<tensor_method_key::number_double_value>, "doubleValue", "()"},
	});

	return jvm_runtime_info::instance().version();
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
}
