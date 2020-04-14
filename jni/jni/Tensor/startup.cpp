#include "android_logger.hpp"
#include "cache_key.hpp"
#include "jvm_runtime_info.hpp"
#include "tensor_cache_key.hpp"
#include "tensor_instantiation.hpp"

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

	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::object>, "java/lang/Object");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::number>, "java/lang/Number");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::byte>, "java/lang/Byte");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::integer>, "java/lang/Integer");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::short_integer>, "java/lang/Short");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::long_integer>, "java/lang/Long");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::single_float>, "java/lang/Float");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::double_float>, "java/lang/Double");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::tensor>, "com/glasssix/common/Tensor");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::null_pointer_exception>, "java/lang/NullPointerException");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::illegal_argument_exception>, "java/lang/IllegalArgumentException");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::index_out_of_bounds_exception>, "java/lang/IndexOutOfBoundsException");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<tensor_class_key::clazz>, "java/lang/Class");
	jvm_runtime_info::instance().add_field_caches(arg_enum_v<tensor_class_key::tensor>,
		{

			{ arg_enum_v<tensor_field_key::tensor_m_impl>, "mImpl", "J" }
		});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<tensor_class_key::tensor>,
		{
			{ arg_enum_v<tensor_method_key::tensor_constructor>, "<init>", "()V"}
		});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<tensor_class_key::object>,
		{
			{ arg_enum_v<tensor_method_key::object_get_class>, "getClass", "()Ljava/lang/Class;"}
		});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<tensor_class_key::number>,
		{
			{ arg_enum_v<tensor_method_key::number_byte_value>, "byteValue", "()B"},
			{ arg_enum_v<tensor_method_key::number_int_value>, "intValue", "()I"},
			{ arg_enum_v<tensor_method_key::number_short_value>, "shortValue", "()S"},
			{ arg_enum_v<tensor_method_key::number_long_value>, "longValue", "()J"},
			{ arg_enum_v<tensor_method_key::number_float_value>, "floatValue", "()F"},
			{ arg_enum_v<tensor_method_key::number_double_value>, "doubleValue", "()D"},
		});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<tensor_class_key::clazz>,
		{
			{ arg_enum_v<tensor_method_key::clazz_get_type_parameters>, "getTypeParameters", "()[Ljava.lang.reflect.TypeVariable;"}
		});

	// Initializes tensor dynamic instantiation.
	utils::tensor_instantiation::initialize();

	return jvm_runtime_info::instance().version();
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
}
