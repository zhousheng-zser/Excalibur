#include "cache_key.hpp"
#include "jvm_thread_env.hpp"
#include "jvm_runtime_info.hpp"
#include "irisvika_cache_key.hpp"

#include <jni.h>

using namespace glasssix::jni;
using utils::arg_enum_v;

JNIEXPORT jint JNI_OnLoad(JavaVM* jvm, void* reserved)
{
	if (!jvm_runtime_info::instance().initialize(jvm))
	{
		return JNI_ERR;
	}

	jvm_thread_env::instance().initialize(jvm, jvm_runtime_info::instance().version());
	
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<irisvika_class_key::face_service>, "com/glasssix/irisviel/FaceService");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<irisvika_class_key::database_record>, "com/glasssix/irisviel/DatabaseRecord");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<irisvika_class_key::database_search_result>, "com/glasssix/irisviel/DatabaseSearchResult");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<irisvika_class_key::null_pointer_exception>, "java/lang/NullPointerException");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<irisvika_class_key::illegal_argument_exception>, "java/lang/IllegalArgumentException");
	jvm_runtime_info::instance().add_class_cache(arg_enum_v<irisvika_class_key::index_out_of_bounds_exception>, "java/lang/IndexOutOfBoundsException");
	jvm_runtime_info::instance().add_field_caches(arg_enum_v<irisvika_class_key::face_service>,
		{
			{ arg_enum_v<irisvika_field_key::face_service_m_impl>, "mImpl", "J" }
		});
	jvm_runtime_info::instance().add_field_caches(arg_enum_v<irisvika_class_key::database_record>,
		{
			{ arg_enum_v<irisvika_field_key::database_record_m_key>, "mKey", "Ljava/lang/String;" },
			{ arg_enum_v<irisvika_field_key::database_record_m_feature>, "mFeature", "[F" }
		});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<irisvika_class_key::database_record>,
		{
			{ arg_enum_v<irisvika_method_key::database_record_constructor>, "<init>", "(Ljava/lang/String;[F)V"}
		});
	jvm_runtime_info::instance().add_method_caches(arg_enum_v<irisvika_class_key::database_search_result>,
		{
			{ arg_enum_v<irisvika_method_key::database_search_result_constructor>, "<init>", "(Lcom/glasssix/irisviel/DatabaseRecord;F)V"}
		});

	return jvm_runtime_info::instance().version();
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
}
