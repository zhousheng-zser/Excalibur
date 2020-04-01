#include "com_glasssix_irisviel_FaceService.h"
#include "cache_key.hpp"
#include "jni_utils.hpp"
#include "jvm_local_ref.hpp"
#include "jvm_runtime_info.hpp"
#include "irisvika_cache_key.hpp"
#include "jvm_field_accessor.hpp"
#include "database_record.hpp"
#include "face_service.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cstddef>

#include <glasssix/scope_guard.hpp>

using namespace glasssix::jni;
using namespace glasssix::irisviel;
using utils::arg_enum_v;
using glasssix::scope_guard;

namespace
{
	jclass clazz_database_record;
	jclass clazz_database_search_result;
	jclass clazz_null_pointer_exception;
	jclass clazz_illegal_argument_exception;

	jmethodID method_database_record_constructor;
	jmethodID method_database_search_result_constructor;

	std::shared_ptr<jvm_field_accessor<jlong>> field_face_service_m_impl;
	std::shared_ptr<jvm_field_accessor<jstring>> field_database_record_m_key;
	std::shared_ptr<jvm_field_accessor<jfloatArray>> field_database_record_m_feature;

	template<typename T = void>
	auto throw_null_pointer_exception(JNIEnv* env)
	{
		return utils::throw_new_exception<T>(env, clazz_null_pointer_exception, "The underlying implementation is null.");
	}

	std::shared_ptr<database_record> get_database_record(JNIEnv* env, jobject obj, jobject record)
	{
		auto feature = field_database_record_m_feature->get(obj);
		auto native_key = utils::to_string(env, field_database_record_m_key->get(obj).get());
		auto native_feature = env->GetFloatArrayElements(feature.get(), nullptr);
		auto dimension = static_cast<int>(env->GetArrayLength(feature.get()));
		scope_guard guard{ [&] { env->ReleaseFloatArrayElements(feature.get(), native_feature, JNI_ABORT); } };
		auto result = database_record::create(dimension);

		result->key(native_key.c_str());
		result->feature(native_feature);

		return result;
	}

	std::vector<std::shared_ptr<database_record>> get_database_records(JNIEnv* env, jobject obj, jobjectArray records)
	{
		std::vector<std::shared_ptr<database_record>> result;
		jsize size = env->GetArrayLength(records);

		for (jsize i = 0; i < size; i++)
		{
			auto item = env->GetObjectArrayElement(records, i);

			result.emplace_back(get_database_record(env, obj, item));
		}

		return result;
	}
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_close(JNIEnv* env, jobject obj)
{
	if (auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj)))
	{
		delete impl;
		field_face_service_m_impl->set(obj, 0);
	}
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_clear(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	return impl ? impl->clear() : throw_null_pointer_exception(env);

}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_removeAll(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	return impl ? impl->remove_all() : throw_null_pointer_exception(env);
}

JNIEXPORT jstring JNICALL Java_com_glasssix_irisviel_FaceService_databaseDirectory(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	return impl ? utils::to_jstring(env, impl->database_directory()) : throw_null_pointer_exception<jstring>(env);
}

JNIEXPORT jstring JNICALL Java_com_glasssix_irisviel_FaceService_cacheDirectory(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	return impl ? utils::to_jstring(env, impl->cache_directory()) : throw_null_pointer_exception<jstring>(env);
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_loadDatabases(JNIEnv* env, jobject obj)
{
	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	return impl ? impl->load_databases() : throw_null_pointer_exception(env);
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_irisviel_FaceService_search(JNIEnv* env, jobject obj, jfloatArray feature, jint top)
{
	if (feature == nullptr)
	{
		return utils::throw_new_exception<jobjectArray>(env, clazz_null_pointer_exception, "The feature cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception<jobjectArray>(env);
	}

	auto native_feature = env->GetFloatArrayElements(feature, nullptr);
	scope_guard guard{ [&] { env->ReleaseFloatArrayElements(feature, native_feature, JNI_ABORT); } };
	auto native_result = impl->search(native_feature, top);
	auto result = env->NewObjectArray(native_result.size(), clazz_database_search_result, nullptr);

	// Fills in the items.
	for (jsize i = 0; i < native_result.size(); i++)
	{
		auto& native_item = native_result[i];
		auto native_record_feature = native_item.data->feature();
		int record_dimension = native_item.data->dimension();
		jvm_local_ref_ex<jfloatArray> record_feature{ env, env->NewFloatArray(record_dimension), true };
		
		// Writes the feature data.
		env->SetFloatArrayRegion(record_feature.get(), 0, record_dimension, native_record_feature);

		// Create a record and searching result.
		jvm_local_ref record{ env,  env->NewObject(clazz_database_record, method_database_record_constructor, utils::to_jstring(env, native_item.data->key()), record_feature.get()), true };
		auto item = env->NewObject(clazz_database_search_result, method_database_search_result_constructor, record.get(), native_item.similarity);
		
		env->SetObjectArrayElement(result, i, item);
	}

	return result;
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_add__Lcom_glasssix_irisviel_DatabaseRecord_2(JNIEnv* env, jobject obj, jobject record)
{
	if (record == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The record cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	impl->add(*get_database_record(env, obj, record));
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_add___3Lcom_glasssix_irisviel_DatabaseRecord_2(JNIEnv* env, jobject obj, jobjectArray records)
{
	if (records == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The records cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	impl->add(get_database_records(env, obj, records));
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_remove__Ljava_lang_String_2(JNIEnv* env, jobject obj, jstring key)
{
	if (key == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The key cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	impl->remove(utils::to_string(env, key));
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_remove___3Ljava_lang_String_2(JNIEnv* env, jobject obj, jobjectArray keys)
{
	if (keys == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The keys cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	std::vector<std::string> native_keys;
	jsize size = env->GetArrayLength(keys);

	for (jsize i = 0; i < size; i++)
	{
		auto item = env->GetObjectArrayElement(keys, i);
		auto native_key = utils::to_string(env, field_database_record_m_key->get(item).get());
		
		native_keys.emplace_back(native_key);
	}

	impl->remove(native_keys);
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_update__Lcom_glasssix_irisviel_DatabaseRecord_2(JNIEnv* env, jobject obj, jobject record)
{
	if (record == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The record cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	impl->update(*get_database_record(env, obj, record));
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_update___3Lcom_glasssix_irisviel_DatabaseRecord_2(JNIEnv* env, jobject obj, jobjectArray records)
{
	if (records == nullptr)
	{
		return utils::throw_new_exception(env, clazz_null_pointer_exception, "The records cannot be null.");
	}

	auto impl = reinterpret_cast<face_service*>(field_face_service_m_impl->get(obj));

	if (impl == nullptr)
	{
		return throw_null_pointer_exception(env);
	}

	impl->update(get_database_records(env, obj, records));
}

JNIEXPORT void JNICALL Java_com_glasssix_irisviel_FaceService_initialize(JNIEnv* env, jobject obj, jint single_database_capacity, jint dimension, jstring working_directory)
{
	clazz_database_record = jvm_runtime_info::instance().get_class_cache(arg_enum_v<irisvika_class_key::database_record>);
	clazz_database_search_result = jvm_runtime_info::instance().get_class_cache(arg_enum_v<irisvika_class_key::database_search_result>);
	clazz_null_pointer_exception = jvm_runtime_info::instance().get_class_cache(arg_enum_v<irisvika_class_key::null_pointer_exception>);
	clazz_illegal_argument_exception = jvm_runtime_info::instance().get_class_cache(arg_enum_v<irisvika_class_key::illegal_argument_exception>);
	method_database_record_constructor = jvm_runtime_info::instance().get_method_cache(arg_enum_v<irisvika_method_key::database_record_constructor>);
	method_database_search_result_constructor = jvm_runtime_info::instance().get_method_cache(arg_enum_v<irisvika_method_key::database_search_result_constructor>);
	field_face_service_m_impl = std::make_shared<jvm_field_accessor<jlong>>(env, jvm_runtime_info::instance().get_field_cache(arg_enum_v<irisvika_field_key::face_service_m_impl>));
	field_database_record_m_key = std::make_shared<jvm_field_accessor<jstring>>(env, jvm_runtime_info::instance().get_field_cache(arg_enum_v<irisvika_field_key::database_record_m_key>));
	field_database_record_m_feature = std::make_shared<jvm_field_accessor<jfloatArray>>(env, jvm_runtime_info::instance().get_field_cache(arg_enum_v<irisvika_field_key::database_record_m_feature>));
	field_face_service_m_impl->set(obj, reinterpret_cast<jlong>(new face_service{ single_database_capacity, dimension, utils::to_string(env, working_directory) }));
}
