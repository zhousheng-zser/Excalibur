#include "Irisvika-jni.hpp"
#include <vector>
#include <cstring>
#include <cstdlib>
#include "knn_service.hpp"

static const char *knn_search_result_path = "com/glasssix/Irisvika/knn_search_result";
static const char *knn_mapping_data_path = "com/glasssix/Irisvika/knn_mapping_data";

static std::string jstring2string(JNIEnv *env, jstring jstr)
{
	char *rtn = nullptr;
	jclass strClazz = env->FindClass("java/lang/String");
	jstring strEncode = env->NewStringUTF("utf-8");
	jmethodID mid = env->GetMethodID(strClazz, "getBytes", "(Ljava/lang/String;)[B");
	jbyteArray bytes = (jbyteArray)env->CallObjectMethod(jstr, mid, strEncode);
	jsize len = env->GetArrayLength(bytes);
	jbyte *p = env->GetByteArrayElements(bytes, JNI_FALSE);
	if(len > 0)
	{
		rtn = (char *)malloc(len + 1);
		memcpy(rtn, p, len);
		rtn[len] = 0;
	}
	
	env->ReleaseByteArrayElements(bytes, p, 0);
	env->DeleteLocalRef(bytes);
	env->DeleteLocalRef(strEncode);
	env->DeleteLocalRef(strClazz);
	
	std::string str(rtn);
	free(rtn);
	
	return str;
}

static jstring char2Jstring(JNIEnv *env, const char *pat, size_t len)
{
	jclass strClazz = env->FindClass("java/lang/String");
	jmethodID mid_String_constructor = env->GetMethodID(strClazz, "<init>", "([BLjava/lang/String;)V");
	jbyteArray bytes = env->NewByteArray(len);
	env->SetByteArrayRegion(bytes, 0, len, (jbyte *)pat);
	jstring encoding = env->NewStringUTF("utf-8");
	
	jstring jstr = (jstring)env->NewObject(strClazz, mid_String_constructor, bytes, encoding);
	
	env->DeleteLocalRef(encoding);
	env->DeleteLocalRef(bytes);
	env->DeleteLocalRef(strClazz);
	
	return jstr;
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init(JNIEnv *env, jobject thiz, jint max_items, jstring new_save_path, jstring tmp_path)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	
	std::string new_save_path_ = jstring2string(env, new_save_path);
	std::string tmp_path_ = jstring2string(env, tmp_path);
	
	glasssix::irisviel::knn_service *pknn_service = new glasssix::irisviel::knn_service(max_items, new_save_path_, tmp_path_);
	env->SetLongField(thiz, fid_mObject, (jlong)pknn_service);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_finalize(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	if(pknn_service != nullptr)
	{
		delete pknn_service;
		pknn_service = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong)pknn_service);
	}
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT jstring JNICALL Java_com_glasssix_Irisvika_Irisvika_save_path(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	std::string path = pknn_service->save_path();
	
	env->DeleteLocalRef(clazz);
	return char2Jstring(env, path.c_str(), path.length());
}

JNIEXPORT jstring JNICALL Java_com_glasssix_Irisvika_Irisvika_tmp_path(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	std::string path = pknn_service->tmp_path();
	
	env->DeleteLocalRef(clazz);
	return char2Jstring(env, path.c_str(), path.length());
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_build(JNIEnv *env, jobject thiz, jobjectArray files)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	std::vector<std::string> files_vec;
	jsize files_num = env->GetArrayLength(files);
	for(int i = 0; i < files_num; i++)
	{
		jstring file = (jstring)env->GetObjectArrayElement(files, i);
		files_vec.push_back(jstring2string(env, file));
		env->DeleteLocalRef(file);
	}
	
	if(files_num)
		pknn_service->build(files_vec);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_search(JNIEnv *env, jobject thiz, jfloatArray query_feature, jint top)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	jsize feature_size = env->GetArrayLength(query_feature);
	std::vector<const float> feature_vec(feature_size);
	
	env->GetFloatArrayRegion(query_feature, 0, feature_size, &feature_vec[0]);
	
	std::vector<glasssix::irisviel::knn_search_result> result_vec = pknn_service->search(feature_vec, top);
	size_t result_size = result_vec.size();
	
	jclass knn_search_result_clazz = env->FindClass(knn_search_result_path);
	std::string func_type = "(L" + knn_mapping_data_path + ";F)V";
	jmethodID mid_knn_search_result_constructor = env->GetMethodID(knn_search_result_clazz, "<init>", func_type.c_str());
	jobjectArray result_array = env->NewObjectArray(result_size, knn_search_result_clazz, nullptr);
	
	jclass knn_mapping_data_clazz = env->FindClass(knn_mapping_data_path);
	jmethodID mid_knn_mapping_data_constructor = env->GetMethodID(knn_mapping_data_clazz, "<init>", "()V");
	jfieldID fid_feature = env->GetFieldID(knn_mapping_data_clazz, "feature", "[F");
	jfieldID fid_key = env->GetFieldID(knn_mapping_data_clazz, "key", "[B");
	for(size_t i = 0; i < result_size; i++)
	{
		jobject data = env->NewObject(knn_mapping_data_clazz, mid_knn_mapping_data_constructor);
		jfloatArray found_feature = env->NewFloatArray(Feature_Size);
		jbyteArray key = env->NewByteArray(33);
		env->SetFloatArrayRegion(found_feature, 0, Feature_Size, result_vec[i].data.feature);
		env->SetByteArrayRegion(key, 0, 33, result_vec[i].data.key);
		env->SetObjectField(data, fid_feature, found_feature);
		env->SetObjectField(data, fid_key, key);
		
		jobject result = env->NewObject(knn_search_result_clazz, mid_knn_search_result_constructor, data, result_vec[i].distance_in_percentage);
		env->SetObjectArrayElement(result_array, i, result);
		
		env->DeleteLocalRef(result);
		env->DeleteLocalRef(key);
		env->DeleteLocalRef(found_feature);
		env->DeleteLocalRef(data);
	}
	
	env->DeleteLocalRef(knn_mapping_data_clazz);
	env->DeleteLocalRef(knn_search_result_clazz);
	env->DeleteLocalRef(clazz);
	
	return result_array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_delete_features(JNIEnv *env, jobject thiz, jobjectArray keys)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	jsize keys_num = env->GetArrayLength(keys);
	std::vector<std::string> keys_vec;
	
	for(size_t i = 0; i < keys_num; i++)
	{
		jstring key = (jstring)env->GetObjectArrayElement(keys, i);
		keys_vec.push_back(jstring2string(env, key));
		env->DeleteLocalRef(key);
	}
	
	jclass strClazz = env->FindClass("java/lang/String");
	
	std::vector<std::string> needs_delete_files_vec = pknn_service->delete_features(keys_vec);
	size_t needs_delete_files_num = needs_delete_files.size();
	jobjectArray needs_delete_files_array = env->NewObjectArray(needs_delete_files_num, strClazz, nullptr);
	for(size_t i = 0; i < needs_delete_files_num; i++)
	{
		jstring file = char2Jstring(env, needs_delete_files_vec[i].c_str(), needs_delete_files_vec[i].length());
		env->SetObjectArrayElement(needs_delete_files_array, i, file);
		env->DeleteLocalRef(file);
	}
	
	env->DeleteLocalRef(strClazz);
	env->DeleteLocalRef(clazz);
	return needs_delete_files_array;
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_delete_feature(JNIEnv *env, jobject thiz, jstring key)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	std::vector<std::string> needs_delete_files_vec = pknn_service->delete_features(jstring2string(env, key));
	
	jclass strClazz = env->FindClass("java/lang/String");
	
	size_t needs_delete_files_num = needs_delete_files.size();
	jobjectArray needs_delete_files_array = env->NewObjectArray(needs_delete_files_num, strClazz, nullptr);
	for(size_t i = 0; i < needs_delete_files_num; i++)
	{
		jstring file = char2Jstring(env, needs_delete_files_vec[i].c_str(), needs_delete_files_vec[i].length());
		env->SetObjectArrayElement(needs_delete_files_array, i, file);
		env->DeleteLocalRef(file);
	}
	
	env->DeleteLocalRef(strClazz);
	env->DeleteLocalRef(clazz);
	return needs_delete_files_array;
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_add_features(JNIEnv *env, jobject thiz, jobjectArray dataObjArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	jclass knn_mapping_data_clazz = jclass knn_mapping_data_clazz = env->FindClass(knn_mapping_data_path);
	jfieldID fid_feature = env->GetFieldID(knn_mapping_data_clazz, "feature", "[F");
	jfieldID fid_key = env->GetFieldID(knn_mapping_data_clazz, "key", "[B");
	
	jsize data_num = env->GetArrayLength(dataObjArray);
	std::vector<glasssix::irisviel::knn_mapping_data> data_vec;
	for(size_t i = 0; i < data_num; i++)
	{
		knn_mapping_data data;
		
		jobject dataObj = env->GetObjectArrayElement(dataObjArray, i);
		
		jfloatArray feature_array = (jfloatArray)env->GetObjectField(dataObj, fid_feature);
		jsize feature_size = env->GetArrayLength(feature_array);
		env->GetFloatArrayRegion(feature_array, 0, feature_size, data.feature);
		
		jbyteArray key_array = (jbyteArray)env->GetObjectField(dataObj, fid_key);
		jsize key_size = env->GetArrayLength(key_array);
		env->GetByteArrayRegion(key_array, 0, key_size, data.key);
		
		data_vec.push_back(data);
		
		env->DeleteLocalRef(key_array);
		env->DeleteLocalRef(feature_array);
		env->DeleteLocalRef(dataObj);
	}
	
	pknn_service->add_features(data_vec);
	
	env->DeleteLocalRef(knn_mapping_data_clazz);
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_add_feature(JNIEnv *env, jobject thiz, jobject dataObj)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	jclass knn_mapping_data_clazz = jclass knn_mapping_data_clazz = env->FindClass(knn_mapping_data_path);
	jfieldID fid_feature = env->GetFieldID(knn_mapping_data_clazz, "feature", "[F");
	jfieldID fid_key = env->GetFieldID(knn_mapping_data_clazz, "key", "[B");
	
	knn_mapping_data data;
	
	jfloatArray feature_array = env->GetObjectField(dataObj, fid_feature);
	jsize feature_size = env->GetArrayLength(feature_array);
	env->GetFloatArrayRegion(feature_array, 0, feature_size, data.feature);
	
	jbyteArray key_array = (jbyteArray)env->GetObjectField(dataObj, fid_key);
	jsize key_size = env->GetArrayLength(key_array);
	env->GetByteArrayRegion(key_array, 0, key_size, data.key);
	
	pknn_service->add_feature(data);
	
	env->DeleteLocalRef(key_array);
	env->DeleteLocalRef(feature_array);
	
	env->DeleteLocalRef(knn_mapping_data_clazz);
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_update(JNIEnv *env, jobject thiz, jobject dataObj)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	jclass knn_mapping_data_clazz = jclass knn_mapping_data_clazz = env->FindClass(knn_mapping_data_path);
	jfieldID fid_feature = env->GetFieldID(knn_mapping_data_clazz, "feature", "[F");
	jfieldID fid_key = env->GetFieldID(knn_mapping_data_clazz, "key", "[B");
	
	knn_mapping_data data;
	
	jfloatArray feature_array = env->GetObjectField(dataObj, fid_feature);
	jsize feature_size = env->GetArrayLength(feature_array);
	env->GetFloatArrayRegion(feature_array, 0, feature_size, data_.feature);
	
	jbyteArray key_array = (jbyteArray)env->GetObjectField(dataObj, fid_key);
	jsize key_size = env->GetArrayLength(key_array);
	env->GetByteArrayRegion(key_array, 0, key_size, data_.key);
	
	pknn_service->update(data);
	
	env->DeleteLocalRef(key_array);
	env->DeleteLocalRef(feature_array);
	
	env->DeleteLocalRef(knn_mapping_data_clazz);
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_update_more(JNIEnv *env, jobject thiz, jobjectArray dataObjArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::irisviel::knn_service *pknn_service = (glasssix::irisviel::knn_service *)p;
	
	jclass knn_mapping_data_clazz = jclass knn_mapping_data_clazz = env->FindClass(knn_mapping_data_path);
	jfieldID fid_feature = env->GetFieldID(knn_mapping_data_clazz, "feature", "[F");
	jfieldID fid_key = env->GetFieldID(knn_mapping_data_clazz, "key", "[B");
	
	jsize data_num = env->GetArrayLength(dataObjArray);
	std::vector<glasssix::irisviel::knn_mapping_data> data_vec;
	for(size_t i = 0; i < data_num; i++)
	{
		knn_mapping_data data;
		
		jobject dataObj = env->GetObjectArrayElement(dataObjArray, i);
		
		jfloatArray feature_array = env->GetObjectField(dataObj, fid_feature);
		jsize feature_size = env->GetArrayLength(feature_array);
		env->GetFloatArrayRegion(feature_array, 0, feature_size, data.feature);
		
		jbyteArray key_array = (jbyteArray)env->GetObjectField(dataObj, fid_key);
		jsize key_size = env->GetArrayLength(key_array);
		env->GetByteArrayRegion(key_array, 0, key_size, data.key);
		
		data_vec.push_back(data);
		
		env->DeleteLocalRef(key_array);
		env->DeleteLocalRef(feature_array);
		env->DeleteLocalRef(dataObj);
	}
	
	pknn_service->update_more(data_vec);
	
	env->DeleteLocalRef(knn_mapping_data_clazz);
	env->DeleteLocalRef(clazz);
}