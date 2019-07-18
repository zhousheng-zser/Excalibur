#include "Irisvika-jni.hpp"
#include "SearchWrapper.hpp"
#include <vector>
#include <string.h>

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init_withData(JNIEnv *env, jobject thiz, jobjectArray baseDataArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	
	jsize baseDataArraySize = env->GetArrayLength(baseDataArray);
	std::vector<const float *> baseData(baseDataArraySize, nullptr);
	
	jfloatArray array0 = (jfloatArray)env->GetObjectArrayElement(baseDataArray, 0);
	jsize dimension = env->GetArrayLength(array0);
	env->DeleteLocalRef(array0);
	
	for(size_t i = 0; i < baseDataArraySize; i++)
	{
		float *data = new float[dimension];
		jfloatArray array = (jfloatArray)env->GetObjectArrayElement(baseDataArray, i);
		env->GetFloatArrayRegion(array, 0, dimension, data);
		baseData[i] = data;
		env->DeleteLocalRef(array);
	}
	
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = new glasssix::Irisvian::SearchWrapper(&baseData, dimension);
	env->SetLongField(thiz, fid_mObject, (jlong)pSearchWrapper);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init(JNIEnv *env, jobject thiz, jint dimension)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = new glasssix::Irisvian::SearchWrapper(dimension);
	env->SetLongField(thiz, fid_mObject, (jlong)pSearchWrapper);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_finalize(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = (glasssix::Irisvian::SearchWrapper *)p;
	if(pSearchWrapper != nullptr)
	{
		delete pSearchWrapper;
		pSearchWrapper = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong)pSearchWrapper);
	}
	
	env->DeleteLocalRef(clazz);
}

std::string jstring2string(JNIEnv *env, jstring jstr)
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

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_loadGraph(JNIEnv *env, jobject thiz, jstring graphPath)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = (glasssix::Irisvian::SearchWrapper *)p;
	
	pSearchWrapper->loadGraph(jstring2string(env, graphPath).c_str());
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_loadGraphwithData(JNIEnv *env, jobject thiz, jstring graphPath, jstring basedataPath)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = (glasssix::Irisvian::SearchWrapper *)p;
	
	pSearchWrapper->loadGraph(jstring2string(env, graphPath).c_str(), jstring2string(env, basedataPath).c_str());
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_optimizeGraph(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = (glasssix::Irisvian::SearchWrapper *)p;
	
	pSearchWrapper->optimizeGraph();
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_searchVector(JNIEnv *env, jobject thiz, jobjectArray queryData, jint topK, jobjectArray returnIDsArray, jobjectArray returnSimilaritiesArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = (glasssix::Irisvian::SearchWrapper *)p;
	
	jsize querySize = env->GetArrayLength(queryData);
	std::vector<const float *> query_vec(querySize, nullptr);
	
	jfloatArray array0 = (jfloatArray)env->GetObjectArrayElement(queryData, 0);
	jsize dimension = env->GetArrayLength(array0);
	env->DeleteLocalRef(array0);
	
	for(size_t i = 0; i < querySize; i++)
	{
		jfloatArray array = (jfloatArray)env->GetObjectArrayElement(queryData, i);
		float *data = new float[dimension];
		env->GetFloatArrayRegion(array, 0, dimension, data);
		
		query_vec.push_back(data);
		env->DeleteLocalRef(array);
	}
	
	std::vector<std::vector<unsigned>> returnIDs;
	std::vector<std::vector<float>> returnSimilarities;
	pSearchWrapper->searchVector(&query_vec, topK, returnIDs, returnSimilarities);
	
	for(size_t i = 0; i < querySize; i++)
	{
		delete[] query_vec[i];
	}
	
	for(size_t i = 0; i < querySize; i++)
	{
		jintArray arrayID = (jintArray)env->GetObjectArrayElement(returnIDsArray, i);
		env->SetIntArrayRegion(arrayID, 0, topK, (jint *)&(returnIDs[i][0]));
		env->DeleteLocalRef(arrayID);
		
		jfloatArray arraySimilarity = (jfloatArray)env->GetObjectArrayElement(returnSimilaritiesArray, i);
		env->SetFloatArrayRegion(arraySimilarity, 0, topK, (jfloat *)&(returnSimilarities[i][0]));
		env->DeleteLocalRef(arraySimilarity);
	}
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_saveResult(JNIEnv *env, jobject thiz, jstring resultPath, jobjectArray returnIDsArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::SearchWrapper *pSearchWrapper = (glasssix::Irisvian::SearchWrapper *)p;
	
	jsize IDSize = env->GetArrayLength(returnIDsArray);	
	jintArray array0 = (jintArray)env->GetObjectArrayElement(returnIDsArray, 0);
	jsize dimension = env->GetArrayLength(array0);
	env->DeleteLocalRef(array0);
	
	std::vector<std::vector<unsigned> > returnIDs(IDSize, std::vector<unsigned>(IDSize));
	
	for(size_t i = 0; i < IDSize; i++)
	{
		jintArray array = (jintArray)env->GetObjectArrayElement(returnIDsArray, i);
		env->GetIntArrayRegion(array, 0, dimension, (jint *)&(returnIDs[i][0]));
		
		env->DeleteLocalRef(array);
	}
	
	pSearchWrapper->saveResult(jstring2string(env, resultPath).c_str(), returnIDs);
	
	env->DeleteLocalRef(clazz);
}