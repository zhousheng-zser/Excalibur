#include "Irisvika-jni.hpp"
#include "IrisvianSearchWrapper.hpp"
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
	
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = new glasssix::Irisvian::IrisvianSearchWrapper(&baseData, dimension);
	env->SetLongField(thiz, fid_mObject, (jlong)pIrisvianSearchWrapper);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init(JNIEnv *env, jobject thiz, jint dimension)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = new glasssix::Irisvian::IrisvianSearchWrapper(dimension);
	env->SetLongField(thiz, fid_mObject, (jlong)pIrisvianSearchWrapper);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_finalize(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	if(pIrisvianSearchWrapper != nullptr)
	{
		delete pIrisvianSearchWrapper;
		pIrisvianSearchWrapper = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong)pIrisvianSearchWrapper);
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
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	pIrisvianSearchWrapper->loadGraph(jstring2string(env, graphPath).c_str());
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_loadGraphwithData(JNIEnv *env, jobject thiz, jstring graphPath, jstring basedataPath)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	pIrisvianSearchWrapper->loadGraph(jstring2string(env, graphPath).c_str(), jstring2string(env, basedataPath).c_str());
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_optimizeGraph(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	pIrisvianSearchWrapper->optimizeGraph();
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_searchVector(JNIEnv *env, jobject thiz, jobjectArray queryData, jint topK, jobjectArray returnIDsArray, jobjectArray returnSimilaritiesArray)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
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
	pIrisvianSearchWrapper->searchVector(&query_vec, topK, returnIDs, returnSimilarities);
	
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
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
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
	
	pIrisvianSearchWrapper->saveResult(jstring2string(env, resultPath).c_str(), returnIDs);
	
	env->DeleteLocalRef(clazz);
}

jstring char2Jstring(JNIEnv *env, const char *pat, size_t len)
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

JNIEXPORT jstring JNICALL Java_com_glasssix_Irisvika_Irisvika_getVersion(JNIEnv *env, jobject thiz)
{
	std::string version = glasssix::Irisvian::IrisvianSearch::getVersion();
	return char2Jstring(env, version.c_str(), version.length());
}

JNIEXPORT jint JNICALL Java_com_glasssix_Irisvika_Irisvika_buildGraph(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	int ret = pIrisvianSearchWrapper->buildGraph();
	
	env->DeleteLocalRef(clazz);
	
	return ret;
}

JNIEXPORT jint JNICALL Java_com_glasssix_Irisvika_Irisvika_buildGraph_withData(JNIEnv *env, jobject thiz, jobjectArray baseData)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	jsize dataSize = env->GetArrayLength(baseData);
	std::vector<const float *> data_vec(dataSize, nullptr);
	
	jfloatArray array0 = (jfloatArray)env->GetObjectArrayElement(baseData, 0);
	jsize dimension = env->GetArrayLength(array0);
	env->DeleteLocalRef(array0);
	
	for(size_t i = 0; i < dataSize; i++)
	{
		jfloatArray array = (jfloatArray)env->GetObjectArrayElement(baseData, i);
		float *data = new float[dimension];
		env->GetFloatArrayRegion(array, 0, dimension, data);
		
		data_vec.push_back(data);
		env->DeleteLocalRef(array);
	}
	
	int ret = pIrisvianSearchWrapper->buildGraph(&data_vec);
	
	env->DeleteLocalRef(clazz);
	
	return ret;
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_saveGraph(JNIEnv *env, jobject thiz, jstring graphPath)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	pIrisvianSearchWrapper->saveGraph(jstring2string(env, graphPath).c_str());
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_saveGraph_withData(JNIEnv *env, jobject thiz, jstring graphPath, jstring baseDataPath)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	pIrisvianSearchWrapper->saveGraph(jstring2string(env, graphPath).c_str(), jstring2string(env, baseDataPath).c_str());
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_getBaseData(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::Irisvian::IrisvianSearchWrapper *pIrisvianSearchWrapper = (glasssix::Irisvian::IrisvianSearchWrapper *)p;
	
	std::vector<const float *> *baseDataPtr = pIrisvianSearchWrapper->getBasedata();
	int dimension = pIrisvianSearchWrapper->getDimension();
	jsize dataSize = baseDataPtr->size();
	
	jclass floatArrayClazz = env->FindClass("[F");
	jjobjectArray baseDataArray = env->NewObjectArray(dataSize, floatArrayClazz, nullptr);

	for (size_t i = 0; i < dataSize; i++)
	{
		jfloatArray data = env->NewFloatArray(dimension);
		
		env->SetFloatArrayRegion(data, 0, dimension, (*baseDataPtr)[i]);
		env->SetObjectArrayElement(baseDataArray, i, data);
		env->DeleteLocalRef(data);
	}
	
	env->DeleteLocalRef(floatArrayClazz);
	env->DeleteLocalRef(clazz);
	
	return baseDataArray;
}