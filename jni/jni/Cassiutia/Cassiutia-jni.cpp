#include "Cassiutia-jni.hpp"
#include "CassiusFeature.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <string>
#include <vector>

JNIEXPORT void JNICALL Java_com_glasssix_Cassiutia_Cassiutia_init(JNIEnv *env, jobject thiz, jint device)
{
	glasssix::cassius::CassiusFeature *pCassius = new glasssix::cassius::CassiusFeature(device);
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	env->SetLongField(thiz, fid_mObject, (jlong)pCassius);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Cassiutia_Cassiutia_finalize(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::cassius::CassiusFeature *pCassius = (glasssix::cassius::CassiusFeature *)p;
	if(pCassius != nullptr)
	{
		delete pCassius;
		pCassius = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong)pCassius);
	}
	
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

JNIEXPORT jstring JNICALL Java_com_glasssix_Cassiutia_Cassiutia_getVersion(JNIEnv *env, jclass clazz)
{
	std::string version = glasssix::cassius::CassiusFeature::getVersion();
	return char2Jstring(env, version.c_str(), version.length());
}

#ifdef USE_OPENCV
JNIEXPORT jfloatArray JNICALL Java_com_glasssix_Cassiutia_Cassiutia_Forward(JNIEnv *env, jobject thiz, jlong MatNativeObj, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::cassius::CassiusFeature *pCassius = (glasssix::cassius::CassiusFeature *)p;
	
	cv::Mat &mat = *(cv::Mat *)MatNativeObj;
	std::vector<std::vector<float> > feature_vecs = pCassius->Forward(mat.data, 1, order);
	
	jsize featureSize = feature_vecs[0].size();
	jfloatArray featureArray = env->NewFloatArray(featureSize);
	env->SetFloatArrayRegion(featureArray, 0, featureSize, feature_vecs[0].data());
	
	env->DeleteLocalRef(clazz);
	
	return featureArray;
}
#endif

JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Cassiutia_Cassiutia_ForwardwithMetaData(JNIEnv *env, jobject thiz, jbyteArray dataArray, jint faceCount, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::cassius::CassiusFeature *pCassius = (glasssix::cassius::CassiusFeature *)p;

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);

	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *)data_vec.data());

	std::vector<std::vector<float> > feature_vecs = pCassius->Forward((unsigned char *)data_vec.data(), faceCount, order);

	jsize featureArraySize = feature_vecs.size();
	jsize dimension = feature_vecs[0].size();

	jclass floatArrayClazz = env->FindClass("[F");

	jobjectArray featureArray = env->NewObjectArray(featureArraySize, floatArrayClazz, nullptr);

	for (size_t i = 0; i < featureArraySize; i++)
	{
		jfloatArray feature = env->NewFloatArray(dimension);
		env->SetFloatArrayRegion(feature, 0, dimension, feature_vecs[i].data());
		env->SetObjectArrayElement(featureArray, i, feature);
		env->DeleteLocalRef(feature);
	}
	
	env->DeleteLocalRef(floatArrayClazz);
	env->DeleteLocalRef(clazz);

	return featureArray;
}
