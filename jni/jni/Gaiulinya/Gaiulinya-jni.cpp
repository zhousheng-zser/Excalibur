#include "Gaiulinya-jni.hpp"
#include "GaiusFeature.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

JNIEXPORT void JNICALL Java_com_glasssix_Gaius_GaiusFeature_init(JNIEnv *env, jobject thiz, jint device)
{
	glasssix::gaius::GaiusFeature *pGaius = new glasssix::gaius::GaiusFeature(device);
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	env->SetLongField(thiz, fid_mObject, (jlong)pGaius);
	
	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Gaius_GaiusFeature_finalize(JNIEnv *env, jobject thiz)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::gaius::GaiusFeature *pGaius = (glasssix::gaius::GaiusFeature *)p;
	if(pGaius != nullptr)
	{
		delete pGaius;
		pGaius = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong)pGaius);
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

JNIEXPORT jstring JNICALL Java_com_glasssix_Gaius_GaiusFeature_getVersion(JNIEnv *env, jclass clazz)
{
	std::string version = glasssix::gaius::GaiusFeature::getVersion();
	return char2Jstring(env, version.c_str(), version.length());
}

JNIEXPORT jfloatArray JNICALL Java_com_glasssix_Gaius_GaiusFeature_Forward(JNIEnv *env, jobject thiz, jlong MatNativeObj, jint order)
{
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::gaius::GaiusFeature *pGaius = (glasssix::gaius::GaiusFeature *)p;
	
	cv::Mat &mat = *(cv::Mat *)MatNativeObj;
	std::vector<std::vector<float> > feature_vecs = pGaius->Forward(mat.data, 1, order);
	
	jsize featureSize = feature_vecs[0].size();
	jfloatArray featureArray = env->NewFloatArray(featureSize);
	env->SetFloatArrayRegion(featureArray, 0, featureSize, feature_vecs[0].data());
	
	env->DeleteLocalRef(clazz);
	
	return featureArray;
}
