#include "Gaiulinya-jni.hpp"
#include "GaiusFeature.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <string>
#include <vector>

#include <task_scheduler.hpp>
#include <business_task_id.hpp>

extern "C" {
JNIEXPORT void JNICALL
Java_com_glasssix_Gaiulinya_Gaiulinya_init(JNIEnv *env, jobject thiz, jint device) {
	glasssix::gaius::GaiusFeature *pGaius = new glasssix::gaius::GaiusFeature(device);
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	env->SetLongField(thiz, fid_mObject, (jlong) pGaius);

	env->DeleteLocalRef(clazz);
}

JNIEXPORT void JNICALL Java_com_glasssix_Gaiulinya_Gaiulinya_finalize(JNIEnv *env, jobject thiz) {
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::gaius::GaiusFeature *pGaius = (glasssix::gaius::GaiusFeature *) p;
	if (pGaius != nullptr) {
		delete pGaius;
		pGaius = nullptr;
		env->SetLongField(thiz, fid_mObject, (jlong) pGaius);
	}

	env->DeleteLocalRef(clazz);
}

jstring char2Jstring(JNIEnv *env, const char *pat, size_t len) {
	jclass strClazz = env->FindClass("java/lang/String");
	jmethodID mid_String_constructor = env->GetMethodID(strClazz, "<init>",
														"([BLjava/lang/String;)V");
	jbyteArray bytes = env->NewByteArray(len);
	env->SetByteArrayRegion(bytes, 0, len, (jbyte *) pat);
	jstring encoding = env->NewStringUTF("utf-8");

	jstring jstr = (jstring) env->NewObject(strClazz, mid_String_constructor, bytes, encoding);

	env->DeleteLocalRef(encoding);
	env->DeleteLocalRef(bytes);
	env->DeleteLocalRef(strClazz);

	return jstr;
}

JNIEXPORT jstring JNICALL
Java_com_glasssix_Gaiulinya_Gaiulinya_getVersion(JNIEnv *env, jclass clazz) {
	std::string version = glasssix::gaius::GaiusFeature::getVersion();
	return char2Jstring(env, version.c_str(), version.length());
}

#ifdef USE_OPENCV
JNIEXPORT jfloatArray JNICALL
Java_com_glasssix_Gaiulinya_Gaiulinya_ForwardbyMat(JNIEnv *env, jobject thiz, jlong MatNativeObj,
												   jint order) {
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::gaius::GaiusFeature *pGaius = (glasssix::gaius::GaiusFeature *) p;

	cv::Mat &mat = *(cv::Mat *) MatNativeObj;

	auto feature_vecs = glasssix::task_scheduler::current().commit(glasssix::business_task_id::extraction_and_alignment, [=] {
		return pGaius->Forward(mat.data, 1, order);
	}).get();

	jsize featureSize = feature_vecs[0].size();
	jfloatArray featureArray = env->NewFloatArray(featureSize);
	env->SetFloatArrayRegion(featureArray, 0, featureSize, feature_vecs[0].data());

	env->DeleteLocalRef(clazz);

	return featureArray;
}
#endif

JNIEXPORT jobjectArray JNICALL
Java_com_glasssix_Gaiulinya_Gaiulinya_ForwardbyMetaData(JNIEnv *env, jobject thiz,
														jbyteArray dataArray, jint faceCount,
														jint order) {
	jclass clazz = env->GetObjectClass(thiz);
	jfieldID fid_mObject = env->GetFieldID(clazz, "mObject", "J");
	jlong p = env->GetLongField(thiz, fid_mObject);
	glasssix::gaius::GaiusFeature *pGaius = (glasssix::gaius::GaiusFeature *) p;

	jsize dataSize = env->GetArrayLength(dataArray);
	std::vector<unsigned char> data_vec(dataSize, 0);

	env->GetByteArrayRegion(dataArray, 0, dataSize, (jbyte *) data_vec.data());


	auto feature_vecs = glasssix::task_scheduler::current().commit(glasssix::business_task_id::extraction_and_alignment, [=] {
		return pGaius->Forward((unsigned char *) data_vec.data(), faceCount, order);
	}).get();

	jsize featureArraySize = feature_vecs.size();
	jsize dimension = feature_vecs[0].size();

	jclass floatArrayClazz = env->FindClass("[F");

	jobjectArray featureArray = env->NewObjectArray(featureArraySize, floatArrayClazz, nullptr);

	for (size_t i = 0; i < featureArraySize; i++) {
		jfloatArray feature = env->NewFloatArray(dimension);

		env->SetFloatArrayRegion(feature, 0, dimension, feature_vecs[i].data());
		env->SetObjectArrayElement(featureArray, i, feature);
		env->DeleteLocalRef(feature);
	}

	env->DeleteLocalRef(floatArrayClazz);
	env->DeleteLocalRef(clazz);

	return featureArray;
}
}
