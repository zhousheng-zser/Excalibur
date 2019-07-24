#ifndef _LONGINUS_JNI_HPP_
#define _LONGINUS_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_init(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_set(JNIEnv *, jobject, jint, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_finalize(JNIEnv *, jobject);
#ifdef USE_OPENCV
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detect(JNIEnv *, jobject, jlong, jint, jfloat, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectwithInfo(JNIEnv *, jobject, jlong, jint, jfloat, jint, jint);
#endif
JNIEXPORT jstring JNICALL Java_com_glasssix_Longimila_Longimila_getVersion(JNIEnv *, jclass);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_match(JNIEnv *, jobject, jobjectArray, jint);
#ifdef USE_OPENCV
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFace(JNIEnv *, jobject, jlong, jobjectArray, jobjectArray);
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignSingleFace(JNIEnv *, jobject, jlong);
#endif
#ifndef TRIAL
#ifdef USE_OPENCV
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectEx(JNIEnv *, jobject, jlong, jint, jfloatArray, jfloat, jint, jint);
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif