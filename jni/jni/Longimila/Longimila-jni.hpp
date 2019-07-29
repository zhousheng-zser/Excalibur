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
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectbyMat(JNIEnv *, jobject, jlong, jint, jfloat, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectwithInfobyMat(JNIEnv *, jobject, jlong, jint, jfloat, jint, jint);
#endif
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectbyMetaData(JNIEnv *, jobject, jbyteArray, jint, jint, jint, jfloat, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectwithInfobyMetaData(JNIEnv *, jobject, jbyteArray, jint, jint, jint, jfloat, jint, jint);
JNIEXPORT jstring JNICALL Java_com_glasssix_Longimila_Longimila_getVersion(JNIEnv *, jclass);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_match(JNIEnv *, jobject, jobjectArray, jint);
#ifdef USE_OPENCV
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFacebyMat(JNIEnv *, jobject, jlong, jobjectArray, jobjectArray);
#endif
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFacebyMetaData(JNIEnv *, jobject, jbyteArray, jint, jint, jobjectArray, jobjectArray);
#ifndef TRIAL
#ifdef USE_OPENCV
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExbyMat(JNIEnv *, jobject, jlong, jint, jfloatArray, jfloat, jint, jint);
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignSingleFacebyMat(JNIEnv *, jobject, jlong);
#endif
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detectExbyMetaData(JNIEnv *, jobject, jbyteArray, jint, jint, jint, jfloatArray, jfloat, jint, jint);
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignSingleFacebyMetaData(JNIEnv *, jobject, jbyteArray, jint, jint);
#endif

#ifdef __cplusplus
}
#endif

#endif