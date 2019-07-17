#ifndef _LONGINUS_JNI_HPP_
#define _LONGINUS_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_init(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_set(JNIEnv *, jobject, jint, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Longimila_Longimila_finalize(JNIEnv *, jobject);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detect_JIFI(JNIEnv *, jobject, jlong, jint, jfloat, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_detect_JIFII(JNIEnv *, jobject, jlong, jint, jfloat, jint, jint);
JNIEXPORT jstring JNICALL Java_com_glasssix_Longimila_Longimila_getVersion(JNIEnv *, jclass);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Longimila_Longimila_match(JNIEnv *, jobject, jobjectArray, jint);
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFace_J_3I_3I(JNIEnv *, jobject, jlong, jintArray, jintArray);
JNIEXPORT jbyteArray JNICALL Java_com_glasssix_Longimila_Longimila_alignFace_J(JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif

#endif