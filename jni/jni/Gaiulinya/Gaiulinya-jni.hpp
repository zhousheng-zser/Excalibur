#ifndef _GAIULINYA_JNI_HPP_
#define _GAIULINYA_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Gaiulinya_Gaiulinya_init(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Gaiulinya_Gaiulinya_finalize(JNIEnv *, jobject);
JNIEXPORT jstring JNICALL Java_com_glasssix_Gaiulinya_Gaiulinya_getVersion(JNIEnv *, jclass);
JNIEXPORT jfloatArray JNICALL Java_com_glasssix_Gaiulinya_Gaiulinya_Forward(JNIEnv *, jobject, jlong, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Gaiulinya_Gaiulinya_ForwardwithMetaData(JNIEnv *, jobject, jbyteArray, jint, jint);


#ifdef __cplusplus
}
#endif

#endif // !_GAIULINYA_JNI_HPP_