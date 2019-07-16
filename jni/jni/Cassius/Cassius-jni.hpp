#ifndef _CASSIUS_JNI_HPP_
#define _CASSIUS_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Cassius_CassiusFeature_init(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Cassius_CassiusFeature_finalize(JNIEnv *, jobject);
JNIEXPORT jstring JNICALL Java_com_glasssix_Cassius_CassiusFeature_getVersion(JNIEnv *, jclass);
JNIEXPORT jfloatArray JNICALL Java_com_glasssix_Cassius_CassiusFeature_Forward(JNIEnv *, jobject, jlong, jint);


#ifdef __cplusplus
}
#endif

#endif