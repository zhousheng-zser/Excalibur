#ifndef _CASSIUTIA_JNI_HPP_
#define _CASSIUTIA_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Cassiutia_Cassiutia_init(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Cassiutia_Cassiutia_finalize(JNIEnv *, jobject);
JNIEXPORT jstring JNICALL Java_com_glasssix_Cassiutia_Cassiutia_getVersion(JNIEnv *, jclass);
JNIEXPORT jfloatArray JNICALL Java_com_glasssix_Cassiutia_Cassiutia_Forward(JNIEnv *, jobject, jlong, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Cassiutia_Cassiutia_ForwardwithMetaData(JNIEnv *, jobject, jbyteArray, jint, jint );


#ifdef __cplusplus
}
#endif

#endif //!_CASSIUTIA_JNI_HPP_