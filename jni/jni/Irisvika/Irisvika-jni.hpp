#ifndef _IRISVIKA_JNI_HPP_
#define _IRISVIKA_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jstring JNICALL Java_com_glasssix_Irisvika_Irisvika_getVersion(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init_withData(JNIEnv *, jobject, jobjectArray);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_finalize(JNIEnv *, jobject);
JNIEXPORT jint JNICALL Java_com_glasssix_Irisvika_Irisvika_buildGraph(JNIEnv *, jobject);
JNIEXPORT jint JNICALL Java_com_glasssix_Irisvika_Irisvika_buildGraph_withData(JNIEnv *, jobject, jobjectArray);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_saveGraph(JNIEnv *, jobject, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_saveGraph_withData(JNIEnv *, jobject, jstring, jstring);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_getBaseData(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_loadGraph(JNIEnv *, jobject, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_loadGraphwithData(JNIEnv *, jobject, jstring, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_optimizeGraph(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_searchVector(JNIEnv *, jobject, jobjectArray, jint, jobjectArray, jobjectArray);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_saveResult(JNIEnv *, jobject, jstring, jobjectArray);

#ifdef __cplusplus
}
#endif

#endif //!_IRISVIKA_JNI_HPP_