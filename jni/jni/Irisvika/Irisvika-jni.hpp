#ifndef _IRISVIKA_JNI_HPP_
#define _IRISVIKA_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_init__3_3F(JNIEnv *, jobject, jobjectArray);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_init_I(JNIEnv *, jobject, jint);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_finalize(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_loadGraph_LJava_lang_String_2(JNIEnv *, jobject, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_loadGraph_LJava_lang_String_2LJava_lang_String_2(JNIEnv *, jobject, jstring, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_optimizeGraph(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_searchVector(JNIEnv *, jobject, jobjectArray, jint, jobjectArray, jobjectArray);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvian_Search_saveResult(JNIEnv *, jobject, jstring, jobjectArray);

#ifdef __cplusplus
}
#endif

#endif //!_IRISVIKA_JNI_HPP_