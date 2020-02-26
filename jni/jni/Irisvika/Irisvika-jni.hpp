#ifndef _IRISVIKA_JNI_HPP_
#define _IRISVIKA_JNI_HPP_

#include <jni.h>
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_init(JNIEnv *, jobject, jint, jstring, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_finalize(JNIEnv *, jobject);
JNIEXPORT jstring JNICALL Java_com_glasssix_Irisvika_Irisvika_save_path(JNIEnv *, jobject);
JNIEXPORT jstring JNICALL Java_com_glasssix_Irisvika_Irisvika_tmp_path(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_build(JNIEnv *, jobject, jobjectArray);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_search(JNIEnv *, jobject, jfloatArray, jint);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_delete_features(JNIEnv *, jobject, jobjectArray);
JNIEXPORT jobjectArray JNICALL Java_com_glasssix_Irisvika_Irisvika_delete_feature(JNIEnv *, jobject, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_add_features(JNIEnv *, jobject, jobjectArray);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_add_feature(JNIEnv *, jobject, jstring);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_update(JNIEnv *, jobject, jobject);
JNIEXPORT void JNICALL Java_com_glasssix_Irisvika_Irisvika_update_more(JNIEnv *, jobject, jobjectArray);

#ifdef __cplusplus
}
#endif

#endif //!_IRISVIKA_JNI_HPP_