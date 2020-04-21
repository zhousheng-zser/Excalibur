#pragma once

#ifndef _Included_com_glasssix_common_NativeNumericUtils
#define _Included_com_glasssix_common_NativeNumericUtils

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToIntUnsigned
 * Signature: ([B)[I
 */
JNIEXPORT jintArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToIntUnsigned(JNIEnv* env, jclass clazz, jbyteArray data);

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToShortUnsigned
 * Signature: ([B)[S
 */
JNIEXPORT jshortArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToShortUnsigned(JNIEnv* env, jclass clazz, jbyteArray data);

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToLongUnsigned
 * Signature: ([B)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToLongUnsigned(JNIEnv* env, jclass clazz, jbyteArray data);

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToFloatUnsigned
 * Signature: ([B)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToFloatUnsigned(JNIEnv* env, jclass clazz, jbyteArray data);

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToDoubleUnsigned
 * Signature: ([B)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToDoubleUnsigned(JNIEnv* env, jclass clazz, jbyteArray data);

#ifdef __cplusplus
}
#endif
#endif
