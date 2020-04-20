#include "com_glasssix_common_NativeNumericUtils.h"

#include <memory>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>

namespace
{
    constexpr bool is_big_endian()
    {
        constexpr std::uint32_t value = 0x01020304U;

        return static_cast<const std::uint8_t&>(value) == 1;
    }

    inline constexpr bool is_big_endian_v = is_big_endian();

    namespace details
    {
        template<typename T, typename Result, std::size_t... Indexes>
        constexpr auto get_numeric_element_helper(T* data, Result& result, std::index_sequence<Indexes...>) -> std::enable_if_t<std::is_arithmetic_v<T>>
        {
            constexpr std::ptrdiff_t total_bits = sizeof(Result) * CHAR_BIT;
            constexpr std::ptrdiff_t max_move_bits = total_bits - CHAR_BIT;
            constexpr std::ptrdiff_t baseline_move_bits = is_big_endian_v ? max_move_bits : 0;
            constexpr std::ptrdiff_t sign = is_big_endian_v ? -1 : 1;
            
            ((result += static_cast<Result>((static_cast<std::uintmax_t>(data[Indexes]) << (baseline_move_bits + sign * static_cast<std::ptrdiff_t>(Indexes) * CHAR_BIT))), ...);
        }
    }

    template<typename T, typename Result>
    constexpr auto get_numeric_element(T* data, Result& result) -> std::enable_if_t<std::is_arithmetic_v<T>>
    {
        details::get_numeric_element_helper(data, result, std::make_index_sequence<sizeof(Result)>{});
    }

    template<typename Result, typename ArrayAllocator, typename ArrayElementRetriver, typename ArrayNativeCleanUp>
    auto byte_to_unsigned(JNIEnv* env, jbyteArray data, ArrayAllocator&& allocator, ArrayElementRetriver&& retriver, ArrayNativeCleanUp&& cleanUp) -> std::enable_if_t<std::is_arithmetic_v<Result>, decltype(std::forward<ArrayAllocator>(allocator)(env, 0))>
    {
        jsize data_size = env->GetArrayLength(data);
        jsize result_size = static_cast<jsize>(data_size / sizeof(Result));
        auto result = std::forward<ArrayAllocator>(allocator)(env, result_size);
        std::shared_ptr<jbyte> native_data{ env->GetByteArrayElements(data, nullptr), [&](jbyte* inner){ env->ReleaseByteArrayElements(data, inner, JNI_ABORT); } };
        std::shared_ptr<Result> native_result{ std::forward<ArrayElementRetriver>(retriver)(env, result, nullptr), [&](Result* inner){ std::forward<ArrayNativeCleanUp>(cleanUp)(env, result, inner, JNI_COMMIT); } };

        auto data_ptr = native_data.get();
        auto result_ptr = native_result.get();

        for (; data_ptr < native_data.get() + data_size; data_ptr += sizeof(Result), result_ptr++)
        {
            get_numeric_element(data_ptr, *result_ptr);
        }

        return result;
    }
}

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToIntUnsigned
 * Signature: ([B)[I
 */
JNIEXPORT jintArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToIntUnsigned(JNIEnv* env, jclass clazz, jbyteArray data)
{
    return byte_to_unsigned<jint>(env, data, env->functions->NewIntArray, env->functions->GetIntArrayElements, env->functions->ReleaseIntArrayElements);
}

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToShortUnsigned
 * Signature: ([B)[S
 */
JNIEXPORT jshortArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToShortUnsigned(JNIEnv* env, jclass clazz, jbyteArray data)
{
    return byte_to_unsigned<jshort>(env, data, env->functions->NewShortArray, env->functions->GetShortArrayElements, env->functions->ReleaseShortArrayElements);
}

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToLongUnsigned
 * Signature: ([B)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToLongUnsigned(JNIEnv* env, jclass clazz, jbyteArray data)
{
    return byte_to_unsigned<jlong>(env, data, env->functions->NewLongArray, env->functions->GetLongArrayElements, env->functions->ReleaseLongArrayElements);
}

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToFloatUnsigned
 * Signature: ([B)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToFloatUnsigned(JNIEnv* env, jclass clazz, jbyteArray data)
{
    return byte_to_unsigned<jfloat>(env, data, env->functions->NewFloatArray, env->functions->GetFloatArrayElements, env->functions->ReleaseFloatArrayElements);
}

/*
 * Class:     com_glasssix_common_NativeNumericUtils
 * Method:    byteToDoubleUnsigned
 * Signature: ([B)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_glasssix_common_NativeNumericUtils_byteToDoubleUnsigned(JNIEnv* env, jclass clazz, jbyteArray data)
{
    return byte_to_unsigned<jdouble>(env, data, env->functions->NewDoubleArray, env->functions->GetDoubleArrayElements, env->functions->ReleaseDoubleArrayElements);
}

#ifdef __cplusplus
}
#endif
