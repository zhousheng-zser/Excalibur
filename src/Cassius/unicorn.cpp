#include "unicorn.hpp"
#include <iostream>
#include <vector>
#include "../../include/Julius/simd_helper.hpp"

#ifdef INT8_DATA
#include "unicorn_int8data.hpp"
#elif defined HALF_DATA
#include "unicorn_halfdata.hpp"
#else
#include "unicorn_data.hpp"
#endif//HALF_DATA

namespace glasssix
{
	namespace cassius
	{
		Unicorn::Unicorn(int device)
		{
			device_ = device;

#if SIMD_TYPE >= SIMDTYPE_SSE
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#ifdef USE_CUDNN
			if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
			}
			cudnn_ready_ = true;
			if (device >= 0)
			{
				int8_quantization_ = false;//do not use int8 in CUDNN
			}
#endif
#endif

#ifdef HALF_DATA
			float quantize_level = USHRT_MAX;
			int8_quantization_ = false;//do not use int8 in HALF_DATA
#else
			float quantize_level = INT_MAX;
#endif//HAL

			if (int8_quantization_)
			{
				Copy_Params(conv1a_weights, Unicorn, quantize_level);//864
				Copy_Params(conv1a_bias, Unicorn, quantize_level);//64
				Copy_Params(relu1a_weights, Unicorn, quantize_level);//32
				Copy_Int8_Params(conv1b, Unicorn);//18432
				Copy_Params(relu1b_weights, Unicorn, quantize_level);//64
				Copy_Int8_Params(conv2_1, Unicorn);//36864
				Copy_Params(relu2_1_weights, Unicorn, quantize_level);//64
				Copy_Int8_Params(conv2_2, Unicorn);//36864
				Copy_Params(relu2_2_weights, Unicorn, quantize_level);//64
				Copy_Int8_Params(conv2, Unicorn);//73728
				Copy_Params(relu2_weights, Unicorn, quantize_level);//128
				Copy_Int8_Params(conv3_1, Unicorn);//147456
				Copy_Params(relu3_1_weights, Unicorn, quantize_level);//128
				Copy_Int8_Params(conv3_2, Unicorn);//147456
				Copy_Params(relu3_2_weights, Unicorn, quantize_level);//128
				Copy_Int8_Params(conv3_3, Unicorn);//147456
				Copy_Params(relu3_3_weights, Unicorn, quantize_level);//128
				Copy_Int8_Params(conv3_4, Unicorn);//147456
				Copy_Params(relu3_4_weights, Unicorn, quantize_level);//128
				Copy_Int8_Params(conv3, Unicorn);//294912
				Copy_Params(relu3_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_1, Unicorn);//589824
				Copy_Params(relu4_1_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_2, Unicorn);//589824
				Copy_Params(relu4_2_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_3, Unicorn);//589824
				Copy_Params(relu4_3_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_4, Unicorn);//589824
				Copy_Params(relu4_4_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_5, Unicorn);//589824
				Copy_Params(relu4_5_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_6, Unicorn);//589824
				Copy_Params(relu4_6_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_7, Unicorn);//589824
				Copy_Params(relu4_7_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_8, Unicorn);//589824
				Copy_Params(relu4_8_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_9, Unicorn);//589824
				Copy_Params(relu4_9_weights, Unicorn, quantize_level);//256
				Copy_Int8_Params(conv4_10, Unicorn);//589824
				Copy_Params(relu4_10_weights, Unicorn, quantize_level);//256,512
				Copy_Int8_Params(conv4, Unicorn);//1179648
				Copy_Params(relu4_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5_1, Unicorn);//2359296
				Copy_Params(relu5_1_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5_2, Unicorn);//2359296
				Copy_Params(relu5_2_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5_3, Unicorn);//2359296
				Copy_Params(relu5_3_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5_4, Unicorn);//2359296
				Copy_Params(relu5_4_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5_5, Unicorn);//2359296
				Copy_Params(relu5_5_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5_6, Unicorn);//2359296
				Copy_Params(relu5_6_weights, Unicorn, quantize_level);//512
				Copy_Int8_Params(conv5, Unicorn);//2359296
				Copy_Params(relu5_weights, Unicorn, quantize_level);//512
			}
			else
			{
#ifdef INT8_DATA
				Copy_Params(conv1a_weights, Unicorn, quantize_level);//64
				Copy_Params(conv1a_bias, Unicorn, quantize_level);//64
				Copy_Params(relu1a_weights, Unicorn, quantize_level);//32
				Copy_Int8_to_FP32_Params(conv1b, Unicorn);//18432
				Copy_Params(conv1b_bias, Unicorn, quantize_level);//128
				Copy_Params(relu1b_weights, Unicorn, quantize_level);//64
				Copy_Int8_to_FP32_Params(conv2_1, Unicorn);//36864
				Copy_Params(conv2_1_bias, Unicorn, quantize_level);//128
				Copy_Params(relu2_1_weights, Unicorn, quantize_level);//64
				Copy_Int8_to_FP32_Params(conv2_2, Unicorn);//36864
				Copy_Params(conv2_2_bias, Unicorn, quantize_level);//128
				Copy_Params(relu2_2_weights, Unicorn, quantize_level);//64
				Copy_Int8_to_FP32_Params(conv2, Unicorn);//73728
				Copy_Params(conv2_bias, Unicorn, quantize_level);//256
				Copy_Params(relu2_weights, Unicorn, quantize_level);//128
				Copy_Int8_to_FP32_Params(conv3_1, Unicorn);//147456
				Copy_Params(conv3_1_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_1_weights, Unicorn, quantize_level);//128
				Copy_Int8_to_FP32_Params(conv3_2, Unicorn);//147456
				Copy_Params(conv3_2_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_2_weights, Unicorn, quantize_level);//128
				Copy_Int8_to_FP32_Params(conv3_3, Unicorn);//147456
				Copy_Params(conv3_3_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_3_weights, Unicorn, quantize_level);//128
				Copy_Int8_to_FP32_Params(conv3_4, Unicorn);//147456
				Copy_Params(conv3_4_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_4_weights, Unicorn, quantize_level);//128
				Copy_Int8_to_FP32_Params(conv3, Unicorn);//294912
				Copy_Params(conv3_bias, Unicorn, quantize_level);//512
				Copy_Params(relu3_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_1, Unicorn);//589824
				Copy_Params(conv4_1_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_1_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_2, Unicorn);//589824
				Copy_Params(conv4_2_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_2_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_3, Unicorn);//589824
				Copy_Params(conv4_3_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_3_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_4, Unicorn);//589824
				Copy_Params(conv4_4_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_4_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_5, Unicorn);//589824
				Copy_Params(conv4_5_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_5_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_6, Unicorn);//589824
				Copy_Params(conv4_6_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_6_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_7, Unicorn);//589824
				Copy_Params(conv4_7_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_7_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_8, Unicorn);//589824
				Copy_Params(conv4_8_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_8_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_9, Unicorn);//589824
				Copy_Params(conv4_9_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_9_weights, Unicorn, quantize_level);//256
				Copy_Int8_to_FP32_Params(conv4_10, Unicorn);//589824
				Copy_Params(conv4_10_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_10_weights, Unicorn, quantize_level);//256,512
				Copy_Int8_to_FP32_Params(conv4, Unicorn);//1179648
				Copy_Params(conv4_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu4_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5_1, Unicorn);//2359296
				Copy_Params(conv5_1_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_1_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5_2, Unicorn);//2359296
				Copy_Params(conv5_2_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_2_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5_3, Unicorn);//2359296
				Copy_Params(conv5_3_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_3_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5_4, Unicorn);//2359296
				Copy_Params(conv5_4_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_4_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5_5, Unicorn);//2359296
				Copy_Params(conv5_5_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_5_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5_6, Unicorn);//2359296
				Copy_Params(conv5_6_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_6_weights, Unicorn, quantize_level);//512
				Copy_Int8_to_FP32_Params(conv5, Unicorn);//2359296
				Copy_Params(conv5_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_weights, Unicorn, quantize_level);//512

#else				
				Copy_Params(conv1a_weights, Unicorn, quantize_level);//864
				Copy_Params(conv1a_bias, Unicorn, quantize_level);//64
				Copy_Params(relu1a_weights, Unicorn, quantize_level);//32
				Copy_Params(conv1b_weights, Unicorn, quantize_level);//18432
				Copy_Params(conv1b_bias, Unicorn, quantize_level);//128
				Copy_Params(relu1b_weights, Unicorn, quantize_level);//64
				Copy_Params(conv2_1_weights, Unicorn, quantize_level);//36864
				Copy_Params(conv2_1_bias, Unicorn, quantize_level);//128
				Copy_Params(relu2_1_weights, Unicorn, quantize_level);//64
				Copy_Params(conv2_2_weights, Unicorn, quantize_level);//36864
				Copy_Params(conv2_2_bias, Unicorn, quantize_level);//128
				Copy_Params(relu2_2_weights, Unicorn, quantize_level);//64
				Copy_Params(conv2_weights, Unicorn, quantize_level);//73728
				Copy_Params(conv2_bias, Unicorn, quantize_level);//256
				Copy_Params(relu2_weights, Unicorn, quantize_level);//128
				Copy_Params(conv3_1_weights, Unicorn, quantize_level);//147456
				Copy_Params(conv3_1_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_1_weights, Unicorn, quantize_level);//128
				Copy_Params(conv3_2_weights, Unicorn, quantize_level);//147456
				Copy_Params(conv3_2_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_2_weights, Unicorn, quantize_level);//128
				Copy_Params(conv3_3_weights, Unicorn, quantize_level);//147456
				Copy_Params(conv3_3_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_3_weights, Unicorn, quantize_level);//128
				Copy_Params(conv3_4_weights, Unicorn, quantize_level);//147456
				Copy_Params(conv3_4_bias, Unicorn, quantize_level);//256
				Copy_Params(relu3_4_weights, Unicorn, quantize_level);//128
				Copy_Params(conv3_weights, Unicorn, quantize_level);//294912
				Copy_Params(conv3_bias, Unicorn, quantize_level);//512
				Copy_Params(relu3_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_1_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_1_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_1_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_2_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_2_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_2_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_3_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_3_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_3_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_4_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_4_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_4_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_5_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_5_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_5_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_6_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_6_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_6_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_7_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_7_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_7_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_8_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_8_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_8_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_9_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_9_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_9_weights, Unicorn, quantize_level);//256
				Copy_Params(conv4_10_weights, Unicorn, quantize_level);//589824
				Copy_Params(conv4_10_bias, Unicorn, quantize_level);//512
				Copy_Params(relu4_10_weights, Unicorn, quantize_level);//256,512
				Copy_Params(conv4_weights, Unicorn, quantize_level);//1179648
				Copy_Params(conv4_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu4_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_1_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_1_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_1_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_2_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_2_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_2_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_3_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_3_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_3_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_4_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_4_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_4_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_5_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_5_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_5_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_6_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_6_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_6_weights, Unicorn, quantize_level);//512
				Copy_Params(conv5_weights, Unicorn, quantize_level);//2359296
				Copy_Params(conv5_bias, Unicorn, quantize_level);//1024
				Copy_Params(relu5_weights, Unicorn, quantize_level);//512
#endif //!INT8_DATA
			}

#ifdef __ARM_NEON
			bool temp_quantization = int8_quantization_;
			int8_quantization_ = false;// conv1a use float32 weights
			Init_Conv_arm_Params(conv1a, 3, 32, 1, 3, 1, 0, true);//nchw:1*3*128*128->1*32*126*126
			int8_quantization_ = temp_quantization;
			Init_PReLU_arm_Params(relu1a, 32, false, false);//nchw:1*32*126*126->1*32*126*126
			Init_Conv_arm_Params(conv1b, 32, 64, 1, 3, 1, 0, true);//nchw:1*32*126*126->1*64*124*124
			Init_PReLU_arm_Params(relu1b, 64, false, false);//nchw:1*64*124*124->1*64*124*124
			Init_Pooling_arm_Params(pool1b, 2, 2, 0, 0);//nchw:1*64*124*124->1*64*62*62
			Init_Conv_arm_Params(conv2_1, 64, 64, 1, 3, 1, 1, true);//nchw:1*64*62*62->1*64*62*62
			Init_PReLU_arm_Params(relu2_1, 64, false, false);//nchw:1*64*62*62->1*64*62*62
			Init_Conv_arm_Params(conv2_2, 64, 64, 1, 3, 1, 1, true);//nchw:1*64*62*62->1*64*62*62
			Init_PReLU_arm_Params(relu2_2, 64, false, false);//nchw:1*64*62*62->1*64*62*62
			Init_Eltwise_Params(res2_2, 0);//nchw:1*64*62*62->1*64*62*62
			Init_Conv_arm_Params(conv2, 64, 128, 1, 3, 1, 0, true);//nchw:1*64*62*62->1*128*60*60
			Init_PReLU_arm_Params(relu2, 128, false, false);//nchw:1*128*60*60->1*128*60*60
			Init_Pooling_arm_Params(pool2, 2, 2, 0, 0);//nchw:1*128*60*60->1*128*30*30
			Init_Conv_arm_Params(conv3_1, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_arm_Params(relu3_1, 128, false, false);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_arm_Params(conv3_2, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_arm_Params(relu3_2, 128, false, false);//nchw:1*128*30*30->1*128*30*30
			Init_Eltwise_Params(res3_2, 0);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_arm_Params(conv3_3, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_arm_Params(relu3_3, 128, false, false);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_arm_Params(conv3_4, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_arm_Params(relu3_4, 128, false, false);//nchw:1*128*30*30->1*128*30*30
			Init_Eltwise_Params(res3_4, 0);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_arm_Params(conv3, 128, 256, 1, 3, 1, 0, true);//nchw:1*128*30*30->1*256*28*28
			Init_PReLU_arm_Params(relu3, 256, false, false);//nchw:1*256*28*28->1*256*28*28
			Init_Pooling_arm_Params(pool3, 2, 2, 0, 0);//nchw:1*256*28*28->1*256*14*14
			Init_Conv_arm_Params(conv4_1, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_1, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_2, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_2, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_2, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_3, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_3, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_4, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_4, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_4, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_5, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_5, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_6, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_6, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_6, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_7, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_7, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_8, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_8, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_8, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_9, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_9, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4_10, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_arm_Params(relu4_10, 256, false, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_10, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_arm_Params(conv4, 256, 512, 1, 3, 1, 0, true);//nchw:1*256*14*14->1*512*12*12
			Init_PReLU_arm_Params(relu4, 512, false, false);//nchw:1*512*12*12->1*512*12*12
			Init_Pooling_arm_Params(pool4, 2, 2, 0, 0);//nchw:1*512*12*12->1*512*6*6
			Init_Conv_arm_Params(conv5_1, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_arm_Params(relu5_1, 512, false, false);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_arm_Params(conv5_2, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_arm_Params(relu5_2, 512, false, false);//nchw:1*512*6*6->1*512*6*6
			Init_Eltwise_Params(res5_2, 0);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_arm_Params(conv5_3, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_arm_Params(relu5_3, 512, false, false);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_arm_Params(conv5_4, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_arm_Params(relu5_4, 512, false, false);//nchw:1*512*6*6->1*512*6*6
			Init_Eltwise_Params(res5_4, 0);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_arm_Params(conv5_5, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_arm_Params(relu5_5, 512, false, false);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_arm_Params(conv5_6, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_arm_Params(relu5_6, 512, false, false);//nchw:1*512*6*6->1*512*6*6
			Init_Eltwise_Params(res5_6, 0);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_arm_Params(conv5, 512, 512, 1, 3, 1, 0, true);//nchw:1*512*6*6->1*512*4*4
			Init_PReLU_arm_Params(relu5, 512, false, false);//nchw:1*512*4*4->1*512*4*4
			Init_Pooling_arm_Params(pool5, 4, 4, 0, 1);//nchw:1*512*4*4->1*512*1*1
			Init_Normalize_Params(normalizer, 1, false);
#else
			bool temp_quantization = int8_quantization_;
			int8_quantization_ = false;// conv1a use float32 weights
			Init_Conv_Params(conv1a, 3, 32, 1, 3, 1, 0, true);//nchw:1*3*128*128->1*32*126*126
			int8_quantization_ = temp_quantization;
			Init_PReLU_Params(relu1a, 32, false);//nchw:1*32*126*126->1*32*126*126
			Init_Conv_Params(conv1b, 32, 64, 1, 3, 1, 0, true);//nchw:1*32*126*126->1*64*124*124
			Init_PReLU_Params(relu1b, 64, false);//nchw:1*64*124*124->1*64*124*124
			Init_Pooling_Params(pool1b, 2, 2, 0, 0);//nchw:1*64*124*124->1*64*62*62
			Init_Conv_Params(conv2_1, 64, 64, 1, 3, 1, 1, true);//nchw:1*64*62*62->1*64*62*62
			Init_PReLU_Params(relu2_1, 64, false);//nchw:1*64*62*62->1*64*62*62
			Init_Conv_Params(conv2_2, 64, 64, 1, 3, 1, 1, true);//nchw:1*64*62*62->1*64*62*62
			Init_PReLU_Params(relu2_2, 64, false);//nchw:1*64*62*62->1*64*62*62
			Init_Eltwise_Params(res2_2, 0);//nchw:1*64*62*62->1*64*62*62
			Init_Conv_Params(conv2, 64, 128, 1, 3, 1, 0, true);//nchw:1*64*62*62->1*128*60*60
			Init_PReLU_Params(relu2, 128, false);//nchw:1*128*60*60->1*128*60*60
			Init_Pooling_Params(pool2, 2, 2, 0, 0);//nchw:1*128*60*60->1*128*30*30
			Init_Conv_Params(conv3_1, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_Params(relu3_1, 128, false);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_Params(conv3_2, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_Params(relu3_2, 128, false);//nchw:1*128*30*30->1*128*30*30
			Init_Eltwise_Params(res3_2, 0);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_Params(conv3_3, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_Params(relu3_3, 128, false);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_Params(conv3_4, 128, 128, 1, 3, 1, 1, true);//nchw:1*128*30*30->1*128*30*30
			Init_PReLU_Params(relu3_4, 128, false);//nchw:1*128*30*30->1*128*30*30
			Init_Eltwise_Params(res3_4, 0);//nchw:1*128*30*30->1*128*30*30
			Init_Conv_Params(conv3, 128, 256, 1, 3, 1, 0, true);//nchw:1*128*30*30->1*256*28*28
			Init_PReLU_Params(relu3, 256, false);//nchw:1*256*28*28->1*256*28*28
			Init_Pooling_Params(pool3, 2, 2, 0, 0);//nchw:1*256*28*28->1*256*14*14
			Init_Conv_Params(conv4_1, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_1, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_2, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_2, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_2, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_3, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_3, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_4, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_4, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_4, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_5, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_5, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_6, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_6, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_6, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_7, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_7, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_8, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_8, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_8, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_9, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_9, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4_10, 256, 256, 1, 3, 1, 1, true);//nchw:1*256*14*14->1*256*14*14
			Init_PReLU_Params(relu4_10, 256, false);//nchw:1*256*14*14->1*256*14*14
			Init_Eltwise_Params(res4_10, 0);//nchw:1*256*14*14->1*256*14*14
			Init_Conv_Params(conv4, 256, 512, 1, 3, 1, 0, true);//nchw:1*256*14*14->1*512*12*12
			Init_PReLU_Params(relu4, 512, false);//nchw:1*512*12*12->1*512*12*12
			Init_Pooling_Params(pool4, 2, 2, 0, 0);//nchw:1*512*12*12->1*512*6*6
			Init_Conv_Params(conv5_1, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_Params(relu5_1, 512, false);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_Params(conv5_2, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_Params(relu5_2, 512, false);//nchw:1*512*6*6->1*512*6*6
			Init_Eltwise_Params(res5_2, 0);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_Params(conv5_3, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_Params(relu5_3, 512, false);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_Params(conv5_4, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_Params(relu5_4, 512, false);//nchw:1*512*6*6->1*512*6*6
			Init_Eltwise_Params(res5_4, 0);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_Params(conv5_5, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_Params(relu5_5, 512, false);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_Params(conv5_6, 512, 512, 1, 3, 1, 1, true);//nchw:1*512*6*6->1*512*6*6
			Init_PReLU_Params(relu5_6, 512, false);//nchw:1*512*6*6->1*512*6*6
			Init_Eltwise_Params(res5_6, 0);//nchw:1*512*6*6->1*512*6*6
			Init_Conv_Params(conv5, 512, 512, 1, 3, 1, 0, true);//nchw:1*512*6*6->1*512*4*4
			Init_PReLU_Params(relu5, 512, false);//nchw:1*512*4*4->1*512*4*4
			Init_Pooling_Params(pool5, 4, 4, 0, 1);//nchw:1*512*4*4->1*512*1*1
			Init_Normalize_Params(normalizer, 1, false);
#endif //!__ARM_NEON
				
		}


		Unicorn::~Unicorn()
		{
			delete conv1a;
			delete relu1a;
			delete conv1b;
			delete relu1b;
			delete pool1b;
			delete conv2_1;
			delete relu2_1;
			delete conv2_2;
			delete relu2_2;
			delete res2_2;
			delete conv2;
			delete relu2;
			delete pool2;
			delete conv3_1;
			delete relu3_1;
			delete conv3_2;
			delete relu3_2;
			delete res3_2;
			delete conv3_3;
			delete relu3_3;
			delete conv3_4;
			delete relu3_4;
			delete res3_4;
			delete conv3;
			delete relu3;
			delete pool3;
			delete conv4_1;
			delete relu4_1;
			delete conv4_2;
			delete relu4_2;
			delete res4_2;
			delete conv4_3;
			delete relu4_3;
			delete conv4_4;
			delete relu4_4;
			delete res4_4;
			delete conv4_5;
			delete relu4_5;
			delete conv4_6;
			delete relu4_6;
			delete res4_6;
			delete conv4_7;
			delete relu4_7;
			delete conv4_8;
			delete relu4_8;
			delete res4_8;
			delete conv4_9;
			delete relu4_9;
			delete conv4_10;
			delete relu4_10;
			delete res4_10;
			delete conv4;
			delete relu4;
			delete pool4;
			delete conv5_1;
			delete relu5_1;
			delete conv5_2;
			delete relu5_2;
			delete res5_2;
			delete conv5_3;
			delete relu5_3;
			delete conv5_4;
			delete relu5_4;
			delete res5_4;
			delete conv5_5;
			delete relu5_5;
			delete conv5_6;
			delete relu5_6;
			delete res5_6;
			delete conv5;
			delete relu5;
			delete pool5;
			delete normalizer;

			FreeHost(relu1a_weights, false);
			FreeHost(relu1b_weights, false);
			FreeHost(relu2_1_weights, false);
			FreeHost(relu2_2_weights, false);
			FreeHost(relu2_weights, false);
			FreeHost(relu3_1_weights, false);
			FreeHost(relu3_2_weights, false);
			FreeHost(relu3_3_weights, false);
			FreeHost(relu3_4_weights, false);
			FreeHost(relu3_weights, false);
			FreeHost(relu4_1_weights, false);
			FreeHost(relu4_2_weights, false);
			FreeHost(relu4_3_weights, false);
			FreeHost(relu4_4_weights, false);
			FreeHost(relu4_5_weights, false);
			FreeHost(relu4_6_weights, false);
			FreeHost(relu4_7_weights, false);
			FreeHost(relu4_8_weights, false);
			FreeHost(relu4_9_weights, false);
			FreeHost(relu4_10_weights, false);
			FreeHost(relu4_weights, false);
			FreeHost(relu5_1_weights, false);
			FreeHost(relu5_2_weights, false);
			FreeHost(relu5_3_weights, false);
			FreeHost(relu5_4_weights, false);
			FreeHost(relu5_5_weights, false);
			FreeHost(relu5_6_weights, false);
			FreeHost(relu5_weights, false);

#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
				CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
#endif
#endif
		} 

		void Unicorn::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1a->Forward(input_data, conv1a_top_data);
			relu1a->Forward_cpu(conv1a_top_data);
			conv1b->Forward(conv1a_top_data, conv1b_top_data);
			relu1b->Forward_cpu(conv1b_top_data);
			pool1b->Forward_cpu(conv1b_top_data, pool1b_top_data);
			conv2_1->Forward(pool1b_top_data, conv2_1_top_data);
			relu2_1->Forward_cpu(conv2_1_top_data);
			conv2_2->Forward(conv2_1_top_data, conv2_2_top_data);
			relu2_2->Forward_cpu(conv2_2_top_data);
			res2_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool1b_top_data, conv2_2_top_data}, res2_2_top_data);	
			conv2->Forward(res2_2_top_data, conv2_top_data);
			relu2->Forward_cpu(conv2_top_data);
			pool2->Forward_cpu(conv2_top_data, pool2_top_data);
			conv3_1->Forward(pool2_top_data, conv3_1_top_data);
			relu3_1->Forward_cpu(conv3_1_top_data);
			conv3_2->Forward(conv3_1_top_data, conv3_2_top_data);
			relu3_2->Forward_cpu(conv3_2_top_data);
			res3_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool2_top_data, conv3_2_top_data}, res3_2_top_data);
			conv3_3->Forward(res3_2_top_data, conv3_3_top_data);
			relu3_3->Forward_cpu(conv3_3_top_data);
			conv3_4->Forward(conv3_3_top_data, conv3_4_top_data);
			relu3_4->Forward_cpu(conv3_4_top_data);
			res3_4->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res3_2_top_data, conv3_4_top_data}, res3_4_top_data);
			conv3->Forward(res3_4_top_data, conv3_top_data);
			relu3->Forward_cpu(conv3_top_data);
			pool3->Forward_cpu(conv3_top_data, pool3_top_data);
			conv4_1->Forward(pool3_top_data, conv4_1_top_data);
			relu4_1->Forward_cpu(conv4_1_top_data);
			conv4_2->Forward(conv4_1_top_data, conv4_2_top_data);
			relu4_2->Forward_cpu(conv4_2_top_data);
			res4_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool3_top_data, conv4_2_top_data}, res4_2_top_data);
			conv4_3->Forward(res4_2_top_data, conv4_3_top_data);
			relu4_3->Forward_cpu(conv4_3_top_data);
			conv4_4->Forward(conv4_3_top_data, conv4_4_top_data);
			relu4_4->Forward_cpu(conv4_4_top_data);
			res4_4->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_2_top_data, conv4_4_top_data}, res4_4_top_data);
			conv4_5->Forward(res4_4_top_data, conv4_5_top_data);
			relu4_5->Forward_cpu(conv4_5_top_data);
			conv4_6->Forward(conv4_5_top_data, conv4_6_top_data);
			relu4_6->Forward_cpu(conv4_6_top_data);
			res4_6->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_4_top_data, conv4_6_top_data}, res4_6_top_data);
			conv4_7->Forward(res4_6_top_data, conv4_7_top_data);
			relu4_7->Forward_cpu(conv4_7_top_data);
			conv4_8->Forward(conv4_7_top_data, conv4_8_top_data);
			relu4_8->Forward_cpu(conv4_8_top_data);
			res4_8->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_6_top_data, conv4_8_top_data}, res4_8_top_data);
			conv4_9->Forward(res4_8_top_data, conv4_9_top_data);
			relu4_9->Forward_cpu(conv4_9_top_data);
			conv4_10->Forward(conv4_9_top_data, conv4_10_top_data);
			relu4_10->Forward_cpu(conv4_10_top_data);
			res4_10->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_8_top_data, conv4_10_top_data}, res4_10_top_data);
			conv4->Forward(res4_10_top_data, conv4_top_data);
			relu4->Forward_cpu(conv4_top_data);
			pool4->Forward_cpu(conv4_top_data, pool4_top_data);
			conv5_1->Forward(pool4_top_data, conv5_1_top_data);
			relu5_1->Forward_cpu(conv5_1_top_data);
			conv5_2->Forward(conv5_1_top_data, conv5_2_top_data);
			relu5_2->Forward_cpu(conv5_2_top_data);
			res5_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool4_top_data, conv5_2_top_data}, res5_2_top_data);
			conv5_3->Forward(res5_2_top_data, conv5_3_top_data);
			relu5_3->Forward_cpu(conv5_3_top_data);
			conv5_4->Forward(conv5_3_top_data, conv5_4_top_data);
			relu5_4->Forward_cpu(conv5_4_top_data);
			res5_4->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res5_2_top_data, conv5_4_top_data}, res5_4_top_data);
			conv5_5->Forward(res5_4_top_data, conv5_5_top_data);
			relu5_5->Forward_cpu(conv5_5_top_data);
			conv5_6->Forward(conv5_5_top_data, conv5_6_top_data);
			relu5_6->Forward_cpu(conv5_6_top_data);
			res5_6->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res5_4_top_data, conv5_6_top_data}, res5_6_top_data);
			conv5->Forward(res5_6_top_data, conv5_top_data);
			relu5->Forward_cpu(conv5_top_data);
			pool5->Forward_cpu(conv5_top_data, pool5_top_data);
			normalizer->Forward_cpu(pool5_top_data);//feature_top_data
		}


#ifdef USE_CUDA
		void Unicorn::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1a->Forward(cublas_handle_, input_data, conv1a_top_data);//concat_top_data
			relu1a->Forward_gpu_native(conv1a_top_data);
			conv1b->Forward(cublas_handle_, conv1a_top_data, conv1b_top_data);
			relu1b->Forward_gpu_native(conv1b_top_data);
			pool1b->Forward_gpu_native(conv1b_top_data, pool1b_top_data);
			conv2_1->Forward(cublas_handle_, pool1b_top_data, conv2_1_top_data);
			relu2_1->Forward_gpu_native(conv2_1_top_data);
			conv2_2->Forward(cublas_handle_, conv2_1_top_data, conv2_2_top_data);
			relu2_2->Forward_gpu_native(conv2_2_top_data);
			res2_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool1b_top_data, conv2_2_top_data}, res2_2_top_data);
			conv2->Forward(cublas_handle_, res2_2_top_data, conv2_top_data);
			relu2->Forward_gpu_native(conv2_top_data);
			pool2->Forward_gpu_native(conv2_top_data, pool2_top_data);
			conv3_1->Forward(cublas_handle_, pool2_top_data, conv3_1_top_data);
			relu3_1->Forward_gpu_native(conv3_1_top_data);
			conv3_2->Forward(cublas_handle_, conv3_1_top_data, conv3_2_top_data);
			relu3_2->Forward_gpu_native(conv3_2_top_data);
			res3_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool2_top_data, conv3_2_top_data}, res3_2_top_data);
			conv3_3->Forward(cublas_handle_, res3_2_top_data, conv3_3_top_data);
			relu3_3->Forward_gpu_native(conv3_3_top_data);
			conv3_4->Forward(cublas_handle_, conv3_3_top_data, conv3_4_top_data);
			relu3_4->Forward_gpu_native(conv3_4_top_data);
			res3_4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res3_2_top_data, conv3_4_top_data}, res3_4_top_data);
			conv3->Forward(cublas_handle_, res3_4_top_data, conv3_top_data);
			relu3->Forward_gpu_native(conv3_top_data);
			pool3->Forward_gpu_native(conv3_top_data, pool3_top_data);
			conv4_1->Forward(cublas_handle_, pool3_top_data, conv4_1_top_data);
			relu4_1->Forward_gpu_native(conv4_1_top_data);
			conv4_2->Forward(cublas_handle_, conv4_1_top_data, conv4_2_top_data);
			relu4_2->Forward_gpu_native(conv4_2_top_data);
			res4_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool3_top_data, conv4_2_top_data}, res4_2_top_data);
			conv4_3->Forward(cublas_handle_, res4_2_top_data, conv4_3_top_data);
			relu4_3->Forward_gpu_native(conv4_3_top_data);
			conv4_4->Forward(cublas_handle_, conv4_3_top_data, conv4_4_top_data);
			relu4_4->Forward_gpu_native(conv4_4_top_data);
			res4_4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_2_top_data, conv4_4_top_data}, res4_4_top_data);
			conv4_5->Forward(cublas_handle_, res4_4_top_data, conv4_5_top_data);
			relu4_5->Forward_gpu_native(conv4_5_top_data);
			conv4_6->Forward(cublas_handle_, conv4_5_top_data, conv4_6_top_data);
			relu4_6->Forward_gpu_native(conv4_6_top_data);
			res4_6->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_4_top_data, conv4_6_top_data}, res4_6_top_data);
			conv4_7->Forward(cublas_handle_, res4_6_top_data, conv4_7_top_data);
			relu4_7->Forward_gpu_native(conv4_7_top_data);
			conv4_8->Forward(cublas_handle_, conv4_7_top_data, conv4_8_top_data);
			relu4_8->Forward_gpu_native(conv4_8_top_data);
			res4_8->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_6_top_data, conv4_8_top_data}, res4_8_top_data);
			conv4_9->Forward(cublas_handle_, res4_8_top_data, conv4_9_top_data);
			relu4_9->Forward_gpu_native(conv4_9_top_data);
			conv4_10->Forward(cublas_handle_, conv4_9_top_data, conv4_10_top_data);
			relu4_10->Forward_gpu_native(conv4_10_top_data);
			res4_10->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_8_top_data, conv4_10_top_data}, res4_10_top_data);
			conv4->Forward(cublas_handle_, res4_10_top_data, conv4_top_data);
			relu4->Forward_gpu_native(conv4_top_data);
			pool4->Forward_gpu_native(conv4_top_data, pool4_top_data);
			conv5_1->Forward(cublas_handle_, pool4_top_data, conv5_1_top_data);
			relu5_1->Forward_gpu_native(conv5_1_top_data);
			conv5_2->Forward(cublas_handle_, conv5_1_top_data, conv5_2_top_data);
			relu5_2->Forward_gpu_native(conv5_2_top_data);
			res5_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool4_top_data, conv5_2_top_data}, res5_2_top_data);
			conv5_3->Forward(cublas_handle_, res5_2_top_data, conv5_3_top_data);
			relu5_3->Forward_gpu_native(conv5_3_top_data);
			conv5_4->Forward(cublas_handle_, conv5_3_top_data, conv5_4_top_data);
			relu5_4->Forward_gpu_native(conv5_4_top_data);
			res5_4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_2_top_data, conv5_4_top_data}, res5_4_top_data);
			conv5_5->Forward(cublas_handle_, res5_4_top_data, conv5_5_top_data);
			relu5_5->Forward_gpu_native(conv5_5_top_data);
			conv5_6->Forward(cublas_handle_, conv5_5_top_data, conv5_6_top_data);
			relu5_6->Forward_gpu_native(conv5_6_top_data);
			res5_6->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_4_top_data, conv5_6_top_data}, res5_6_top_data);
			conv5->Forward(cublas_handle_, res5_6_top_data, conv5_top_data);
			relu5->Forward_gpu_native(conv5_top_data);
			pool5->Forward_gpu_native(conv5_top_data, pool5_top_data);
			normalizer->Forward_gpu_native(pool5_top_data);//feature_top_data
		}
#ifdef USE_CUDNN
		void Unicorn::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1a->Forward(cudnn_handle_, input_data, conv1a_top_data);//concat_top_data
			relu1a->Forward_gpu_native(conv1a_top_data);
			conv1b->Forward(cudnn_handle_, conv1a_top_data, conv1b_top_data);
			relu1b->Forward_gpu_native(conv1b_top_data);
			pool1b->Forward_gpu_cudnn(conv1b_top_data, pool1b_top_data);
			conv2_1->Forward(cudnn_handle_, pool1b_top_data, conv2_1_top_data);
			relu2_1->Forward_gpu_native(conv2_1_top_data);
			conv2_2->Forward(cudnn_handle_, conv2_1_top_data, conv2_2_top_data);
			relu2_2->Forward_gpu_native(conv2_2_top_data);
			res2_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool1b_top_data, conv2_2_top_data}, res2_2_top_data);
			conv2->Forward(cudnn_handle_, res2_2_top_data, conv2_top_data);
			relu2->Forward_gpu_native(conv2_top_data);
			pool2->Forward_gpu_cudnn(conv2_top_data, pool2_top_data);
			conv3_1->Forward(cudnn_handle_, pool2_top_data, conv3_1_top_data);
			relu3_1->Forward_gpu_native(conv3_1_top_data);
			conv3_2->Forward(cudnn_handle_, conv3_1_top_data, conv3_2_top_data);
			relu3_2->Forward_gpu_native(conv3_2_top_data);
			res3_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool2_top_data, conv3_2_top_data}, res3_2_top_data);
			conv3_3->Forward(cudnn_handle_, res3_2_top_data, conv3_3_top_data);
			relu3_3->Forward_gpu_native(conv3_3_top_data);
			conv3_4->Forward(cudnn_handle_, conv3_3_top_data, conv3_4_top_data);
			relu3_4->Forward_gpu_native(conv3_4_top_data);
			res3_4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res3_2_top_data, conv3_4_top_data}, res3_4_top_data);
			conv3->Forward(cudnn_handle_, res3_4_top_data, conv3_top_data);
			relu3->Forward_gpu_native(conv3_top_data);
			pool3->Forward_gpu_cudnn(conv3_top_data, pool3_top_data);
			conv4_1->Forward(cudnn_handle_, pool3_top_data, conv4_1_top_data);
			relu4_1->Forward_gpu_native(conv4_1_top_data);
			conv4_2->Forward(cudnn_handle_, conv4_1_top_data, conv4_2_top_data);
			relu4_2->Forward_gpu_native(conv4_2_top_data);
			res4_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool3_top_data, conv4_2_top_data}, res4_2_top_data);
			conv4_3->Forward(cudnn_handle_, res4_2_top_data, conv4_3_top_data);
			relu4_3->Forward_gpu_native(conv4_3_top_data);
			conv4_4->Forward(cudnn_handle_, conv4_3_top_data, conv4_4_top_data);
			relu4_4->Forward_gpu_native(conv4_4_top_data);
			res4_4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_2_top_data, conv4_4_top_data}, res4_4_top_data);
			conv4_5->Forward(cudnn_handle_, res4_4_top_data, conv4_5_top_data);
			relu4_5->Forward_gpu_native(conv4_5_top_data);
			conv4_6->Forward(cudnn_handle_, conv4_5_top_data, conv4_6_top_data);
			relu4_6->Forward_gpu_native(conv4_6_top_data);
			res4_6->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_4_top_data, conv4_6_top_data}, res4_6_top_data);
			conv4_7->Forward(cudnn_handle_, res4_6_top_data, conv4_7_top_data);
			relu4_7->Forward_gpu_native(conv4_7_top_data);
			conv4_8->Forward(cudnn_handle_, conv4_7_top_data, conv4_8_top_data);
			relu4_8->Forward_gpu_native(conv4_8_top_data);
			res4_8->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_6_top_data, conv4_8_top_data}, res4_8_top_data);
			conv4_9->Forward(cudnn_handle_, res4_8_top_data, conv4_9_top_data);
			relu4_9->Forward_gpu_native(conv4_9_top_data);
			conv4_10->Forward(cudnn_handle_, conv4_9_top_data, conv4_10_top_data);
			relu4_10->Forward_gpu_native(conv4_10_top_data);
			res4_10->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_8_top_data, conv4_10_top_data}, res4_10_top_data);
			conv4->Forward(cudnn_handle_, res4_10_top_data, conv4_top_data);
			relu4->Forward_gpu_native(conv4_top_data);
			pool4->Forward_gpu_cudnn(conv4_top_data, pool4_top_data);
			conv5_1->Forward(cudnn_handle_, pool4_top_data, conv5_1_top_data);
			relu5_1->Forward_gpu_native(conv5_1_top_data);
			conv5_2->Forward(cudnn_handle_, conv5_1_top_data, conv5_2_top_data);
			relu5_2->Forward_gpu_native(conv5_2_top_data);
			res5_2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool4_top_data, conv5_2_top_data}, res5_2_top_data);
			conv5_3->Forward(cudnn_handle_, res5_2_top_data, conv5_3_top_data);
			relu5_3->Forward_gpu_native(conv5_3_top_data);
			conv5_4->Forward(cudnn_handle_, conv5_3_top_data, conv5_4_top_data);
			relu5_4->Forward_gpu_native(conv5_4_top_data);
			res5_4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_2_top_data, conv5_4_top_data}, res5_4_top_data);
			conv5_5->Forward(cudnn_handle_, res5_4_top_data, conv5_5_top_data);
			relu5_5->Forward_gpu_native(conv5_5_top_data);
			conv5_6->Forward(cudnn_handle_, conv5_5_top_data, conv5_6_top_data);
			relu5_6->Forward_gpu_native(conv5_6_top_data);
			res5_6->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_4_top_data, conv5_6_top_data}, res5_6_top_data);
			conv5->Forward(cudnn_handle_, res5_6_top_data, conv5_top_data);
			relu5->Forward_gpu_native(conv5_top_data);
			pool5->Forward_gpu_cudnn(conv5_top_data, pool5_top_data);
			normalizer->Forward_gpu_native(pool5_top_data);//feature_top_data
		}
#endif
#endif

		std::vector<std::vector<float> > Unicorn::Forward(const float* input_data, int num, int order)
		{
			if (num <= 0)
			{
				LOG(FATAL) << "no human face information!!!";
				return std::vector<std::vector<float> >();
			}

			std::vector<std::vector<float> > feature;

			if (order == 0)//NCHW
			{
				tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 3, 128, 128}, device_, NCHW));
			}
			else//NHWC
			{
				tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 128, 128, 3}, device_, NHWC));
			}

			if (device_ < 0)
			{
				float* tensor_data = tensor_float_data->mutable_cpu_data();
				memcpy(tensor_data, input_data, num * 3 * 128 * 128 * sizeof(float));
				tensor_operation_cpu::preprocess_tensors_cpu(tensor_float_data, tensor_float_data);
				Forward_cpu(tensor_float_data);
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp(512);
					std::memcpy(temp.data(), get_pool5()->cpu_data() + i * 512, 512 * sizeof(float));
					feature.push_back(temp);
				}
				return feature;
			}
			else
			{
#ifdef USE_CUDA
				float* tensor_data = tensor_float_data->mutable_gpu_data();
				cudaMemcpy(tensor_data, input_data, num * 3 * 128 * 128 * sizeof(float), cudaMemcpyDefault);
				tensor_operation_gpu::preprocess_tensors_gpu(tensor_float_data, tensor_float_data);
#ifdef USE_CUDNN
				Forward_gpu_cudnn(tensor_float_data);
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp(512);
					std::memcpy(temp.data(), get_pool5()->cpu_data() + i * 512, 512 * sizeof(float));
					feature.push_back(temp);
				}
				return feature;
#endif
				Forward_gpu_native(tensor_float_data);
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp(512);
					std::memcpy(temp.data(), get_pool5()->cpu_data() + i * 512, 512 * sizeof(float));
					feature.push_back(temp);
				}
				return feature;
#else
				NO_GPU;
				return feature;
#endif
			}
		}

		std::vector<std::vector<float> > Unicorn::Forward(const unsigned char* input_data, int num, int order)
		{
			if (num <= 0)
			{
				LOG(FATAL) << "no human face information!!!";
				return std::vector<std::vector<float> >();
			}

			std::vector<std::vector<float> > feature;

			if (order == 0)//NCHW
			{
				tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 3, 128, 128}, device_, NCHW));
				tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 3, 128, 128}, device_, NCHW));
			}
			else//NHWC
			{
				tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 128, 128, 3}, device_, NHWC));
				tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 128, 128, 3}, device_, NHWC));
			}

			if (device_ < 0)
			{
				unsigned char* tensor_data = tensor_unsigned_char_data->mutable_cpu_data();
				memcpy(tensor_data, input_data, num * 3 * 128 * 128 * sizeof(unsigned char));
				tensor_operation_cpu::preprocess_tensors_cpu(tensor_unsigned_char_data, tensor_float_data);
				Forward_cpu(tensor_float_data);
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp(512);
					std::memcpy(temp.data(), get_pool5()->cpu_data() + i * 512, 512 * sizeof(float));
					feature.push_back(temp);
				}
				return feature;
			}
			else
			{
#ifdef USE_CUDA
				unsigned char* tensor_data = tensor_unsigned_char_data->mutable_gpu_data();
				cudaMemcpy(tensor_data, input_data, num * 3 * 128 * 128 * sizeof(unsigned char), cudaMemcpyDefault);
				tensor_operation_gpu::preprocess_tensors_gpu(tensor_unsigned_char_data, tensor_float_data);
#ifdef USE_CUDNN
				Forward_gpu_cudnn(tensor_float_data);
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp(512);
					std::memcpy(temp.data(), get_pool5()->cpu_data() + i * 512, 512 * sizeof(float));
					feature.push_back(temp);
				}
				return feature;
#endif
				Forward_gpu_native(tensor_float_data);
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp(512);
					std::memcpy(temp.data(), get_pool5()->cpu_data() + i * 512, 512 * sizeof(float));
					feature.push_back(temp);
				}
				return feature;
#else
				NO_GPU;
				return feature;
#endif
			}
		}
	}
}
#undef HALF