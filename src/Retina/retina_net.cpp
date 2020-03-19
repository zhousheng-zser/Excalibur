#include "retina_net.hpp"

#ifdef INT8_DATA
#include "retina_net_int8_data.hpp"
#else
#include "retina_net_data.hpp"
#endif
#include "../../include/Excalibur/prune.hpp"
#include <glasssix/timer.hpp>

#ifdef CALC_LAYERS
#include <glasssix/profiler.hpp>
#endif

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		Retina_net::Retina_net(int device)
		{
			device_ = device;
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#if SIMD_TYPE >= SIMDTYPE_SSE
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif
			
			float quantize_level = INT_MAX;

			if (int8_quantization_)
			{
				Copy_Int8_Params(mobilenet0_conv0_fwd, retina_net);
				Copy_Params(mobilenet0_conv1_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv1_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv2_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv2_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv3_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv3_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv4_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv4_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv5_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv5_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv6_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv6_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv7_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv7_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv8_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv8_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv9_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv9_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv10_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv10_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv11_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv11_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv12_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv12_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv13_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv13_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv14_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv14_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv15_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv15_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv16_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv16_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv17_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv17_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv18_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv18_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv19_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv19_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv20_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv20_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv21_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv21_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv22_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv22_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv23_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv23_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv24_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv24_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv25_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv25_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv26_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv26_fwd_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_lateral_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_lateral_bias, retina_net, quantize_level);
				Copy_Int8_Params(rf_c3_det_conv1, retina_net);
				Copy_Int8_Params(rf_c3_det_context_conv1, retina_net);
				Copy_Int8_Params(rf_c3_det_context_conv2, retina_net);
				Copy_Int8_Params(rf_c3_det_context_conv3_1, retina_net);
				Copy_Int8_Params(rf_c3_det_context_conv3_2, retina_net);
				Copy_Params(face_rpn_cls_score_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride32_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride32_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride32_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_lateral_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_lateral_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_upsampling_weights, retina_net, quantize_level);
				Copy_Int8_Params(rf_c2_aggr, retina_net);
				Copy_Int8_Params(rf_c2_det_conv1, retina_net);
				Copy_Int8_Params(rf_c2_det_context_conv1, retina_net);
				Copy_Int8_Params(rf_c2_det_context_conv2, retina_net);
				Copy_Int8_Params(rf_c2_det_context_conv3_1, retina_net);
				Copy_Int8_Params(rf_c2_det_context_conv3_2, retina_net);
				Copy_Params(face_rpn_cls_score_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride16_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride16_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride16_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_red_conv_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_red_conv_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_upsampling_weights, retina_net, quantize_level);
				Copy_Int8_Params(rf_c1_aggr, retina_net);
				Copy_Int8_Params(rf_c1_det_conv1, retina_net);
				Copy_Int8_Params(rf_c1_det_context_conv1, retina_net);
				Copy_Int8_Params(rf_c1_det_context_conv2, retina_net);
				Copy_Int8_Params(rf_c1_det_context_conv3_1, retina_net);
				Copy_Int8_Params(rf_c1_det_context_conv3_2, retina_net);
				Copy_Params(face_rpn_cls_score_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride8_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride8_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride8_bias, retina_net, quantize_level);
			}
			else
			{
#ifdef INT8_DATA
				Copy_Int8_to_FP32_Params(mobilenet0_conv0_fwd, retina_net);
				Copy_Params(mobilenet0_conv0_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv1_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv1_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv2_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv2_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv3_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv3_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv4_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv4_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv5_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv5_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv6_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv6_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv7_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv7_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv8_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv8_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv9_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv9_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv10_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv10_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv11_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv11_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv12_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv12_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv13_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv13_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv14_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv14_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv15_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv15_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv16_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv16_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv17_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv17_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv18_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv18_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv19_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv19_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv20_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv20_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv21_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv21_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv22_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv22_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv23_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv23_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv24_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv24_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv25_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv25_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv26_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv26_fwd_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_lateral_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_lateral_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c3_det_conv1, retina_net);
				Copy_Params(rf_c3_det_conv1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c3_det_context_conv1, retina_net);
				Copy_Params(rf_c3_det_context_conv1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c3_det_context_conv2, retina_net);
				Copy_Params(rf_c3_det_context_conv2_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c3_det_context_conv3_1, retina_net);
				Copy_Params(rf_c3_det_context_conv3_1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c3_det_context_conv3_2, retina_net);
				Copy_Params(rf_c3_det_context_conv3_2_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride32_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride32_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride32_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_lateral_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_lateral_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_upsampling_weights, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c2_aggr, retina_net);
				Copy_Params(rf_c2_aggr_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c2_det_conv1, retina_net);
				Copy_Params(rf_c2_det_conv1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c2_det_context_conv1, retina_net);
				Copy_Params(rf_c2_det_context_conv1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c2_det_context_conv2, retina_net);
				Copy_Params(rf_c2_det_context_conv2_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c2_det_context_conv3_1, retina_net);
				Copy_Params(rf_c2_det_context_conv3_1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c2_det_context_conv3_2, retina_net);
				Copy_Params(rf_c2_det_context_conv3_2_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride16_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride16_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride16_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_red_conv_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_red_conv_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_upsampling_weights, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c1_aggr, retina_net);
				Copy_Params(rf_c1_aggr_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c1_det_conv1, retina_net);
				Copy_Params(rf_c1_det_conv1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c1_det_context_conv1, retina_net);
				Copy_Params(rf_c1_det_context_conv1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c1_det_context_conv2, retina_net);
				Copy_Params(rf_c1_det_context_conv2_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c1_det_context_conv3_1, retina_net);
				Copy_Params(rf_c1_det_context_conv3_1_bias, retina_net, quantize_level);
				Copy_Int8_to_FP32_Params(rf_c1_det_context_conv3_2, retina_net);
				Copy_Params(rf_c1_det_context_conv3_2_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride8_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride8_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride8_bias, retina_net, quantize_level);
#else
				Copy_Params(mobilenet0_conv0_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv0_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv1_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv1_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv2_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv2_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv3_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv3_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv4_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv4_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv5_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv5_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv6_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv6_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv7_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv7_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv8_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv8_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv9_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv9_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv10_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv10_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv11_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv11_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv12_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv12_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv13_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv13_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv14_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv14_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv15_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv15_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv16_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv16_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv17_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv17_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv18_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv18_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv19_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv19_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv20_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv20_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv21_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv21_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv22_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv22_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv23_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv23_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv24_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv24_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv25_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv25_fwd_bias, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv26_fwd_weights, retina_net, quantize_level);
				Copy_Params(mobilenet0_conv26_fwd_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_lateral_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_lateral_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_det_conv1_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_det_conv1_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv1_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv1_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv2_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv2_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv3_1_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv3_1_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv3_2_weights, retina_net, quantize_level);
				Copy_Params(rf_c3_det_context_conv3_2_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride32_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride32_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride32_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride32_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_lateral_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_lateral_bias, retina_net, quantize_level);
				Copy_Params(rf_c3_upsampling_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_aggr_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_aggr_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_det_conv1_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_det_conv1_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv1_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv1_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv2_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv2_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv3_1_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv3_1_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv3_2_weights, retina_net, quantize_level);
				Copy_Params(rf_c2_det_context_conv3_2_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride16_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride16_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride16_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride16_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_red_conv_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_red_conv_bias, retina_net, quantize_level);
				Copy_Params(rf_c2_upsampling_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_aggr_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_aggr_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_det_conv1_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_det_conv1_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv1_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv1_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv2_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv2_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv3_1_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv3_1_bias, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv3_2_weights, retina_net, quantize_level);
				Copy_Params(rf_c1_det_context_conv3_2_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_cls_score_stride8_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_bbox_pred_stride8_bias, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride8_weights, retina_net, quantize_level);
				Copy_Params(face_rpn_landmark_pred_stride8_bias, retina_net, quantize_level);
#endif // INT8_DATA

			}
			

#ifdef __ARM_NEON

            Init_Conv_arm_Params(mobilenet0_conv0_fwd, 3, 8, 1, 3, 2, 1, true);//1*3*320*320->1*8*160*160
			Init_ReLU_arm_Params(mobilenet0_relu0_fwd, 8, true);//1*8*160*160->1*8*160*160
			Init_Conv_arm_Params(mobilenet0_conv1_fwd, 8, 8, 8, 3, 1, 1, true);//1*8*160*160->1*8*160*160
			Init_ReLU_arm_Params(mobilenet0_relu1_fwd, 8, true);//1*8*160*160->1*8*160*160
			Init_Conv_arm_Params(mobilenet0_conv2_fwd, 8, 16, 1, 1, 1, 0, true);//1*8*160*160->1*16*160*160
			Init_ReLU_arm_Params(mobilenet0_relu2_fwd, 16, true);//1*16*160*160->1*16*160*160
			Init_Conv_arm_Params(mobilenet0_conv3_fwd, 16, 16, 16, 3, 2, 1, true);//1*16*160*160->1*16*80*80
			Init_ReLU_arm_Params(mobilenet0_relu3_fwd, 16, true);//1*16*80*80->1*16*80*80
			Init_Conv_arm_Params(mobilenet0_conv4_fwd, 16, 32, 1, 1, 1, 0, true);//1*16*80*80->1*32*80*80
			Init_ReLU_arm_Params(mobilenet0_relu4_fwd, 32, true);//1*32*80*80->1*32*80*80
			Init_Conv_arm_Params(mobilenet0_conv5_fwd, 32, 32, 32, 3, 1, 1, true);//1*32*80*80->1*32*80*80
			Init_ReLU_arm_Params(mobilenet0_relu5_fwd, 32, true);//1*32*80*80->1*32*80*80
			Init_Conv_arm_Params(mobilenet0_conv6_fwd, 32, 32, 1, 1, 1, 0, true);//1*32*80*80->1*32*80*80
			Init_ReLU_arm_Params(mobilenet0_relu6_fwd, 32, true);//1*32*80*80->1*32*80*80
			Init_Conv_arm_Params(mobilenet0_conv7_fwd, 32, 32, 32, 3, 2, 1, true);//1*32*80*80->1*32*40*40
			Init_ReLU_arm_Params(mobilenet0_relu7_fwd, 32, true);//1*32*40*40->1*32*40*40
			Init_Conv_arm_Params(mobilenet0_conv8_fwd, 32, 64, 1, 1, 1, 0, true);//1*32*40*40->1*64*40*40
			Init_ReLU_arm_Params(mobilenet0_relu8_fwd, 64, true);//1*64*40*40->1*64*40*40
			Init_Conv_arm_Params(mobilenet0_conv9_fwd, 64, 64, 64, 3, 1, 1, true);//1*64*40*40->1*64*40*40
			Init_ReLU_arm_Params(mobilenet0_relu9_fwd, 64, true);//1*64*40*40->1*64*40*40
			Init_Conv_arm_Params(mobilenet0_conv10_fwd, 64, 64, 1, 1, 1, 0, true);//1*64*40*40->1*64*40*40
			Init_ReLU_arm_Params(mobilenet0_relu10_fwd, 64, true);//1*64*40*40->1*64*40*40
			Init_Conv_arm_Params(mobilenet0_conv11_fwd, 64, 64, 64, 3, 2, 1, true);//1*64*40*40->1*64*20*20
			Init_ReLU_arm_Params(mobilenet0_relu11_fwd, 64, true);//1*64*20*20->1*64*20*20
			Init_Conv_arm_Params(mobilenet0_conv12_fwd, 64, 128, 1, 1, 1, 0, true);//1*64*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu12_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv13_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu13_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv14_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu14_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv15_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu15_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv16_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu16_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv17_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu17_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv18_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu18_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv19_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu19_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv20_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu20_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv21_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu21_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv22_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
			Init_ReLU_arm_Params(mobilenet0_relu22_fwd, 128, true);//1*128*20*20->1*128*20*20
			Init_Conv_arm_Params(mobilenet0_conv23_fwd, 128, 128, 128, 3, 2, 1, true);//1*128*20*20->1*128*10*10
			Init_ReLU_arm_Params(mobilenet0_relu23_fwd, 128, true);//1*128*10*10->1*128*10*10
			Init_Conv_arm_Params(mobilenet0_conv24_fwd, 128, 256, 1, 1, 1, 0, true);//1*128*10*10->1*256*10*10
			Init_ReLU_arm_Params(mobilenet0_relu24_fwd, 256, true);//1*256*10*10->1*256*10*10
			Init_Conv_arm_Params(mobilenet0_conv25_fwd, 256, 256, 256, 3, 1, 1, true);//1*256*10*10->1*256*10*10
			Init_ReLU_arm_Params(mobilenet0_relu25_fwd, 256, true);//1*256*10*10->1*256*10*10
			Init_Conv_arm_Params(mobilenet0_conv26_fwd, 256, 256, 1, 1, 1, 0, true);//1*256*10*10->1*256*10*10
			Init_ReLU_arm_Params(mobilenet0_relu26_fwd, 256, true);//1*256*10*10->1*256*10*10
			Init_Conv_arm_Params(rf_c3_lateral, 256, 64, 1, 1, 1, 0, true);//1*256*10*10->1*64*10*10
			Init_ReLU_arm_Params(rf_c3_lateral_relu, 64, true);//1*64*10*10->1*64*10*10
			Init_Conv_arm_Params(rf_c3_det_conv1, 64, 32, 1, 3, 1, 1, true);//1*64*10*10->1*32*10*10
			Init_Conv_arm_Params(rf_c3_det_context_conv1, 64, 16, 1, 3, 1, 1, true);//1*64*10*10->1*16*10*10
			Init_ReLU_arm_Params(rf_c3_det_context_conv1_relu, 16, true);//1*16*10*10->1*16*10*10
			Init_Conv_arm_Params(rf_c3_det_context_conv2, 16, 16, 1, 3, 1, 1, true);//1*16*10*10->1*16*10*10
			Init_Conv_arm_Params(rf_c3_det_context_conv3_1, 16, 16, 1, 3, 1, 1, true);//1*16*10*10->1*16*10*10
			Init_ReLU_arm_Params(rf_c3_det_context_conv3_1_relu, 16, true);//1*16*10*10->1*16*10*10
			Init_Conv_arm_Params(rf_c3_det_context_conv3_2, 16, 16, 1, 3, 1, 1, true);//1*16*10*10->1*16*10*10
			Init_Concat_Params(rf_c3_det_concat, 1);//1*32*10*10 + 1*16*10*10 + 1*16*10*10 -> 1*64*10*10
			Init_ReLU_arm_Params(rf_c3_det_concat_relu, 64, true);//1*64*10*10->1*64*10*10
			Init_Conv_arm_Params(face_rpn_cls_score_stride32, 64, 4, 1, 1, 1, 0, true);//1*64*10*10->1*4*10*10
			Init_Reshape_arm_Params(face_rpn_cls_score_reshape_stride32, 0, 2, -1, 0);//1*4*10*10->1*2*20*10
			Init_Softmax_arm_Params(face_rpn_cls_prob_stride32, 2);//1*2*20*10->1*2*20*10
			Init_Reshape_arm_Params(face_rpn_cls_prob_reshape_stride32, 0, 4, -1, 0);//1*2*20*10->1*4*10*10
			Init_Conv_arm_Params(face_rpn_bbox_pred_stride32, 64, 8, 1, 1, 1, 0, true);//1*64*10*10->1*8*10*10
			Init_Conv_arm_Params(face_rpn_landmark_pred_stride32, 64, 20, 1, 1, 1, 0, true);//1*64*10*10->1*20*10*10
			Init_Conv_arm_Params(rf_c2_lateral, 128, 64, 1, 1, 1, 0, true);//1*128*20*20->1*64*20*20
			Init_ReLU_arm_Params(rf_c2_lateral_relu, 64, true);//1*64*20*20->1*64*20*20
			Init_Deconv_arm_Params(rf_c3_upsampling, 64, 64, 64, 4, 2, 1, false);//1*64*10*10->1*64*20*20
			Init_Eltwise_arm_Params(_plus0, 0);//1*64*20*20->1*64*20*20
			Init_Conv_arm_Params(rf_c2_aggr, 64, 64, 1, 3, 1, 1, true);//1*64*20*20->1*64*20*20
			Init_ReLU_arm_Params(rf_c2_aggr_relu, 64, true);//1*64*20*20->1*64*20*20
			Init_Conv_arm_Params(rf_c2_det_conv1, 64, 32, 1, 3, 1, 1, true);//1*64*20*20->1*32*20*20
			Init_Conv_arm_Params(rf_c2_det_context_conv1, 64, 16, 1, 3, 1, 1, true);//1*64*20*20->1*16*20*20
			Init_ReLU_arm_Params(rf_c2_det_context_conv1_relu, 16, true);//1*16*20*20->1*16*20*20
			Init_Conv_arm_Params(rf_c2_det_context_conv2, 16, 16, 1, 3, 1, 1, true);//1*16*20*20->1*16*20*20
			Init_Conv_arm_Params(rf_c2_det_context_conv3_1, 16, 16, 1, 3, 1, 1, true);//1*16*20*20->1*16*20*20
			Init_ReLU_arm_Params(rf_c2_det_context_conv3_1_relu, 16, true);//1*16*20*20->1*16*20*20
			Init_Conv_arm_Params(rf_c2_det_context_conv3_2, 16, 16, 1, 3, 1, 1, true);//1*16*20*20->1*16*20*20
			Init_Concat_Params(rf_c2_det_concat, 1);//1*32*20*20 + 1*16*20*20 + 1*16*20*20 -> 1*64*20*20
			Init_ReLU_arm_Params(rf_c2_det_concat_relu, 64, true);//1*64*20*20->1*64*20*20
			Init_Conv_arm_Params(face_rpn_cls_score_stride16, 64, 4, 1, 1, 1, 0, true);//1*64*20*20->1*4*20*20
			Init_Reshape_arm_Params(face_rpn_cls_score_reshape_stride16, 0, 2, -1, 0);//1*4*20*20->1*2*40*20
			Init_Softmax_arm_Params(face_rpn_cls_prob_stride16, 666);//1*2*40*20->1*2*40*20
			Init_Reshape_arm_Params(face_rpn_cls_prob_reshape_stride16, 0, 4, -1, 0);//1*2*40*20->1*4*20*20
			Init_Conv_arm_Params(face_rpn_bbox_pred_stride16, 64, 8, 1, 1, 1, 0, true);//1*64*20*20->1*8*20*20
			Init_Conv_arm_Params(face_rpn_landmark_pred_stride16, 64, 20, 1, 1, 1, 0, true);//1*64*20*20->1*20*20*20
			Init_Conv_arm_Params(rf_c1_red_conv, 64, 64, 1, 1, 1, 0, true);//1*64*40*40->1*64*40*40
			Init_ReLU_arm_Params(rf_c1_red_conv_relu, 64, true);//1*64*40*40->1*64*40*40
			Init_Deconv_arm_Params(rf_c2_upsampling, 64, 64, 64, 4, 2, 1, false);//1*64*20*20->1*64*40*40
			Init_Eltwise_arm_Params(_plus1, 0);//1*64*40*40->1*64*40*40
			Init_Conv_arm_Params(rf_c1_aggr, 64, 64, 1, 3, 1, 1, true);//1*64*40*40->1*64*40*40
			Init_ReLU_arm_Params(rf_c1_aggr_relu, 64, true);//1*64*40*40->1*64*40*40
			Init_Conv_arm_Params(rf_c1_det_conv1, 64, 32, 1, 3, 1, 1, true);//1*64*40*40->1*32*40*40
			Init_Conv_arm_Params(rf_c1_det_context_conv1, 64, 16, 1, 3, 1, 1, true);//1*64*40*40->1*16*40*40
			Init_ReLU_arm_Params(rf_c1_det_context_conv1_relu, 16, true);//1*16*40*40->1*16*40*40
			Init_Conv_arm_Params(rf_c1_det_context_conv2, 16, 16, 1, 3, 1, 1, true);//1*64*40*40->1*16*40*40
			Init_Conv_arm_Params(rf_c1_det_context_conv3_1, 16, 16, 1, 3, 1, 1, true);//1*16*40*40->1*16*40*40
			Init_ReLU_arm_Params(rf_c1_det_context_conv3_1_relu, 16, true);//1*16*40*40->1*16*40*40
			Init_Conv_arm_Params(rf_c1_det_context_conv3_2, 16, 16, 1, 3, 1, 1, true);//1*16*40*40->1*16*40*40
			Init_Concat_Params(rf_c1_det_concat, 1);//1*32*40*40 + 1*16*40*40 + 1*16*40*40 -> 1*64*40*40
			Init_ReLU_arm_Params(rf_c1_det_concat_relu, 64, true);//1*64*40*40->1*64*40*40
			Init_Conv_arm_Params(face_rpn_cls_score_stride8, 64, 4, 1, 1, 1, 0, true);//1*64*40*40->1*4*40*40
			Init_Reshape_arm_Params(face_rpn_cls_score_reshape_stride8, 0, 2, -1, 0);//1*4*40*40->1*2*80*40
			Init_Softmax_arm_Params(face_rpn_cls_prob_stride8, 2);//1*2*80*40->1*2*80*40
			Init_Reshape_arm_Params(face_rpn_cls_prob_reshape_stride8, 0, 4, -1, 0);//1*2*80*40->1*4*40*40
			Init_Conv_arm_Params(face_rpn_bbox_pred_stride8, 64, 8, 1, 1, 1, 0, true);//1*64*40*40->1*8*40*40
			Init_Conv_arm_Params(face_rpn_landmark_pred_stride8, 64, 20, 1, 1, 1, 0, true);//1*64*40*40->1*20*40*40
#else
            Init_Conv_Params(mobilenet0_conv0_fwd, 3, 8, 1, 3, 2, 1, true);//1*3*320*320->1*8*160*160
            Init_ReLU_Params(mobilenet0_relu0_fwd, 8, true);//1*8*160*160->1*8*160*160
            Init_Conv_Params(mobilenet0_conv1_fwd, 8, 8, 8, 3, 1, 1, true);//1*8*160*160->1*8*160*160
            Init_ReLU_Params(mobilenet0_relu1_fwd, 8, true);//1*8*160*160->1*8*160*160
            Init_Conv_Params(mobilenet0_conv2_fwd, 8, 16, 1, 1, 1, 0, true);//1*8*160*160->1*16*160*160
            Init_ReLU_Params(mobilenet0_relu2_fwd, 16, true);//1*16*160*160->1*16*160*160
            Init_Conv_Params(mobilenet0_conv3_fwd, 16, 16, 16, 3, 2, 1, true);//1*16*160*160->1*16*80*80
            Init_ReLU_Params(mobilenet0_relu3_fwd, 16, true);//1*16*80*80->1*16*80*80
            Init_Conv_Params(mobilenet0_conv4_fwd, 16, 32, 1, 1, 1, 0, true);//1*16*80*80->1*32*80*80
            Init_ReLU_Params(mobilenet0_relu4_fwd, 32, true);//1*32*80*80->1*32*80*80
            Init_Conv_Params(mobilenet0_conv5_fwd, 32, 32, 32, 3, 1, 1, true);//1*32*80*80->1*32*80*80
            Init_ReLU_Params(mobilenet0_relu5_fwd, 32, true);//1*32*80*80->1*32*80*80
            Init_Conv_Params(mobilenet0_conv6_fwd, 32, 32, 1, 1, 1, 0, true);//1*32*80*80->1*32*80*80
            Init_ReLU_Params(mobilenet0_relu6_fwd, 32, true);//1*32*80*80->1*32*80*80
            Init_Conv_Params(mobilenet0_conv7_fwd, 32, 32, 32, 3, 2, 1, true);//1*32*80*80->1*32*40*40
            Init_ReLU_Params(mobilenet0_relu7_fwd, 32, true);//1*32*40*40->1*32*40*40
            Init_Conv_Params(mobilenet0_conv8_fwd, 32, 64, 1, 1, 1, 0, true);//1*32*40*40->1*64*40*40
            Init_ReLU_Params(mobilenet0_relu8_fwd, 64, true);//1*64*40*40->1*64*40*40
            Init_Conv_Params(mobilenet0_conv9_fwd, 64, 64, 64, 3, 1, 1, true);//1*64*40*40->1*64*40*40
            Init_ReLU_Params(mobilenet0_relu9_fwd, 64, true);//1*64*40*40->1*64*40*40
            Init_Conv_Params(mobilenet0_conv10_fwd, 64, 64, 1, 1, 1, 0, true);//1*64*40*40->1*64*40*40
            Init_ReLU_Params(mobilenet0_relu10_fwd, 64, true);//1*64*40*40->1*64*40*40
            Init_Conv_Params(mobilenet0_conv11_fwd, 64, 64, 64, 3, 2, 1, true);//1*64*40*40->1*64*20*20
            Init_ReLU_Params(mobilenet0_relu11_fwd, 64, true);//1*64*20*20->1*64*20*20
            Init_Conv_Params(mobilenet0_conv12_fwd, 64, 128, 1, 1, 1, 0, true);//1*64*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu12_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv13_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu13_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv14_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu14_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv15_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu15_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv16_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu16_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv17_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu17_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv18_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu18_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv19_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu19_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv20_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu20_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv21_fwd, 128, 128, 128, 3, 1, 1, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu21_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv22_fwd, 128, 128, 1, 1, 1, 0, true);//1*128*20*20->1*128*20*20
            Init_ReLU_Params(mobilenet0_relu22_fwd, 128, true);//1*128*20*20->1*128*20*20
            Init_Conv_Params(mobilenet0_conv23_fwd, 128, 128, 128, 3, 2, 1, true);//1*128*20*20->1*128*10*10
            Init_ReLU_Params(mobilenet0_relu23_fwd, 128, true);//1*128*10*10->1*128*10*10
            Init_Conv_Params(mobilenet0_conv24_fwd, 128, 256, 1, 1, 1, 0, true);//1*128*10*10->1*256*10*10
            Init_ReLU_Params(mobilenet0_relu24_fwd, 256, true);//1*256*10*10->1*256*10*10
            Init_Conv_Params(mobilenet0_conv25_fwd, 256, 256, 256, 3, 1, 1, true);//1*256*10*10->1*256*10*10
            Init_ReLU_Params(mobilenet0_relu25_fwd, 256, true);//1*256*10*10->1*256*10*10
            Init_Conv_Params(mobilenet0_conv26_fwd, 256, 256, 1, 1, 1, 0, true);//1*256*10*10->1*256*10*10
            Init_ReLU_Params(mobilenet0_relu26_fwd, 256, true);//1*256*10*10->1*256*10*10
            Init_Conv_Params(rf_c3_lateral, 256, 64, 1, 1, 1, 0, true);//1*256*10*10->1*64*10*10
            Init_ReLU_Params(rf_c3_lateral_relu, 64, true);//1*64*10*10->1*64*10*10
            Init_Conv_Params(rf_c3_det_conv1, 64, 32, 1, 3, 1, 1, true);//1*64*10*10->1*32*10*10
            Init_Conv_Params(rf_c3_det_context_conv1, 64, 16, 1, 3, 1, 1, true);//1*64*10*10->1*16*10*10
            Init_ReLU_Params(rf_c3_det_context_conv1_relu, 16, true);//1*16*10*10->1*16*10*10
            Init_Conv_Params(rf_c3_det_context_conv2, 16, 16, 1, 3, 1, 1, true);//1*16*10*10->1*16*10*10
            Init_Conv_Params(rf_c3_det_context_conv3_1, 16, 16, 1, 3, 1, 1, true);//1*16*10*10->1*16*10*10
            Init_ReLU_Params(rf_c3_det_context_conv3_1_relu, 16, true);//1*16*10*10->1*16*10*10
            Init_Conv_Params(rf_c3_det_context_conv3_2, 16, 16, 1, 3, 1, 1, true);//1*16*10*10->1*16*10*10
            Init_Concat_Params(rf_c3_det_concat, 1);//1*32*10*10 + 1*16*10*10 + 1*16*10*10 -> 1*64*10*10
            Init_ReLU_Params(rf_c3_det_concat_relu, 64, true);//1*64*10*10->1*64*10*10
            Init_Conv_Params(face_rpn_cls_score_stride32, 64, 4, 1, 1, 1, 0, true);//1*64*10*10->1*4*10*10
            Init_Reshape_Params(face_rpn_cls_score_reshape_stride32, 0, 2, -1, 0);//1*4*10*10->1*2*20*10
            Init_Softmax_Params(face_rpn_cls_prob_stride32, 2);//1*2*20*10->1*2*20*10
            Init_Reshape_Params(face_rpn_cls_prob_reshape_stride32, 0, 4, -1, 0);//1*2*20*10->1*4*10*10
            Init_Conv_Params(face_rpn_bbox_pred_stride32, 64, 8, 1, 1, 1, 0, true);//1*64*10*10->1*8*10*10
            Init_Conv_Params(face_rpn_landmark_pred_stride32, 64, 20, 1, 1, 1, 0, true);//1*64*10*10->1*20*10*10
            Init_Conv_Params(rf_c2_lateral, 128, 64, 1, 1, 1, 0, true);//1*128*20*20->1*64*20*20
            Init_ReLU_Params(rf_c2_lateral_relu, 64, true);//1*64*20*20->1*64*20*20
            Init_Deconv_Params(rf_c3_upsampling, 64, 64, 64, 4, 2, 1, false);//1*64*10*10->1*64*20*20
            Init_Eltwise_Params(_plus0, 0);//1*64*20*20->1*64*20*20
            Init_Conv_Params(rf_c2_aggr, 64, 64, 1, 3, 1, 1, true);//1*64*20*20->1*64*20*20
            Init_ReLU_Params(rf_c2_aggr_relu, 64, true);//1*64*20*20->1*64*20*20
            Init_Conv_Params(rf_c2_det_conv1, 64, 32, 1, 3, 1, 1, true);//1*64*20*20->1*32*20*20
            Init_Conv_Params(rf_c2_det_context_conv1, 64, 16, 1, 3, 1, 1, true);//1*64*20*20->1*16*20*20
            Init_ReLU_Params(rf_c2_det_context_conv1_relu, 16, true);//1*16*20*20->1*16*20*20
            Init_Conv_Params(rf_c2_det_context_conv2, 16, 16, 1, 3, 1, 1, true);//1*16*20*20->1*16*20*20
            Init_Conv_Params(rf_c2_det_context_conv3_1, 16, 16, 1, 3, 1, 1, true);//1*16*20*20->1*16*20*20
            Init_ReLU_Params(rf_c2_det_context_conv3_1_relu, 16, true);//1*16*20*20->1*16*20*20
            Init_Conv_Params(rf_c2_det_context_conv3_2, 16, 16, 1, 3, 1, 1, true);//1*16*20*20->1*16*20*20
            Init_Concat_Params(rf_c2_det_concat, 1);//1*32*20*20 + 1*16*20*20 + 1*16*20*20 -> 1*64*20*20
            Init_ReLU_Params(rf_c2_det_concat_relu, 64, true);//1*64*20*20->1*64*20*20
            Init_Conv_Params(face_rpn_cls_score_stride16, 64, 4, 1, 1, 1, 0, true);//1*64*20*20->1*4*20*20
            Init_Reshape_Params(face_rpn_cls_score_reshape_stride16, 0, 2, -1, 0);//1*4*20*20->1*2*40*20
            Init_Softmax_Params(face_rpn_cls_prob_stride16, 666);//1*2*40*20->1*2*40*20
            Init_Reshape_Params(face_rpn_cls_prob_reshape_stride16, 0, 4, -1, 0);//1*2*40*20->1*4*20*20
            Init_Conv_Params(face_rpn_bbox_pred_stride16, 64, 8, 1, 1, 1, 0, true);//1*64*20*20->1*8*20*20
            Init_Conv_Params(face_rpn_landmark_pred_stride16, 64, 20, 1, 1, 1, 0, true);//1*64*20*20->1*20*20*20
            Init_Conv_Params(rf_c1_red_conv, 64, 64, 1, 1, 1, 0, true);//1*64*40*40->1*64*40*40
            Init_ReLU_Params(rf_c1_red_conv_relu, 64, true);//1*64*40*40->1*64*40*40
            Init_Deconv_Params(rf_c2_upsampling, 64, 64, 64, 4, 2, 1, false);//1*64*20*20->1*64*40*40
            Init_Eltwise_Params(_plus1, 0);//1*64*40*40->1*64*40*40
            Init_Conv_Params(rf_c1_aggr, 64, 64, 1, 3, 1, 1, true);//1*64*40*40->1*64*40*40
            Init_ReLU_Params(rf_c1_aggr_relu, 64, true);//1*64*40*40->1*64*40*40
            Init_Conv_Params(rf_c1_det_conv1, 64, 32, 1, 3, 1, 1, true);//1*64*40*40->1*32*40*40
            Init_Conv_Params(rf_c1_det_context_conv1, 64, 16, 1, 3, 1, 1, true);//1*64*40*40->1*16*40*40
            Init_ReLU_Params(rf_c1_det_context_conv1_relu, 16, true);//1*16*40*40->1*16*40*40
            Init_Conv_Params(rf_c1_det_context_conv2, 16, 16, 1, 3, 1, 1, true);//1*64*40*40->1*16*40*40
            Init_Conv_Params(rf_c1_det_context_conv3_1, 16, 16, 1, 3, 1, 1, true);//1*16*40*40->1*16*40*40
            Init_ReLU_Params(rf_c1_det_context_conv3_1_relu, 16, true);//1*16*40*40->1*16*40*40
            Init_Conv_Params(rf_c1_det_context_conv3_2, 16, 16, 1, 3, 1, 1, true);//1*16*40*40->1*16*40*40
            Init_Concat_Params(rf_c1_det_concat, 1);//1*32*40*40 + 1*16*40*40 + 1*16*40*40 -> 1*64*40*40
            Init_ReLU_Params(rf_c1_det_concat_relu, 64, true);//1*64*40*40->1*64*40*40
            Init_Conv_Params(face_rpn_cls_score_stride8, 64, 4, 1, 1, 1, 0, true);//1*64*40*40->1*4*40*40
            Init_Reshape_Params(face_rpn_cls_score_reshape_stride8, 0, 2, -1, 0);//1*4*40*40->1*2*80*40
            Init_Softmax_Params(face_rpn_cls_prob_stride8, 2);//1*2*80*40->1*2*80*40
            Init_Reshape_Params(face_rpn_cls_prob_reshape_stride8, 0, 4, -1, 0);//1*2*80*40->1*4*40*40
            Init_Conv_Params(face_rpn_bbox_pred_stride8, 64, 8, 1, 1, 1, 0, true);//1*64*40*40->1*8*40*40
            Init_Conv_Params(face_rpn_landmark_pred_stride8, 64, 20, 1, 1, 1, 0, true);//1*64*40*40->1*20*40*40
#endif // __ARM_NEON
			
		}

		Retina_net::~Retina_net()
		{
			delete mobilenet0_conv0_fwd;
			delete mobilenet0_relu0_fwd;
			delete mobilenet0_conv1_fwd;
			delete mobilenet0_relu1_fwd;
			delete mobilenet0_conv2_fwd;
			delete mobilenet0_relu2_fwd;
			delete mobilenet0_conv3_fwd;
			delete mobilenet0_relu3_fwd;
			delete mobilenet0_conv4_fwd;
			delete mobilenet0_relu4_fwd;
			delete mobilenet0_conv5_fwd;
			delete mobilenet0_relu5_fwd;
			delete mobilenet0_conv6_fwd;
			delete mobilenet0_relu6_fwd;
			delete mobilenet0_conv7_fwd;
			delete mobilenet0_relu7_fwd;
			delete mobilenet0_conv8_fwd;
			delete mobilenet0_relu8_fwd;
			delete mobilenet0_conv9_fwd;
			delete mobilenet0_relu9_fwd;
			delete mobilenet0_conv10_fwd;
			delete mobilenet0_relu10_fwd;
			delete mobilenet0_conv11_fwd;
			delete mobilenet0_relu11_fwd;
			delete mobilenet0_conv12_fwd;
			delete mobilenet0_relu12_fwd;
			delete mobilenet0_conv13_fwd;
			delete mobilenet0_relu13_fwd;
			delete mobilenet0_conv14_fwd;
			delete mobilenet0_relu14_fwd;
			delete mobilenet0_conv15_fwd;
			delete mobilenet0_relu15_fwd;
			delete mobilenet0_conv16_fwd;
			delete mobilenet0_relu16_fwd;
			delete mobilenet0_conv17_fwd;
			delete mobilenet0_relu17_fwd;
			delete mobilenet0_conv18_fwd;
			delete mobilenet0_relu18_fwd;
			delete mobilenet0_conv19_fwd;
			delete mobilenet0_relu19_fwd;
			delete mobilenet0_conv20_fwd;
			delete mobilenet0_relu20_fwd;
			delete mobilenet0_conv21_fwd;
			delete mobilenet0_relu21_fwd;
			delete mobilenet0_conv22_fwd;
			delete mobilenet0_relu22_fwd;
			delete mobilenet0_conv23_fwd;
			delete mobilenet0_relu23_fwd;
			delete mobilenet0_conv24_fwd;
			delete mobilenet0_relu24_fwd;
			delete mobilenet0_conv25_fwd;
			delete mobilenet0_relu25_fwd;
			delete mobilenet0_conv26_fwd;
			delete mobilenet0_relu26_fwd;
			delete rf_c3_lateral;
			delete rf_c3_lateral_relu;
			delete rf_c3_det_conv1;
			delete rf_c3_det_context_conv1;
			delete rf_c3_det_context_conv1_relu;
			delete rf_c3_det_context_conv2;
			delete rf_c3_det_context_conv3_1;
			delete rf_c3_det_context_conv3_1_relu;
			delete rf_c3_det_context_conv3_2;
			delete rf_c3_det_concat;
			delete rf_c3_det_concat_relu;
			delete face_rpn_cls_score_stride32;
			delete face_rpn_cls_score_reshape_stride32;
			delete face_rpn_cls_prob_stride32;
			delete face_rpn_cls_prob_reshape_stride32;
			delete face_rpn_bbox_pred_stride32;
			delete face_rpn_landmark_pred_stride32;
			delete rf_c2_lateral;
			delete rf_c2_lateral_relu;
			delete rf_c3_upsampling;
			delete _plus0;
			delete rf_c2_aggr;
			delete rf_c2_aggr_relu;
			delete rf_c2_det_conv1;
			delete rf_c2_det_context_conv1;
			delete rf_c2_det_context_conv1_relu;
			delete rf_c2_det_context_conv2;
			delete rf_c2_det_context_conv3_1;
			delete rf_c2_det_context_conv3_1_relu;
			delete rf_c2_det_context_conv3_2;
			delete rf_c2_det_concat;
			delete rf_c2_det_concat_relu;
			delete face_rpn_cls_score_stride16;
			delete face_rpn_cls_score_reshape_stride16;
			delete face_rpn_cls_prob_stride16;
			delete face_rpn_cls_prob_reshape_stride16;
			delete face_rpn_bbox_pred_stride16;
			delete face_rpn_landmark_pred_stride16;
			delete rf_c1_red_conv;
			delete rf_c1_red_conv_relu;
			delete rf_c2_upsampling;
			delete _plus1;
			delete rf_c1_aggr;
			delete rf_c1_aggr_relu;
			delete rf_c1_det_conv1;
			delete rf_c1_det_context_conv1;
			delete rf_c1_det_context_conv1_relu;
			delete rf_c1_det_context_conv2;
			delete rf_c1_det_context_conv3_1;
			delete rf_c1_det_context_conv3_1_relu;
			delete rf_c1_det_context_conv3_2;
			delete rf_c1_det_concat;
			delete rf_c1_det_concat_relu;
			delete face_rpn_cls_score_stride8;
			delete face_rpn_cls_score_reshape_stride8;
			delete face_rpn_cls_prob_stride8;
			delete face_rpn_cls_prob_reshape_stride8;
			delete face_rpn_bbox_pred_stride8;
			delete face_rpn_landmark_pred_stride8;

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

		void Retina_net::Forward_cpu(const std::shared_ptr<tensor<float>> &input_data)
		{
			mobilenet0_conv0_fwd->Forward(input_data, mobilenet0_conv0_fwd_top_data);
			mobilenet0_relu0_fwd->Forward_cpu(mobilenet0_conv0_fwd_top_data);
			mobilenet0_conv1_fwd->Forward(mobilenet0_conv0_fwd_top_data, mobilenet0_conv1_fwd_top_data);
			mobilenet0_relu1_fwd->Forward_cpu(mobilenet0_conv1_fwd_top_data);
			mobilenet0_conv2_fwd->Forward(mobilenet0_conv1_fwd_top_data, mobilenet0_conv2_fwd_top_data);
			mobilenet0_relu2_fwd->Forward_cpu(mobilenet0_conv2_fwd_top_data);
			mobilenet0_conv3_fwd->Forward(mobilenet0_conv2_fwd_top_data, mobilenet0_conv3_fwd_top_data);
			mobilenet0_relu3_fwd->Forward_cpu(mobilenet0_conv3_fwd_top_data);
			mobilenet0_conv4_fwd->Forward(mobilenet0_conv3_fwd_top_data, mobilenet0_conv4_fwd_top_data);
			mobilenet0_relu4_fwd->Forward_cpu(mobilenet0_conv4_fwd_top_data);
			mobilenet0_conv5_fwd->Forward(mobilenet0_conv4_fwd_top_data, mobilenet0_conv5_fwd_top_data);
			mobilenet0_relu5_fwd->Forward_cpu(mobilenet0_conv5_fwd_top_data);
			mobilenet0_conv6_fwd->Forward(mobilenet0_conv5_fwd_top_data, mobilenet0_conv6_fwd_top_data);
			mobilenet0_relu6_fwd->Forward_cpu(mobilenet0_conv6_fwd_top_data);
			mobilenet0_conv7_fwd->Forward(mobilenet0_conv6_fwd_top_data, mobilenet0_conv7_fwd_top_data);
			mobilenet0_relu7_fwd->Forward_cpu(mobilenet0_conv7_fwd_top_data);
			mobilenet0_conv8_fwd->Forward(mobilenet0_conv7_fwd_top_data, mobilenet0_conv8_fwd_top_data);
			mobilenet0_relu8_fwd->Forward_cpu(mobilenet0_conv8_fwd_top_data);
			mobilenet0_conv9_fwd->Forward(mobilenet0_conv8_fwd_top_data, mobilenet0_conv9_fwd_top_data);
			mobilenet0_relu9_fwd->Forward_cpu(mobilenet0_conv9_fwd_top_data);
			mobilenet0_conv10_fwd->Forward(mobilenet0_conv9_fwd_top_data, mobilenet0_conv10_fwd_top_data);
			mobilenet0_relu10_fwd->Forward_cpu(mobilenet0_conv10_fwd_top_data);
			mobilenet0_conv11_fwd->Forward(mobilenet0_conv10_fwd_top_data, mobilenet0_conv11_fwd_top_data);
			mobilenet0_relu11_fwd->Forward_cpu(mobilenet0_conv11_fwd_top_data);
			mobilenet0_conv12_fwd->Forward(mobilenet0_conv11_fwd_top_data, mobilenet0_conv12_fwd_top_data);
			mobilenet0_relu12_fwd->Forward_cpu(mobilenet0_conv12_fwd_top_data);
			mobilenet0_conv13_fwd->Forward(mobilenet0_conv12_fwd_top_data, mobilenet0_conv13_fwd_top_data);
			mobilenet0_relu13_fwd->Forward_cpu(mobilenet0_conv13_fwd_top_data);
			mobilenet0_conv14_fwd->Forward(mobilenet0_conv13_fwd_top_data, mobilenet0_conv14_fwd_top_data);
			mobilenet0_relu14_fwd->Forward_cpu(mobilenet0_conv14_fwd_top_data);
			mobilenet0_conv15_fwd->Forward(mobilenet0_conv14_fwd_top_data, mobilenet0_conv15_fwd_top_data);
			mobilenet0_relu15_fwd->Forward_cpu(mobilenet0_conv15_fwd_top_data);
			mobilenet0_conv16_fwd->Forward(mobilenet0_conv15_fwd_top_data, mobilenet0_conv16_fwd_top_data);
			mobilenet0_relu16_fwd->Forward_cpu(mobilenet0_conv16_fwd_top_data);
			mobilenet0_conv17_fwd->Forward(mobilenet0_conv16_fwd_top_data, mobilenet0_conv17_fwd_top_data);
			mobilenet0_relu17_fwd->Forward_cpu(mobilenet0_conv17_fwd_top_data);
			mobilenet0_conv18_fwd->Forward(mobilenet0_conv17_fwd_top_data, mobilenet0_conv18_fwd_top_data);
			mobilenet0_relu18_fwd->Forward_cpu(mobilenet0_conv18_fwd_top_data);
			mobilenet0_conv19_fwd->Forward(mobilenet0_conv18_fwd_top_data, mobilenet0_conv19_fwd_top_data);
			mobilenet0_relu19_fwd->Forward_cpu(mobilenet0_conv19_fwd_top_data);
			mobilenet0_conv20_fwd->Forward(mobilenet0_conv19_fwd_top_data, mobilenet0_conv20_fwd_top_data);
			mobilenet0_relu20_fwd->Forward_cpu(mobilenet0_conv20_fwd_top_data);
			mobilenet0_conv21_fwd->Forward(mobilenet0_conv20_fwd_top_data, mobilenet0_conv21_fwd_top_data);
			mobilenet0_relu21_fwd->Forward_cpu(mobilenet0_conv21_fwd_top_data);
			mobilenet0_conv22_fwd->Forward(mobilenet0_conv21_fwd_top_data, mobilenet0_conv22_fwd_top_data);
			mobilenet0_relu22_fwd->Forward_cpu(mobilenet0_conv22_fwd_top_data);
			mobilenet0_conv23_fwd->Forward(mobilenet0_conv22_fwd_top_data, mobilenet0_conv23_fwd_top_data);
			mobilenet0_relu23_fwd->Forward_cpu(mobilenet0_conv23_fwd_top_data);
			mobilenet0_conv24_fwd->Forward(mobilenet0_conv23_fwd_top_data, mobilenet0_conv24_fwd_top_data);
			mobilenet0_relu24_fwd->Forward_cpu(mobilenet0_conv24_fwd_top_data);
			mobilenet0_conv25_fwd->Forward(mobilenet0_conv24_fwd_top_data, mobilenet0_conv25_fwd_top_data);
			mobilenet0_relu25_fwd->Forward_cpu(mobilenet0_conv25_fwd_top_data);
			mobilenet0_conv26_fwd->Forward(mobilenet0_conv25_fwd_top_data, mobilenet0_conv26_fwd_top_data);
			mobilenet0_relu26_fwd->Forward_cpu(mobilenet0_conv26_fwd_top_data);
			rf_c3_lateral->Forward(mobilenet0_conv26_fwd_top_data, rf_c3_lateral_top_data);
			rf_c3_lateral_relu->Forward_cpu(rf_c3_lateral_top_data);
			rf_c3_det_conv1->Forward(rf_c3_lateral_top_data, rf_c3_det_conv1_top_data);
			rf_c3_det_context_conv1->Forward(rf_c3_lateral_top_data, rf_c3_det_context_conv1_top_data);
			rf_c3_det_context_conv1_relu->Forward_cpu(rf_c3_det_context_conv1_top_data);
			rf_c3_det_context_conv2->Forward(rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv2_top_data);
			rf_c3_det_context_conv3_1->Forward(rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv3_1_top_data);
			rf_c3_det_context_conv3_1_relu->Forward_cpu(rf_c3_det_context_conv3_1_top_data);
			rf_c3_det_context_conv3_2->Forward(rf_c3_det_context_conv3_1_top_data, rf_c3_det_context_conv3_2_top_data);
			rf_c3_det_concat->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{rf_c3_det_conv1_top_data, rf_c3_det_context_conv2_top_data, rf_c3_det_context_conv3_2_top_data}, rf_c3_det_concat_top_data);
			rf_c3_det_concat_relu->Forward_cpu(rf_c3_det_concat_top_data);
			face_rpn_cls_score_stride32->Forward(rf_c3_det_concat_top_data, face_rpn_cls_score_stride32_top_data);
			face_rpn_cls_score_reshape_stride32->Forward(face_rpn_cls_score_stride32_top_data, face_rpn_cls_score_reshape_stride32_top_data);
			face_rpn_cls_prob_stride32->Forward_cpu(face_rpn_cls_score_reshape_stride32_top_data, face_rpn_cls_prob_stride32_top_data);
			face_rpn_cls_prob_reshape_stride32->Forward(face_rpn_cls_prob_stride32_top_data, face_rpn_cls_prob_reshape_stride32_top_data);
			face_rpn_bbox_pred_stride32->Forward(rf_c3_det_concat_top_data, face_rpn_bbox_pred_stride32_top_data);
			face_rpn_landmark_pred_stride32->Forward(rf_c3_det_concat_top_data, face_rpn_landmark_pred_stride32_top_data);
			rf_c2_lateral->Forward(mobilenet0_conv22_fwd_top_data, rf_c2_lateral_top_data);
			rf_c2_lateral_relu->Forward_cpu(rf_c2_lateral_top_data);
			rf_c3_upsampling->Forward(rf_c3_lateral_top_data, rf_c3_upsampling_top_data);
			_plus0->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{rf_c3_upsampling_top_data, rf_c2_lateral_top_data}, _plus0_top_data);
			rf_c2_aggr->Forward(_plus0_top_data, rf_c2_aggr_top_data);
			rf_c2_aggr_relu->Forward_cpu(rf_c2_aggr_top_data);
			rf_c2_det_conv1->Forward(rf_c2_aggr_top_data, rf_c2_det_conv1_top_data);
			rf_c2_det_context_conv1->Forward(rf_c2_aggr_top_data, rf_c2_det_context_conv1_top_data);
			rf_c2_det_context_conv1_relu->Forward_cpu(rf_c2_det_context_conv1_top_data);
			rf_c2_det_context_conv2->Forward(rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv2_top_data);
			rf_c2_det_context_conv3_1->Forward(rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv3_1_top_data);
			rf_c2_det_context_conv3_1_relu->Forward_cpu(rf_c2_det_context_conv3_1_top_data);
			rf_c2_det_context_conv3_2->Forward(rf_c2_det_context_conv3_1_top_data, rf_c2_det_context_conv3_2_top_data);
			rf_c2_det_concat->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{rf_c2_det_conv1_top_data, rf_c2_det_context_conv2_top_data, rf_c2_det_context_conv3_2_top_data}, rf_c2_det_concat_top_data);
			rf_c2_det_concat_relu->Forward_cpu(rf_c2_det_concat_top_data);
			face_rpn_cls_score_stride16->Forward(rf_c2_det_concat_top_data, face_rpn_cls_score_stride16_top_data);
			face_rpn_cls_score_reshape_stride16->Forward(face_rpn_cls_score_stride16_top_data, face_rpn_cls_score_reshape_stride16_top_data);
			face_rpn_cls_prob_stride16->Forward_cpu(face_rpn_cls_score_reshape_stride16_top_data, face_rpn_cls_prob_stride16_top_data);
			face_rpn_cls_prob_reshape_stride16->Forward(face_rpn_cls_prob_stride16_top_data, face_rpn_cls_prob_reshape_stride16_top_data);
			face_rpn_bbox_pred_stride16->Forward(rf_c2_det_concat_top_data, face_rpn_bbox_pred_stride16_top_data);
			face_rpn_landmark_pred_stride16->Forward(rf_c2_det_concat_top_data, face_rpn_landmark_pred_stride16_top_data);
			rf_c1_red_conv->Forward(mobilenet0_conv10_fwd_top_data, rf_c1_red_conv_top_data);
			rf_c1_red_conv_relu->Forward_cpu(rf_c1_red_conv_top_data);
			rf_c2_upsampling->Forward(rf_c2_aggr_top_data, rf_c2_upsampling_top_data);
			_plus1->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{rf_c1_red_conv_top_data, rf_c2_upsampling_top_data}, _plus1_top_data);
			rf_c1_aggr->Forward(_plus1_top_data, rf_c1_aggr_top_data);
			rf_c1_aggr_relu->Forward_cpu(rf_c1_aggr_top_data);
			rf_c1_det_conv1->Forward(rf_c1_aggr_top_data, rf_c1_det_conv1_top_data);
			rf_c1_det_context_conv1->Forward(rf_c1_aggr_top_data, rf_c1_det_context_conv1_top_data);
			rf_c1_det_context_conv1_relu->Forward_cpu(rf_c1_det_context_conv1_top_data);
			rf_c1_det_context_conv2->Forward(rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv2_top_data);
			rf_c1_det_context_conv3_1->Forward(rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv3_1_top_data);
			rf_c1_det_context_conv3_1_relu->Forward_cpu(rf_c1_det_context_conv3_1_top_data);
			rf_c1_det_context_conv3_2->Forward(rf_c1_det_context_conv3_1_top_data, rf_c1_det_context_conv3_2_top_data);
			rf_c1_det_concat->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{rf_c1_det_conv1_top_data, rf_c1_det_context_conv2_top_data, rf_c1_det_context_conv3_2_top_data}, rf_c1_det_concat_top_data);
			rf_c1_det_concat_relu->Forward_cpu(rf_c1_det_concat_top_data);
			face_rpn_cls_score_stride8->Forward(rf_c1_det_concat_top_data, face_rpn_cls_score_stride8_top_data);
			face_rpn_cls_score_reshape_stride8->Forward(face_rpn_cls_score_stride8_top_data, face_rpn_cls_score_reshape_stride8_top_data);
			face_rpn_cls_prob_stride8->Forward_cpu(face_rpn_cls_score_reshape_stride8_top_data, face_rpn_cls_prob_stride8_top_data);
			face_rpn_cls_prob_reshape_stride8->Forward(face_rpn_cls_prob_stride8_top_data, face_rpn_cls_prob_reshape_stride8_top_data);
			face_rpn_bbox_pred_stride8->Forward(rf_c1_det_concat_top_data, face_rpn_bbox_pred_stride8_top_data);
			face_rpn_landmark_pred_stride8->Forward(rf_c1_det_concat_top_data, face_rpn_landmark_pred_stride8_top_data);
		}

#ifdef USE_CUDA
#ifdef USE_CUDNN
		void Retina_net::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> &input_data)
		{
			mobilenet0_conv0_fwd->Forward(cudnn_handle_, input_data, mobilenet0_conv0_fwd_top_data);
			mobilenet0_relu0_fwd->Forward_gpu_native(mobilenet0_conv0_fwd_top_data);
			mobilenet0_conv1_fwd->Forward(cudnn_handle_, mobilenet0_conv0_fwd_top_data, mobilenet0_conv1_fwd_top_data);
			mobilenet0_relu1_fwd->Forward_gpu_native(mobilenet0_conv1_fwd_top_data);
			mobilenet0_conv2_fwd->Forward(cudnn_handle_, mobilenet0_conv1_fwd_top_data, mobilenet0_conv2_fwd_top_data);
			mobilenet0_relu2_fwd->Forward_gpu_native(mobilenet0_conv2_fwd_top_data);
			mobilenet0_conv3_fwd->Forward(cudnn_handle_, mobilenet0_conv2_fwd_top_data, mobilenet0_conv3_fwd_top_data);
			mobilenet0_relu3_fwd->Forward_gpu_native(mobilenet0_conv3_fwd_top_data);
			mobilenet0_conv4_fwd->Forward(cudnn_handle_, mobilenet0_conv3_fwd_top_data, mobilenet0_conv4_fwd_top_data);
			mobilenet0_relu4_fwd->Forward_gpu_native(mobilenet0_conv4_fwd_top_data);
			mobilenet0_conv5_fwd->Forward(cudnn_handle_, mobilenet0_conv4_fwd_top_data, mobilenet0_conv5_fwd_top_data);
			mobilenet0_relu5_fwd->Forward_gpu_native(mobilenet0_conv5_fwd_top_data);
			mobilenet0_conv6_fwd->Forward(cudnn_handle_, mobilenet0_conv5_fwd_top_data, mobilenet0_conv6_fwd_top_data);
			mobilenet0_relu6_fwd->Forward_gpu_native(mobilenet0_conv6_fwd_top_data);
			mobilenet0_conv7_fwd->Forward(cudnn_handle_, mobilenet0_conv6_fwd_top_data, mobilenet0_conv7_fwd_top_data);
			mobilenet0_relu7_fwd->Forward_gpu_native(mobilenet0_conv7_fwd_top_data);
			mobilenet0_conv8_fwd->Forward(cudnn_handle_, mobilenet0_conv7_fwd_top_data, mobilenet0_conv8_fwd_top_data);
			mobilenet0_relu8_fwd->Forward_gpu_native(mobilenet0_conv8_fwd_top_data);
			mobilenet0_conv9_fwd->Forward(cudnn_handle_, mobilenet0_conv8_fwd_top_data, mobilenet0_conv9_fwd_top_data);
			mobilenet0_relu9_fwd->Forward_gpu_native(mobilenet0_conv9_fwd_top_data);
			mobilenet0_conv10_fwd->Forward(cudnn_handle_, mobilenet0_conv9_fwd_top_data, mobilenet0_conv10_fwd_top_data);
			mobilenet0_relu10_fwd->Forward_gpu_native(mobilenet0_conv10_fwd_top_data);
			mobilenet0_conv11_fwd->Forward(cudnn_handle_, mobilenet0_conv10_fwd_top_data, mobilenet0_conv11_fwd_top_data);
			mobilenet0_relu11_fwd->Forward_gpu_native(mobilenet0_conv11_fwd_top_data);
			mobilenet0_conv12_fwd->Forward(cudnn_handle_, mobilenet0_conv11_fwd_top_data, mobilenet0_conv12_fwd_top_data);
			mobilenet0_relu12_fwd->Forward_gpu_native(mobilenet0_conv12_fwd_top_data);
			mobilenet0_conv13_fwd->Forward(cudnn_handle_, mobilenet0_conv12_fwd_top_data, mobilenet0_conv13_fwd_top_data);
			mobilenet0_relu13_fwd->Forward_gpu_native(mobilenet0_conv13_fwd_top_data);
			mobilenet0_conv14_fwd->Forward(cudnn_handle_, mobilenet0_conv13_fwd_top_data, mobilenet0_conv14_fwd_top_data);
			mobilenet0_relu14_fwd->Forward_gpu_native(mobilenet0_conv14_fwd_top_data);
			mobilenet0_conv15_fwd->Forward(cudnn_handle_, mobilenet0_conv14_fwd_top_data, mobilenet0_conv15_fwd_top_data);
			mobilenet0_relu15_fwd->Forward_gpu_native(mobilenet0_conv15_fwd_top_data);
			mobilenet0_conv16_fwd->Forward(cudnn_handle_, mobilenet0_conv15_fwd_top_data, mobilenet0_conv16_fwd_top_data);
			mobilenet0_relu16_fwd->Forward_gpu_native(mobilenet0_conv16_fwd_top_data);
			mobilenet0_conv17_fwd->Forward(cudnn_handle_, mobilenet0_conv16_fwd_top_data, mobilenet0_conv17_fwd_top_data);
			mobilenet0_relu17_fwd->Forward_gpu_native(mobilenet0_conv17_fwd_top_data);
			mobilenet0_conv18_fwd->Forward(cudnn_handle_, mobilenet0_conv17_fwd_top_data, mobilenet0_conv18_fwd_top_data);
			mobilenet0_relu18_fwd->Forward_gpu_native(mobilenet0_conv18_fwd_top_data);
			mobilenet0_conv19_fwd->Forward(cudnn_handle_, mobilenet0_conv18_fwd_top_data, mobilenet0_conv19_fwd_top_data);
			mobilenet0_relu19_fwd->Forward_gpu_native(mobilenet0_conv19_fwd_top_data);
			mobilenet0_conv20_fwd->Forward(cudnn_handle_, mobilenet0_conv19_fwd_top_data, mobilenet0_conv20_fwd_top_data);
			mobilenet0_relu20_fwd->Forward_gpu_native(mobilenet0_conv20_fwd_top_data);
			mobilenet0_conv21_fwd->Forward(cudnn_handle_, mobilenet0_conv20_fwd_top_data, mobilenet0_conv21_fwd_top_data);
			mobilenet0_relu21_fwd->Forward_gpu_native(mobilenet0_conv21_fwd_top_data);
			mobilenet0_conv22_fwd->Forward(cudnn_handle_, mobilenet0_conv21_fwd_top_data, mobilenet0_conv22_fwd_top_data);
			mobilenet0_relu22_fwd->Forward_gpu_native(mobilenet0_conv22_fwd_top_data);
			mobilenet0_conv23_fwd->Forward(cudnn_handle_, mobilenet0_conv22_fwd_top_data, mobilenet0_conv23_fwd_top_data);
			mobilenet0_relu23_fwd->Forward_gpu_native(mobilenet0_conv23_fwd_top_data);
			mobilenet0_conv24_fwd->Forward(cudnn_handle_, mobilenet0_conv23_fwd_top_data, mobilenet0_conv24_fwd_top_data);
			mobilenet0_relu24_fwd->Forward_gpu_native(mobilenet0_conv24_fwd_top_data);
			mobilenet0_conv25_fwd->Forward(cudnn_handle_, mobilenet0_conv24_fwd_top_data, mobilenet0_conv25_fwd_top_data);
			mobilenet0_relu25_fwd->Forward_gpu_native(mobilenet0_conv25_fwd_top_data);
			mobilenet0_conv26_fwd->Forward(cudnn_handle_, mobilenet0_conv25_fwd_top_data, mobilenet0_conv26_fwd_top_data);
			mobilenet0_relu26_fwd->Forward_gpu_native(mobilenet0_conv26_fwd_top_data);
			rf_c3_lateral->Forward(cudnn_handle_, mobilenet0_conv26_fwd_top_data, rf_c3_lateral_top_data);
			rf_c3_lateral_relu->Forward_gpu_native(rf_c3_lateral_top_data);
			rf_c3_det_conv1->Forward(cudnn_handle_, rf_c3_lateral_top_data, rf_c3_det_conv1_top_data);
			rf_c3_det_context_conv1->Forward(cudnn_handle_, rf_c3_lateral_top_data, rf_c3_det_context_conv1_top_data);
			rf_c3_det_context_conv1_relu->Forward_gpu_native(rf_c3_det_context_conv1_top_data);
			rf_c3_det_context_conv2->Forward(cudnn_handle_, rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv2_top_data);
			rf_c3_det_context_conv3_1->Forward(cudnn_handle_, rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv3_1_top_data);
			rf_c3_det_context_conv3_1_relu->Forward_gpu_native(rf_c3_det_context_conv3_1_top_data);
			rf_c3_det_context_conv3_2->Forward(cudnn_handle_, rf_c3_det_context_conv3_1_top_data, rf_c3_det_context_conv3_2_top_data);
			rf_c3_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c3_det_conv1_top_data, rf_c3_det_context_conv2_top_data, rf_c3_det_context_conv3_2_top_data}, rf_c3_det_concat_top_data);
			rf_c3_det_concat_relu->Forward_gpu_native(rf_c3_det_concat_top_data);
			face_rpn_cls_score_stride32->Forward(cudnn_handle_, rf_c3_det_concat_top_data, face_rpn_cls_score_stride32_top_data);
			face_rpn_cls_score_reshape_stride32->Forward(face_rpn_cls_score_stride32_top_data, face_rpn_cls_score_reshape_stride32_top_data);
			face_rpn_cls_prob_stride32->Forward_gpu_cudnn(face_rpn_cls_score_reshape_stride32_top_data, face_rpn_cls_prob_stride32_top_data);
			face_rpn_cls_prob_reshape_stride32->Forward(face_rpn_cls_prob_stride32_top_data, face_rpn_cls_prob_reshape_stride32_top_data);
			face_rpn_bbox_pred_stride32->Forward(cudnn_handle_, rf_c3_det_concat_top_data, face_rpn_bbox_pred_stride32_top_data);
			face_rpn_landmark_pred_stride32->Forward(cudnn_handle_, rf_c3_det_concat_top_data, face_rpn_landmark_pred_stride32_top_data);
			rf_c2_lateral->Forward(cudnn_handle_, mobilenet0_conv22_fwd_top_data, rf_c2_lateral_top_data);
			rf_c2_lateral_relu->Forward_gpu_native(rf_c2_lateral_top_data);
			rf_c3_upsampling->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_upsampling_top_data);
			_plus0->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{rf_c3_upsampling_top_data, rf_c2_lateral_top_data}, _plus0_top_data);
			rf_c2_aggr->Forward(cudnn_handle_, _plus0_top_data, rf_c2_aggr_top_data);
			rf_c2_aggr_relu->Forward_gpu_native(rf_c2_aggr_top_data);
			rf_c2_det_conv1->Forward(cudnn_handle_, rf_c2_aggr_top_data, rf_c2_det_conv1_top_data);
			rf_c2_det_context_conv1->Forward(cudnn_handle_, rf_c2_aggr_top_data, rf_c2_det_context_conv1_top_data);
			rf_c2_det_context_conv1_relu->Forward_gpu_native(rf_c2_det_context_conv1_top_data);
			rf_c2_det_context_conv2->Forward(cudnn_handle_, rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv2_top_data);
			rf_c2_det_context_conv3_1->Forward(cudnn_handle_, rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv3_1_top_data);
			rf_c2_det_context_conv3_1_relu->Forward_gpu_native(rf_c2_det_context_conv3_1_top_data);
			rf_c2_det_context_conv3_2->Forward(cudnn_handle_, rf_c2_det_context_conv3_1_top_data, rf_c2_det_context_conv3_2_top_data);
			rf_c2_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c2_det_conv1_top_data, rf_c2_det_context_conv2_top_data, rf_c2_det_context_conv3_2_top_data}, rf_c2_det_concat_top_data);
			rf_c2_det_concat_relu->Forward_gpu_native(rf_c2_det_concat_top_data);
			face_rpn_cls_score_stride16->Forward(cudnn_handle_, rf_c2_det_concat_top_data, face_rpn_cls_score_stride16_top_data);
			face_rpn_cls_score_reshape_stride16->Forward(face_rpn_cls_score_stride16_top_data, face_rpn_cls_score_reshape_stride16_top_data);
			face_rpn_cls_prob_stride16->Forward_gpu_cudnn(face_rpn_cls_score_reshape_stride16_top_data, face_rpn_cls_prob_stride16_top_data);
			face_rpn_cls_prob_reshape_stride16->Forward(face_rpn_cls_prob_stride16_top_data, face_rpn_cls_prob_reshape_stride16_top_data);
			face_rpn_bbox_pred_stride16->Forward(cudnn_handle_, rf_c2_det_concat_top_data, face_rpn_bbox_pred_stride16_top_data);
			face_rpn_landmark_pred_stride16->Forward(cudnn_handle_, rf_c2_det_concat_top_data, face_rpn_landmark_pred_stride16_top_data);
			rf_c1_red_conv->Forward(cudnn_handle_, mobilenet0_conv10_fwd_top_data, rf_c1_red_conv_top_data);
			rf_c1_red_conv_relu->Forward_gpu_native(rf_c1_red_conv_top_data);
			rf_c2_upsampling->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_upsampling_top_data);
			_plus1->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{rf_c1_red_conv_top_data, rf_c2_upsampling_top_data}, _plus1_top_data);
			rf_c1_aggr->Forward(cudnn_handle_, _plus1_top_data, rf_c1_aggr_top_data);
			rf_c1_aggr_relu->Forward_gpu_native(rf_c1_aggr_top_data);
			rf_c1_det_conv1->Forward(cudnn_handle_, rf_c1_aggr_top_data, rf_c1_det_conv1_top_data);
			rf_c1_det_context_conv1->Forward(cudnn_handle_, rf_c1_aggr_top_data, rf_c1_det_context_conv1_top_data);
			rf_c1_det_context_conv1_relu->Forward_gpu_native(rf_c1_det_context_conv1_top_data);
			rf_c1_det_context_conv2->Forward(cudnn_handle_, rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv2_top_data);
			rf_c1_det_context_conv3_1->Forward(cudnn_handle_, rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv3_1_top_data);
			rf_c1_det_context_conv3_1_relu->Forward_gpu_native(rf_c1_det_context_conv3_1_top_data);
			rf_c1_det_context_conv3_2->Forward(cudnn_handle_, rf_c1_det_context_conv3_1_top_data, rf_c1_det_context_conv3_2_top_data);
			rf_c1_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c1_det_conv1_top_data, rf_c1_det_context_conv2_top_data, rf_c1_det_context_conv3_2_top_data}, rf_c1_det_concat_top_data);
			rf_c1_det_concat_relu->Forward_gpu_native(rf_c1_det_concat_top_data);
			face_rpn_cls_score_stride8->Forward(cudnn_handle_, rf_c1_det_concat_top_data, face_rpn_cls_score_stride8_top_data);
			face_rpn_cls_score_reshape_stride8->Forward(face_rpn_cls_score_stride8_top_data, face_rpn_cls_score_reshape_stride8_top_data);
			face_rpn_cls_prob_stride8->Forward_gpu_cudnn(face_rpn_cls_score_reshape_stride8_top_data, face_rpn_cls_prob_stride8_top_data);
			face_rpn_cls_prob_reshape_stride8->Forward(face_rpn_cls_prob_stride8_top_data, face_rpn_cls_prob_reshape_stride8_top_data);
			face_rpn_bbox_pred_stride8->Forward(cudnn_handle_, rf_c1_det_concat_top_data, face_rpn_bbox_pred_stride8_top_data);
			face_rpn_landmark_pred_stride8->Forward(cudnn_handle_, rf_c1_det_concat_top_data, face_rpn_landmark_pred_stride8_top_data);
		}
#else

//#define CALC_LAYERS
#ifdef CALC_LAYERS
void Retina_net::Forward_gpu_native(const std::shared_ptr<tensor<float>> &input_data)
{
	Profiler *profiler_ = Profiler::Get();
	profiler_->TurnON();

	profiler_->ScopeStart("conv0");
	mobilenet0_conv0_fwd->Forward(cublas_handle_, input_data, mobilenet0_conv0_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu0_fwd->Forward_gpu_native(mobilenet0_conv0_fwd_top_data);

	profiler_->ScopeStart("conv1");
	mobilenet0_conv1_fwd->Forward(cublas_handle_, mobilenet0_conv0_fwd_top_data, mobilenet0_conv1_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu1_fwd->Forward_gpu_native(mobilenet0_conv1_fwd_top_data);
	
	profiler_->ScopeStart("conv2");
	mobilenet0_conv2_fwd->Forward(cublas_handle_, mobilenet0_conv1_fwd_top_data, mobilenet0_conv2_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu2_fwd->Forward_gpu_native(mobilenet0_conv2_fwd_top_data);

	profiler_->ScopeStart("conv3");
	mobilenet0_conv3_fwd->Forward(cublas_handle_, mobilenet0_conv2_fwd_top_data, mobilenet0_conv3_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu3_fwd->Forward_gpu_native(mobilenet0_conv3_fwd_top_data);

	profiler_->ScopeStart("conv4");
	mobilenet0_conv4_fwd->Forward(cublas_handle_, mobilenet0_conv3_fwd_top_data, mobilenet0_conv4_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu4_fwd->Forward_gpu_native(mobilenet0_conv4_fwd_top_data);

	profiler_->ScopeStart("conv5");
	mobilenet0_conv5_fwd->Forward(cublas_handle_, mobilenet0_conv4_fwd_top_data, mobilenet0_conv5_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu5_fwd->Forward_gpu_native(mobilenet0_conv5_fwd_top_data);

	profiler_->ScopeStart("conv6");
	mobilenet0_conv6_fwd->Forward(cublas_handle_, mobilenet0_conv5_fwd_top_data, mobilenet0_conv6_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu6_fwd->Forward_gpu_native(mobilenet0_conv6_fwd_top_data);

	profiler_->ScopeStart("conv7");
	mobilenet0_conv7_fwd->Forward(cublas_handle_, mobilenet0_conv6_fwd_top_data, mobilenet0_conv7_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu7_fwd->Forward_gpu_native(mobilenet0_conv7_fwd_top_data);

	profiler_->ScopeStart("conv8");
	mobilenet0_conv8_fwd->Forward(cublas_handle_, mobilenet0_conv7_fwd_top_data, mobilenet0_conv8_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu8_fwd->Forward_gpu_native(mobilenet0_conv8_fwd_top_data);

	profiler_->ScopeStart("conv9");
	mobilenet0_conv9_fwd->Forward(cublas_handle_, mobilenet0_conv8_fwd_top_data, mobilenet0_conv9_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu9_fwd->Forward_gpu_native(mobilenet0_conv9_fwd_top_data);

	profiler_->ScopeStart("conv10");
	mobilenet0_conv10_fwd->Forward(cublas_handle_, mobilenet0_conv9_fwd_top_data, mobilenet0_conv10_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu10_fwd->Forward_gpu_native(mobilenet0_conv10_fwd_top_data);

	profiler_->ScopeStart("conv11");
	mobilenet0_conv11_fwd->Forward(cublas_handle_, mobilenet0_conv10_fwd_top_data, mobilenet0_conv11_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu11_fwd->Forward_gpu_native(mobilenet0_conv11_fwd_top_data);

	profiler_->ScopeStart("conv12");
	mobilenet0_conv12_fwd->Forward(cublas_handle_, mobilenet0_conv11_fwd_top_data, mobilenet0_conv12_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu12_fwd->Forward_gpu_native(mobilenet0_conv12_fwd_top_data);

	profiler_->ScopeStart("conv13");
	mobilenet0_conv13_fwd->Forward(cublas_handle_, mobilenet0_conv12_fwd_top_data, mobilenet0_conv13_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu13_fwd->Forward_gpu_native(mobilenet0_conv13_fwd_top_data);
	
	profiler_->ScopeStart("conv14");
	mobilenet0_conv14_fwd->Forward(cublas_handle_, mobilenet0_conv13_fwd_top_data, mobilenet0_conv14_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu14_fwd->Forward_gpu_native(mobilenet0_conv14_fwd_top_data);
	
	profiler_->ScopeStart("conv15");
	mobilenet0_conv15_fwd->Forward(cublas_handle_, mobilenet0_conv14_fwd_top_data, mobilenet0_conv15_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu15_fwd->Forward_gpu_native(mobilenet0_conv15_fwd_top_data);
	
	profiler_->ScopeStart("conv16");
	mobilenet0_conv16_fwd->Forward(cublas_handle_, mobilenet0_conv15_fwd_top_data, mobilenet0_conv16_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu16_fwd->Forward_gpu_native(mobilenet0_conv16_fwd_top_data);
	
	profiler_->ScopeStart("conv17");
	mobilenet0_conv17_fwd->Forward(cublas_handle_, mobilenet0_conv16_fwd_top_data, mobilenet0_conv17_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu17_fwd->Forward_gpu_native(mobilenet0_conv17_fwd_top_data);
	
	profiler_->ScopeStart("conv18");
	mobilenet0_conv18_fwd->Forward(cublas_handle_, mobilenet0_conv17_fwd_top_data, mobilenet0_conv18_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu18_fwd->Forward_gpu_native(mobilenet0_conv18_fwd_top_data);
	
	profiler_->ScopeStart("conv19");
	mobilenet0_conv19_fwd->Forward(cublas_handle_, mobilenet0_conv18_fwd_top_data, mobilenet0_conv19_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu19_fwd->Forward_gpu_native(mobilenet0_conv19_fwd_top_data);
	
	profiler_->ScopeStart("conv20");
	mobilenet0_conv20_fwd->Forward(cublas_handle_, mobilenet0_conv19_fwd_top_data, mobilenet0_conv20_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu20_fwd->Forward_gpu_native(mobilenet0_conv20_fwd_top_data);
	
	profiler_->ScopeStart("conv21");
	mobilenet0_conv21_fwd->Forward(cublas_handle_, mobilenet0_conv20_fwd_top_data, mobilenet0_conv21_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu21_fwd->Forward_gpu_native(mobilenet0_conv21_fwd_top_data);
	
	profiler_->ScopeStart("conv22");
	mobilenet0_conv22_fwd->Forward(cublas_handle_, mobilenet0_conv21_fwd_top_data, mobilenet0_conv22_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu22_fwd->Forward_gpu_native(mobilenet0_conv22_fwd_top_data);
	
	profiler_->ScopeStart("conv23");
	mobilenet0_conv23_fwd->Forward(cublas_handle_, mobilenet0_conv22_fwd_top_data, mobilenet0_conv23_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu23_fwd->Forward_gpu_native(mobilenet0_conv23_fwd_top_data);
	
	profiler_->ScopeStart("conv24");
	mobilenet0_conv24_fwd->Forward(cublas_handle_, mobilenet0_conv23_fwd_top_data, mobilenet0_conv24_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu24_fwd->Forward_gpu_native(mobilenet0_conv24_fwd_top_data);
	
	profiler_->ScopeStart("conv25");
	mobilenet0_conv25_fwd->Forward(cublas_handle_, mobilenet0_conv24_fwd_top_data, mobilenet0_conv25_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu25_fwd->Forward_gpu_native(mobilenet0_conv25_fwd_top_data);
	
	profiler_->ScopeStart("conv26");
	mobilenet0_conv26_fwd->Forward(cublas_handle_, mobilenet0_conv25_fwd_top_data, mobilenet0_conv26_fwd_top_data);
	profiler_->ScopeEnd();
	mobilenet0_relu26_fwd->Forward_gpu_native(mobilenet0_conv26_fwd_top_data);
	
	profiler_->ScopeStart("rf_c3_lateral");
	rf_c3_lateral->Forward(cublas_handle_, mobilenet0_conv26_fwd_top_data, rf_c3_lateral_top_data);
	profiler_->ScopeEnd();
	rf_c3_lateral_relu->Forward_gpu_native(rf_c3_lateral_top_data);
	
	profiler_->ScopeStart("rf_c3_det_conv1");
	rf_c3_det_conv1->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_det_conv1_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c3_det_context_conv1");
	rf_c3_det_context_conv1->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_det_context_conv1_top_data);
	profiler_->ScopeEnd();
	rf_c3_det_context_conv1_relu->Forward_gpu_native(rf_c3_det_context_conv1_top_data);
	
	profiler_->ScopeStart("rf_c3_det_context_conv2");
	rf_c3_det_context_conv2->Forward(cublas_handle_, rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv2_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c3_det_context_conv3_1");
	rf_c3_det_context_conv3_1->Forward(cublas_handle_, rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv3_1_top_data);
	profiler_->ScopeEnd();
	rf_c3_det_context_conv3_1_relu->Forward_gpu_native(rf_c3_det_context_conv3_1_top_data);
	
	profiler_->ScopeStart("rf_c3_det_context_conv3_2");
	rf_c3_det_context_conv3_2->Forward(cublas_handle_, rf_c3_det_context_conv3_1_top_data, rf_c3_det_context_conv3_2_top_data);
	profiler_->ScopeEnd();

	rf_c3_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c3_det_conv1_top_data, rf_c3_det_context_conv2_top_data, rf_c3_det_context_conv3_2_top_data}, rf_c3_det_concat_top_data);
	rf_c3_det_concat_relu->Forward_gpu_native(rf_c3_det_concat_top_data);
	
	profiler_->ScopeStart("face_rpn_cls_score_stride32");
	face_rpn_cls_score_stride32->Forward(cublas_handle_, rf_c3_det_concat_top_data, face_rpn_cls_score_stride32_top_data);
	profiler_->ScopeEnd();

	face_rpn_cls_score_reshape_stride32->Forward(face_rpn_cls_score_stride32_top_data, face_rpn_cls_score_reshape_stride32_top_data);
	face_rpn_cls_prob_stride32->Forward_gpu_native(face_rpn_cls_score_reshape_stride32_top_data, face_rpn_cls_prob_stride32_top_data);
	face_rpn_cls_prob_reshape_stride32->Forward(face_rpn_cls_prob_stride32_top_data, face_rpn_cls_prob_reshape_stride32_top_data);
	
	profiler_->ScopeStart("face_rpn_bbox_pred_stride32");
	face_rpn_bbox_pred_stride32->Forward(cublas_handle_, rf_c3_det_concat_top_data, face_rpn_bbox_pred_stride32_top_data);
	profiler_->ScopeEnd();
	
	profiler_->ScopeStart("face_rpn_landmark_pred_stride32");
	face_rpn_landmark_pred_stride32->Forward(cublas_handle_, rf_c3_det_concat_top_data, face_rpn_landmark_pred_stride32_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c2_lateral");
	rf_c2_lateral->Forward(cublas_handle_, mobilenet0_conv22_fwd_top_data, rf_c2_lateral_top_data);
	profiler_->ScopeEnd();
	rf_c2_lateral_relu->Forward_gpu_native(rf_c2_lateral_top_data);
	
	profiler_->ScopeStart("rf_c3_upsampling");
	rf_c3_upsampling->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_upsampling_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("_plus0");
	_plus0->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{rf_c3_upsampling_top_data, rf_c2_lateral_top_data}, _plus0_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c2_aggr");
	rf_c2_aggr->Forward(cublas_handle_, _plus0_top_data, rf_c2_aggr_top_data);
	profiler_->ScopeEnd();
	rf_c2_aggr_relu->Forward_gpu_native(rf_c2_aggr_top_data);
	
	profiler_->ScopeStart("rf_c2_det_conv1");
	rf_c2_det_conv1->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_det_conv1_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c2_det_context_conv1");
	rf_c2_det_context_conv1->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_det_context_conv1_top_data);
	profiler_->ScopeEnd();
	rf_c2_det_context_conv1_relu->Forward_gpu_native(rf_c2_det_context_conv1_top_data);
	
	profiler_->ScopeStart("rf_c2_det_context_conv2");
	rf_c2_det_context_conv2->Forward(cublas_handle_, rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv2_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c2_det_context_conv3_1");
	rf_c2_det_context_conv3_1->Forward(cublas_handle_, rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv3_1_top_data);
	profiler_->ScopeEnd();
	rf_c2_det_context_conv3_1_relu->Forward_gpu_native(rf_c2_det_context_conv3_1_top_data);
	
	profiler_->ScopeStart("rf_c2_det_context_conv3_2");
	rf_c2_det_context_conv3_2->Forward(cublas_handle_, rf_c2_det_context_conv3_1_top_data, rf_c2_det_context_conv3_2_top_data);
	profiler_->ScopeEnd();
	rf_c2_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c2_det_conv1_top_data, rf_c2_det_context_conv2_top_data, rf_c2_det_context_conv3_2_top_data}, rf_c2_det_concat_top_data);
	rf_c2_det_concat_relu->Forward_gpu_native(rf_c2_det_concat_top_data);
	
	profiler_->ScopeStart("face_rpn_cls_score_stride16");
	face_rpn_cls_score_stride16->Forward(cublas_handle_, rf_c2_det_concat_top_data, face_rpn_cls_score_stride16_top_data);
	profiler_->ScopeEnd();

	face_rpn_cls_score_reshape_stride16->Forward(face_rpn_cls_score_stride16_top_data, face_rpn_cls_score_reshape_stride16_top_data);
	face_rpn_cls_prob_stride16->Forward_gpu_native(face_rpn_cls_score_reshape_stride16_top_data, face_rpn_cls_prob_stride16_top_data);
	face_rpn_cls_prob_reshape_stride16->Forward(face_rpn_cls_prob_stride16_top_data, face_rpn_cls_prob_reshape_stride16_top_data);
	
	profiler_->ScopeStart("face_rpn_bbox_pred_stride16");
	face_rpn_bbox_pred_stride16->Forward(cublas_handle_, rf_c2_det_concat_top_data, face_rpn_bbox_pred_stride16_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("face_rpn_landmark_pred_stride16");
	face_rpn_landmark_pred_stride16->Forward(cublas_handle_, rf_c2_det_concat_top_data, face_rpn_landmark_pred_stride16_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c1_red_conv");
	rf_c1_red_conv->Forward(cublas_handle_, mobilenet0_conv10_fwd_top_data, rf_c1_red_conv_top_data);
	profiler_->ScopeEnd();
	rf_c1_red_conv_relu->Forward_gpu_native(rf_c1_red_conv_top_data);
	
	profiler_->ScopeStart("rf_c2_upsampling");
	rf_c2_upsampling->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_upsampling_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("_plus1");
	_plus1->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{rf_c1_red_conv_top_data, rf_c2_upsampling_top_data}, _plus1_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c1_aggr");
	rf_c1_aggr->Forward(cublas_handle_, _plus1_top_data, rf_c1_aggr_top_data);
	profiler_->ScopeEnd();
	rf_c1_aggr_relu->Forward_gpu_native(rf_c1_aggr_top_data);
	
	profiler_->ScopeStart("rf_c1_det_conv1");
	rf_c1_det_conv1->Forward(cublas_handle_, rf_c1_aggr_top_data, rf_c1_det_conv1_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c1_det_context_conv1");
	rf_c1_det_context_conv1->Forward(cublas_handle_, rf_c1_aggr_top_data, rf_c1_det_context_conv1_top_data);
	profiler_->ScopeEnd();
	rf_c1_det_context_conv1_relu->Forward_gpu_native(rf_c1_det_context_conv1_top_data);
	
	profiler_->ScopeStart("rf_c1_det_context_conv2");
	rf_c1_det_context_conv2->Forward(cublas_handle_, rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv2_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("rf_c1_det_context_conv3_1");
	rf_c1_det_context_conv3_1->Forward(cublas_handle_, rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv3_1_top_data);
	profiler_->ScopeEnd();
	rf_c1_det_context_conv3_1_relu->Forward_gpu_native(rf_c1_det_context_conv3_1_top_data);
	
	profiler_->ScopeStart("rf_c1_det_context_conv3_2");
	rf_c1_det_context_conv3_2->Forward(cublas_handle_, rf_c1_det_context_conv3_1_top_data, rf_c1_det_context_conv3_2_top_data);
	profiler_->ScopeEnd();

	rf_c1_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c1_det_conv1_top_data, rf_c1_det_context_conv2_top_data, rf_c1_det_context_conv3_2_top_data}, rf_c1_det_concat_top_data);
	rf_c1_det_concat_relu->Forward_gpu_native(rf_c1_det_concat_top_data);
	
	profiler_->ScopeStart("face_rpn_cls_score_stride8");
	face_rpn_cls_score_stride8->Forward(cublas_handle_, rf_c1_det_concat_top_data, face_rpn_cls_score_stride8_top_data);
	profiler_->ScopeEnd();

	face_rpn_cls_score_reshape_stride8->Forward(face_rpn_cls_score_stride8_top_data, face_rpn_cls_score_reshape_stride8_top_data);
	face_rpn_cls_prob_stride8->Forward_gpu_native(face_rpn_cls_score_reshape_stride8_top_data, face_rpn_cls_prob_stride8_top_data);
	face_rpn_cls_prob_reshape_stride8->Forward(face_rpn_cls_prob_stride8_top_data, face_rpn_cls_prob_reshape_stride8_top_data);
	
	profiler_->ScopeStart("face_rpn_bbox_pred_stride8");
	face_rpn_bbox_pred_stride8->Forward(cublas_handle_, rf_c1_det_concat_top_data, face_rpn_bbox_pred_stride8_top_data);
	profiler_->ScopeEnd();

	profiler_->ScopeStart("face_rpn_landmark_pred_stride8");
	face_rpn_landmark_pred_stride8->Forward(cublas_handle_, rf_c1_det_concat_top_data, face_rpn_landmark_pred_stride8_top_data);
	profiler_->ScopeEnd();

	profiler_->TurnOFF();
	profiler_->DumpProfile("D:/retina_face.json");
}
#else

    void Retina_net::Forward_gpu_native(const std::shared_ptr<tensor<float>> &input_data)
    {
    	mobilenet0_conv0_fwd->Forward(cublas_handle_, input_data, mobilenet0_conv0_fwd_top_data);
    	mobilenet0_relu0_fwd->Forward_gpu_native(mobilenet0_conv0_fwd_top_data);
    	mobilenet0_conv1_fwd->Forward(cublas_handle_, mobilenet0_conv0_fwd_top_data, mobilenet0_conv1_fwd_top_data);
    	mobilenet0_relu1_fwd->Forward_gpu_native(mobilenet0_conv1_fwd_top_data);
    	mobilenet0_conv2_fwd->Forward(cublas_handle_, mobilenet0_conv1_fwd_top_data, mobilenet0_conv2_fwd_top_data);
    	mobilenet0_relu2_fwd->Forward_gpu_native(mobilenet0_conv2_fwd_top_data);
    	mobilenet0_conv3_fwd->Forward(cublas_handle_, mobilenet0_conv2_fwd_top_data, mobilenet0_conv3_fwd_top_data);
    	mobilenet0_relu3_fwd->Forward_gpu_native(mobilenet0_conv3_fwd_top_data);
    	mobilenet0_conv4_fwd->Forward(cublas_handle_, mobilenet0_conv3_fwd_top_data, mobilenet0_conv4_fwd_top_data);
    	mobilenet0_relu4_fwd->Forward_gpu_native(mobilenet0_conv4_fwd_top_data);
    	mobilenet0_conv5_fwd->Forward(cublas_handle_, mobilenet0_conv4_fwd_top_data, mobilenet0_conv5_fwd_top_data);
    	mobilenet0_relu5_fwd->Forward_gpu_native(mobilenet0_conv5_fwd_top_data);
    	mobilenet0_conv6_fwd->Forward(cublas_handle_, mobilenet0_conv5_fwd_top_data, mobilenet0_conv6_fwd_top_data);
    	mobilenet0_relu6_fwd->Forward_gpu_native(mobilenet0_conv6_fwd_top_data);
    	mobilenet0_conv7_fwd->Forward(cublas_handle_, mobilenet0_conv6_fwd_top_data, mobilenet0_conv7_fwd_top_data);
    	mobilenet0_relu7_fwd->Forward_gpu_native(mobilenet0_conv7_fwd_top_data);
    	mobilenet0_conv8_fwd->Forward(cublas_handle_, mobilenet0_conv7_fwd_top_data, mobilenet0_conv8_fwd_top_data);
    	mobilenet0_relu8_fwd->Forward_gpu_native(mobilenet0_conv8_fwd_top_data);
    	mobilenet0_conv9_fwd->Forward(cublas_handle_, mobilenet0_conv8_fwd_top_data, mobilenet0_conv9_fwd_top_data);
    	mobilenet0_relu9_fwd->Forward_gpu_native(mobilenet0_conv9_fwd_top_data);
    	mobilenet0_conv10_fwd->Forward(cublas_handle_, mobilenet0_conv9_fwd_top_data, mobilenet0_conv10_fwd_top_data);
    	mobilenet0_relu10_fwd->Forward_gpu_native(mobilenet0_conv10_fwd_top_data);
    	mobilenet0_conv11_fwd->Forward(cublas_handle_, mobilenet0_conv10_fwd_top_data, mobilenet0_conv11_fwd_top_data);
    	mobilenet0_relu11_fwd->Forward_gpu_native(mobilenet0_conv11_fwd_top_data);
    	mobilenet0_conv12_fwd->Forward(cublas_handle_, mobilenet0_conv11_fwd_top_data, mobilenet0_conv12_fwd_top_data);
    	mobilenet0_relu12_fwd->Forward_gpu_native(mobilenet0_conv12_fwd_top_data);
    	mobilenet0_conv13_fwd->Forward(cublas_handle_, mobilenet0_conv12_fwd_top_data, mobilenet0_conv13_fwd_top_data);
    	mobilenet0_relu13_fwd->Forward_gpu_native(mobilenet0_conv13_fwd_top_data);
    	mobilenet0_conv14_fwd->Forward(cublas_handle_, mobilenet0_conv13_fwd_top_data, mobilenet0_conv14_fwd_top_data);
    	mobilenet0_relu14_fwd->Forward_gpu_native(mobilenet0_conv14_fwd_top_data);
    	mobilenet0_conv15_fwd->Forward(cublas_handle_, mobilenet0_conv14_fwd_top_data, mobilenet0_conv15_fwd_top_data);
    	mobilenet0_relu15_fwd->Forward_gpu_native(mobilenet0_conv15_fwd_top_data);
    	mobilenet0_conv16_fwd->Forward(cublas_handle_, mobilenet0_conv15_fwd_top_data, mobilenet0_conv16_fwd_top_data);
    	mobilenet0_relu16_fwd->Forward_gpu_native(mobilenet0_conv16_fwd_top_data);
    	mobilenet0_conv17_fwd->Forward(cublas_handle_, mobilenet0_conv16_fwd_top_data, mobilenet0_conv17_fwd_top_data);
    	mobilenet0_relu17_fwd->Forward_gpu_native(mobilenet0_conv17_fwd_top_data);
    	mobilenet0_conv18_fwd->Forward(cublas_handle_, mobilenet0_conv17_fwd_top_data, mobilenet0_conv18_fwd_top_data);
    	mobilenet0_relu18_fwd->Forward_gpu_native(mobilenet0_conv18_fwd_top_data);
    	mobilenet0_conv19_fwd->Forward(cublas_handle_, mobilenet0_conv18_fwd_top_data, mobilenet0_conv19_fwd_top_data);
    	mobilenet0_relu19_fwd->Forward_gpu_native(mobilenet0_conv19_fwd_top_data);
    	mobilenet0_conv20_fwd->Forward(cublas_handle_, mobilenet0_conv19_fwd_top_data, mobilenet0_conv20_fwd_top_data);
    	mobilenet0_relu20_fwd->Forward_gpu_native(mobilenet0_conv20_fwd_top_data);
    	mobilenet0_conv21_fwd->Forward(cublas_handle_, mobilenet0_conv20_fwd_top_data, mobilenet0_conv21_fwd_top_data);
    	mobilenet0_relu21_fwd->Forward_gpu_native(mobilenet0_conv21_fwd_top_data);
    	mobilenet0_conv22_fwd->Forward(cublas_handle_, mobilenet0_conv21_fwd_top_data, mobilenet0_conv22_fwd_top_data);
    	mobilenet0_relu22_fwd->Forward_gpu_native(mobilenet0_conv22_fwd_top_data);
    	mobilenet0_conv23_fwd->Forward(cublas_handle_, mobilenet0_conv22_fwd_top_data, mobilenet0_conv23_fwd_top_data);
    	mobilenet0_relu23_fwd->Forward_gpu_native(mobilenet0_conv23_fwd_top_data);
    	mobilenet0_conv24_fwd->Forward(cublas_handle_, mobilenet0_conv23_fwd_top_data, mobilenet0_conv24_fwd_top_data);
    	mobilenet0_relu24_fwd->Forward_gpu_native(mobilenet0_conv24_fwd_top_data);
    	mobilenet0_conv25_fwd->Forward(cublas_handle_, mobilenet0_conv24_fwd_top_data, mobilenet0_conv25_fwd_top_data);
    	mobilenet0_relu25_fwd->Forward_gpu_native(mobilenet0_conv25_fwd_top_data);
    	mobilenet0_conv26_fwd->Forward(cublas_handle_, mobilenet0_conv25_fwd_top_data, mobilenet0_conv26_fwd_top_data);
    	mobilenet0_relu26_fwd->Forward_gpu_native(mobilenet0_conv26_fwd_top_data);
    	rf_c3_lateral->Forward(cublas_handle_, mobilenet0_conv26_fwd_top_data, rf_c3_lateral_top_data);
    	rf_c3_lateral_relu->Forward_gpu_native(rf_c3_lateral_top_data);
    	rf_c3_det_conv1->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_det_conv1_top_data);
    	rf_c3_det_context_conv1->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_det_context_conv1_top_data);
    	rf_c3_det_context_conv1_relu->Forward_gpu_native(rf_c3_det_context_conv1_top_data);
    	rf_c3_det_context_conv2->Forward(cublas_handle_, rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv2_top_data);
    	rf_c3_det_context_conv3_1->Forward(cublas_handle_, rf_c3_det_context_conv1_top_data, rf_c3_det_context_conv3_1_top_data);
    	rf_c3_det_context_conv3_1_relu->Forward_gpu_native(rf_c3_det_context_conv3_1_top_data);
    	rf_c3_det_context_conv3_2->Forward(cublas_handle_, rf_c3_det_context_conv3_1_top_data, rf_c3_det_context_conv3_2_top_data);
    	rf_c3_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c3_det_conv1_top_data, rf_c3_det_context_conv2_top_data, rf_c3_det_context_conv3_2_top_data}, rf_c3_det_concat_top_data);
    	rf_c3_det_concat_relu->Forward_gpu_native(rf_c3_det_concat_top_data);
    	face_rpn_cls_score_stride32->Forward(cublas_handle_, rf_c3_det_concat_top_data, face_rpn_cls_score_stride32_top_data);
    	face_rpn_cls_score_reshape_stride32->Forward(face_rpn_cls_score_stride32_top_data, face_rpn_cls_score_reshape_stride32_top_data);
    	face_rpn_cls_prob_stride32->Forward_gpu_native(face_rpn_cls_score_reshape_stride32_top_data, face_rpn_cls_prob_stride32_top_data);
    	face_rpn_cls_prob_reshape_stride32->Forward(face_rpn_cls_prob_stride32_top_data, face_rpn_cls_prob_reshape_stride32_top_data);
    	face_rpn_bbox_pred_stride32->Forward(cublas_handle_, rf_c3_det_concat_top_data, face_rpn_bbox_pred_stride32_top_data);
    	face_rpn_landmark_pred_stride32->Forward(cublas_handle_, rf_c3_det_concat_top_data, face_rpn_landmark_pred_stride32_top_data);
    	rf_c2_lateral->Forward(cublas_handle_, mobilenet0_conv22_fwd_top_data, rf_c2_lateral_top_data);
    	rf_c2_lateral_relu->Forward_gpu_native(rf_c2_lateral_top_data);
    	rf_c3_upsampling->Forward(cublas_handle_, rf_c3_lateral_top_data, rf_c3_upsampling_top_data);
    	_plus0->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{rf_c3_upsampling_top_data, rf_c2_lateral_top_data}, _plus0_top_data);
    	rf_c2_aggr->Forward(cublas_handle_, _plus0_top_data, rf_c2_aggr_top_data);
    	rf_c2_aggr_relu->Forward_gpu_native(rf_c2_aggr_top_data);
    	rf_c2_det_conv1->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_det_conv1_top_data);
    	rf_c2_det_context_conv1->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_det_context_conv1_top_data);
    	rf_c2_det_context_conv1_relu->Forward_gpu_native(rf_c2_det_context_conv1_top_data);
    	rf_c2_det_context_conv2->Forward(cublas_handle_, rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv2_top_data);
    	rf_c2_det_context_conv3_1->Forward(cublas_handle_, rf_c2_det_context_conv1_top_data, rf_c2_det_context_conv3_1_top_data);
    	rf_c2_det_context_conv3_1_relu->Forward_gpu_native(rf_c2_det_context_conv3_1_top_data);
    	rf_c2_det_context_conv3_2->Forward(cublas_handle_, rf_c2_det_context_conv3_1_top_data, rf_c2_det_context_conv3_2_top_data);
    	rf_c2_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c2_det_conv1_top_data, rf_c2_det_context_conv2_top_data, rf_c2_det_context_conv3_2_top_data}, rf_c2_det_concat_top_data);
    	rf_c2_det_concat_relu->Forward_gpu_native(rf_c2_det_concat_top_data);
    	face_rpn_cls_score_stride16->Forward(cublas_handle_, rf_c2_det_concat_top_data, face_rpn_cls_score_stride16_top_data);
    	face_rpn_cls_score_reshape_stride16->Forward(face_rpn_cls_score_stride16_top_data, face_rpn_cls_score_reshape_stride16_top_data);
    	face_rpn_cls_prob_stride16->Forward_gpu_native(face_rpn_cls_score_reshape_stride16_top_data, face_rpn_cls_prob_stride16_top_data);
    	face_rpn_cls_prob_reshape_stride16->Forward(face_rpn_cls_prob_stride16_top_data, face_rpn_cls_prob_reshape_stride16_top_data);
    	face_rpn_bbox_pred_stride16->Forward(cublas_handle_, rf_c2_det_concat_top_data, face_rpn_bbox_pred_stride16_top_data);
    	face_rpn_landmark_pred_stride16->Forward(cublas_handle_, rf_c2_det_concat_top_data, face_rpn_landmark_pred_stride16_top_data);
    	rf_c1_red_conv->Forward(cublas_handle_, mobilenet0_conv10_fwd_top_data, rf_c1_red_conv_top_data);
    	rf_c1_red_conv_relu->Forward_gpu_native(rf_c1_red_conv_top_data);
    	rf_c2_upsampling->Forward(cublas_handle_, rf_c2_aggr_top_data, rf_c2_upsampling_top_data);
    	_plus1->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{rf_c1_red_conv_top_data, rf_c2_upsampling_top_data}, _plus1_top_data);
    	rf_c1_aggr->Forward(cublas_handle_, _plus1_top_data, rf_c1_aggr_top_data);
    	rf_c1_aggr_relu->Forward_gpu_native(rf_c1_aggr_top_data);
    	rf_c1_det_conv1->Forward(cublas_handle_, rf_c1_aggr_top_data, rf_c1_det_conv1_top_data);
    	rf_c1_det_context_conv1->Forward(cublas_handle_, rf_c1_aggr_top_data, rf_c1_det_context_conv1_top_data);
    	rf_c1_det_context_conv1_relu->Forward_gpu_native(rf_c1_det_context_conv1_top_data);
    	rf_c1_det_context_conv2->Forward(cublas_handle_, rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv2_top_data);
    	rf_c1_det_context_conv3_1->Forward(cublas_handle_, rf_c1_det_context_conv1_top_data, rf_c1_det_context_conv3_1_top_data);
    	rf_c1_det_context_conv3_1_relu->Forward_gpu_native(rf_c1_det_context_conv3_1_top_data);
    	rf_c1_det_context_conv3_2->Forward(cublas_handle_, rf_c1_det_context_conv3_1_top_data, rf_c1_det_context_conv3_2_top_data);
    	rf_c1_det_concat->Forward_gpu_native(std::vector<std::shared_ptr<tensor<float>>>{rf_c1_det_conv1_top_data, rf_c1_det_context_conv2_top_data, rf_c1_det_context_conv3_2_top_data}, rf_c1_det_concat_top_data);
    	rf_c1_det_concat_relu->Forward_gpu_native(rf_c1_det_concat_top_data);
    	face_rpn_cls_score_stride8->Forward(cublas_handle_, rf_c1_det_concat_top_data, face_rpn_cls_score_stride8_top_data);
    	face_rpn_cls_score_reshape_stride8->Forward(face_rpn_cls_score_stride8_top_data, face_rpn_cls_score_reshape_stride8_top_data);
    	face_rpn_cls_prob_stride8->Forward_gpu_native(face_rpn_cls_score_reshape_stride8_top_data, face_rpn_cls_prob_stride8_top_data);
    	face_rpn_cls_prob_reshape_stride8->Forward(face_rpn_cls_prob_stride8_top_data, face_rpn_cls_prob_reshape_stride8_top_data);
    	face_rpn_bbox_pred_stride8->Forward(cublas_handle_, rf_c1_det_concat_top_data, face_rpn_bbox_pred_stride8_top_data);
    	face_rpn_landmark_pred_stride8->Forward(cublas_handle_, rf_c1_det_concat_top_data, face_rpn_landmark_pred_stride8_top_data);
    }
#endif // CALC_LAYERS
#endif // USE_CUDNN
#endif // USE_CUDA

	}
}