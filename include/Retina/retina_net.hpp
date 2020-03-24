#ifndef _Retina_net_HPP_
#define _Retina_net_HPP_

#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class Retina_net
		{
			Declear_Params(mobilenet0_conv0_fwd);
			Declear_Params(mobilenet0_conv1_fwd);
			Declear_Params(mobilenet0_conv2_fwd);
			Declear_Params(mobilenet0_conv3_fwd);
			Declear_Params(mobilenet0_conv4_fwd);
			Declear_Params(mobilenet0_conv5_fwd);
			Declear_Params(mobilenet0_conv6_fwd);
			Declear_Params(mobilenet0_conv7_fwd);
			Declear_Params(mobilenet0_conv8_fwd);
			Declear_Params(mobilenet0_conv9_fwd);
			Declear_Params(mobilenet0_conv10_fwd);
			Declear_Params(mobilenet0_conv11_fwd);
			Declear_Params(mobilenet0_conv12_fwd);
			Declear_Params(mobilenet0_conv13_fwd);
			Declear_Params(mobilenet0_conv14_fwd);
			Declear_Params(mobilenet0_conv15_fwd);
			Declear_Params(mobilenet0_conv16_fwd);
			Declear_Params(mobilenet0_conv17_fwd);
			Declear_Params(mobilenet0_conv18_fwd);
			Declear_Params(mobilenet0_conv19_fwd);
			Declear_Params(mobilenet0_conv20_fwd);
			Declear_Params(mobilenet0_conv21_fwd);
			Declear_Params(mobilenet0_conv22_fwd);
			Declear_Params(mobilenet0_conv23_fwd);
			Declear_Params(mobilenet0_conv24_fwd);
			Declear_Params(mobilenet0_conv25_fwd);
			Declear_Params(mobilenet0_conv26_fwd);
			Declear_Params(rf_c3_lateral);
			Declear_Params(rf_c3_det_conv1);
			Declear_Params(rf_c3_det_context_conv1);
			Declear_Params(rf_c3_det_context_conv2);
			Declear_Params(rf_c3_det_context_conv3_1);
			Declear_Params(rf_c3_det_context_conv3_2);
			Declear_Params(face_rpn_cls_score_stride32);
			Declear_Params(face_rpn_bbox_pred_stride32);
			Declear_Params(face_rpn_landmark_pred_stride32);
			Declear_Params(rf_c2_lateral);
			Declear_Params(rf_c3_upsampling);
			Declear_Params(rf_c2_aggr);
			Declear_Params(rf_c2_det_conv1);
			Declear_Params(rf_c2_det_context_conv1);
			Declear_Params(rf_c2_det_context_conv2);
			Declear_Params(rf_c2_det_context_conv3_1);
			Declear_Params(rf_c2_det_context_conv3_2);
			Declear_Params(face_rpn_cls_score_stride16);
			Declear_Params(face_rpn_bbox_pred_stride16);
			Declear_Params(face_rpn_landmark_pred_stride16);
			Declear_Params(rf_c1_red_conv);
			Declear_Params(rf_c2_upsampling);
			Declear_Params(rf_c1_aggr);
			Declear_Params(rf_c1_det_conv1);
			Declear_Params(rf_c1_det_context_conv1);
			Declear_Params(rf_c1_det_context_conv2);
			Declear_Params(rf_c1_det_context_conv3_1);
			Declear_Params(rf_c1_det_context_conv3_2);
			Declear_Params(face_rpn_cls_score_stride8);
			Declear_Params(face_rpn_bbox_pred_stride8);
			Declear_Params(face_rpn_landmark_pred_stride8);

			//
			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;

			std::shared_ptr<tensor<float>> tensor_float_data = nullptr;
			std::shared_ptr<tensor<unsigned char>> tensor_unsigned_char_data = nullptr;
			//

			Declear_Opration(baseconv, mobilenet0_conv0_fwd);
			Neuron_Name(mobilenet0_conv0_fwd);
			Declear_Opration(prelu, mobilenet0_relu0_fwd);
			Neuron_Name(mobilenet0_relu0_fwd);
			Declear_Opration(baseconv, mobilenet0_conv1_fwd);
			Neuron_Name(mobilenet0_conv1_fwd);
			Declear_Opration(prelu, mobilenet0_relu1_fwd);
			Neuron_Name(mobilenet0_relu1_fwd);
			Declear_Opration(baseconv, mobilenet0_conv2_fwd);
			Neuron_Name(mobilenet0_conv2_fwd);
			Declear_Opration(prelu, mobilenet0_relu2_fwd);
			Neuron_Name(mobilenet0_relu2_fwd);
			Declear_Opration(baseconv, mobilenet0_conv3_fwd);
			Neuron_Name(mobilenet0_conv3_fwd);
			Declear_Opration(prelu, mobilenet0_relu3_fwd);
			Neuron_Name(mobilenet0_relu3_fwd);
			Declear_Opration(baseconv, mobilenet0_conv4_fwd);
			Neuron_Name(mobilenet0_conv4_fwd);
			Declear_Opration(prelu, mobilenet0_relu4_fwd);
			Neuron_Name(mobilenet0_relu4_fwd);
			Declear_Opration(baseconv, mobilenet0_conv5_fwd);
			Neuron_Name(mobilenet0_conv5_fwd);
			Declear_Opration(prelu, mobilenet0_relu5_fwd);
			Neuron_Name(mobilenet0_relu5_fwd);
			Declear_Opration(baseconv, mobilenet0_conv6_fwd);
			Neuron_Name(mobilenet0_conv6_fwd);
			Declear_Opration(prelu, mobilenet0_relu6_fwd);
			Neuron_Name(mobilenet0_relu6_fwd);
			Declear_Opration(baseconv, mobilenet0_conv7_fwd);
			Neuron_Name(mobilenet0_conv7_fwd);
			Declear_Opration(prelu, mobilenet0_relu7_fwd);
			Neuron_Name(mobilenet0_relu7_fwd);
			Declear_Opration(baseconv, mobilenet0_conv8_fwd);
			Neuron_Name(mobilenet0_conv8_fwd);
			Declear_Opration(prelu, mobilenet0_relu8_fwd);
			Neuron_Name(mobilenet0_relu8_fwd);
			Declear_Opration(baseconv, mobilenet0_conv9_fwd);
			Neuron_Name(mobilenet0_conv9_fwd);
			Declear_Opration(prelu, mobilenet0_relu9_fwd);
			Neuron_Name(mobilenet0_relu9_fwd);
			Declear_Opration(baseconv, mobilenet0_conv10_fwd);
			Neuron_Name(mobilenet0_conv10_fwd);
			Declear_Opration(prelu, mobilenet0_relu10_fwd);
			Neuron_Name(mobilenet0_relu10_fwd);
			Declear_Opration(baseconv, mobilenet0_conv11_fwd);
			Neuron_Name(mobilenet0_conv11_fwd);
			Declear_Opration(prelu, mobilenet0_relu11_fwd);
			Neuron_Name(mobilenet0_relu11_fwd);
			Declear_Opration(baseconv, mobilenet0_conv12_fwd);
			Neuron_Name(mobilenet0_conv12_fwd);
			Declear_Opration(prelu, mobilenet0_relu12_fwd);
			Neuron_Name(mobilenet0_relu12_fwd);
			Declear_Opration(baseconv, mobilenet0_conv13_fwd);
			Neuron_Name(mobilenet0_conv13_fwd);
			Declear_Opration(prelu, mobilenet0_relu13_fwd);
			Neuron_Name(mobilenet0_relu13_fwd);
			Declear_Opration(baseconv, mobilenet0_conv14_fwd);
			Neuron_Name(mobilenet0_conv14_fwd);
			Declear_Opration(prelu, mobilenet0_relu14_fwd);
			Neuron_Name(mobilenet0_relu14_fwd);
			Declear_Opration(baseconv, mobilenet0_conv15_fwd);
			Neuron_Name(mobilenet0_conv15_fwd);
			Declear_Opration(prelu, mobilenet0_relu15_fwd);
			Neuron_Name(mobilenet0_relu15_fwd);
			Declear_Opration(baseconv, mobilenet0_conv16_fwd);
			Neuron_Name(mobilenet0_conv16_fwd);
			Declear_Opration(prelu, mobilenet0_relu16_fwd);
			Neuron_Name(mobilenet0_relu16_fwd);
			Declear_Opration(baseconv, mobilenet0_conv17_fwd);
			Neuron_Name(mobilenet0_conv17_fwd);
			Declear_Opration(prelu, mobilenet0_relu17_fwd);
			Neuron_Name(mobilenet0_relu17_fwd);
			Declear_Opration(baseconv, mobilenet0_conv18_fwd);
			Neuron_Name(mobilenet0_conv18_fwd);
			Declear_Opration(prelu, mobilenet0_relu18_fwd);
			Neuron_Name(mobilenet0_relu18_fwd);
			Declear_Opration(baseconv, mobilenet0_conv19_fwd);
			Neuron_Name(mobilenet0_conv19_fwd);
			Declear_Opration(prelu, mobilenet0_relu19_fwd);
			Neuron_Name(mobilenet0_relu19_fwd);
			Declear_Opration(baseconv, mobilenet0_conv20_fwd);
			Neuron_Name(mobilenet0_conv20_fwd);
			Declear_Opration(prelu, mobilenet0_relu20_fwd);
			Neuron_Name(mobilenet0_relu20_fwd);
			Declear_Opration(baseconv, mobilenet0_conv21_fwd);
			Neuron_Name(mobilenet0_conv21_fwd);
			Declear_Opration(prelu, mobilenet0_relu21_fwd);
			Neuron_Name(mobilenet0_relu21_fwd);
			Declear_Opration(baseconv, mobilenet0_conv22_fwd);
			Neuron_Name(mobilenet0_conv22_fwd);
			Declear_Opration(prelu, mobilenet0_relu22_fwd);
			Neuron_Name(mobilenet0_relu22_fwd);
			Declear_Opration(baseconv, mobilenet0_conv23_fwd);
			Neuron_Name(mobilenet0_conv23_fwd);
			Declear_Opration(prelu, mobilenet0_relu23_fwd);
			Neuron_Name(mobilenet0_relu23_fwd);
			Declear_Opration(baseconv, mobilenet0_conv24_fwd);
			Neuron_Name(mobilenet0_conv24_fwd);
			Declear_Opration(prelu, mobilenet0_relu24_fwd);
			Neuron_Name(mobilenet0_relu24_fwd);
			Declear_Opration(baseconv, mobilenet0_conv25_fwd);
			Neuron_Name(mobilenet0_conv25_fwd);
			Declear_Opration(prelu, mobilenet0_relu25_fwd);
			Neuron_Name(mobilenet0_relu25_fwd);
			Declear_Opration(baseconv, mobilenet0_conv26_fwd);
			Neuron_Name(mobilenet0_conv26_fwd);
			Declear_Opration(prelu, mobilenet0_relu26_fwd);
			Neuron_Name(mobilenet0_relu26_fwd);
			Declear_Opration(baseconv, rf_c3_lateral);
			Neuron_Name(rf_c3_lateral);
			Declear_Opration(prelu, rf_c3_lateral_relu);
			Neuron_Name(rf_c3_lateral_relu);
			Declear_Opration(baseconv, rf_c3_det_conv1);
			Neuron_Name(rf_c3_det_conv1);
			Declear_Opration(baseconv, rf_c3_det_context_conv1);
			Neuron_Name(rf_c3_det_context_conv1);
			Declear_Opration(prelu, rf_c3_det_context_conv1_relu);
			Neuron_Name(rf_c3_det_context_conv1_relu);
			Declear_Opration(baseconv, rf_c3_det_context_conv2);
			Neuron_Name(rf_c3_det_context_conv2);
			Declear_Opration(baseconv, rf_c3_det_context_conv3_1);
			Neuron_Name(rf_c3_det_context_conv3_1);
			Declear_Opration(prelu, rf_c3_det_context_conv3_1_relu);
			Neuron_Name(rf_c3_det_context_conv3_1_relu);
			Declear_Opration(baseconv, rf_c3_det_context_conv3_2);
			Neuron_Name(rf_c3_det_context_conv3_2);
			Declear_Opration(concat, rf_c3_det_concat);
			Neuron_Name(rf_c3_det_concat);
			Declear_Opration(prelu, rf_c3_det_concat_relu);
			Neuron_Name(rf_c3_det_concat_relu);
			Declear_Opration(baseconv, face_rpn_cls_score_stride32);
			Neuron_Name(face_rpn_cls_score_stride32);
			Declear_Opration(reshape, face_rpn_cls_score_reshape_stride32);
			Neuron_Name(face_rpn_cls_score_reshape_stride32);
			Declear_Opration(softmax, face_rpn_cls_prob_stride32);
			Neuron_Name(face_rpn_cls_prob_stride32);
			Declear_Opration(reshape, face_rpn_cls_prob_reshape_stride32);
			Neuron_Name(face_rpn_cls_prob_reshape_stride32);
			Declear_Opration(baseconv, face_rpn_bbox_pred_stride32);
			Neuron_Name(face_rpn_bbox_pred_stride32);
			Declear_Opration(baseconv, face_rpn_landmark_pred_stride32);
			Neuron_Name(face_rpn_landmark_pred_stride32);
			Declear_Opration(baseconv, rf_c2_lateral);
			Neuron_Name(rf_c2_lateral);
			Declear_Opration(prelu, rf_c2_lateral_relu);
			Neuron_Name(rf_c2_lateral_relu);
			Declear_Opration(baseconv, rf_c3_upsampling);
			Neuron_Name(rf_c3_upsampling);
			Declear_Opration(eltwise, _plus0);
			Neuron_Name(_plus0);
			Declear_Opration(baseconv, rf_c2_aggr);
			Neuron_Name(rf_c2_aggr);
			Declear_Opration(prelu, rf_c2_aggr_relu);
			Neuron_Name(rf_c2_aggr_relu);
			Declear_Opration(baseconv, rf_c2_det_conv1);
			Neuron_Name(rf_c2_det_conv1);
			Declear_Opration(baseconv, rf_c2_det_context_conv1);
			Neuron_Name(rf_c2_det_context_conv1);
			Declear_Opration(prelu, rf_c2_det_context_conv1_relu);
			Neuron_Name(rf_c2_det_context_conv1_relu);
			Declear_Opration(baseconv, rf_c2_det_context_conv2);
			Neuron_Name(rf_c2_det_context_conv2);
			Declear_Opration(baseconv, rf_c2_det_context_conv3_1);
			Neuron_Name(rf_c2_det_context_conv3_1);
			Declear_Opration(prelu, rf_c2_det_context_conv3_1_relu);
			Neuron_Name(rf_c2_det_context_conv3_1_relu);
			Declear_Opration(baseconv, rf_c2_det_context_conv3_2);
			Neuron_Name(rf_c2_det_context_conv3_2);
			Declear_Opration(concat, rf_c2_det_concat);
			Neuron_Name(rf_c2_det_concat);
			Declear_Opration(prelu, rf_c2_det_concat_relu);
			Neuron_Name(rf_c2_det_concat_relu);
			Declear_Opration(baseconv, face_rpn_cls_score_stride16);
			Neuron_Name(face_rpn_cls_score_stride16);
			Declear_Opration(reshape, face_rpn_cls_score_reshape_stride16);
			Neuron_Name(face_rpn_cls_score_reshape_stride16);
			Declear_Opration(softmax, face_rpn_cls_prob_stride16);
			Neuron_Name(face_rpn_cls_prob_stride16);
			Declear_Opration(reshape, face_rpn_cls_prob_reshape_stride16);
			Neuron_Name(face_rpn_cls_prob_reshape_stride16);
			Declear_Opration(baseconv, face_rpn_bbox_pred_stride16);
			Neuron_Name(face_rpn_bbox_pred_stride16);
			Declear_Opration(baseconv, face_rpn_landmark_pred_stride16);
			Neuron_Name(face_rpn_landmark_pred_stride16);
			Declear_Opration(baseconv, rf_c1_red_conv);
			Neuron_Name(rf_c1_red_conv);
			Declear_Opration(prelu, rf_c1_red_conv_relu);
			Neuron_Name(rf_c1_red_conv_relu);
			Declear_Opration(baseconv, rf_c2_upsampling);
			Neuron_Name(rf_c2_upsampling);
			Declear_Opration(eltwise, _plus1);
			Neuron_Name(_plus1);
			Declear_Opration(baseconv, rf_c1_aggr);
			Neuron_Name(rf_c1_aggr);
			Declear_Opration(prelu, rf_c1_aggr_relu);
			Neuron_Name(rf_c1_aggr_relu);
			Declear_Opration(baseconv, rf_c1_det_conv1);
			Neuron_Name(rf_c1_det_conv1);
			Declear_Opration(baseconv, rf_c1_det_context_conv1);
			Neuron_Name(rf_c1_det_context_conv1);
			Declear_Opration(prelu, rf_c1_det_context_conv1_relu);
			Neuron_Name(rf_c1_det_context_conv1_relu);
			Declear_Opration(baseconv, rf_c1_det_context_conv2);
			Neuron_Name(rf_c1_det_context_conv2);
			Declear_Opration(baseconv, rf_c1_det_context_conv3_1);
			Neuron_Name(rf_c1_det_context_conv3_1);
			Declear_Opration(prelu, rf_c1_det_context_conv3_1_relu);
			Neuron_Name(rf_c1_det_context_conv3_1_relu);
			Declear_Opration(baseconv, rf_c1_det_context_conv3_2);
			Neuron_Name(rf_c1_det_context_conv3_2);
			Declear_Opration(concat, rf_c1_det_concat);
			Neuron_Name(rf_c1_det_concat);
			Declear_Opration(prelu, rf_c1_det_concat_relu);
			Neuron_Name(rf_c1_det_concat_relu);
			Declear_Opration(baseconv, face_rpn_cls_score_stride8);
			Neuron_Name(face_rpn_cls_score_stride8);
			Declear_Opration(reshape, face_rpn_cls_score_reshape_stride8);
			Neuron_Name(face_rpn_cls_score_reshape_stride8);
			Declear_Opration(softmax, face_rpn_cls_prob_stride8);
			Neuron_Name(face_rpn_cls_prob_stride8);
			Declear_Opration(reshape, face_rpn_cls_prob_reshape_stride8);
			Neuron_Name(face_rpn_cls_prob_reshape_stride8);
			Declear_Opration(baseconv, face_rpn_bbox_pred_stride8);
			Neuron_Name(face_rpn_bbox_pred_stride8);
			Declear_Opration(baseconv, face_rpn_landmark_pred_stride8);
			Neuron_Name(face_rpn_landmark_pred_stride8);

#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_gpu_native(const std::shared_ptr<tensor<float>> &input_data);
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> &input_data);
#endif 
#endif
			void Forward_cpu(const std::shared_ptr<tensor<float>> &input_data);

		public:
			Retina_net(int device = -1);
			~Retina_net();

			std::vector<std::vector<std::tuple<std::vector<int>, const float*>>> Forward(const std::shared_ptr<tensor<unsigned char>> &input)
			{
				CHECK_EQ(input->order(), NCHW);

				if (device_ < 0)
				{
					tensor_operation_cpu::type_converter_cpu(input, tensor_float_data);
					Forward_cpu(tensor_float_data);
				}
				else
				{
#ifdef USE_CUDA
					tensor_operation_gpu::type_converter_gpu(input, tensor_float_data);
#ifdef USE_CUDNN
					Forward_gpu_cudnn(tensor_float_data);
#else
					Forward_gpu_native(tensor_float_data);
#endif

#else
					NO_GPU;
#endif
				}

				std::vector<std::vector<std::tuple<std::vector<int>, const float*>>> tuple_result;
				std::vector<std::tuple<std::vector<int>, const float*>> temp_tuple;

				temp_tuple.emplace_back(face_rpn_cls_prob_reshape_stride32_top_data->data_shape(), get_face_rpn_cls_prob_reshape_stride32()->cpu_data());
				temp_tuple.emplace_back(face_rpn_bbox_pred_stride32_top_data->data_shape(), get_face_rpn_bbox_pred_stride32()->cpu_data());
				temp_tuple.emplace_back(face_rpn_landmark_pred_stride32_top_data->data_shape(), get_face_rpn_landmark_pred_stride32()->cpu_data());
				tuple_result.push_back(temp_tuple);

				temp_tuple.clear();
				temp_tuple.emplace_back(face_rpn_cls_prob_reshape_stride16_top_data->data_shape(), get_face_rpn_cls_prob_reshape_stride16()->cpu_data());
				temp_tuple.emplace_back(face_rpn_bbox_pred_stride16_top_data->data_shape(), get_face_rpn_bbox_pred_stride16()->cpu_data());
				temp_tuple.emplace_back(face_rpn_landmark_pred_stride16_top_data->data_shape(), get_face_rpn_landmark_pred_stride16()->cpu_data());
				tuple_result.push_back(temp_tuple);

				temp_tuple.clear();
				temp_tuple.emplace_back(face_rpn_cls_prob_reshape_stride8_top_data->data_shape(), get_face_rpn_cls_prob_reshape_stride8()->cpu_data());
				temp_tuple.emplace_back(face_rpn_bbox_pred_stride8_top_data->data_shape(), get_face_rpn_bbox_pred_stride8()->cpu_data());
				temp_tuple.emplace_back(face_rpn_landmark_pred_stride8_top_data->data_shape(), get_face_rpn_landmark_pred_stride8()->cpu_data());
				tuple_result.push_back(temp_tuple);

				return tuple_result;
			}
		};
	}
}

#endif