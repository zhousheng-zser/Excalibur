#include "./include/landmarkNet.hpp"
#include "./include/landmarkNet_data.hpp"
#include "../Excalibur/tensor_utils.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		LandmarkNet::LandmarkNet(int device)
		{
			float quantize_level = INT_MAX;
			Copy_Params(conv1_weights, LandmarkNet, quantize_level);//144
			Copy_Params(conv1_bias, LandmarkNet, quantize_level);//16
			Copy_Params(prelu1_weights, LandmarkNet, quantize_level);//16
			Copy_Params(conv1_dw_weights, LandmarkNet, quantize_level);//144
			Copy_Params(conv1_dw_bias, LandmarkNet, quantize_level);//16
			Copy_Params(prelu1_dw_weights, LandmarkNet, quantize_level);//16
			Copy_Params(conv2_weights, LandmarkNet, quantize_level);//512
			Copy_Params(conv2_bias, LandmarkNet, quantize_level);//32
			Copy_Params(conv2_dw_weights, LandmarkNet, quantize_level);//288
			Copy_Params(conv2_dw_bias, LandmarkNet, quantize_level);//32
			Copy_Params(prelu2_dw_weights, LandmarkNet, quantize_level);//32
			Copy_Params(conv3_weights, LandmarkNet, quantize_level);//1024
			Copy_Params(conv3_bias, LandmarkNet, quantize_level);//32
			Copy_Params(conv3_dw_weights, LandmarkNet, quantize_level);//288
			Copy_Params(conv3_dw_bias, LandmarkNet, quantize_level);//32
			Copy_Params(prelu3_dw_weights, LandmarkNet, quantize_level);//32
			Copy_Params(conv4_weights, LandmarkNet, quantize_level);//2048
			Copy_Params(conv4_bias, LandmarkNet, quantize_level);//64
			Copy_Params(conv4_dw_weights, LandmarkNet, quantize_level);//576
			Copy_Params(conv4_dw_bias, LandmarkNet, quantize_level);//64
			Copy_Params(prelu4_dw_weights, LandmarkNet, quantize_level);//64
			Copy_Params(conv5_weights, LandmarkNet, quantize_level);//147456
			Copy_Params(conv5_bias, LandmarkNet, quantize_level);//256
			Copy_Params(prelu5_weights, LandmarkNet, quantize_level);//256
			Copy_Params(conv6_1_weights, LandmarkNet, quantize_level);//256
			Copy_Params(conv6_1_bias, LandmarkNet, quantize_level);//1
			Copy_Params(conv6_2_weights, LandmarkNet, quantize_level);//768
			Copy_Params(conv6_2_bias, LandmarkNet, quantize_level);//3
			Copy_Params(conv6_3_weights, LandmarkNet, quantize_level);//2560
			Copy_Params(conv6_3_bias, LandmarkNet, quantize_level);//10
			//
			device_ = device;
#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#endif

			Init_Conv_Params(conv1, 1, 16, 3, 2, 1, true);
			Init_PReLU_Params(prelu1, 16, false);
			Init_DepthConv_Params(conv1_dw, 16, 16, 3, 1, 1, true);
			Init_PReLU_Params(prelu1_dw, 16, false);
			Init_Conv_Params(conv2, 16, 32, 1, 1, 0, true);
			Init_DepthConv_Params(conv2_dw, 32, 32, 3, 2, 1, true);
			Init_PReLU_Params(prelu2_dw, 32, false);
			Init_Conv_Params(conv3, 32, 32, 1, 1, 0, true);
			Init_DepthConv_Params(conv3_dw, 32, 32, 3, 2, 1, true);
			Init_PReLU_Params(prelu3_dw, 32, false);
			Init_Conv_Params(conv4, 32, 64, 1, 1, 0, true);
			Init_DepthConv_Params(conv4_dw, 64, 64, 3, 2, 1, true);
			Init_PReLU_Params(prelu4_dw, 64, false);
			Init_InnerProduct_Params(conv5, 64, 3, 3, 256, true);
			Init_PReLU_Params(prelu5, 256, false);
			Init_InnerProduct_Params(conv6_1, 256, 1, 1, 1, true);
			Init_InnerProduct_Params(conv6_2, 256, 1, 1, 3, true);
			Init_InnerProduct_Params(conv6_3, 256, 1, 1, 10, true);
		}


		LandmarkNet::~LandmarkNet()
		{
			delete conv1;
			delete prelu1;
			delete conv1_dw;
			delete prelu1_dw;
			delete conv2;
			delete conv2_dw;
			delete prelu2_dw;
			delete conv3;
			delete conv3_dw;
			delete prelu3_dw;
			delete conv4;
			delete conv4_dw;
			delete prelu4_dw;
			delete conv5;
			delete prelu5;
			delete conv6_1;
			delete sigmoid1;
			delete conv6_2;
			delete conv6_3;
#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#endif
		}

		void LandmarkNet::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward_cpu(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			conv1_dw->Forward_cpu(conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_cpu(conv1_dw_top_data);
			conv2->Forward_cpu(conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward_cpu(conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_cpu(conv2_dw_top_data);
			conv3->Forward_cpu(conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward_cpu(conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_cpu(conv3_dw_top_data);
			conv4->Forward_cpu(conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward_cpu(conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_cpu(conv4_dw_top_data);
			conv5->Forward_cpu(conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_cpu(conv5_top_data);
			conv6_1->Forward_cpu(conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_cpu(conv6_1_top_data);
			conv6_2->Forward_cpu(conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_cpu(conv5_top_data, conv6_3_top_data);
		}

#ifdef USE_CUDA
		void LandmarkNet::Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward_native_gpu(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_native_gpu(conv1_top_data);
			conv1_dw->Forward_native_gpu(cublas_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_native_gpu(conv1_dw_top_data);
			conv2->Forward_native_gpu(cublas_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward_native_gpu(cublas_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_native_gpu(conv2_dw_top_data);
			conv3->Forward_native_gpu(cublas_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward_native_gpu(cublas_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_native_gpu(conv3_dw_top_data);
			conv4->Forward_native_gpu(cublas_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward_native_gpu(cublas_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_native_gpu(conv4_dw_top_data);
			conv5->Forward_native_gpu(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_native_gpu(conv5_top_data);
			conv6_1->Forward_native_gpu(cublas_handle_, conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_native_gpu(conv6_1_top_data);
			conv6_2->Forward_native_gpu(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_native_gpu(cublas_handle_, conv5_top_data, conv6_3_top_data);
		}

#ifdef USE_CUDNN
		void LandmarkNet::Forward_cudnn_gpu(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif

			conv1->Forward_cudnn_gpu(input_data, conv1_top_data);
			prelu1->Forward_native_gpu(conv1_top_data);
			conv1_dw->Forward_cudnn_gpu(conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_native_gpu(conv1_dw_top_data);
			conv2->Forward_cudnn_gpu(conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward_cudnn_gpu(conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_native_gpu(conv2_dw_top_data);
			conv3->Forward_cudnn_gpu(conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward_cudnn_gpu(conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_native_gpu(conv3_dw_top_data);
			conv4->Forward_cudnn_gpu(conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward_cudnn_gpu(conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_native_gpu(conv4_dw_top_data);
			conv5->Forward_native_gpu(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_native_gpu(conv5_top_data);
			conv6_1->Forward_native_gpu(cublas_handle_, conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_native_gpu(conv6_1_top_data);
			conv6_2->Forward_native_gpu(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_native_gpu(cublas_handle_, conv5_top_data, conv6_3_top_data);
		}
#endif 
#endif

		void LandmarkNet::getParam(std::vector<std::vector<float> > &keypointParam, unsigned num)
		{
			for (size_t i = 0; i < num; i++)
			{
				std::vector<float> temp;
				const float* conv_6_1_data = get_conv6_1()->cpu_data() + 1 * i;
				const float* conv_6_2_data = get_conv6_2()->cpu_data() + 3 * i;
				const float* conv_6_3_data = get_conv6_3()->cpu_data() + 10 * i;
				for (size_t j = 0; j < 1; j++)
				{
					temp.push_back(conv_6_1_data[j]);
				}
				for (size_t j = 0; j < 3; j++)
				{
					temp.push_back(conv_6_2_data[j]);
				}
				for (size_t j = 0; j < 10; j++)
				{
					temp.push_back(conv_6_3_data[j]);
				}
				keypointParam.push_back(temp);
			}
		}


		std::vector<unsigned char> LandmarkNet::alignFace(const unsigned char* origine, int n, int channels, 
			int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks)
		{
			CHECK_EQ(channels, 1);
			std::shared_ptr<tensor<unsigned char>> ori_image, ROI, rotated_ROI, final_mat, final_mat_gray, color_img, resized_color_img;
			std::vector<std::shared_ptr<tensor<unsigned char>>> src_vector;
			int num = bbox.size();
			std::vector<unsigned char> res;
			res.resize(num * 3 * 128 * 128);

			if (device_ < 0)
			{
				ori_image.reset(new tensor<unsigned char>(std::vector<int>{n, channels, height, width}, device_, NCHW));
				memcpy(ori_image->mutable_cpu_data(), origine, n * channels * height * width * sizeof(unsigned char));

				for (size_t i = 0; i < landmarks.size(); i++)
				{
					src_vector.clear();
					CHECK_EQ(landmarks[i].size() / 2, 5);
					rectangle<int> MarginRect = rectangle<int>(bbox[i][0] - bbox[i][3] * 0.2,
						bbox[i][1] - bbox[i][2] * 0.2,
						bbox[i][3] * 1.4f,
						bbox[i][2] * 1.4f);

					tensor_operation_cpu::safty_cut_cpu(ori_image, ROI, &MarginRect);

					point<float> ldmk5[5];
					for (size_t j = 0; j < landmarks[i].size() / 2; j++)
					{
						ldmk5[j] = point<float>(landmarks[i][2 * j] - MarginRect.x, landmarks[i][2 * j + 1] - MarginRect.y);
					}
					point<float> center_eye = point<float>((ldmk5[0].x + ldmk5[1].x) / 2, (ldmk5[0].y + ldmk5[1].y) / 2);
					point<float> center_mouth = point<float>((ldmk5[3].x + ldmk5[4].x) / 2, (ldmk5[3].y + ldmk5[4].y) / 2);
					point<float> center = point<float>((center_eye.x + center_mouth.x) / 2, (center_eye.y + center_mouth.y) / 2);
					double tan = (center_eye.x - center_mouth.x) / (center_eye.y - center_mouth.y);
					double arctan = atan(tan) * 180 / 3.1415926;

					tensor_operation_cpu::rotate_with_points_cpu(ROI, rotated_ROI, center, -1 * arctan);

					double distance = sqrt((center_eye.x - center_mouth.x) * (center_eye.x - center_mouth.x) + (center_eye.y - center_mouth.y) * (center_eye.y - center_mouth.y));
					double cos = (center_mouth.y - center_eye.y) / distance;
					double sin = (center_mouth.x - center_eye.x) / distance;
					point<float> new_center_eye = point<float>(center_eye.x + (float)(sin * distance / 2), (float)(center_eye.y - (1 - cos) * distance / 2));
					point<float> new_center_mouth = point<float>(center_mouth.x - (float)(sin * distance / 2), (float)(center_mouth.y + (1 - cos) * distance / 2));
					rectangle<float> final_rect = rectangle<float>(new_center_eye.x - distance,
						new_center_eye.y - distance / 2,
						distance * 2, distance * 2);
					tensor_operation_cpu::safty_cut_cpu(rotated_ROI, final_mat, &final_rect);
					tensor_operation_cpu::equalize_hist_cpu(final_mat, final_mat);

					for (size_t i = 0; i < 3; i++)
					{
						src_vector.push_back(final_mat);
					}

					tensor_operation_cpu::merge_channel_cpu(src_vector, color_img);
					tensor_operation_cpu::resize_cpu(color_img, resized_color_img, 128, 128);

					memcpy(&(res[0]) + i * 3 * 128 * 128 * sizeof(unsigned char), resized_color_img->cpu_data(), 3 * 128 * 128 * sizeof(unsigned char));
				}
			}
			else
			{
#ifdef USE_CUDA
				ori_image.reset(new tensor<unsigned char>(std::vector<int>{n, channels, height, width}, device_, NCHW));
				cudaMemcpy(ori_image->mutable_gpu_data(), origine, n * channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);

				for (size_t i = 0; i < landmarks.size(); i++)
				{
					src_vector.clear();
					CHECK_EQ(landmarks[i].size() / 2, 5);
					rectangle<int> MarginRect = rectangle<int>(bbox[i][0] - bbox[i][3] * 0.2,
						bbox[i][1] - bbox[i][2] * 0.2,
						bbox[i][3] * 1.4f,
						bbox[i][2] * 1.4f);

					tensor_operation_gpu::safty_cut_gpu(ori_image, ROI, &MarginRect);

					point<float> ldmk5[5];
					for (size_t j = 0; j < landmarks[i].size() / 2; j++)
					{
						ldmk5[j] = point<float>(landmarks[i][2 * j] - MarginRect.x, landmarks[i][2 * j + 1] - MarginRect.y);
					}
					point<float> center_eye = point<float>((ldmk5[0].x + ldmk5[1].x) / 2, (ldmk5[0].y + ldmk5[1].y) / 2);
					point<float> center_mouth = point<float>((ldmk5[3].x + ldmk5[4].x) / 2, (ldmk5[3].y + ldmk5[4].y) / 2);
					point<float> center = point<float>((center_eye.x + center_mouth.x) / 2, (center_eye.y + center_mouth.y) / 2);
					double tan = (center_eye.x - center_mouth.x) / (center_eye.y - center_mouth.y);
					double arctan = atan(tan) * 180 / 3.1415926;

					tensor_operation_gpu::rotate_with_points_gpu(ROI, rotated_ROI, center, -1 * arctan);

					double distance = sqrt((center_eye.x - center_mouth.x) * (center_eye.x - center_mouth.x) + (center_eye.y - center_mouth.y) * (center_eye.y - center_mouth.y));
					double cos = (center_mouth.y - center_eye.y) / distance;
					double sin = (center_mouth.x - center_eye.x) / distance;
					point<float> new_center_eye = point<float>(center_eye.x + (float)(sin * distance / 2), (float)(center_eye.y - (1 - cos) * distance / 2));
					point<float> new_center_mouth = point<float>(center_mouth.x - (float)(sin * distance / 2), (float)(center_mouth.y + (1 - cos) * distance / 2));
					rectangle<float> final_rect = rectangle<float>(new_center_eye.x - distance,
						new_center_eye.y - distance / 2,
						distance * 2, distance * 2);
					tensor_operation_gpu::safty_cut_gpu(rotated_ROI, final_mat, &final_rect);
					tensor_operation_gpu::equalize_hist_gpu(final_mat, final_mat);

					for (size_t i = 0; i < 3; i++)
					{
						src_vector.push_back(final_mat);
					}

					tensor_operation_gpu::merge_channel_gpu(src_vector, color_img);
					tensor_operation_gpu::resize_gpu(color_img, resized_color_img, 128, 128);

					cudaMemcpy(&(res[0]) + i * 3 * 128 * 128 * sizeof(unsigned char), resized_color_img->gpu_data(), 3 * 128 * 128 * sizeof(unsigned char), cudaMemcpyDefault);
				}

#else
				NO_GPU;
#endif // USE_CUDA

			}

			return res;
		}
	}
}