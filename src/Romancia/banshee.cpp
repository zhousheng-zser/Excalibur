#include "banshee.hpp"
#include "banshee_data.hpp"
#include <glasssix/tensor.hpp>
#include <glasssix/syncedmem.hpp>

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		Banshee::Banshee(int device)
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

			float quantize_level = INT_MAX;

			if (int8_quantization_)
			{
				Copy_Int8_Params(conv1, Banshee);
				Copy_Params(prelu1_weights, Banshee, quantize_level);//16
				Copy_Int8_Params(conv1_dw, Banshee);
				Copy_Params(prelu1_dw_weights, Banshee, quantize_level);//16
				Copy_Int8_Params(conv2, Banshee);
				Copy_Int8_Params(conv2_dw, Banshee);
				Copy_Params(prelu2_dw_weights, Banshee, quantize_level);//32
				Copy_Int8_Params(conv3, Banshee);
				Copy_Int8_Params(conv3_dw, Banshee);
				Copy_Params(prelu3_dw_weights, Banshee, quantize_level);//32
				Copy_Int8_Params(conv4, Banshee);
				Copy_Int8_Params(conv4_dw, Banshee);
				Copy_Params(prelu4_dw_weights, Banshee, quantize_level);//64
				Copy_Params(conv5_weights, Banshee, quantize_level);//147456
				Copy_Params(conv5_bias, Banshee, quantize_level);//256
				Copy_Params(prelu5_weights, Banshee, quantize_level);//256
				Copy_Params(conv6_1_weights, Banshee, quantize_level);//256
				Copy_Params(conv6_1_bias, Banshee, quantize_level);//1
				Copy_Params(conv6_2_weights, Banshee, quantize_level);//768
				Copy_Params(conv6_2_bias, Banshee, quantize_level);//3
				Copy_Params(conv6_3_weights, Banshee, quantize_level);//2560
				Copy_Params(conv6_3_bias, Banshee, quantize_level);//10
			}
			else
			{
				Copy_Params(conv1_weights, Banshee, quantize_level);//144
				Copy_Params(conv1_bias, Banshee, quantize_level);//16
				Copy_Params(prelu1_weights, Banshee, quantize_level);//16
				Copy_Params(conv1_dw_weights, Banshee, quantize_level);//144
				Copy_Params(conv1_dw_bias, Banshee, quantize_level);//16
				Copy_Params(prelu1_dw_weights, Banshee, quantize_level);//16
				Copy_Params(conv2_weights, Banshee, quantize_level);//512
				Copy_Params(conv2_bias, Banshee, quantize_level);//32
				Copy_Params(conv2_dw_weights, Banshee, quantize_level);//288
				Copy_Params(conv2_dw_bias, Banshee, quantize_level);//32
				Copy_Params(prelu2_dw_weights, Banshee, quantize_level);//32
				Copy_Params(conv3_weights, Banshee, quantize_level);//1024
				Copy_Params(conv3_bias, Banshee, quantize_level);//32
				Copy_Params(conv3_dw_weights, Banshee, quantize_level);//288
				Copy_Params(conv3_dw_bias, Banshee, quantize_level);//32
				Copy_Params(prelu3_dw_weights, Banshee, quantize_level);//32
				Copy_Params(conv4_weights, Banshee, quantize_level);//2048
				Copy_Params(conv4_bias, Banshee, quantize_level);//64
				Copy_Params(conv4_dw_weights, Banshee, quantize_level);//576
				Copy_Params(conv4_dw_bias, Banshee, quantize_level);//64
				Copy_Params(prelu4_dw_weights, Banshee, quantize_level);//64
				Copy_Params(conv5_weights, Banshee, quantize_level);//147456
				Copy_Params(conv5_bias, Banshee, quantize_level);//256
				Copy_Params(prelu5_weights, Banshee, quantize_level);//256
				Copy_Params(conv6_1_weights, Banshee, quantize_level);//256
				Copy_Params(conv6_1_bias, Banshee, quantize_level);//1
				Copy_Params(conv6_2_weights, Banshee, quantize_level);//768
				Copy_Params(conv6_2_bias, Banshee, quantize_level);//3
				Copy_Params(conv6_3_weights, Banshee, quantize_level);//2560
				Copy_Params(conv6_3_bias, Banshee, quantize_level);//10
			}
			

			//			
#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}

#ifdef USE_CUDNN
			if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
			}
			cudnn_ready_ = true;
#endif // USE_CUDNN
#endif
                        
			Init_Conv_Params(conv1, 1, 16, 1, 3, 2, 1, true);//nchw:1*1*48*48->1*16*24*24
			Init_PReLU_Params(prelu1, 16, false);//nchw:1*16*24*24->1*16*24*24
			Init_Conv_Params(conv1_dw, 16, 16, 16, 3, 1, 1, true);//nchw:1*16*24*24->1*16*24*24
			Init_PReLU_Params(prelu1_dw, 16, false);//nchw:1*16*24*24->1*16*24*24
			Init_Conv_Params(conv2, 16, 32, 1, 1, 1, 0, true);//nchw:1*16*24*24->1*32*24*24
			Init_Conv_Params(conv2_dw, 32, 32, 32, 3, 2, 1, true);//nchw:1*32*24*24->1*32*12*12
			Init_PReLU_Params(prelu2_dw, 32, false);//nchw:1*32*12*12->1*32*12*12
			Init_Conv_Params(conv3, 32, 32, 1, 1, 1, 0, true);//nchw:1*32*12*12->1*32*12*12
			Init_Conv_Params(conv3_dw, 32, 32, 32, 3, 2, 1, true);//nchw:1*32*12*12->1*32*6*6
			Init_PReLU_Params(prelu3_dw, 32, false);//nchw:1*32*6*6->1*32*6*6
			Init_Conv_Params(conv4, 32, 64, 1, 1, 1, 0, true);//nchw:1*32*6*6->1*64*6*6
			Init_Conv_Params(conv4_dw, 64, 64, 64, 3, 2, 1, true);//nchw:1*64*6*6->1*64*3*3
			Init_PReLU_Params(prelu4_dw, 64, false);//nchw:1*64*3*3->1*64*3*3
			Init_InnerProduct_Params(conv5, 64, 3, 3, 256, true);//nchw:1*64*3*3->1*256*1*1
			Init_PReLU_Params(prelu5, 256, false);//nchw:1*256*1*1->1*256*1*1
			Init_InnerProduct_Params(conv6_1, 256, 1, 1, 1, true);//nchw:1*256*1*1->1*1*1*1
			Init_Sigmoid_Params(sigmoid1);
			Init_InnerProduct_Params(conv6_2, 256, 1, 1, 3, true);//nchw:1*256*1*1->1*3*1*1
			Init_InnerProduct_Params(conv6_3, 256, 1, 1, 10, true);//nchw:1*256*1*1->1*10*1*1
		}


		Banshee::~Banshee()
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

			FreeHost(prelu1_weights, false);
			FreeHost(prelu1_dw_weights, false);
			FreeHost(prelu2_dw_weights, false);
			FreeHost(prelu3_dw_weights, false);
			FreeHost(prelu4_dw_weights, false);
			FreeHost(prelu5_weights, false);

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

		void Banshee::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			conv1_dw->Forward(conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_cpu(conv1_dw_top_data);
			conv2->Forward(conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_cpu(conv2_dw_top_data);
			conv3->Forward(conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_cpu(conv3_dw_top_data);
			conv4->Forward(conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_cpu(conv4_dw_top_data);
			conv5->Forward_cpu(conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_cpu(conv5_top_data);
			conv6_1->Forward_cpu(conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_cpu(conv6_1_top_data);
			conv6_2->Forward_cpu(conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_cpu(conv5_top_data, conv6_3_top_data);
		}

#ifdef USE_CUDA
		void Banshee::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cublas_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2->Forward(cublas_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(cublas_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv3->Forward(cublas_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(cublas_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv4->Forward(cublas_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(cublas_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_gpu_native(conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_3_top_data);
		}

#ifdef USE_CUDNN
		void Banshee::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cudnn_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2->Forward(cudnn_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(cudnn_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv3->Forward(cudnn_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(cudnn_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv4->Forward(cudnn_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(cudnn_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_gpu_native(conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_3_top_data);
		}
#endif 
#endif

		void Banshee::getParam(std::vector<std::vector<float> > &keypointParam, unsigned num)
		{
			if (device_ < 0)
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
			else
			{
#ifdef USE_CUDA
				for (size_t i = 0; i < num; i++)
				{
					std::vector<float> temp;
					std::vector<float> conv_6_1(1), conv_6_2(3), conv_6_3(10);
					cudaMemcpy(conv_6_1.data(), get_conv6_1()->gpu_data(), 1 * sizeof(float), cudaMemcpyDefault);
					cudaMemcpy(conv_6_2.data(), get_conv6_2()->gpu_data(), 3 * sizeof(float), cudaMemcpyDefault);
					cudaMemcpy(conv_6_3.data(), get_conv6_3()->gpu_data(), 10 * sizeof(float), cudaMemcpyDefault);

					for (size_t j = 0; j < 1; j++)
					{
						temp.push_back(conv_6_1[j]);
					}
					for (size_t j = 0; j < 3; j++)
					{
						temp.push_back(conv_6_2[j]);
					}
					for (size_t j = 0; j < 10; j++)
					{
						temp.push_back(conv_6_3[j]);
					}
					keypointParam.push_back(temp);
			}
#else
				NO_GPU;
#endif // USE_CUDA

			}
		}


		std::vector<unsigned char> Banshee::alignFace(const unsigned char* origine, int n, int channels,
			int height, int width)
		{
			if (n <= 0)
			{
				LOG(FATAL) << "no human face information!!!";
				return std::vector<unsigned char>();
			}

			CHECK_EQ(n, 1);
			CHECK_EQ(channels, 1);

			std::shared_ptr<tensor<unsigned char>> ori_image, resized_img, ROI, rotated_ROI, final_mat, final_mat_gray, color_img, resized_color_img;
			std::vector<std::shared_ptr<tensor<unsigned char>>> src_vector;
			std::vector<unsigned char> res;
			res.resize(n * 3 * 128 * 128);

			if (device_ < 0)
			{
				ori_image.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
				memcpy(ori_image->mutable_cpu_data(), origine, 1 * channels * height * width * sizeof(unsigned char));

				rectangle<int> MarginRect = rectangle<int>((float)width / 7,
					(float)height / 7,
					(float)height * 5 / 7,
					(float)width * 5 / 7);

				tensor_operation_cpu::safty_cut_cpu(ori_image, ROI, &MarginRect);

				tensor_operation_cpu::resize_cpu(ROI, resized_img, 48, 48);
				Forward(resized_img->cpu_data(), 1, 0);

				std::vector<std::vector<float> > landmarks;
				getParam(landmarks, 1);

				point<float> ldmk5[5];
				for (size_t j = 0; j < 5; j++)
				{
					ldmk5[j] = point<float>(landmarks[0][4 + 2 * j] * (float)width * 5 / 7 + (float)width / 7, landmarks[0][4 + 2 * j + 1] * (float)height * 5 / 7 + (float)height / 7);
				}
				point<float> center_eye = point<float>((ldmk5[0].x + ldmk5[1].x) / 2, (ldmk5[0].y + ldmk5[1].y) / 2);
				point<float> center_mouth = point<float>((ldmk5[3].x + ldmk5[4].x) / 2, (ldmk5[3].y + ldmk5[4].y) / 2);
				point<float> center = point<float>((center_eye.x + center_mouth.x) / 2, (center_eye.y + center_mouth.y) / 2);
				double tan = (center_eye.x - center_mouth.x) / (center_eye.y - center_mouth.y);
				double arctan = atan(tan) * 180 / 3.1415926;

				tensor_operation_cpu::rotate_with_points_cpu(ori_image, rotated_ROI, center, -1 * arctan);

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

				memcpy(res.data(), resized_color_img->cpu_data(), 3 * 128 * 128 * sizeof(unsigned char));
			}
			else
			{
#ifdef USE_CUDA
				ori_image.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
				cudaMemcpy(ori_image->mutable_gpu_data(), origine, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);

				rectangle<int> MarginRect = rectangle<int>((float)width / 7,
					(float)height / 7,
					(float)height * 5 / 7,
					(float)width * 5 / 7);

				tensor_operation_gpu::safty_cut_gpu(ori_image, ROI, &MarginRect);

				tensor_operation_gpu::resize_gpu(ROI, resized_img, 48, 48);
				Forward(resized_img->gpu_data(), 1, 0);

				std::vector<std::vector<float> > landmarks;
				getParam(landmarks, 1);

				point<float> ldmk5[5];
				for (size_t j = 0; j < 5; j++)
				{
					ldmk5[j] = point<float>(landmarks[0][4 + 2 * j] * (float)width * 5 / 7 + (float)width / 7, landmarks[0][4 + 2 * j + 1] * (float)height * 5 / 7 + (float)height / 7);
				}


				point<float> center_eye = point<float>((ldmk5[0].x + ldmk5[1].x) / 2, (ldmk5[0].y + ldmk5[1].y) / 2);
				point<float> center_mouth = point<float>((ldmk5[3].x + ldmk5[4].x) / 2, (ldmk5[3].y + ldmk5[4].y) / 2);
				point<float> center = point<float>((center_eye.x + center_mouth.x) / 2, (center_eye.y + center_mouth.y) / 2);
				double tan = (center_eye.x - center_mouth.x) / (center_eye.y - center_mouth.y);
				double arctan = atan(tan) * 180 / 3.1415926;

				tensor_operation_gpu::rotate_with_points_gpu(ori_image, rotated_ROI, center, -1 * arctan);

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

				cudaMemcpy(res.data(), resized_color_img->gpu_data(), 3 * 128 * 128 * sizeof(unsigned char), cudaMemcpyDefault);

#else
				NO_GPU;
#endif // USE_CUDA

			}

			
			return res;
		}


		std::vector<unsigned char> Banshee::alignFace(const unsigned char* origine, int n, int channels, 
			int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks)
		{
			if (n <= 0)
			{
				LOG(FATAL) << "no human face information!!!";
				return std::vector<unsigned char>();
			}

			CHECK_EQ(n, bbox.size());
			CHECK_EQ(n, landmarks.size());
			CHECK_EQ(channels, 1);
			std::shared_ptr<tensor<unsigned char>> ori_image, ROI, rotated_ROI, final_mat, final_mat_gray, color_img, resized_color_img;
			std::vector<std::shared_ptr<tensor<unsigned char>>> src_vector;
			std::vector<unsigned char> res;
			res.resize(n * 3 * 128 * 128);

			if (device_ < 0)
			{
				ori_image.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
				memcpy(ori_image->mutable_cpu_data(), origine, 1 * channels * height * width * sizeof(unsigned char));

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
				ori_image.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_, NCHW));
				cudaMemcpy(ori_image->mutable_gpu_data(), origine, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);

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
