#include "black_white_vsl.hpp"
#include "blur_vsl_net_data.hpp"
#include <glasssix/tensor.hpp>
#include <glasssix/syncedmem.hpp>

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		Black_white_vsl::Black_white_vsl(int device) {};
		Black_white_vsl::~Black_white_vsl() {};



		/// <summary>
		/// detect on visible image, whether black and white photo detected, return true if pass(not black and white photo)
		/// </summary>
		/// <param name="vsl_color_image">visible image data</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="bbox">detected humanface bboxes</param>
		/// <param name="landmarks">detected humanface landmarks</param>
		/// <param name="thresh">thresh[0]: threshold value of black-white-judge, 30 by default; thresh[1]: not in use</param>
		/// <param name="value">value[0]: return value of black-white-judge score; value[1]: not in use</param>
		/// <param name="order">order type of visible image: NCHW(0) / NHWC(1)</param>
		bool Black_white_vsl::judge(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order)
		{
			std::shared_ptr<tensor<unsigned char>> face_vsl;

			//get judge area: 1*3*48*48
			{
				CHECK_EQ(bbox.size(), 1);
				CHECK_EQ(landmarks.size(), 1);
				CHECK_EQ(landmarks[0].size() / 2, 5);

				std::shared_ptr<tensor<unsigned char>> image_vsl;
				if (order == 0)
				{
					image_vsl.reset(new tensor<unsigned char>(std::vector<int>{1, 3, height, width}, -1, NCHW));
				}
				else if (order == 1)
				{
					image_vsl.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, 3}, -1, NHWC));
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				memcpy(image_vsl->mutable_cpu_data(), vsl_color_image, 1 * 3 * height * width * sizeof(unsigned char));
				
				glasssix::excalibur::rectangle<int> faceRect = glasssix::excalibur::rectangle<int>(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]);

				tensor_operation_cpu::safty_cut_cpu(image_vsl, face_vsl, &faceRect);
				tensor_operation_cpu::resize_cpu(face_vsl, face_vsl, 48, 48);
			}

			//mean
			{
				std::vector<std::shared_ptr<tensor<unsigned char>>> channels;
				tensor_operation_cpu::split_channel_cpu(face_vsl, channels);
				
				CHECK_EQ(channels.size(), 3);
				std::shared_ptr<tensor<unsigned char>> temp1, temp2, mean_tensor;
				tensor_operation_cpu::absdiff_cpu(channels[0], channels[1], temp1);
				tensor_operation_cpu::absdiff_cpu(channels[1], channels[2], temp2);
				tensor_operation_cpu::add_channel_cpu(temp1, temp2, mean_tensor);
				tensor_operation_cpu::absdiff_cpu(channels[0], channels[2], temp1);
				tensor_operation_cpu::add_channel_cpu(temp1, mean_tensor, mean_tensor);

				value[0] = tensor_operation_cpu::mean_array(mean_tensor);

				if (value[0] < thresh[0])
				{
					return false;//black and white photo detected
				}
				else
				{
					return true;//real human or color photo detected
				}
			}
		}
	}
}