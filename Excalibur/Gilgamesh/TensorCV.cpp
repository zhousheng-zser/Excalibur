#include "TensorCV.hpp"

using namespace excalibur;

namespace glasssix
{
	namespace gilgamesh
	{
		void tensorcv::resize(Tensor^ src, Tensor^ %dst, int new_height,
			int new_width, InterpolationType type, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, new_height, new_width, device);
				resize_cpu(src, dst, new_height, new_width, type);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::rotate(Tensor^ src, Tensor^ %dst, float theta,
			int center_x, int center_y, InterpolationType type, float v, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, src->Height, src->Width, device);
				rotate_cpu(src, dst, theta, center_x, center_y, type, v);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::flip(Tensor^ src, Tensor^ %dst, FlipType axis, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, src->Height, src->Width, device);
				flip_cpu(src, dst, axis);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::rgb2gray(Tensor^ src, Tensor^ %dst, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, 1, src->Height, src->Width, device);
				rgb2gray_cpu(src, dst);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::copy_make_border(Tensor^ src, Tensor^ %dst, int top, int bottom,
			int left, int right, BorderType type, float v, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, src->Height + bottom + top, src->Width + left + right, device);
				copy_make_border_cpu(src, dst, top, bottom, left, right, type, v);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::copy_cut_border(Tensor^ src, Tensor^ %dst, int top, int bottom, 
			int left, int right, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, src->Height - top - bottom, src->Width - left - right, device);
				copy_cut_border_cpu(src, dst, top, bottom, left, right);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::draw_rectangle(Tensor^ %dst, rectangle^ rect, int thickness, color^ color_, int device)
		{
			if (device < 0)
			{
				draw_rectangle_cpu(dst, rect, thickness, color_);
			}
			else
			{
				// No implementation
				return;
			}
		}

		//
		void tensorcv::resize(UTensor^ src, UTensor^ %dst, int new_height,
			int new_width, InterpolationType type, int device)
		{
			if (device < 0)
			{
				dst = gcnew UTensor(src->Num, src->Channel, new_height, new_width, device);
				resize_cpu(src, dst, new_height, new_width, type);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::rotate(UTensor^ src, UTensor^ %dst, float theta,
			int center_x, int center_y, InterpolationType type, unsigned char v, int device)
		{
			if (device < 0)
			{
				dst = gcnew UTensor(src->Num, src->Channel, src->Height, src->Width, device);
				rotate_cpu(src, dst, theta, center_x, center_y, type, v);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::flip(UTensor^ src, UTensor^ %dst, FlipType axis, int device)
		{
			if (device < 0)
			{
				dst = gcnew UTensor(src->Num, src->Channel, src->Height, src->Width, device);
				flip_cpu(src, dst, axis);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::rgb2gray(UTensor^ src, UTensor^ %dst, int device)
		{
			if (device < 0)
			{
				dst = gcnew UTensor(src->Num, 1, src->Height, src->Width, device);
				rgb2gray_cpu(src, dst);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::copy_make_border(UTensor^ src, UTensor^ %dst, int top, int bottom,
			int left, int right, BorderType type, unsigned char v, int device)
		{
			if (device < 0)
			{
				dst = gcnew UTensor(src->Num, src->Channel, src->Height + bottom + top, src->Width + left + right, device);
				copy_make_border_cpu(src, dst, top, bottom, left, right, type, v);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::copy_cut_border(UTensor^ src, UTensor^ %dst, int top, int bottom,
			int left, int right, int device)
		{
			if (device < 0)
			{
				dst = gcnew UTensor(src->Num, src->Channel, src->Height - top - bottom, src->Width - left - right, device);
				copy_cut_border_cpu(src, dst, top, bottom, left, right);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::draw_rectangle(UTensor^ %dst, rectangle^ rect, int thickness, color^ color_, int device)
		{
			if (device < 0)
			{
				draw_rectangle_cpu(dst, rect, thickness, color_);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::convert2UTensor(Tensor^ src, UTensor^ %dst)
		{
			int device = src->Device;
			dst = gcnew UTensor(src->Num, src->Channel, src->Height, src->Width, device);
			if (device < 0)
			{
				tensoroperation::type_convertor_cpu(src->data->getdata(), dst->data->getdata());
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::convert2Tensor(UTensor^ src, Tensor^ %dst)
		{
			int device = src->Device;
			dst = gcnew Tensor(src->Num, src->Channel, src->Height, src->Width, device);
			if (device < 0)
			{
				tensoroperation::type_convertor_cpu(src->data->getdata(), dst->data->getdata());
			}
			else
			{
				// No implementation
				return;
			}
		}



		void tensorcv::equalize_hist(Tensor^ src, Tensor^ %dst, int device)
		{
			dst = gcnew Tensor(src->Num, src->Channel, src->Height, src->Width, device);
			if (device < 0)
			{
				tensoroperation::equalize_hist_cpu(src->data->getdata(), dst->data->getdata());
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::equalize_hist(UTensor^ src, UTensor^ %dst, int device)
		{
			dst = gcnew UTensor(src->Num, src->Channel, src->Height, src->Width, device);
			if (device < 0)
			{
				tensoroperation::equalize_hist_cpu(src->data->getdata(), dst->data->getdata());
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::lbp_feature(Tensor^ src, Tensor^ %dst, LbpType type, int device)
		{
			if (device < 0)
			{
				switch (type)
				{
				case glasssix::gilgamesh::LbpType::Native:
					dst = gcnew Tensor(src->Num, src->Channel, src->Height - 2, src->Width - 2, device);
					tensoroperation::lbp_feature_cpu(src->data->getdata(), dst->data->getdata(), lbpType::Native);
					break;
				case glasssix::gilgamesh::LbpType::RI:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::U2:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::RIU2:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::HF:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::LTP:
					NOT_IMPLEMENTED;
					break;
				default:
					LOG(ERROR) << "Un-supported LBP type.";
					break;
				}
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::lbp_feature(UTensor^ src, UTensor^ %dst, LbpType type, int device)
		{
			if (device < 0)
			{
				switch (type)
				{
				case glasssix::gilgamesh::LbpType::Native:
					dst = gcnew UTensor(src->Num, src->Channel, src->Height - 2, src->Width - 2, device);
					tensoroperation::lbp_feature_cpu(src->data->getdata(), dst->data->getdata(), lbpType::Native);
					break;
				case glasssix::gilgamesh::LbpType::RI:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::U2:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::RIU2:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::HF:
					NOT_IMPLEMENTED;
					break;
				case glasssix::gilgamesh::LbpType::LTP:
					NOT_IMPLEMENTED;
					break;
				default:
					LOG(ERROR) << "Un-supported LBP type.";
					break;
				}
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::mblbp_feature(Tensor^ src, Tensor^ %dst, int block_h,
			int block_w, int stride_h, int stride_w, int device)
		{
			dst = gcnew Tensor(src->Num, src->Channel, 
				(src->Height - 3 * block_h) / stride_h + 1, (src->Width - 3 * block_w) / stride_w + 1, device);
			if (device < 0)
			{
				tensoroperation::mblbp_feature_cpu(src->data->getdata(), dst->data->getdata(),
					block_h, block_w, stride_h, stride_w);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::mblbp_feature(UTensor^ src, UTensor^ %dst, int block_h,
			int block_w, int stride_h, int stride_w, int device)
		{
			dst = gcnew UTensor(src->Num, src->Channel,
				(src->Height - 3 * block_h) / stride_h + 1, (src->Width - 3 * block_w) / stride_w + 1, device);
			if (device < 0)
			{
				tensoroperation::mblbp_feature_cpu(src->data->getdata(), dst->data->getdata(),
					block_h, block_w, stride_h, stride_w);
			}
			else
			{
				// No implementation
				return;
			}
		}

		///PRIVATE FUNCTIONS
		void tensorcv::resize_cpu(Tensor^ src, Tensor^ %dst, int new_height,
			int new_width, InterpolationType type)
		{
			switch (type)
			{
			case glasssix::gilgamesh::InterpolationType::Nearest:
				tensoroperation::resize_cpu(src->data->getdata(), dst->data->getdata(),
					new_height, new_width, interpolationType::Nearest);
				break;
			case glasssix::gilgamesh::InterpolationType::Bilinear:
				tensoroperation::resize_cpu(src->data->getdata(),
					dst->data->getdata(),
					new_height, new_width, interpolationType::Bilinear);
				break;
			case glasssix::gilgamesh::InterpolationType::Cubic:
				tensoroperation::resize_cpu(src->data->getdata(), dst->data->getdata(),
					new_height, new_width, interpolationType::Cubic);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
			int center_x, int center_y, InterpolationType type, float v)
		{
			switch (type)
			{
			case glasssix::gilgamesh::InterpolationType::Nearest:
				tensoroperation::rotate_cpu(src->data->getdata(), dst->data->getdata(),
					theta, center_x, center_y, interpolationType::Nearest, v);
				break;
			case glasssix::gilgamesh::InterpolationType::Bilinear:
				tensoroperation::rotate_cpu(src->data->getdata(), dst->data->getdata(),
					theta, center_x, center_y, interpolationType::Bilinear, v);
				break;
			case glasssix::gilgamesh::InterpolationType::Cubic:
				tensoroperation::rotate_cpu(src->data->getdata(), dst->data->getdata(),
					theta, center_x, center_y, interpolationType::Cubic, v);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::flip_cpu(Tensor^ src, Tensor^ %dst, FlipType axis)
		{
			switch (axis)
			{
			case glasssix::gilgamesh::FlipType::C_Wise:
				tensoroperation::flip_cpu(src->data->getdata(), dst->data->getdata(), flipType::C_Wise);
				break;
			case glasssix::gilgamesh::FlipType::W_Wise:
				tensoroperation::flip_cpu(src->data->getdata(), dst->data->getdata(), flipType::W_Wise);
				break;
			case glasssix::gilgamesh::FlipType::H_Wise:
				tensoroperation::flip_cpu(src->data->getdata(), dst->data->getdata(), flipType::H_Wise);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::rgb2gray_cpu(Tensor^ src, Tensor^ %dst)
		{
			tensoroperation::rgb2gray_cpu(src->data->getdata(), dst->data->getdata());
		}

		void tensorcv::copy_make_border_cpu(Tensor^ src, Tensor^ %dst, int top, int bottom,
			int left, int right, BorderType type, float v)
		{
			switch (type)
			{
			case glasssix::gilgamesh::BorderType::Border_Constant:
				tensoroperation::copy_make_border_cpu(src->data->getdata(), dst->data->getdata(), top, bottom,
					left, right, borderType::Border_Constant, v);
				break;
			case glasssix::gilgamesh::BorderType::Border_Replicate:
				tensoroperation::copy_make_border_cpu(src->data->getdata(), dst->data->getdata(), top, bottom,
					left, right, borderType::Border_Replicate, v);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::copy_cut_border_cpu(Tensor^ src, Tensor^ %dst, int top, int bottom, int left, int right)
		{
			tensoroperation::copy_cut_border_cpu(src->data->getdata(), dst->data->getdata(), top, bottom, left, right);
		}

		void tensorcv::draw_rectangle_cpu(Tensor^ %dst, rectangle^ rect, int thickness, color^ color_)
		{
			tensoroperation::draw_rectangle_cpu(dst->data->getdata(), (rect->rect), thickness, (color_->c));
		}

		//
		void tensorcv::resize_cpu(UTensor^ src, UTensor^ %dst, int new_height,
			int new_width, InterpolationType type)
		{
			switch (type)
			{
			case glasssix::gilgamesh::InterpolationType::Nearest:
				tensoroperation::resize_cpu(src->data->getdata(), dst->data->getdata(),
					new_height, new_width, interpolationType::Nearest);
				break;
			case glasssix::gilgamesh::InterpolationType::Bilinear:
				tensoroperation::resize_cpu(src->data->getdata(),
					dst->data->getdata(),
					new_height, new_width, interpolationType::Bilinear);
				break;
			case glasssix::gilgamesh::InterpolationType::Cubic:
				tensoroperation::resize_cpu(src->data->getdata(), dst->data->getdata(),
					new_height, new_width, interpolationType::Cubic);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::rotate_cpu(UTensor^ src, UTensor^ %dst, float theta,
			int center_x, int center_y, InterpolationType type, unsigned char v)
		{
			switch (type)
			{
			case glasssix::gilgamesh::InterpolationType::Nearest:
				tensoroperation::rotate_cpu(src->data->getdata(), dst->data->getdata(),
					theta, center_x, center_y, interpolationType::Nearest, v);
				break;
			case glasssix::gilgamesh::InterpolationType::Bilinear:
				tensoroperation::rotate_cpu(src->data->getdata(), dst->data->getdata(),
					theta, center_x, center_y, interpolationType::Bilinear, v);
				break;
			case glasssix::gilgamesh::InterpolationType::Cubic:
				tensoroperation::rotate_cpu(src->data->getdata(), dst->data->getdata(),
					theta, center_x, center_y, interpolationType::Cubic, v);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::flip_cpu(UTensor^ src, UTensor^ %dst, FlipType axis)
		{
			switch (axis)
			{
			case glasssix::gilgamesh::FlipType::C_Wise:
				tensoroperation::flip_cpu(src->data->getdata(), dst->data->getdata(), flipType::C_Wise);
				break;
			case glasssix::gilgamesh::FlipType::W_Wise:
				tensoroperation::flip_cpu(src->data->getdata(), dst->data->getdata(), flipType::W_Wise);
				break;
			case glasssix::gilgamesh::FlipType::H_Wise:
				tensoroperation::flip_cpu(src->data->getdata(), dst->data->getdata(), flipType::H_Wise);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::rgb2gray_cpu(UTensor^ src, UTensor^ %dst)
		{
			tensoroperation::rgb2gray_cpu(src->data->getdata(), dst->data->getdata());
		}

		void tensorcv::copy_make_border_cpu(UTensor^ src, UTensor^ %dst, int top, int bottom,
			int left, int right, BorderType type, unsigned char v)
		{
			switch (type)
			{
			case glasssix::gilgamesh::BorderType::Border_Constant:
				tensoroperation::copy_make_border_cpu(src->data->getdata(), dst->data->getdata(), top, bottom,
					left, right, borderType::Border_Constant, v);
				break;
			case glasssix::gilgamesh::BorderType::Border_Replicate:
				tensoroperation::copy_make_border_cpu(src->data->getdata(), dst->data->getdata(), top, bottom,
					left, right, borderType::Border_Replicate, v);
				break;
			default:
				NOT_IMPLEMENTED;
				break;
			}
		}

		void tensorcv::copy_cut_border_cpu(UTensor^ src, UTensor^ %dst, int top, int bottom, int left, int right)
		{
			tensoroperation::copy_cut_border_cpu(src->data->getdata(), dst->data->getdata(), top, bottom, left, right);
		}

		void tensorcv::draw_rectangle_cpu(UTensor^ %dst, rectangle^ rect, int thickness, color^ color_)
		{
			tensoroperation::draw_rectangle_cpu(dst->data->getdata(), (rect->rect), thickness, (color_->c));
		}

	}
}