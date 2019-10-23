#include "arm/conv_arm.hpp"
#include "arm/conv_arm_func.hpp"

#ifdef __ARM_NEON
#include "arm_neon.h"
#endif
#include <iostream>

//#define __ARM_NEON

//#define F43

glasssix::excalibur::conv_arm::conv_arm(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device, bool int8_quantization)
	: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, -1, false)
{
	use_sgemm1x1 = false;
	conv3x3s2 = false;
	use_winograd3x3 = false;

	if (group_ == 1)
	{
		if (kernelSize_ == 1 && stride_ == 1)
		{
			//if (input_Channel_ >= 64 && output_Channel_ >= 64)
				use_sgemm1x1 = true;
		}
		else if (kernelSize_ == 3 && stride_ == 2)
		{
			conv3x3s2 = true;
		}
		else if (kernelSize_ == 3 && stride_ == 1)
		{
			if (input_Channel_ >= 16 && output_Channel_ >= 16)
				use_winograd3x3 = true;
		}
	}
}

void glasssix::excalibur::conv_arm::set_bias(float * bias)
{
	baseconv::set_bias(bias);
}

void glasssix::excalibur::conv_arm::set_weights(float * weights)
{
	baseconv::set_weights(weights);
	if (group_ == 1)
	{
		if (use_sgemm1x1)
		{
			conv1x1s1_sgemm_transform_kernel_neon(weights_, weights_transformed_, input_Channel_, output_Channel_);
		}
		else if (conv3x3s2)
		{
			conv3x3s2_transform_kernel_neon(weights_, weights_transformed_, input_Channel_, output_Channel_);
		}
		else if (use_winograd3x3)
		{

#ifdef __ARM_NEON
			conv3x3s1_winograd64_transform_kernel_neon5(weights_, weights_transformed_, input_Channel_, output_Channel_);
#else
#ifdef F43
			conv3x3s1_winograd43_transform_kernel_sse(weights_, weights_transformed_vec_, input_Channel_, output_Channel_);
#else
			conv3x3s1_winograd23_transform_kernel_sse(weights_, weights_transformed_, input_Channel_, output_Channel_);
#endif // F43
#endif // __ARM_NEON
					
		}

		conv_im2col_sgemm_transform_kernel_neon(weights_, weights_sgemm_, input_Channel_, output_Channel_, kernelSize_ * kernelSize_);
	}
}

void glasssix::excalibur::conv_arm::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	order_ = bottom->order();
	if (order_ == NCHW)
	{
		int n = bottom->num();
		int w = bottom->width();
		int h = bottom->height();
		int c = bottom->channels();
		std::shared_ptr<tensor<float> > bottom_bordered = bottom;

		if (pad_ > 0)
		{
			tensor_operation_cpu::make_border_cpu(bottom, bottom_bordered, pad_, pad_, pad_, pad_);
			w = bottom_bordered->width();
			h = bottom_bordered->height();
		}
		int outw = (w - kernelSize_) / stride_ + 1;
		int outh = (h - kernelSize_) / stride_ + 1;

		top.reset(new tensor<float>(std::vector<int> {n, output_Channel_, outh, outw }, -1, NCHW));

		if (group_ == 1)
		{
			//if (use_winograd3x3 && w <= 120 && h <= 120)
			if (use_winograd3x3)
			{
#ifdef __ARM_NEON
				conv3x3s1_winograd64_neon5(bottom_bordered, top, weights_transformed_, bias_, bias_term_);
#else
#ifdef F43
				conv3x3s1_winograd43_sse(bottom_bordered, top, weights_transformed_vec_, bias_, bias_term_);
#else
				conv3x3s1_winograd23_sse(bottom_bordered, top, weights_transformed_, bias_, bias_term_);
#endif // F43
#endif // __ARM_NEON			
			}
			else if (use_sgemm1x1)
			{
				conv1x1s1_sgemm_neon(bottom_bordered, top, weights_transformed_, bias_, bias_term_);
			}
			else if (kernelSize_ == 1 && stride_ == 2)
			{
				conv_im2col_sgemm_neon(bottom_bordered, top, weights_sgemm_, bias_, kernelSize_, kernelSize_, stride_, stride_, bias_term_);
			}
			else if (conv3x3s2)
			{

#ifdef __ARM_NEON
				if (outw >= 8 && outh >= 8)
				{
					conv3x3s2_packed_neon(bottom_bordered, top, weights_transformed_, bias_, bias_term_);
				}
				else
				{
					conv_im2col_sgemm_neon(bottom_bordered, top, weights_sgemm_, bias_, kernelSize_, kernelSize_, stride_, stride_, bias_term_);
				}
#else
				conv3x3s2_sse(bottom_bordered, top, weights_, bias_, bias_term_);
#endif // __ARM_NEON

			}
			else if (kernelSize_ == 3 && stride_ == 1)
			{

#ifdef __ARM_NEON
				conv3x3s1_neon(bottom_bordered, top, weights_, bias_, bias_term_);
#else
				conv3x3s1_sse(bottom_bordered, top, weights_, bias_, bias_term_);
#endif // __ARM_NEON
				
			}
			else if (kernelSize_ == 1 && stride_ == 1)
			{

#ifdef __ARM_NEON
				conv1x1s1_neon(bottom_bordered, top, weights_, bias_, bias_term_);
#else
				conv1x1s1_sse(bottom_bordered, top, weights_, bias_, bias_term_);
#endif // __ARM_NEON
				
			}
			else
			{
				conv_im2col_sgemm_neon(bottom_bordered, top, weights_sgemm_, bias_, kernelSize_, kernelSize_, stride_, stride_, bias_term_);
			}
		}
		else
		{
			if (group_ == input_Channel_ && group_ == output_Channel_)
			{
				if (kernelSize_ == 3)
				{
					if (stride_ == 1)
					{

#ifdef __ARM_NEON
						convdw3x3s1_neon(bottom_bordered, top, weights_, bias_, bias_term_);
#else
						convdw3x3s1_sse(bottom_bordered, top, weights_, bias_, bias_term_);
#endif // __ARM_NEON
						
					}						
					else if (stride_ == 2)
					{

#ifdef __ARM_NEON
						convdw3x3s2_neon(bottom_bordered, top, weights_, bias_, bias_term_);
#else
						convdw3x3s2_sse(bottom_bordered, top, weights_, bias_, bias_term_);
#endif // __ARM_NEON
						
					}
					else
						NOT_IMPLEMENTED;
				}
				else
					NOT_IMPLEMENTED;
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}
