#include "arm/pooling_arm.hpp"
#include "tensor_operation_cpu.hpp"
#include "arm/pooling_arm_func.hpp" 
#include <cfloat>

void glasssix::excalibur::pooling_arm::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	order_ = bottom->order();

	if ((kernel_ != 2 && kernel_ != 3) || stride_ != 2 || order_ != NCHW || type_ != MAX)
	{
		pooling::Forward_cpu(bottom, top);
		return;
	}

	int num = bottom->num();
	int w = bottom->width();
	int h = bottom->height();
	int channels = bottom->channels();

	std::shared_ptr<tensor<float> > bottom_bordered = bottom;

	int wtailpad = 0, htailpad = 0;
	int wtail = (w + 2 * pad_ - kernel_) % stride_;
	int htail = (h + 2 * pad_ - kernel_) % stride_;
	
	if (wtail || htail)
	{
		if (wtail)
			wtailpad = stride_ - wtail;
		if (htail)
			htailpad = stride_ - htail;

		tensor_operation_cpu::make_border_cpu(bottom, bottom_bordered, pad_, pad_ + htailpad, pad_, pad_ + wtailpad, Border_Constant, -FLT_MAX);
		w = bottom_bordered->width();
		h = bottom_bordered->height();
	}

	int outw = (w - kernel_) / stride_ + 1;
	int outh = (h - kernel_) / stride_ + 1;

	top.reset(new tensor<float>(std::vector<int>{num, channels, outh, outw}, -1, NCHW));

	if (kernel_ == 2)
		pooling2x2s2_max_neon(bottom_bordered, top);
	else if (kernel_ == 3)
		pooling3x3s2_max_neon(bottom_bordered, top);
}
