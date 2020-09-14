#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/arm/operation_deconvolutiondepthwise_arm.hpp"
#include "../../../include/Excalibur/operation_make_border.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_deconvolutiondepthwise_arm<Dtype>::operation_deconvolutiondepthwise_arm(const operation_param & param) : operation_general_conv<Dtype>(param)
		{
		}

		template<typename Dtype>
		int operation_deconvolutiondepthwise_arm<Dtype>::init_weights()
		{
			this->input_channel_ = this->weight_data_size_ / this->group_ / (this->output_channel_ / this->group_) / (this->kernel_size_w_ * this->kernel_size_h_) *  this->group_;

			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (!this->params_.int8_quantization_)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < this->weight_data_size_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += this->weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < this->output_channel_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += this->output_channel_ * sizeof(float);
				}

				const float *p = this->weights_f32_[0]->cpu_data();
				int kernel_size = this->kernel_size_w_ * this->kernel_size_h_;
				if (this->output_channel_ == this->group_)
				{
					kernel_pack1_.reset(new memory::tensor<float>(std::vector<int>{this->input_channel_*this->output_channel_ * kernel_size / this->group_}, this->params_.device_));
					float *pt = kernel_pack1_->mutable_cpu_data();
					for (int i = 0; i < (this->input_channel_ / this->group_)*(this->output_channel_ / this->group_)*this->group_; i++)
					{
						for (int k = 0; k < kernel_size; k++)
						{
							pt[kernel_size - 1 - k] = p[k];
						}

						p += kernel_size;
						pt += kernel_size;
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			return mem;
		}

		template<typename Dtype>
		int operation_deconvolutiondepthwise_arm<Dtype>::init_weights(FILE * fp)
		{
			this->input_channel_ = this->weight_data_size_ / this->group_ / (this->output_channel_ / this->group_) / (this->kernel_size_w_ * this->kernel_size_h_) *  this->group_;
			int quantize_tag;
			fread(&quantize_tag, 1, sizeof(int), fp);
			int mem = 0;
			if (quantize_tag == 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, this->weight_data_size_ * sizeof(float), fp);
				mem += this->weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, this->output_channel_ * sizeof(float), fp);
					mem += this->output_channel_ * sizeof(float);
				}

				const float *p = this->weights_f32_[0]->cpu_data();
				int kernel_size = this->kernel_size_w_ * this->kernel_size_h_;
				if (this->output_channel_ == this->group_)
				{
					kernel_pack1_.reset(new memory::tensor<float>(std::vector<int>{this->input_channel_*this->output_channel_ * kernel_size / this->group_}, this->params_.device_));
					float *pt = kernel_pack1_->mutable_cpu_data();
					for (int i = 0; i < (this->input_channel_ / this->group_)*(this->output_channel_ / this->group_)*this->group_; i++)
					{
						for (int k = 0; k < kernel_size; k++)
						{
							pt[kernel_size - 1 - k] = p[k];
						}

						p += kernel_size;
						pt += kernel_size;
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			return mem;
		}

		template<typename Dtype>
		void operation_deconvolutiondepthwise_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);

			int n = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();
			int bottom_cstep = w * h;

			int kernel_size = this->kernel_size_w_ * this->kernel_size_h_;
			int outw = (w - 1) * this->stride_w_ + this->kernel_size_w_;
			int outh = (h - 1) * this->stride_h_ + this->kernel_size_h_;
			int top_cstep = outw * outh;

			tops[0].reset(new memory::tensor<float>(std::vector<int> {n, this->output_channel_, outh, outw }, -1, memory::NCHW));

			memory::orderType order = bottoms[0]->order();
			if (!((order == memory::NCHW) || (order == memory::NHWC)))
			{
				NOT_IMPLEMENTED;
			}

			if (order == memory::NHWC)
			{
				bottoms[0]->convert_order();
			}

			const float *bias_data = nullptr;
			if (this->bias_term_)
				bias_data = this->weights_f32_[1]->cpu_data();

			if (this->input_channel_ == this->group_ && this->output_channel_ == this->group_)
			{
				const float *kerenel_pack1_data = kernel_pack1_->cpu_data();
				for (int num_i = 0; num_i < n; num_i++)
				{
					float *top_data = tops[0]->mutable_cpu_data() + num_i * this->output_channel_ * top_cstep;
					const float *bottom_data = bottoms[0]->cpu_data() + num_i * this->input_channel_ * bottom_cstep;
#ifdef _OPENMP
#pragma omp parallel for
#endif
					for (int g = 0; g < this->input_channel_; g++)
					{
						float* outptr = top_data + g * top_cstep;
						const float* kptr = (const float*)kerenel_pack1_data + kernel_size * g;
						const float *inptr = bottom_data + g * bottom_cstep;

						for (int i = 0; i < outh; i++)
						{
							for (int j = 0; j < outw; j++)
							{
								float sum = 0.f;

								if (this->bias_term_)
								{
									sum = bias_data[g];
								}

								for (int y = 0; y < this->kernel_size_h_; y++)
								{
									int sys = (i + y - (this->kernel_size_h_ - 1));
									if (sys < 0 || sys % this->stride_h_ != 0)
										continue;

									int sy = sys / this->stride_h_;
									if (sy >= h)
										continue;

									const float* sptr = inptr + sy * w;

									for (int x = 0; x < this->kernel_size_w_; x++)
									{
										int sxs = (j + x - (this->kernel_size_w_ - 1));
										if (sxs < 0 || sxs % this->stride_w_ != 0)
											continue;

										int sx = sxs / this->stride_w_;
										if (sx >= w)
											continue;

										float val = sptr[sx];

										int k = y * this->stride_h_ + x;

										float w = kptr[k];

										sum += val * w;
									}
								}

								outptr[j] = sum;
							}

							outptr += outw;
						}
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			if (this->pad_bottom_ > 0 || this->pad_left_ > 0 || this->pad_right_ > 0 || this->pad_top_ > 0)
			{
				make_border(tops[0], tops[0], this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_);
			}
		}

		INSTANCE_CLASS(operation_deconvolutiondepthwise_arm);
//		REGISTE(operation_deconvolutiondepthwise_arm);
	}
}