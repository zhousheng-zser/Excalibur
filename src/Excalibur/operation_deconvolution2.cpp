#include "../../include/Excalibur/operation_deconvolution2.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/operation_cut_border.hpp"
#include "../../include/Excalibur/operation_make_border.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_deconvolution2<Dtype>::operation_deconvolution2(const operation_param& param) : operation_general_conv<Dtype>(param)
		{
		}

		template<typename Dtype>
		int operation_deconvolution2<Dtype>::init_weights(FILE* fp)
		{
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
				return mem;
			}
			else
			{
				NOT_IMPLEMENTED;
				return 0;
			}
		}

		template<typename Dtype>
		int operation_deconvolution2<Dtype>::init_weights()
		{
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
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			return mem;
		}

		template<typename Dtype>
		void operation_deconvolution2<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);

			int num = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();
			int channels = bottoms[0]->channels();

			const int kernel_extent_w = this->dilation_w_ * (this->kernel_size_w_ - 1) + 1;
			const int kernel_extent_h = this->dilation_h_ * (this->kernel_size_h_ - 1) + 1;

			int outw = (w - 1) * this->stride_w_ + kernel_extent_w;
			int outh = (h - 1) * this->stride_h_ + kernel_extent_h;

			std::shared_ptr<memory::tensor<float>> top_bordered;
			top_bordered.reset(new memory::tensor<float>(std::vector<int>{num, this->output_channel_, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

			const float* weight_data = this->weights_f32_[0]->cpu_data();
			const float* bias_data = this->weights_f32_[1]->cpu_data();

			const int maxk = this->kernel_size_w_ * this->kernel_size_h_;

			// kernel offsets
			std::vector<int> _space_ofs(maxk);
			int* space_ofs = &_space_ofs[0];
			{
				int p1 = 0;
				int p2 = 0;
				int gap = outw * this->dilation_h_ - this->kernel_size_w_ * this->dilation_w_;
				for (int i = 0; i < this->kernel_size_h_; i++)
				{
					for (int j = 0; j < this->kernel_size_w_; j++)
					{
						space_ofs[p1] = p2;
						p1++;
						p2 += this->dilation_w_;
					}
					p2 += gap;
				}
			}

			for (int n = 0; n < num; n++)
			{
				const float* bottom_data = bottoms[0]->cpu_data() + bottoms[0]->offset(n);
				float* top_bordered_data = top_bordered->mutable_cpu_data() + top_bordered->offset(n);

				// num_output
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = 0; p < this->output_channel_; p++)
				{
					float* out = top_bordered_data + p * outw * outh;

					const float bias = this->bias_term_ ? bias_data[p] : 0.f;

					for (int i = 0; i < outw * outh; i++)
					{
						out[i] = bias;
					}

					for (int i = 0; i < h; i++)
					{
						for (int j = 0; j < w; j++)
						{
							float* outptr = out + (i * this->stride_h_) * outw + j * this->stride_w_;

							const float* kptr = weight_data + maxk * channels * p;

							// channels
							for (int q = 0; q < channels; q++)
							{
								const float* m = bottom_data + (q)*w * h;
								float val = *(m + (i)*w + j);

								for (int k = 0; k < maxk; k++)
								{
									float w = kptr[k];
									outptr[space_ofs[k]] += val * w;
								}

								kptr += maxk;
							}
						}
					}
				}
			}
			if (this->pad_left_ > 0 || this->pad_right_ > 0 || this->pad_top_ > 0 || this->pad_bottom_ > 0)
			{
				if (this->output_pad_bottom_ > 0 || this->output_pad_right_ > 0)
					make_border(top_bordered, top_bordered, 0, this->output_pad_bottom_, 0, this->output_pad_right_);
				cut_border_cpu(top_bordered, tops[0], this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_);
			}
			else if (this->output_dim_h_ > 0 && this->output_dim_w_ > 0)
			{
				if (this->output_pad_right_ > 0 || this->output_pad_bottom_ > 0)
					make_border(top_bordered, top_bordered, 0, this->output_pad_bottom_, 0, this->output_pad_right_);

				int wcut = top_bordered->width() - this->output_dim_w_;
				int hcut = top_bordered->height() - this->output_dim_h_;

				if (this->pad_left_ == -233 || this->pad_right_ == -233 || this->pad_top_ == -233 || this->pad_bottom_ == -233)
				{
					// onnx padding=SAME_UPPER
					cut_border_cpu(top_bordered, tops[0], hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2);
				}
				else if (this->pad_left_ == -234 || this->pad_right_ == -234 || this->pad_top_ == -234 || this->pad_bottom_ == -234)
				{
					// onnx padding=SAME_LOWER
					cut_border_cpu(top_bordered, tops[0], hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2);
				}
			}
			else
			{
				if (this->output_pad_right_ > 0 || this->output_pad_bottom_ > 0)
					make_border(top_bordered, tops[0], 0, this->output_pad_bottom_, 0, this->output_pad_right_);
				else
					tops[0] = top_bordered;
			}
		}


#ifndef USE_CUDA
		STUB_GPU(operation_deconvolution2);
#endif

		INSTANCE_CLASS(operation_deconvolution2);
		REGISTE(operation_deconvolution2);
	}
}