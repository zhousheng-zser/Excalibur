#ifndef _QUANTIZELAYER_HPP_
#define _QUANTIZELAYER_HPP_

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <glasssix\tensor.hpp>

#define QUANTIZE_NUM 127
#define INTERVAL_NUM 2048
using namespace glasssix::excalibur;
using namespace std;

class QuantizeLayer
{
public:
	QuantizeLayer() = default;

	QuantizeLayer(std::string layer_name, std::string blob_name, int group)
	{
		layer_name_ = layer_name;
		blob_name_ = blob_name;
		group_ = group;

		weight_scale_tensor.reset(new tensor<float>(std::vector<int>{ group_ }));
		blob_distubution_tensor.reset(new tensor<float>(std::vector<int>{ INTERVAL_NUM }));

		weight_scale_ = weight_scale_tensor->mutable_cpu_data();
		blob_distubution_ = blob_distubution_tensor->mutable_cpu_data();
		
		memset(weight_scale_, 0, group_ * sizeof(float));
		memset(blob_distubution_, 0, INTERVAL_NUM * sizeof(float));
	}

	~QuantizeLayer() {}

	std::shared_ptr<tensor<float> > weight_scale_tensor;
	std::shared_ptr<tensor<float> > blob_distubution_tensor;
   

	float *weight_scale_;
	float *blob_distubution_;
	float blob_max_ = 0.0f;
	float blob_distubution_interval_ = 0.0f;	
	float blob_scale_ = 1.0f;
	
	std::string layer_name_;
	std::string blob_name_;
	int group_;

	void quantize_weight(const float* weight_data, int length)
	{
		if (length % group_ != 0)
		{
			std::cout << "weight_data length error!!!" << std::endl;
			return;
		}

		for (int g = 0; g < group_; g++)
		{
			float min_val = FLT_MAX;
			float max_val = FLT_MIN;
			int interval = length / group_;
			int offset = g * interval;
			for (int i = 0; i < interval; i++)
			{
				int index = offset + i;
				if (weight_data[index] < min_val)
				{
					min_val = weight_data[index];
				}
				if (weight_data[index] > max_val)
				{
					max_val = weight_data[index];
				}
			}

			float threshold = std::max(abs(max_val), abs(min_val));
			if (threshold < 0.0001)
			{
				weight_scale_[g] = 0;
			}
			else
			{
				weight_scale_[g] = QUANTIZE_NUM / threshold;
			}

			std::cout << "layer:" << layer_name_ << ", group:" << g << ", weight_scale:" << weight_scale_[g] << std::endl;
		}
	}


	void initial_blob_max(const float* blob_data, int length)//bottom_blob
	{
		float min_val = FLT_MAX;
		float max_val = FLT_MIN;

		for (int i = 0; i < length; i++)
		{
			if (blob_data[i] < min_val)
			{
				min_val = blob_data[i];
			}
			
			if (blob_data[i] > max_val)
			{
				max_val = blob_data[i];
			}
		}

		blob_max_ = std::max(blob_max_, std::max(abs(min_val),abs(max_val)));
	}


	void initial_blob_distubution_interval()
	{
		blob_distubution_interval_ = blob_max_ / INTERVAL_NUM;
		std::cout << std::setw(10) << layer_name_ 
			      << ",max_val:" << blob_max_
			      << ",interval:" << blob_distubution_interval_ << std::endl;
	}


	void initial_histograms(const float* blob_data, int length)//bottom_blob
	{
	    std::shared_ptr<tensor<float>> temp_distubution_tensor;
		temp_distubution_tensor.reset(new tensor<float>(std::vector<int>{INTERVAL_NUM}));
		float* temp_distubution = temp_distubution_tensor->mutable_cpu_data();
		memset(temp_distubution, 0, INTERVAL_NUM * sizeof(float));

		for (int i = 0; i < length; i++)
		{
			int index = blob_data[i] / blob_distubution_interval_;
			if (index >= 0)
			{
				if (index == INTERVAL_NUM)
				{
					index--;
				}
				temp_distubution[index]++;
			}
		}

		for (int i = 0; i < INTERVAL_NUM; i++)
		{
			blob_distubution_[i] += temp_distubution[i];
		}
	}


	void quantize_blob()
	{
		int length = INTERVAL_NUM - 1;
		int target_bin = 128;
		std::shared_ptr<tensor<float> > kl_divergence_tensor;
		kl_divergence_tensor.reset(new tensor<float>(std::vector<int>{length - target_bin}));
		float *kl_divergence = kl_divergence_tensor->mutable_cpu_data();
		memset(kl_divergence, 0, (length - target_bin) * sizeof(float));

		std::shared_ptr<tensor<float> > distribution_tensor;
		distribution_tensor.reset(new tensor<float>(std::vector<int>{length}));
		float *distribution = distribution_tensor->mutable_cpu_data();
		for (int i = 0; i < length; i++)
		{
			distribution[i] = blob_distubution_[i + 1];
		}

		float threshold_sum = 0;
		for (int i = target_bin; i < length; i++)
		{
			threshold_sum += distribution[i];
		}

		for (int threshold = target_bin; threshold < length; threshold++)
		{
			std::shared_ptr<tensor<int> > is_nonzeros_tensor;
			is_nonzeros_tensor.reset(new tensor<int>(std::vector<int>{threshold}));
			int *is_nonzeros = is_nonzeros_tensor->mutable_cpu_data();

			std::shared_ptr<tensor<float> > sliced_nd_hist_tensor;
			sliced_nd_hist_tensor.reset(new tensor<float>(std::vector<int>{threshold}));
			float *sliced_nd_hist = sliced_nd_hist_tensor->mutable_cpu_data();

			for (int i = 0; i < threshold; i++)
			{
				sliced_nd_hist[i] = distribution[i];
				if (sliced_nd_hist[i] != 0)
				{
					is_nonzeros[i] = 1;
				}
				else
				{
					is_nonzeros[i] = 0;
				}
			}
			sliced_nd_hist[threshold - 1] += threshold_sum;
			threshold_sum -= distribution[threshold];

			std::shared_ptr<tensor<float> > quantized_bins_tensor;
			quantized_bins_tensor.reset(new tensor<float>(std::vector<int>{target_bin}));
			float *quantized_bins = quantized_bins_tensor->mutable_cpu_data();
			memset(quantized_bins, 0, target_bin * sizeof(float));

			int num_merged_bins = threshold / target_bin;

			for (int i = 0; i < target_bin; i++)
			{
				int start = i * num_merged_bins;
				int stop = start + num_merged_bins;
				float sum = 0;
				for (int j = start; j < stop; j++)
				{
					sum += sliced_nd_hist[j];
				}
				quantized_bins[i] = sum;
			}

			for (int i = target_bin * num_merged_bins; i < threshold; i++)
			{
				quantized_bins[target_bin - 1] += sliced_nd_hist[i];
			}

			std::shared_ptr<tensor<float> > q_tensor;
			q_tensor.reset(new tensor<float>(std::vector<int>{threshold}));
			float *q = q_tensor->mutable_cpu_data();
			memset(q, 0, threshold * sizeof(float));

			for (int i = 0; i < target_bin; i++)
			{
				int start = i * num_merged_bins;
				int stop;
				if (i == target_bin - 1)
				{
					stop = threshold - 1;
				}
				else
				{
					stop = start + num_merged_bins;
				}

				int num = 0;
				for (int j = start; j < stop; j++)
				{
					num += is_nonzeros[j];
				}

				if (num)
				{
					for (int j = start; j < stop; j++)
					{
						q[j] = float(quantized_bins[i]) / num;
					}
				}
			}

			//smooth
			for (int i = 0; i < threshold; i++)
			{
				if (sliced_nd_hist[i] == 0)
				{
					sliced_nd_hist[i] = 0.0001;
				}

				if (q[i] == 0)
				{
					q[i] = 0.0001;
				}
			}

			float divergence = 0;
			for (int i = 0; i < threshold; i++)
			{
				divergence += sliced_nd_hist[i] * log(sliced_nd_hist[i] / q[i]);
			}

			kl_divergence[threshold - target_bin] = divergence;
		}

		float min = FLT_MAX;
		int min_pos = 0;
		for (int i = 0; i < length - target_bin; i++)
		{
			if (kl_divergence[i] < min)
			{
				min = kl_divergence[i];
				min_pos = i;
			}
		}

		int threshold_bin = min_pos + target_bin;
		blob_scale_ = QUANTIZE_NUM / (threshold_bin + 0.5f) / blob_distubution_interval_;
		std::cout << "layer:" << std::setw(10) << layer_name_ << ",bottom_scale:" << blob_scale_ << std::endl;
	}
};

#endif // !_QUANTIZELAYER_HPP_
