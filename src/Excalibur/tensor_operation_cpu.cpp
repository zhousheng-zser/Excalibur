#include "../../include/Excalibur/tensor_operation_cpu.hpp"
using namespace glasssix::excalibur;
using namespace glasssix::memory;

const unsigned char LBPMAP[5][256] =
{
	//59 mapping
	{ 1,   2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
	12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
	17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
	23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
	30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
	37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
	43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
	48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58 },

	//63mapping small face no overlap feature
	{ 0,   14,  51,  30,  5,   0,   47,  26,  52,  0,   0,   0,   49,  0,   58,  37,
	15,  0,   0,   0,   0,   0,   0,   0,   31,  0,   0,   0,   29,  0,   41,
	56,  34,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,   0,   20,  0,   0,   0,   0,   0,   0,   0,   13,  0,   0,   0,   7,
	0,   46,  57,  2,   61,  0,   0,   22,  0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   60,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   0,   24,  0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   0,   0,   8,   0,   0,   0,   0,   0,   0,   0,   3,
	0,   0,   0,   10,  0,   45,  54,  32,  19,  0,   9,   0,   0,   0,   6,
	0,   0,   0,   0,   0,   0,   0,   43,  0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   0,   0,   0,   0,   55,  0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,   0,   0,   0,   0,   0,   59,  0,   0,   0,   17,  23,  12,  0,   4,
	0,   62,  0,   11,  0,   0,   0,   0,   0,   0,   0,   44,  0,   0,   0,
	0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   53,  33,  21,
	0,   36,  0,   0,   0,   50,  0,   0,   0,   0,   0,   0,   0,   28,  18,
	25,  0,   38,  0,   0,   0,   40,  35,  39,  0,   16,  48,  42,  27,  1 },

	//63mapping big face no overlap feature
	{ 0,    5,    44,   27,   9,    0,    43,   33,   47,   0,    0,    0,    42,   0,    61,
	46,   4,    56,   0,    0,    0,    0,    0,    0,    26,   0,    0,    0,    31,   0,
	45,   51,   28,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
	0,    0,    0,    22,   0,    0,    0,    0,    0,    0,    0,    16,   0,    0,    0,
	15,   0,    53,   58,   2,    59,   0,    0,    48,   0,    0,    0,    0,    0,    0,
	0,    0,    0,    0,    0,    57,   0,    0,    0,    0,    0,    0,    0,    62,   0,
	0,    0,    0,    0,    0,    0,    17,   0,    0,    0,    0,    0,    0,    0,    0,
	0,    0,    0,    0,    0,    0,    0,    7,    0,    0,    0,    0,    0,    0,    0,
	3,    0,    0,    0,    12,   0,    49,   54,   32,   23,   0,    18,   0,    0,    0,
	14,   0,    0,    0,    0,    0,    0,    0,    52,   0,    0,    0,    0,    0,    0,
	0,    0,    0,    0,    0,    0,    0,    0,    0,    60,   0,    0,    0,    0,    0,
	0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
	0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    37,   20,   8,    0,
	6,    0,    0,    0,    13,   0,    0,    0,    0,    0,    0,    0,    50,   0,    0,
	0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    55,   24,
	11,   0,    34,   0,    0,    0,    39,   0,    0,    0,    0,    0,    0,    0,    21,
	10,   29,   0,    41,   0,    0,    0,    36,   30,   40,   0,    25,   38,   35,   19,   1 },

	//63mapping small face overlap feature
	{ 1,     3,     43,     21,     1,     0,      31,     12,     44,     0,      0,      0,      33,     0,      40,
	19,     4,     0,      0,      0,      0,      0,      0,      0,      24,     0,      0,      0,      13,     0,
	20,     38,     36,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      8,     0,      0,      0,      0,      0,      0,      0,      1,     0,      0,      0,
	1,     0,      26,     39,     1,     49,     0,      0,      17,     0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      48,     0,      0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      11,     0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      1,     0,      0,      0,      0,      0,      0,      0,
	1,     0,      0,      0,      1,     0,      30,     46,     35,     7,     0,      1,     0,      0,      0,
	1,     0,      0,      0,      0,      0,      0,      0,      22,     0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      37,     0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      47,     0,      0,      0,      10,     9,     1,     0,
	1,     50,     0,      0,      1,     0,      0,      0,      0,      0,      0,      0,      29,     0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      45,     18,
	5,     0,      16,     0,      0,      0,      34,     0,      0,      0,      0,      0,      0,      0,      25,
	2,     15,     0,      27,     0,      0,      0,      41,     14,     28,     0,      6,     32,     42,     23, 	1 },

	//63mapping big face overlap feature
	{ 1,     1,     43,     17,     1,     0,      29,     21,     44,     0,      0,      0,      28,     0,      42,
	24,     1,     47,     0,      0,      0,      0,      0,      0,      14,     0,      0,      0,      18,     0,
	23,     38,     34,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      11,     0,      0,      0,      0,      0,      0,      0,      5,     0,      0,      0,
	3,     0,      26,     45,     1,     49,     0,      0,      39,     0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      48,     0,      0,      0,      0,      0,      0,      0,      52,     0,
	0,      0,      0,      0,      0,      0,      6,     0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      1,     0,      0,      0,      0,      0,      0,      0,
	1,     0,      0,      0,      4,     0,      35,     50,     36,     16,     0,      8,     0,      0,      0,
	2,     0,      0,      0,      0,      0,      0,      0,      27,     0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      46,     0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      31,     9,     1,     0,
	1,     0,      0,      0,      7,     0,      0,      0,      0,      0,      0,      0,      37,     0,      0,
	0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      51,     10,
	1,     0,      19,     0,      0,      0,      30,     0,      0,      0,      0,      0,      0,      0,      15,
	1,     22,     0,      33,     0,      0,      0,      41,     13,     32,     0,      20,     25,     40,     12, 	1 },
};


/// <summary>
/// convert between different datatype of tensor
/// </summary>
/// <param name="src">original tensor</param>
/// <param name="dst">new tensor</param>
template <typename DtypeSRC, typename DtypeDST>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const DtypeSRC* src_data = src->cpu_data();
	std::shared_ptr<tensor<DtypeDST>> dst_temp;
	dst_temp.reset(new tensor<DtypeDST>(src->data_shape(), src->device(), src->order()));
	DtypeDST* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

	for (int i = 0; i < length; i++)
	{
		dst_data[i] = DtypeDST(src_data[i]);
	}

	dst = std::make_shared<tensor<DtypeDST>>(dst_temp->clone());
}



/// <summary>
/// convert between different datatype of tensor
/// </summary>
/// <param name="src">original tensor</param>
/// <param name="dst">new tensor</param>
template <typename DtypeSRC, typename DtypeDST>
void tensor_operation_cpu::type_converter_cpu(const tensor<DtypeSRC> &src, tensor<DtypeDST> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const DtypeSRC* src_data = src.cpu_data();
	tensor<DtypeDST> dst_temp = tensor<DtypeDST>(src.data_shape(), src.device(), src.order());
	DtypeDST* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

	for (int i = 0; i < length; i++)
	{
		dst_data[i] = DtypeDST(src_data[i]);
	}

	dst = dst_temp.clone();
}



/// <summary>
/// preprocess tensor
/// </summary>
/// <param name="src">original tensor</param>
/// <param name="dst">new tensor</param>
template <typename DtypeSRC, typename DtypeDST>
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst, float means[3], float var)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src->num();
	int channel = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	std::shared_ptr<tensor<DtypeDST>> dst_temp;
	dst_temp.reset(new tensor<DtypeDST>(src->data_shape(), src->device(), src->order()));
	DtypeDST* dst_data = dst_temp->mutable_cpu_data();
	const DtypeSRC* src_data = src->cpu_data();

	if (channel == 3)
	{
		if (src->order() == NCHW)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								DtypeDST((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
		}
		else if (src->order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								DtypeDST((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						DtypeDST((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
	}

	dst = std::make_shared<tensor<DtypeDST>>(dst_temp->clone());
}



/// <summary>
/// preprocess tensor
/// </summary>
/// <param name="src">original tensor</param>
/// <param name="dst">new tensor</param>
template <typename DtypeSRC, typename DtypeDST>
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<DtypeSRC> &src, tensor<DtypeDST> &dst, float means[3], float var)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src.num();
	int channel = src.channels();
	int height = src.height();
	int width = src.width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	tensor<DtypeDST> dst_temp = tensor<DtypeDST>(src.data_shape(), src.device(), src.order());
	DtypeDST* dst_data = dst_temp.mutable_cpu_data();
	const DtypeSRC* src_data = src.cpu_data();

	if (channel == 3)
	{
		if (src.order() == NCHW)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								DtypeDST((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
		}
		else if (src.order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								DtypeDST((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						DtypeDST((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
	}

	dst = dst_temp.clone();
}



template<>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<int>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const float* src_data = src->cpu_data();
	std::shared_ptr<tensor<int>> dst_temp;
	dst_temp.reset(new tensor<int>(src->data_shape(), src->device(), src->order()));
	int* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_float32 = mm_load_ps(src_data + index * mm_align_size);
		temp_int32 = mm_cvtps_epi32(temp_float32);
		mm_store_si((mm_typei*)(dst_data + index * mm_align_size), temp_int32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = int(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = int(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = std::make_shared<tensor<int>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::type_converter_cpu(const tensor<float> &src, tensor<int> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const float* src_data = src.cpu_data();
	tensor<int> dst_temp = tensor<int>(src.data_shape(), src.device(), src.order());
	int* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_float32 = mm_load_ps(src_data + index * mm_align_size);
		temp_int32 = mm_cvtps_epi32(temp_float32);
		mm_store_si((mm_typei*)(dst_data + index * mm_align_size), temp_int32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = int(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = int(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<float>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const int* src_data = src->cpu_data();
	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_int32 = mm_load_si((mm_typei const *)(src_data + index * mm_align_size));
		temp_float32 = mm_cvtepi32_ps(temp_int32);
		mm_store_ps((float*)(dst_data + index * mm_align_size), temp_float32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = float(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = float(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::type_converter_cpu(const tensor<int> &src, tensor<float> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const int* src_data = src.cpu_data();
	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_int32 = mm_load_si((mm_typei const *)(src_data + index * mm_align_size));
		temp_float32 = mm_cvtepi32_ps(temp_int32);
		mm_store_ps((float*)(dst_data + index * mm_align_size), temp_float32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = float(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = float(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<int>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const unsigned char* src_data = src->cpu_data();
	std::shared_ptr<tensor<int>> dst_temp;
	dst_temp.reset(new tensor<int>(src->data_shape(), src->device(), src->order()));
	int* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_uint8;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepu8_epi32(temp_uint8);
		mm_store_si((mm_typei*)(dst_data + index * mm_align_size), temp_int32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = int(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = int(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = std::make_shared<tensor<int>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::type_converter_cpu(const tensor<unsigned char> &src, tensor<int> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const unsigned char* src_data = src.cpu_data();
	tensor<int> dst_temp = tensor<int>(src.data_shape(), src.device(), src.order());
	int* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_uint8;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepu8_epi32(temp_uint8);
		mm_store_si((mm_typei*)(dst_data + index * mm_align_size), temp_int32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = int(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = int(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const unsigned char* src_data = src->cpu_data();
	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_uint8;
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepu8_epi32(temp_uint8);
		temp_float32 = mm_cvtepi32_ps(temp_int32);
		mm_store_ps((float*)(dst_data + index * mm_align_size), temp_float32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = float(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = float(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::type_converter_cpu(const tensor<unsigned char> &src, tensor<float> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const unsigned char* src_data = src.cpu_data();
	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_uint8;
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepu8_epi32(temp_uint8);
		temp_float32 = mm_cvtepi32_ps(temp_int32);
		mm_store_ps((float*)(dst_data + index * mm_align_size), temp_float32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = float(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = float(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<signed char>> &src, std::shared_ptr<tensor<int>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const signed char* src_data = src->cpu_data();
	std::shared_ptr<tensor<int>> dst_temp;
	dst_temp.reset(new tensor<int>(src->data_shape(), src->device(), src->order()));
	int* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_int8;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepi8_epi32(temp_int8);
		mm_store_si((mm_typei*)(dst_data + index * mm_align_size), temp_int32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = int(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = int(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = std::make_shared<tensor<int>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::type_converter_cpu(const tensor<signed char> &src, tensor<int> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const signed char* src_data = src.cpu_data();
	tensor<int> dst_temp = tensor<int>(src.data_shape(), src.device(), src.order());
	int* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_int8;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepi8_epi32(temp_int8);
		mm_store_si((mm_typei*)(dst_data + index * mm_align_size), temp_int32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = int(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = int(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::type_converter_cpu(const std::shared_ptr<tensor<signed char>> &src, std::shared_ptr<tensor<float>> &dst)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const signed char* src_data = src->cpu_data();
	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	int length = src->count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_int8;
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepi8_epi32(temp_int8);
		temp_float32 = mm_cvtepi32_ps(temp_int32);
		mm_store_ps((float*)(dst_data + index * mm_align_size), temp_float32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = float(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = float(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::type_converter_cpu(const tensor<signed char> &src, tensor<float> &dst)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	const signed char* src_data = src.cpu_data();
	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	int length = src.count();

#if SIMD_TYPE >= SIMDTYPE_SSE
	__m128i temp_int8;
	mm_type temp_float32;
	mm_typei temp_int32;
	int circle_num = length / mm_align_size;
	int index = 0;

	for (; index < circle_num; index++)
	{
		temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
		temp_int32 = mm_cvtepi8_epi32(temp_int8);
		temp_float32 = mm_cvtepi32_ps(temp_int32);
		mm_store_ps((float*)(dst_data + index * mm_align_size), temp_float32);
	}

	for (index *= mm_align_size; index < length; index++)
	{
		dst_data[index] = float(src_data[index]);
	}
#else
	for (int i = 0; i < length; i++)
	{
		dst_data[i] = float(src_data[i]);
	}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src->num();
	int channel = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	const float* src_data = src->cpu_data();

	if (channel == 3)
	{
		if (src->order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			mm_type temp_float32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_float32 = mm_load_ps((const float*)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = (src_data[n_offset + ch_offset + index] - means[ch]) * var;
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								(src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var;
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src->order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								(src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var;
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{

#if SIMD_TYPE >= SIMDTYPE_SSE
		mm_type temp_float32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_float32 = mm_load_ps((const float*)(src_data + n_offset + index * mm_align_size));
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = (src_data[n_offset + index] - means[0]) * var;
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						(src_data[offset + subsub_offset + w] - means[0]) * var;
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<float> &src, tensor<float> &dst, float means[3], float var)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src.num();
	int channel = src.channels();
	int height = src.height();
	int width = src.width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	const float* src_data = src.cpu_data();

	if (channel == 3)
	{
		if (src.order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			mm_type temp_float32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_float32 = mm_load_ps((const float*)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = (src_data[n_offset + ch_offset + index] - means[ch]) * var;
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								(src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var;
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src.order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								(src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var;
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{
#if SIMD_TYPE >= SIMDTYPE_SSE
		mm_type temp_float32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_float32 = mm_load_ps((const float*)(src_data + n_offset + index * mm_align_size));
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = (src_data[n_offset + index] - means[0]) * var;
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						(src_data[offset + subsub_offset + w] - means[0]) * var;
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src->num();
	int channel = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	const int* src_data = src->cpu_data();

	if (channel == 3)
	{
		if (src->order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			mm_type temp_float32;
			mm_typei temp_int32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_int32 = mm_load_si((mm_typei const *)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_float32 = mm_cvtepi32_ps(temp_int32);
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = float((src_data[n_offset + ch_offset + index] - means[ch]) * var);
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								float((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src->order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								float((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{
#if SIMD_TYPE >= SIMDTYPE_SSE
		mm_type temp_float32;
		mm_typei temp_int32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_int32 = mm_load_si((mm_typei const *)(src_data + n_offset + index * mm_align_size));
				temp_float32 = mm_cvtepi32_ps(temp_int32);
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = float((src_data[n_offset + index] - means[0]) * var);
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						float((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<int> &src, tensor<float> &dst, float means[3], float var)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src.num();
	int channel = src.channels();
	int height = src.height();
	int width = src.width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	const int* src_data = src.cpu_data();

	if (channel == 3)
	{
		if (src.order() == NCHW)
		{


#if SIMD_TYPE >= SIMDTYPE_SSE
			mm_type temp_float32;
			mm_typei temp_int32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_int32 = mm_load_si((mm_typei const *)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_float32 = mm_cvtepi32_ps(temp_int32);
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = float((src_data[n_offset + ch_offset + index] - means[ch]) * var);
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								float((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src.order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								float((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{

#if SIMD_TYPE >= SIMDTYPE_SSE
		mm_type temp_float32;
		mm_typei temp_int32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_int32 = mm_load_si((mm_typei const *)(src_data + n_offset + index * mm_align_size));
				temp_float32 = mm_cvtepi32_ps(temp_int32);
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = float((src_data[n_offset + index] - means[0]) * var);
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						float((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src->num();
	int channel = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	const unsigned char* src_data = src->cpu_data();

	if (channel == 3)
	{

		if (src->order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			__m128i temp_uint8;
			mm_type temp_float32;
			mm_typei temp_int32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_int32 = mm_cvtepu8_epi32(temp_uint8);
						temp_float32 = mm_cvtepi32_ps(temp_int32);
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = float((src_data[n_offset + ch_offset + index] - means[ch]) * var);
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								(float(src_data[offset + sub_offset + subsub_offset + w]) - means[c]) * var;
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src->order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								float((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{

#if SIMD_TYPE >= SIMDTYPE_SSE
		__m128i temp_uint8;
		mm_type temp_float32;
		mm_typei temp_int32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
				temp_int32 = mm_cvtepu8_epi32(temp_uint8);
				temp_float32 = mm_cvtepi32_ps(temp_int32);
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = float((src_data[n_offset + index] - means[0]) * var);
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						float((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<unsigned char> &src, tensor<float> &dst, float means[3], float var)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src.num();
	int channel = src.channels();
	int height = src.height();
	int width = src.width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	const unsigned char* src_data = src.cpu_data();

	if (channel == 3)
	{

		if (src.order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			__m128i temp_uint8;
			mm_type temp_float32;
			mm_typei temp_int32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_int32 = mm_cvtepu8_epi32(temp_uint8);
						temp_float32 = mm_cvtepi32_ps(temp_int32);
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = float((src_data[n_offset + ch_offset + index] - means[ch]) * var);
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								float((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src.order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								float((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{

#if SIMD_TYPE >= SIMDTYPE_SSE
		__m128i temp_uint8;
		mm_type temp_float32;
		mm_typei temp_int32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_uint8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
				temp_int32 = mm_cvtepu8_epi32(temp_uint8);
				temp_float32 = mm_cvtepi32_ps(temp_int32);
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = float((src_data[n_offset + index] - means[0]) * var);
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						float((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = dst_temp.clone();
}


template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<signed char>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var)
{
	if (src->device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src->num();
	int channel = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	std::shared_ptr<tensor<float>> dst_temp;
	dst_temp.reset(new tensor<float>(src->data_shape(), src->device(), src->order()));
	float* dst_data = dst_temp->mutable_cpu_data();
	const signed char* src_data = src->cpu_data();

	if (channel == 3)
	{

		if (src->order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			__m128i temp_int8;
			mm_type temp_float32;
			mm_typei temp_int32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_int32 = mm_cvtepi8_epi32(temp_int8);
						temp_float32 = mm_cvtepi32_ps(temp_int32);
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = float((src_data[n_offset + ch_offset + index] - means[ch]) * var);
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								float((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src->order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								float((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{

#if SIMD_TYPE >= SIMDTYPE_SSE
		__m128i temp_int8;
		mm_type temp_float32;
		mm_typei temp_int32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
				temp_int32 = mm_cvtepi8_epi32(temp_int8);
				temp_float32 = mm_cvtepi32_ps(temp_int32);
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = float((src_data[n_offset + index] - means[0]) * var);
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						float((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = std::make_shared<tensor<float>>(dst_temp->clone());
}

template<>
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<signed char> &src, tensor<float> &dst, float means[3], float var)
{
	if (src.device() >= 0)
	{
		LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
		return;
	}

	int num = src.num();
	int channel = src.channels();
	int height = src.height();
	int width = src.width();
	int offset = height * width;
	int num_offset = channel * height * width;

	if (!(channel == 1 || channel == 3))
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	tensor<float> dst_temp = tensor<float>(src.data_shape(), src.device(), src.order());
	float* dst_data = dst_temp.mutable_cpu_data();
	const signed char* src_data = src.cpu_data();

	if (channel == 3)
	{

		if (src.order() == NCHW)
		{
#if SIMD_TYPE >= SIMDTYPE_SSE
			__m128i temp_int8;
			mm_type temp_float32;
			mm_typei temp_int32;
			mm_type mean_float32[] = { mm_set1_ps(means[0]), mm_set1_ps(means[1]), mm_set1_ps(means[2]) };
			mm_type var_float32 = mm_set1_ps(var);
			int circle_num = offset / mm_align_size;
			int index = 0;

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				for (int ch = 0; ch < 3; ch++)
				{
					int ch_offset = ch * offset;
					index = 0;

					for (; index < circle_num; index++)
					{
						temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + n_offset + ch_offset + index * mm_align_size));
						temp_int32 = mm_cvtepi8_epi32(temp_int8);
						temp_float32 = mm_cvtepi32_ps(temp_int32);
						temp_float32 = mm_sub_ps(temp_float32, mean_float32[ch]);
						temp_float32 = mm_mul_ps(temp_float32, var_float32);
						mm_store_ps((float*)(dst_data + n_offset + ch_offset + index * mm_align_size), temp_float32);
					}

					for (index *= mm_align_size; index < offset; index++)
					{
						dst_data[n_offset + ch_offset + index] = float((src_data[n_offset + ch_offset + index] - means[ch]) * var);
					}
				}
			}
#else
			for (int n = 0; n < num; n++)
			{
				int offset = n * channel * height * width;
				for (int c = 0; c < 3; c++)
				{
					int sub_offset = c * height * width;
					for (int h = 0; h < height; h++)
					{
						int subsub_offset = h * width;
						for (int w = 0; w < width; w++)
						{
							dst_data[offset + sub_offset + subsub_offset + w] =
								float((src_data[offset + sub_offset + subsub_offset + w] - means[c]) * var);
						}
					}
				}
			}
#endif //!SIMD_TYPE >= SIMDTYPE_SSE

		}
		else if (src.order() == NHWC)
		{
			for (int n = 0; n < num; n++)
			{
				int offset = n * 3 * height * width;
				for (int h = 0; h < height; h++)
				{
					int sub_offset = h * width * 3;
					for (int w = 0; w < width; w++)
					{
						int subsub_offset = w * 3;
						for (int c = 0; c < 3; c++)
						{
							dst_data[offset + sub_offset + subsub_offset + c] =
								float((src_data[offset + sub_offset + subsub_offset + c] - means[c]) * var);
						}
					}
				}
			}
		}
		else
		{
			NOT_IMPLEMENTED;
		}
	}
	else if (channel == 1)
	{

#if SIMD_TYPE >= SIMDTYPE_SSE
		__m128i temp_int8;
		mm_type temp_float32;
		mm_typei temp_int32;
		mm_type mean_float32 = mm_set1_ps(means[0]);
		mm_type var_float32 = mm_set1_ps(var);
		int circle_num = offset / mm_align_size;
		int index = 0;

		for (int n = 0; n < num; n++)
		{
			int n_offset = n * num_offset;
			index = 0;
			for (; index < circle_num; index++)
			{
				temp_int8 = _mm_loadl_epi64((__m128i const*)(src_data + index * mm_align_size));
				temp_int32 = mm_cvtepi8_epi32(temp_int8);
				temp_float32 = mm_cvtepi32_ps(temp_int32);
				temp_float32 = mm_sub_ps(temp_float32, mean_float32);
				temp_float32 = mm_mul_ps(temp_float32, var_float32);
				mm_store_ps((float*)(dst_data + n_offset + index * mm_align_size), temp_float32);
			}

			for (index *= mm_align_size; index < offset; index++)
			{
				dst_data[n_offset + index] = float((src_data[n_offset + index] - means[0]) * var);
			}
		}
#else
		for (int n = 0; n < num; n++)
		{
			int offset = n * height * width;
			for (int h = 0; h < height; h++)
			{
				int subsub_offset = h * width;
				for (int w = 0; w < width; w++)
				{
					dst_data[offset + subsub_offset + w] =
						float((src_data[offset + subsub_offset + w] - means[0]) * var);
				}
			}
		}
#endif//!SIMD_TYPE >= SIMDTYPE_SSE
	}

	dst = dst_temp.clone();
}
