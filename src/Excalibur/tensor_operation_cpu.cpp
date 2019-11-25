#include "../../include/Excalibur/tensor_operation_cpu.hpp"
using namespace glasssix::excalibur;


/// <summary>
/// convert between different datatype of tensor
/// </summary>
/// <param name="src">original tensor</param>
/// <param name="dst">new tensor</param>
template <typename DtypeSRC, typename DtypeDST>
void type_converter_cpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst)
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
void type_converter_cpu(const tensor<DtypeSRC> &src, tensor<DtypeDST> &dst)
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
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<DtypeSRC> &src, tensor<DtypeDST> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
		temp_uint8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_uint8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_uint8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_uint8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_int8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_int8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_int8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
		temp_int8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<float> &src, tensor<float> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<float>> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<int> &src, tensor<float> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
						temp_uint8 = _mm_loadu_si64((void const*)(src_data + n_offset + ch_offset + index * mm_align_size));
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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
				temp_uint8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<unsigned char> &src, tensor<float> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
						temp_uint8 = _mm_loadu_si64((void const*)(src_data + n_offset + ch_offset + index * mm_align_size));
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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
				temp_uint8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
void tensor_operation_cpu::preprocess_tensors_cpu(const std::shared_ptr<tensor<signed char>> &src, std::shared_ptr<tensor<float>> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
						temp_int8 = _mm_loadu_si64((void const*)(src_data + n_offset + ch_offset + index * mm_align_size));
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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
				temp_int8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
void tensor_operation_cpu::preprocess_tensors_cpu(const tensor<signed char> &src, tensor<float> &dst)
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
		float means[] = { 104.f, 117.0f, 124.f };
		float var = 0.0078125f;

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
						temp_int8 = _mm_loadu_si64((void const*)(src_data + n_offset + ch_offset + index * mm_align_size));
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
		float means[] = { 127.5f };
		float var = 0.0078125f;

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
				temp_int8 = _mm_loadu_si64((void const*)(src_data + index * mm_align_size));
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
