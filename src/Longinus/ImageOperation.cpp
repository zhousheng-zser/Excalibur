#include "ImageOperation.hpp"
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#define ICV_WARP_SHIFT          10
#define ICV_WARP_SHIFT2         15
#define ICV_SHIFT_DIFF          (ICV_WARP_SHIFT2-ICV_WARP_SHIFT)
#define ICV_WARP_MASK           ((1 << ICV_WARP_SHIFT) - 1)

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#define ICV_WARP_MUL_ONE_8U(x)  ((x) << ICV_WARP_SHIFT)
#define ICV_WARP_DESCALE_8U(x)  CV_DESCALE((x), ICV_WARP_SHIFT*2)
#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

typedef struct CvResizeAlpha
{
	int idx;
	int ialpha;
}CvResizeAlpha;


static int icvResize_Bilinear_8u_C1(const unsigned char* src, int srcstep, int swidth, int sheight,
	unsigned char* dst, int dststep, int dwidth, int dheight,
	int xmax,
	const CvResizeAlpha* xofs,
	const CvResizeAlpha* yofs,
	int* buf0, int* buf1)
{
	int prev_sy0 = -1, prev_sy1 = -1;
	int k, dx, dy;

	srcstep /= sizeof(src[0]);
	dststep /= sizeof(dst[0]);

	for (dy = 0; dy < dheight; dy++, dst += dststep)
	{
		int fy = yofs[dy].ialpha, *swap_t;
		int sy0 = yofs[dy].idx, sy1 = sy0 + (fy > 0 && sy0 < sheight - 1);

		if (sy0 == prev_sy0 && sy1 == prev_sy1)
			k = 2;
		else if (sy0 == prev_sy1)
		{
			CV_SWAP(buf0, buf1, swap_t);
			k = 1;
		}
		else
			k = 0;

		for (; k < 2; k++)
		{
			int* _buf = k == 0 ? buf0 : buf1;
			const unsigned char* _src;
			int sy = k == 0 ? sy0 : sy1;
			if (k == 1 && sy1 == sy0)
			{
				memcpy(buf1, buf0, dwidth * sizeof(buf0[0]));
				continue;
			}

			_src = src + sy * srcstep;
			for (dx = 0; dx < xmax; dx++)
			{
				int sx = xofs[dx].idx;
				int fx = xofs[dx].ialpha;
				int t = _src[sx];
				_buf[dx] = ICV_WARP_MUL_ONE_8U(t) + fx * (_src[sx + 1] - t);
			}

			for (; dx < dwidth; dx++)
				_buf[dx] = ICV_WARP_MUL_ONE_8U(_src[xofs[dx].idx]);
		}

		prev_sy0 = sy0;
		prev_sy1 = sy1;

		if (sy0 == sy1)
			for (dx = 0; dx < dwidth; dx++)
				dst[dx] = (unsigned char)ICV_WARP_DESCALE_8U(ICV_WARP_MUL_ONE_8U(buf0[dx]));
		else
			for (dx = 0; dx < dwidth; dx++)
				dst[dx] = (unsigned char)ICV_WARP_DESCALE_8U(ICV_WARP_MUL_ONE_8U(buf0[dx]) +
					fy * (buf1[dx] - buf0[dx]));
	}

	return 1;
}

void glasssix::longinus::myResize(const unsigned char * psrc, int swidth, int sheight, int sstep,
	unsigned char * pdst, int dwidth, int dheight, int dstep)
{
	void* temp_buf = 0;

	int scale_x, scale_y;
	int sx, sy, dx, dy;
	int xmax = dwidth, buf_size;
	int *buf0, *buf1;
	CvResizeAlpha *xofs, *yofs;
	int fx_1024x, fy_1024x;

	scale_x = ((swidth << ICV_WARP_SHIFT2) + dwidth / 2) / dwidth;
	scale_y = ((sheight << ICV_WARP_SHIFT2) + dheight / 2) / dheight;


	buf_size = dwidth * 2 * sizeof(int) + (dwidth + dheight) * sizeof(CvResizeAlpha);
	temp_buf = buf0 = (int*)malloc(buf_size);
	buf1 = buf0 + dwidth;
	xofs = (CvResizeAlpha*)(buf1 + dwidth);
	yofs = xofs + dwidth;

	for (dx = 0; dx < dwidth; dx++)
	{
		fx_1024x = ((dx * 2 + 1)*scale_x - (1 << ICV_WARP_SHIFT2)) / 2;
		sx = (fx_1024x >> ICV_WARP_SHIFT2);
		fx_1024x = ((fx_1024x - (sx << ICV_WARP_SHIFT2)) >> ICV_SHIFT_DIFF);

		if (sx < 0)
			sx = 0, fx_1024x = 0;

		if (sx >= swidth - 1)
		{
			fx_1024x = 0, sx = swidth - 1;
			if (xmax >= dwidth)
				xmax = dx;
		}

		xofs[dx].idx = sx;
		xofs[dx].ialpha = fx_1024x;
	}

	for (dy = 0; dy < dheight; dy++)
	{
		fy_1024x = ((dy * 2 + 1)*scale_y - (1 << ICV_WARP_SHIFT2)) / 2;
		sy = (fy_1024x >> ICV_WARP_SHIFT2);
		fy_1024x = ((fy_1024x - (sy << ICV_WARP_SHIFT2)) >> ICV_SHIFT_DIFF);

		if (sy < 0)
			sy = 0, fy_1024x = 0;

		yofs[dy].idx = sy;
		yofs[dy].ialpha = fy_1024x;
	}

	icvResize_Bilinear_8u_C1(psrc, sstep, swidth, sheight, pdst,
		dstep, dwidth, dheight, xmax, xofs, yofs, buf0, buf1);

	free(temp_buf);
}

void glasssix::longinus::myIntegral(const unsigned char *pSrc, int width, int height, int step, int *pSum, int sum_width)
{
	int x, y;
	int s;
	const unsigned char *psrc = pSrc;
	int *psum = pSum;
	int src_step = step;
	int sum_step = sum_width;

	if (pSrc == NULL || pSum == NULL)
	{
		fprintf(stderr, "%s: NULL pointer.\n", __FUNCTION__);
		return;
	}
	if (width <= 0 || height <= 0 || step <= 0)
	{
		fprintf(stderr, "%s: Invalid image size.\n", __FUNCTION__);
		return;
	}

	//the first row
	for (x = 0; x < width + 1; x++)
		psum[x] = 0;
	//the first column
	for (y = 1; y < height + 1; y++)
		psum[y * sum_step] = 0;

	for (y = 1, psum += sum_step;
		y < height + 1;
		y++, psrc += src_step, psum += sum_step)
	{
		for (x = 1, s = 0; x < width + 1; x++)
		{
			s += (psrc[x - 1]);
			psum[x] = psum[x - sum_step] + s;
		}
	}

	return;
}

void glasssix::longinus::myFlip(const unsigned char * psrc, int width, int height, int step,
	unsigned char * pdst)
{
	const unsigned char * s;
	unsigned char * d;

	for (int r = 0; r < height; r++)
	{
		s = psrc + r * step;
		d = pdst + r * step;
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int c = 0; c < width; c++)
		{
			d[c] = s[width - 1 - c];
		}
	}
}

void glasssix::longinus::resize_cpu_bilinear(const unsigned char* src_data, int old_height, int old_width, int channels,
	unsigned char* dst_data, int new_height, int new_width)
{
	int src_offset = old_height * old_width;
	int dst_offset = new_width * new_height;
	int* c_src_offset = new int[channels];
	int* c_dst_offset = new int[channels];
	for (int c = 0; c < channels; c++)
	{
		c_src_offset[c] = c * src_offset;
		c_dst_offset[c] = c * dst_offset;
		for (int h = 0; h < new_height; h++)
		{
			//Ordinate location in the oringinal image
			float y = (static_cast<float>(h) + 0.5f) * static_cast<float>(old_height) / static_cast<float>(new_height) - 0.5f;
			int iy = (int)y;
			if (h == new_height - 1)
			{
				iy = iy - 1;
			}
			unsigned char* pDstLine = dst_data + c_dst_offset[c] + h * new_width;
			for (int w = 0; w < new_width; w++)
			{
				//Abscissa location in the oringinal image
				float x = (static_cast<float>(w) + 0.5f) * static_cast<float>(old_width) / static_cast<float>(new_width) - 0.5f;
				//u and v is decimal part of x and y, enlarge 2048 times to avoid floating operations
				int u = static_cast<int> ((y - static_cast<int>(y)) * 2048);
				int v = static_cast<int> ((x - static_cast<int>(x)) * 2048);
				//ix, iy to store intergal part of x and y
				int ix = (int)x;
				if (w == new_width - 1)
				{
					ix = ix - 1;
				}
				pDstLine[w] = (src_data[c_src_offset[c] + iy * old_width + ix] * (2048 - u)*(2048 - v)
					+ src_data[c_src_offset[c] + iy * old_width + ix + 1] * (2048 - u)*v
					+ src_data[c_src_offset[c] + (iy + 1) * old_width + ix] * u*(2048 - v)
					+ src_data[c_src_offset[c] + (iy + 1) * old_width + ix + 1] * u*v) >> 22;
			}
		}
	}
	delete[] c_src_offset;
	delete[] c_dst_offset;
}

void glasssix::longinus::matrix_transpose(const unsigned char *src_data, int height, int width, unsigned char *dst_data)
{
	int dst_width = height;
	int dst_height = width;

	for (int row = 0; row < dst_height; ++row)
	{
		for (int col = 0; col < dst_width; ++col)
		{
			dst_data[row * dst_width + col] = src_data[col * width + row];
		}
	}
}

float blurscore(const unsigned char *data, int width, int height)
{
	float blur_val = 0.0f;
	float kernel[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
	float *BVer = new float[width * height];
	float *BHor = new float[width * height];

	float filter_data = 0.0;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (i < 4 || i > height - 5)
			{
				BVer[i * width + j] = data[i * width + j];
			}
			else
			{
				filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] + kernel[2] * data[(i - 2) * width + j] +
					kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i)* width + j] + kernel[5] * data[(i + 1) * width + j] +
					kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] + kernel[8] * data[(i + 4) * width + j];
				BVer[i * width + j] = filter_data;
			}

			if (j < 4 || j > width - 5)
			{
				BHor[i * width + j] = data[i * width + j];
			}
			else
			{
				filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] + kernel[2] * data[i * width + (j - 2)] +
					kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + (j + 1)] +
					kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] + kernel[8] * data[i * width + (j + 4)];
				BHor[i * width + j] = filter_data;
			}

		}
	}

	float D_Fver = 0.0;
	float D_FHor = 0.0;
	float D_BVer = 0.0;
	float D_BHor = 0.0;
	float s_FVer = 0.0;
	float s_FHor = 0.0;
	float s_Vver = 0.0;
	float s_VHor = 0.0;
	for (int i = 1; i < height; ++i)
	{
		for (int j = 1; j < width; ++j)
		{
			D_Fver = std::abs((float)data[i * width + j] - (float)data[(i - 1) * width + j]);
			s_FVer += D_Fver;
			D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
			s_Vver += std::max((float)0.0, D_Fver - D_BVer);

			D_FHor = std::abs((float)data[i * width + j] - (float)data[i * width + (j - 1)]);
			s_FHor += D_FHor;
			D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
			s_VHor += std::max((float)0.0, D_FHor - D_BHor);
		}
	}
	float b_FVer = (s_FVer - s_Vver) / s_FVer;
	float b_FHor = (s_FHor - s_VHor) / s_FHor;
	blur_val = std::max(b_FVer, b_FHor);

	delete[] BVer;
	delete[] BHor;

	//
	float clarity = 1.0f - blur_val;

	float T1 = 0.0f;
	float T2 = 1.0f;
	if (clarity <= T1)
	{
		clarity = 0.0;
	}
	else if (clarity >= T2)
	{
		clarity = 1.0;
	}
	else
	{
		clarity = (clarity - T1) / (T2 - T1);
	}
	return clarity;
}