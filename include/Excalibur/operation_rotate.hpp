#ifndef _OPERATION_ROTATE_HPP_
#define _OPERATION_ROTATE_HPP_

#include <memory>
#include "Primitives/logger.hpp"
#include "operation_resize.hpp"
#include "saturate.hpp"

using namespace glasssix;

namespace glasssix
{
    namespace excalibur
    {
        const int INTER_REMAP_COEF_BITS = 15;
        const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;
        enum InterpolationMasks
        {
            INTER_BITS = 5,
            INTER_BITS2 = INTER_BITS * 2,
            INTER_TAB_SIZE = 1 << INTER_BITS,
            INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
        };

        enum
        {
            SHIFT = INTER_REMAP_COEF_BITS,
            DELTA = 1 << (INTER_REMAP_COEF_BITS - 1)
        };

        static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

        template <typename Dtype>
        class point
        {
        public:
            Dtype x;
            Dtype y;

            point()
            {
                x = Dtype(0);
                y = Dtype(0);
            }

            point(Dtype x, Dtype y)
            {
                this->x = x;
                this->y = y;
            }

            point &operator=(const point &r)
            {
                if (this == &r)
                {
                    return *this;
                }
                x = r.x;
                y = r.y;
                return *this;
            }

            point(const point &r)
            {
                x = r.x;
                y = r.y;
            }

            float distance(const point &r)
            {
                return sqrt((x - r.x) * (x - r.x) * 1.0f + (y - r.y) * (y - r.y) * 1.0f);
            }
        };

        static inline void initInterTab1D(float *tab, int tabsz)
        {
            float scale = 1.f / tabsz;
            for (int i = 0; i < tabsz; i++, tab += 2)
            {
                tab[0] = 1.f - i * scale;
                tab[1] = i * scale;
            }
        }

        // init Bilinear interpolation coef table
        static inline const short *initInterTab2D()
        {
            short *itab = BilinearTab_i[0][0];
            int ksize = 2;
            float _tab[8 * INTER_TAB_SIZE] = {0.0f};

            int i, j, k1, k2;
            initInterTab1D(_tab, INTER_TAB_SIZE);
            for (i = 0; i < INTER_TAB_SIZE; i++)
            {
                for (j = 0; j < INTER_TAB_SIZE; j++, itab += ksize * ksize)
                {
                    int isum = 0;
                    for (k1 = 0; k1 < ksize; k1++)
                    {
                        float vy = _tab[i * ksize + k1];
                        for (k2 = 0; k2 < ksize; k2++)
                        {
                            float v = vy * _tab[j * ksize + k2];
                            isum += itab[k1 * ksize + k2] = saturate_cast<short>(v * INTER_REMAP_COEF_SCALE);
                        }
                    }

                    if (isum != INTER_REMAP_COEF_SCALE)
                    {
                        int diff = isum - INTER_REMAP_COEF_SCALE;
                        int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                        if (diff < 0)
                            itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
                        else
                            itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
                    }
                }
            }
            itab -= INTER_TAB_SIZE2 * ksize * ksize;
            return (const short *)itab;
        }

        // calculate invert matrix A^(-1) version2
        inline static bool GetMatrixInverseV2(std::vector<std::vector<double>> &src, std::vector<std::vector<double>> &des)
        {
            // calculate value of |A|
            double D = src[0][0] * src[1][1] - src[0][1] * src[1][0];
            if (D == 0)
                return false;
            D = 1 / D;

            des[0][0] = src[1][1] * D;
            des[0][1] = -src[0][1] * D;
            des[0][2] = (src[0][1] * src[1][2] - src[0][2] * src[1][1]) * D;
            des[1][0] = -src[1][0] * D;
            des[1][1] = src[0][0] * D;
            des[1][2] = (src[0][2] * src[1][0] - src[0][0] * src[1][2]) * D;
            return true;
        }

        //copy data to src
        template <typename Dtype>
        static inline void copy_data(std::shared_ptr<memory::tensor<Dtype>> &dst, Dtype *dpart, int s_width, int height, int x_start, int y_start, size_t sstep, size_t dstep)
        {
            if (dst->order() == memory::NHWC)
            {
                Dtype *pDst = dst->mutable_cpu_data() + y_start * sstep + x_start * 3;
                for (int y = 0; y < height; ++y)
                {
                    std::copy(dpart, dpart + dstep, pDst);
                    // move ptr
                    pDst += sstep;
                    dpart += dstep;
                }
            }
            else if (dst->order() == memory::NCHW)
            {
                Dtype *pDst = dst->mutable_cpu_data() + y_start * s_width + x_start;
                for (int i = 0; i < 3; ++i)
                {
                    std::copy(dpart + i * dstep, dpart + i * dstep + dstep, pDst + i * sstep);
                }
            }
        }

        // remapBilinear
        template <typename Dtype>
        static void remapBilinear(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, Dtype *dpart, int d_width, int d_height, ushort *_XY, int x_start, int y_start, Dtype fill_pixel_value, ushort *bufa, const short *ctab)
        {
            int s_width = src->width();
            int s_height = src->height();

            const int cn = src->channels();
            const Dtype *S0 = src->cpu_data();

            if (src->order() == memory::NHWC)
            {
                size_t dstep = d_width * cn;
                size_t sstep = s_width * cn;
                Dtype *D = nullptr;

                for (int dy = 0; dy < d_height; ++dy)
                {
                    const ushort *FXY = bufa + dy * d_width;
                    D = dpart + dy * dstep;
                    const ushort *XY = _XY + dy * d_width * 2;

                    for (int dx = 0; dx < d_width; ++dx, D += cn)
                    {
                        ushort sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                        if (sx >= s_width || sx + 1 < 0 || sy >= s_height || sy + 1 < 0)
                        {
                            for (int k = 0; k < cn; k++)
                            {
                                D[k] = fill_pixel_value;
                            }
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            const Dtype *v0, *v1, *v2, *v3;
                            const short *w = ctab + FXY[dx] * 4;
                            sx0 = sx, sx1 = sx + 1, sy0 = sy, sy1 = sy + 1;

                            v0 = S0 + sy0 * sstep + sx0 * cn;
                            v1 = S0 + sy0 * sstep + sx1 * cn;
                            v2 = S0 + sy1 * sstep + sx0 * cn;
                            v3 = S0 + sy1 * sstep + sx1 * cn;

                            for (int k = 0; k < cn; k++)
                            {
                                D[k] = saturate_cast<Dtype>((int(v0[k] * w[0] + v1[k] * w[1] + v2[k] * w[2] + v3[k] * w[3]) + DELTA) >> SHIFT);
                            }
                        }
                    }
                }
                // copy dpart data to dst
                copy_data<Dtype>(dst, dpart, s_width, d_height, x_start, y_start, sstep, dstep);
            }
            else if (src->order() == memory::NCHW)
            {
                Dtype *D = nullptr;
                const int dstep = d_width * d_height;
                const int sstep = s_width * s_height;
                for (int dy = 0; dy < d_height; ++dy)
                {
                    const ushort *FXY = bufa + dy * d_width;
                    D = dpart + dy * d_width;
                    const ushort *XY = _XY + dy * d_width * 2;

                    for (int dx = 0; dx < d_width; ++dx, ++D)
                    {
                        ushort sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                        if (sx >= s_width || sx + 1 < 0 || sy >= s_height || sy + 1 < 0)
                        {
                            for (int k = 0; k < cn; k++)
                            {
                                D[k * dstep] = fill_pixel_value;
                            }
                        }
                        else
                        {
                            int sx0, sx1, sy0, sy1;
                            const Dtype *v0, *v1, *v2, *v3;
                            const short *w = ctab + FXY[dx] * 4;
                            sx0 = sx, sx1 = sx + 1, sy0 = sy, sy1 = sy + 1;

                            v0 = S0 + sy0 * s_width + sx0 * cn;
                            v1 = S0 + sy0 * s_width + sx1 * cn;
                            v2 = S0 + sy1 * s_width + sx0 * cn;
                            v3 = S0 + sy1 * s_width + sx1 * cn;

                            for (int k = 0; k < cn; k++)
                            {
                                D[k] = saturate_cast<Dtype>((int(v0[k * sstep] * w[0] + v1[k * sstep] * w[1] + v2[k * sstep] * w[2] + v3[k * sstep] * w[3]) + DELTA) >> SHIFT);
                            }
                        }
                    }
                }
                // copy dpart data to dst
                copy_data<Dtype>(dst, dpart, s_width, d_height, x_start, y_start, sstep, dstep);
            }
        }

        template <typename Dtype, typename Ptype>
        static void rotate_with_points_cpu(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>> &dst, const point<Ptype> &center, float theta, float scale = 1.0f, Dtype fill_pixel_value = 0, interpolationType type = Bilinear)
        {
            if (src->device() >= 0)
            {
                LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
                return;
            }

            if (fabs(theta) <= 1e-6)
            {
                dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
                return;
            }

            const int BLOCK_SZ = 64;
            ushort XY[BLOCK_SZ * BLOCK_SZ * 2];
            int channels = src->channels();
            int height = src->height();
            int width = src->width();

            // int _abdelta[width * 2] = {0};
            int *_abdelta = new int[width * 2];
            int *adelta = _abdelta, *bdelta = adelta + width;
            const int AB_BITS = MAX(10, (int)INTER_BITS);
            const int AB_SCALE = 1 << AB_BITS;
            int round_delta = AB_SCALE / INTER_TAB_SIZE / 2, x, y, x1, y1;

            std::vector<std::vector<double>> M(2, std::vector<double>(3));
            std::vector<std::vector<double>> reverse_M(2, std::vector<double>(3));

            double rad = theta * (PI / 180);
            double cosa = cos(rad);
            double sina = sin(rad);
            double a = scale * cosa;
            double b = scale * sina;

            M[0][0] = a;
            M[0][1] = b;
            M[0][2] = (1 - a) * (double)center.x - b * (double)center.y;
            M[1][0] = -1 * b;
            M[1][1] = a;
            M[1][2] = b * (double)center.x + (1 - a) * (double)center.y;

            bool isInverted = GetMatrixInverseV2(M, reverse_M);
            if (!isInverted)
            {
                LOG(FATAL) << "cannot rotate!!!";
                return;
            }

            const short *ctab = initInterTab2D();

            for (int x = 0; x < width; x++)
            {
                adelta[x] = int(reverse_M[0][0] * x * AB_SCALE);
                bdelta[x] = int(reverse_M[1][0] * x * AB_SCALE);
            }

            const int bh0 = 32;
            const int bw0 = 128;
            int x_start = 0, y_start = 0;
            Dtype *dpart = nullptr;
            for (y = 0; y < height; y += bh0)
            {
                y_start = y;
                for (x = 0; x < width; x += bw0)
                {
                    x_start = x;
                    int bw = std::min(bw0, width - x);
                    int bh = std::min(bh0, height - y);
                    dpart = new Dtype[bh * bw * channels];
                    // ushort bufa[bh * bw] = {0};
                    ushort *bufa = new ushort[bh * bw];
                    ushort *A = nullptr;

                    for (y1 = 0; y1 < bh; y1++)
                    {
                        ushort *xy = XY + y1 * bw * 2;
                        int X0 = int((reverse_M[0][1] * (y + y1) + reverse_M[0][2]) * AB_SCALE) + round_delta;
                        int Y0 = int((reverse_M[1][1] * (y + y1) + reverse_M[1][2]) * AB_SCALE) + round_delta;
                        A = bufa + y1 * bw;
                        for (x1 = 0; x1 < bw; x1++)
                        {
                            int X = (X0 + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                            int Y = (Y0 + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                            xy[x1 * 2] = ushort(X >> INTER_BITS);
                            xy[x1 * 2 + 1] = ushort(Y >> INTER_BITS);
                            A[x1] = (ushort)((short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                                     (X & (INTER_TAB_SIZE - 1))) &
                                             (INTER_TAB_SIZE2 - 1));
                        }
                    }

                    //remapBilinear
                    remapBilinear<Dtype>(src, dst, dpart, bw, bh, XY, x_start, y_start, fill_pixel_value, bufa, ctab);
                    delete[] dpart;
                    delete[] bufa;
                }
            }
            delete[] _abdelta;
        }
    }
}
#endif