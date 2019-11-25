#include "julius_asum.hpp"
#include "julius_axpby.hpp"
#include "julius_dot.hpp"
#include "julius_sdot.hpp"
#include "julius_nrm2.hpp"
#include "julius_scal.hpp"
#include "julius_gemv.hpp"
#include "julius_gemm.hpp"
#include "julius.hpp"

namespace glasssix
{
	namespace excalibur
	{
		float cblas_sasum(const int n, const float *x, const int incx)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0f;
			}
			CHECK_GT(incx, 0);
			return juliusblas::cblas_sasum(n, x, incx);
		}

		double cblas_dasum(const int n, const double *x, const int incx)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0;
			}
			CHECK_GT(incx, 0);
			return juliusblas::cblas_dasum(n, x, incx);
		}

		void cblas_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy)
		{
			cblas_saxpby(n, alpha, x, incx, 1.0f, y, incy);
		}

		void cblas_daxpy(const int n, const double alpha, const double *x, const int incx, double *y, const int incy)
		{
			cblas_daxpby(n, alpha, x, incx, 1.0, y, incy);
		}

		void cblas_saxpby(const int n, const float alpha, const float* x,
			const int incx, const float beta, float* y, const int incy)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return;
			}
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			juliusblas::cblas_saxpby(n, alpha, x, incx, beta, y, incy);
		}

		void cblas_daxpby(const int n, const double alpha, const double* x,
			const int incx, const double beta, double* y, const int incy)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return;
			}
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			juliusblas::cblas_daxpby(n, alpha, x, incx, beta, y, incy);
		}

		float cblas_sdsdot(const int n, const float alpha, const float *x, const int incx, const float *y, const int incy)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return alpha;
			}
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			return juliusblas::cblas_sdsdot(n, alpha, x, incx, y, incy);
		}

		double cblas_dsdot(const int n, const float *x, const int incx, const float *y, const int incy)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0;
			}
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			return juliusblas::cblas_dsdot(n, x, incx, y, incy);
		}

		float  cblas_sdot(const int n, const float  *x, const int incx, const float  *y, const int incy)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0f;
			}
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			return juliusblas::cblas_sdot(n, x, incx, y, incy);
		}

		double cblas_ddot(const int n, const double *x, const int incx, const double *y, const int incy)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0;
			}
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			return juliusblas::cblas_ddot(n, x, incx, y, incy);
		}

		float cblas_snrm2(const int n, const float *x, const int incx)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0f;
			}
			CHECK_GT(incx, 0);
			return juliusblas::cblas_snrm2(n, x, incx);
		}

		double cblas_dnrm2(const int n, const double *x, const int incx)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return 0.0;
			}
			CHECK_GT(incx, 0);
			return juliusblas::cblas_dnrm2(n, x, incx);
		}

		void cblas_sscal(const int n, const float alpha, float *x, const int incx)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return;
			}
			CHECK_GT(incx, 0);
			juliusblas::cblas_sscal(n, alpha, x, incx);
		}

		void cblas_dscal(const int n, const double alpha, double *x, const int incx)
		{
			if (n <= 0)
			{
				LOG(ERROR) << "Illegal vector size.";
				return;
			}
			CHECK_GT(incx, 0);
			juliusblas::cblas_dscal(n, alpha, x, incx);
		}

		void cblas_sgemv(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int M, const int N,
			const float alpha, const float  *A, const int lda, const float  *x, const int incx, const float beta, float  *y, const int incy)
		{
			CHECK_GT(M, 0);
			CHECK_GT(N, 0);
			CHECK_GT(lda, 0);
			CHECK_GT(incx, 0);
			CHECK_GT(incy, 0);
			switch (order)
			{
			case CblasRowMajor:
				if (trans == CblasNoTrans)
				{
					juliusblas::cblas_sgemv_AnoTrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else if (trans == CblasTrans)
				{
					juliusblas::cblas_sgemv_ATrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemv.";
				}
				break;
			case CblasColMajor:
				NOT_IMPLEMENTED << " with CblasColMajor in Julius sgemv.";
				break;
			default:
				if (trans == CblasNoTrans)
				{
					juliusblas::cblas_sgemv_AnoTrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else if (trans == CblasTrans)
				{
					juliusblas::cblas_sgemv_ATrans(M, N, alpha, A, lda, x, incx, beta, y, incy);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemv.";
				}
				break;
			}
		}

		void cblas_dgemv(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int m, const int n,
			const double alpha, const double  *a, const int lda, const double  *x, const int incx, const double beta, double  *y, const int incy)
		{
			NOT_IMPLEMENTED;
		}

		void cblas_sgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb,
			const float beta, float* C, const int ldc)
		{
			CHECK_GT(M, 0);
			CHECK_GT(N, 0);
			CHECK_GT(K, 0);
			CHECK_GT(lda, 0);
			CHECK_GT(ldb, 0);
			CHECK_GT(ldc, 0);
			switch (Order)
			{
			case CblasRowMajor:
				if (TransA == CblasNoTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasNoTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemm.";
				}
				break;
			case CblasColMajor:
				NOT_IMPLEMENTED << " with CblasColMajor in Julius sgemm.";
				break;
			default:
				if (TransA == CblasNoTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasNoTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_AnoTrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_sgemm_ATrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemm.";
				}
				break;
			}
		}

		void cblas_dgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb,
			const double beta, double* C, const int ldc)
		{
			NOT_IMPLEMENTED;
		}


		void cblas_fgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const float alpha, const signed char* A, const int lda, const signed char* B, const int ldb,
			const float beta, int* C, const int ldc)
		{
			CHECK_GT(M, 0);
			CHECK_GT(N, 0);
			CHECK_GT(K, 0);
			CHECK_GT(lda, 0);
			CHECK_GT(ldb, 0);
			CHECK_GT(ldc, 0);
			switch (Order)
			{
			case CblasRowMajor:
				if (TransA == CblasNoTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_fgemm_AnoTrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_fgemm_ATrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasNoTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_fgemm_AnoTrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_fgemm_ATrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemm.";
				}
				break;
			case CblasColMajor:
				NOT_IMPLEMENTED << " with CblasColMajor in Julius sgemm.";
				break;
			default:
				if (TransA == CblasNoTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_fgemm_AnoTrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasNoTrans)
				{
					juliusblas::cblas_fgemm_ATrans_BnoTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasNoTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_fgemm_AnoTrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else if (TransA == CblasTrans && TransB == CblasTrans)
				{
					juliusblas::cblas_fgemm_ATrans_BTrans(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
				}
				else
				{
					NOT_IMPLEMENTED << " error trans type in Julius sgemm.";
				}
				break;
			}
		}


		void cblas_hgemm(const enum CBLAS_LAYOUT Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
			const int M, const int N, const int K, const unsigned short alpha, const unsigned short* A, const int lda, const unsigned short* B, const int ldb,
			const unsigned short beta, unsigned short* C, const int ldc)
		{
			float* f_A = nullptr;
			float* f_B = nullptr;
			float* f_C = new float[M * ldc];
			if (TransA == CBLAS_TRANSPOSE::CblasNoTrans)
			{
				f_A = new float[M * lda];
				half2float(A, f_A, M * lda);
			}
			else if(TransA == CBLAS_TRANSPOSE::CblasTrans)
			{
				f_A = new float[K * lda];
				half2float(A, f_A, K * lda);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			if (TransB == CBLAS_TRANSPOSE::CblasNoTrans)
			{
				f_B = new float[K * ldb];
				half2float(B, f_B, K * ldb);
			}
			else if (TransB == CBLAS_TRANSPOSE::CblasTrans)
			{
				f_B = new float[N * ldb];
				half2float(B, f_B, N * ldb);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			float f_alpha = float16_to_float32(alpha);
			float f_beta = float16_to_float32(beta);
			cblas_sgemm(Order, TransA, TransB, M, N, K, f_alpha, f_A, lda, f_B, ldb, f_beta, f_C, ldc);
			float2half(f_C, C, M * ldc);
			delete[] f_A;
			delete[] f_B;
			delete[] f_C;
		}
	}
}