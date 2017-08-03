#include "Excalibur_MathFunctions.hpp"
//#include <boost/thread.hpp>

namespace Excalibur
{
	// Make sure each thread can have different values.
	//static boost::thread_specific_ptr<Excalibur_MathFunctions> thread_instance_;

	Excalibur_MathFunctions::Excalibur_MathFunctions()
	{
		mode = CPU;//default CPU
		gpu_device_ = -1;
//#ifdef USE_CUDA
//		cublas_handle_ = NULL;
//		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) 
//		{
//			LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
//		}
//#ifdef USE_CUDNN
//		cudnn_handle_ = NULL;
//		if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS) 
//		{
//			LOG(ERROR) << "Cannot create cuDNN handle. cuDNN won't be available.";
//		}
//#endif
//#endif
	}


	Excalibur_MathFunctions::Excalibur_MathFunctions(int gpu_decvice, Avalon mode_)
	{
		mode = mode_;
		gpu_device_ = gpu_decvice;
#ifdef USE_CUDA
		if (mode==GPU)
		{
			//setgpu device
			cublas_handle_ = NULL;
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#ifdef USE_CUDNN
			cudnn_handle_ = NULL;
			if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create cuDNN handle. cuDNN won't be available.";
			}
#endif
		}
#endif
	}


	Excalibur_MathFunctions::~Excalibur_MathFunctions()
	{
#ifdef USE_CUDA
		if (mode == GPU)//if(cublas_handle_)
			CUBLAS_CHECK(cublasDestroy(cublas_handle_));
#ifdef USE_CUDNN
		if (cudnn_handle_) 
			CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
#endif
#endif
	}

	/*Excalibur_MathFunctions& Excalibur_MathFunctions::Get() 
	{
		if (!thread_instance_.get()) 
		{
			thread_instance_.reset(new Excalibur_MathFunctions());
		}
		return *(thread_instance_.get());
	}*/

	void Excalibur_MathFunctions::set_Avalon(Avalon mode_)
	{
		mode = mode_;
	}


	template <>
	void Excalibur_MathFunctions::excalibur_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float* A, const float* B, const float beta, float* C)
	{
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			ldb, beta, C, N);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C)
	{
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			ldb, beta, C, N);
	}
#ifdef USE_MKL
	template <>
	void Excalibur_MathFunctions::excalibur_cpu_gemm_batch<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, 
		const int M, const int N, const int K, const float alpha, const float** A, const float** B, const float beta,
		float** C, const int group_count, const int group_size)
	{
		//alloc temp data
		MKL_INT* lda = new MKL_INT[group_count];
		MKL_INT* ldb = new MKL_INT[group_count];
		CBLAS_TRANSPOSE* TransA_array = new CBLAS_TRANSPOSE[group_count];
		CBLAS_TRANSPOSE* TransB_array = new CBLAS_TRANSPOSE[group_count];
		MKL_INT* M_array = new MKL_INT[group_count];
		MKL_INT* N_array = new MKL_INT[group_count];
		MKL_INT* K_array = new MKL_INT[group_count];
		MKL_INT* group_size_array = new MKL_INT[group_count];
		float* alpha_array = new float[group_count];
		float* beta_array = new float[group_count];
		for (int i = 0; i < group_count; i++)
		{
			lda[i] = (TransA == CblasNoTrans) ? K : M;
			ldb[i] = (TransB == CblasNoTrans) ? N : K;
			TransA_array[i] = TransA;
			TransB_array[i] = TransB;
			group_size_array[i] = group_size;
			M_array[i] = M;
			K_array[i] = K;
			N_array[i] = N;
		}
		/*excalibur_cpu_set(group_count, M, M_array);
		excalibur_cpu_set(group_count, K, K_array);
		excalibur_cpu_set(group_count, N, N_array);*/
		excalibur_cpu_set(group_count, alpha, alpha_array);
		excalibur_cpu_set(group_count, beta, beta_array);
		//calculate batch gemm
		cblas_sgemm_batch(CblasRowMajor, TransA_array, TransB_array, M_array, N_array, K_array, alpha_array, A, lda, B,
			ldb, beta_array, C, N_array, group_count, group_size_array);
		//delete temp data
		delete lda, ldb, TransA_array, TransB_array, M_array, N_array, K_array, group_size_array, alpha_array, beta_array;
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_gemm_batch<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
		const int M, const int N, const int K, const double alpha, const double** A, const double** B, const double beta,
		double** C, const int group_count, const int group_size)
	{
		//alloc temp data
		MKL_INT* lda = new MKL_INT[group_count];
		MKL_INT* ldb = new MKL_INT[group_count];
		CBLAS_TRANSPOSE* TransA_array = new CBLAS_TRANSPOSE[group_count];
		CBLAS_TRANSPOSE* TransB_array = new CBLAS_TRANSPOSE[group_count];
		MKL_INT* M_array = new MKL_INT[group_count];
		MKL_INT* N_array = new MKL_INT[group_count];
		MKL_INT* K_array = new MKL_INT[group_count];
		MKL_INT* group_size_array = new MKL_INT[group_count];
		double* alpha_array = new double[group_count];
		double* beta_array = new double[group_count];
		for (int i = 0; i < group_count; i++)
		{
			lda[i] = (TransA == CblasNoTrans) ? K : M;
			ldb[i] = (TransB == CblasNoTrans) ? N : K;
			TransA_array[i] = TransA;
			TransB_array[i] = TransB;
			group_size_array[i] = group_size;
			M_array[i] = M;
			K_array[i] = K;
			N_array[i] = N;
		}
		/*excalibur_cpu_set(group_count, M, M_array);
		excalibur_cpu_set(group_count, K, K_array);
		excalibur_cpu_set(group_count, N, N_array);*/
		excalibur_cpu_set(group_count, alpha, alpha_array);
		excalibur_cpu_set(group_count, beta, beta_array);
		//calculate batch gemm
		cblas_dgemm_batch(CblasRowMajor, TransA_array, TransB_array, M_array, N_array, K_array, alpha_array, A, lda, B,
			ldb, beta_array, C, N_array, group_count, group_size_array);
		//delete temp data
		delete lda, ldb, TransA_array, TransB_array, M_array, N_array, K_array, group_size_array, alpha_array, beta_array;
	}
#endif
	template <>
	void Excalibur_MathFunctions::excalibur_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, const float* A, const float* x, const float beta, float* y)
	{
		cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, const double* A, const double* x, const double beta, double* y)
	{
		cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_axpy<float>(const int N, const float alpha, const float* X, float* Y)
	{
		cblas_saxpy(N, alpha, X, 1, Y, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_axpy<double>(const int N, const double alpha, const double* X, double* Y)
	{
		cblas_daxpy(N, alpha, X, 1, Y, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_axpby<float>(const int N, const float alpha, const float* X, const float beta, float* Y)
	{
		cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_axpby<double>(const int N, const double alpha, const double* X, const double beta, double* Y)
	{
		cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
	}

	template <typename Dtype>
	void Excalibur_MathFunctions::excalibur_cpu_copy(const int N, const Dtype* X, Dtype* Y)
	{
		memcpy(Y, X, sizeof(Dtype) * N);
	}

	template void Excalibur_MathFunctions::excalibur_cpu_copy<int>(const int N, const int* X, int* Y);
	template void Excalibur_MathFunctions::excalibur_cpu_copy<unsigned int>(const int N, const unsigned int* X,
		unsigned int* Y);
	template void Excalibur_MathFunctions::excalibur_cpu_copy<float>(const int N, const float* X, float* Y);
	template void Excalibur_MathFunctions::excalibur_cpu_copy<double>(const int N, const double* X, double* Y);


	template <typename Dtype>
	void Excalibur_MathFunctions::excalibur_cpu_set(const int N, const Dtype alpha, Dtype* X)
	{
		if (alpha == 0) 
		{
			memset(X, 0, sizeof(Dtype) * N);
			return;
		}
		for (int i = 0; i < N; ++i) 
		{
			X[i] = alpha;
		}
	}
	template void Excalibur_MathFunctions::excalibur_cpu_set<int>(const int N, const int alpha, int* Y);
	template void Excalibur_MathFunctions::excalibur_cpu_set<float>(const int N, const float alpha, float* Y);
	template void Excalibur_MathFunctions::excalibur_cpu_set<double>(const int N, const double alpha, double* Y);

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_add_scalar(const int N, const float alpha, float* Y) 
	{
		for (int i = 0; i < N; ++i) 
		{
			Y[i] += alpha;
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_add_scalar(const int N, const double alpha, double* Y) 
	{
		for (int i = 0; i < N; ++i) 
		{
			Y[i] += alpha;
		}
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_scal<float>(const int N, const float alpha, float *X) 
	{
		cblas_sscal(N, alpha, X, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_scal<double>(const int N, const double alpha, double *X) 
	{
		cblas_dscal(N, alpha, X, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_add<float>(const int n, const float* a, const float* b,
		float* y) 
	{
		vsAdd(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_add<double>(const int n, const double* a, const double* b,
		double* y) 
	{
		vdAdd(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_sub<float>(const int n, const float* a, const float* b,
		float* y) 
	{
		vsSub(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_sub<double>(const int n, const double* a, const double* b,
		double* y) 
	{
		vdSub(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_mul<float>(const int n, const float* a, const float* b,
		float* y) 
	{
		vsMul(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_mul<double>(const int n, const double* a, const double* b,
		double* y) 
	{
		vdMul(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_div<float>(const int n, const float* a, const float* b,
		float* y) 
	{
		vsDiv(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_div<double>(const int n, const double* a, const double* b,
		double* y) 
	{
		vdDiv(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_powx<float>(const int n, const float* a, const float b,
		float* y) 
	{
		vsPowx(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_powx<double>(const int n, const double* a, const double b,
		double* y) 
	{
		vdPowx(n, a, b, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_sqr<float>(const int n, const float* a, float* y) 
	{
		vsSqr(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_sqr<double>(const int n, const double* a, double* y) 
	{
		vdSqr(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_exp<float>(const int n, const float* a, float* y) 
	{
		vsExp(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_exp<double>(const int n, const double* a, double* y) 
	{
		vdExp(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_log<float>(const int n, const float* a, float* y) 
	{
		vsLn(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_log<double>(const int n, const double* a, double* y) 
	{
		vdLn(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_abs<float>(const int n, const float* a, float* y) 
	{
		vsAbs(n, a, y);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_abs<double>(const int n, const double* a, double* y) 
	{
		vdAbs(n, a, y);
	}

	template <>
	float Excalibur_MathFunctions::excalibur_cpu_strided_dot<float>(const int n, const float* x, const int incx,
		const float* y, const int incy)
	{
		return cblas_sdot(n, x, incx, y, incy);
	}

	template <>
	double Excalibur_MathFunctions::excalibur_cpu_strided_dot<double>(const int n, const double* x,
		const int incx, const double* y, const int incy) 
	{
		return cblas_ddot(n, x, incx, y, incy);
	}

	template <>
	float Excalibur_MathFunctions::excalibur_cpu_asum<float>(const int n, const float* x) {
		return cblas_sasum(n, x, 1);
	}

	template <>
	double Excalibur_MathFunctions::excalibur_cpu_asum<double>(const int n, const double* x) {
		return cblas_dasum(n, x, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_scale<float>(const int n, const float alpha, const float *x,
		float* y) {
		cblas_scopy(n, x, 1, y, 1);
		cblas_sscal(n, alpha, y, 1);
	}

	template <>
	void Excalibur_MathFunctions::excalibur_cpu_scale<double>(const int n, const double alpha, const double *x,
		double* y) {
		cblas_dcopy(n, x, 1, y, 1);
		cblas_dscal(n, alpha, y, 1);
	}
}
