#include <stdlib.h>

#define A( i,j ) a[ (j)*lda + (i) ]

#define m_r 0x100000000LL  
#define c_r 0xB16  
#define a_r 0x5DEECE66DLL  

static unsigned long long seed = 1;

double drand48(void)
{
	seed = (a_r * seed + c_r) & 0xFFFFFFFFFFFFLL;
	unsigned int x = seed >> 16;
	return  ((double)x / (double)m_r);

}

void random_matrix(int m, int n, double *a, int lda)
{
	//double drand48();
	int i, j;

	for (j = 0; j<n; j++)
		for (i = 0; i<m; i++)
			A(i, j) = 2.0 * drand48() - 1.0;
}