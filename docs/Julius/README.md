# Julius

A SIMD supported [BLAS](http://www.netlib.org/blas/) library. Now only some important interfaces were implementated. Given our usage, there is no plans for complex value supporting(i.e. only *float* and *double* datatypes will be implemented) currently.

## Implementation Status

| Function |Native Code|  SSE	 |	AVX   |	AVX512|
|:--------:|:---------:|:-------:|:------:|:-----:|
| *s/dasum* |   NT     |  NT/F   |  NT/F  |  NT/F |
| *s/daxpy* |   TP     |  TP/F   |  TP/F  |  F    |
| *s/daxpby*|   TP     |  TP/F   |  TP/F  |  F    |
| *s/dcopy* |   F      |  F      |  F     |  F    |
| *sdsdot*  |   TP     |  F      |  F     |  F    |
| *dsdot*   |   TP     |  F      |  F     |  F    |
| *s/dnrm2* |   TP     |  NT/F   |  NT/F  |  NT/F |
| *s/ddot*  |   TP     |  F      |    F   |  F    |
| *s/dscal* |   TP     |  NT/F   |  NT/F  |  F    |
| *s/damax* |   F      |  F      |    F   |  F    |
| *s/damin* |   F      |  F      |    F   |  F    |
| *s/dgemv* |   TP/F   |  TP/F   |  TP/F  |  F    |
| *s/dgemm* |   TP/F   |  TP/F   |  TP/F  |  F    |
| *fgemm*   |   F      |  F      |    F   |  F    |


- TP: Implementated and test passed;
- F: Not implementated;
- TE: Implementated but error exists;
- --: No implementation;
- NT: Implementated but not test yet;
  
## Performance Status

- OS: Windows10 v1803(x64)
- CPU: Intel Core i7-8700k
- Memory: 32GB DDR4 2400MHz
- Compiler: MSVC v1910

### ?gemv


### ?gemm
