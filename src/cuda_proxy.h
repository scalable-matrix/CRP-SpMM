// Wrap up some CUDA operations for non-CUDA modules
#ifndef __CUDA_PROXY_H__
#define __CUDA_PROXY_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// This can be called before calling MPI_Init()
void select_cuda_device_by_mpi_local_rank();

// ========== Memory operations ========== //

void cuda_set_rt_dev_id(const int dev_id);

void cuda_memcpy_h2d(const void *hptr, void *dptr, const size_t bytes);

void cuda_memcpy_d2h(const void *dptr, void *hptr, const size_t bytes);

void cuda_memcpy_d2d(const void *dptr_src, void *dptr_dst, const size_t bytes);

void cuda_memcpy_auto(const void *src, void *dst, const size_t bytes);

void cuda_malloc_dev(void **dptr_, const size_t bytes);

void cuda_malloc_host(void **hptr_, const size_t bytes);

void cuda_memset_dev(void *dptr, const int value, const size_t bytes);

void cuda_free_dev(void *dptr);

void cuda_free_host(void *hptr);

void cuda_device_sync();

void cuda_stream_sync(void *stream_p);

// Copy a row-major matrix to another row-major matrix
void cuda_copy_matrix(
    size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd
);

// ========== CUSPARSE operations ========== //

// CUSPARSE CSR SpMM C := alpha * A * B + beta * C
// Input parameters:
//   m, n, k     : A is m-by-k, B is k-by-n, C is m-by-n
//   alpha, beta : Scaling factors
//   A_nnz       : Number of non-zero elements in A
//   A_rowptr_h  : Size m + 1, CSR row pointer array of A, on host
//   A_colidx_h  : Size A_nnz, CSR column index array of A, on host
//   A_val_h     : Size A_nnz, CSR value array of A, on host
//   B_h         : Size k * ldB, row-major B matrix, on host
//   ldB         : Leading dimension of B_h, >= n
//   ldC         : Leading dimension of C_h, >= n
// Output parameters:
//   C_h : Size m * ldC, row-major C matrix, on host
void cuda_cusparse_csr_spmm(
    const int m, const int n, const int k, const double alpha, 
    const int A_nnz, const int *A_rowptr_h, const int *A_colidx_h, const double *A_val_h,
    const double *B_h, const int ldB, const double beta, double *C_h, const int ldC
);

#ifdef __cplusplus
}
#endif

#endif 