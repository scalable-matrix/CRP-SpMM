#ifndef __SPMAT_PART_H__
#define __SPMAT_PART_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Partition a CSR matrix into multiple row blocks s.t. each row block
// has contiguous rows and roughly the same number of non-zero elements
// Input parameters:
//   nrow    : number of rows in the matrix
//   row_ptr : Size nrow + 1, CSR row pointer array
//   nblk    : number of row blocks
// Output parameter:
//   rblk_ptr : Size nblk + 1, row block pointer array
void csr_mat_row_partition(const int nrow, const int *row_ptr, const int nblk, int *rblk_ptr);

// Prime factorization
// Input parameter:
//   n : number to be factorized
// Output parameters:
//   <return> : nfac, number of prime factors
//   *factors : Size nfac, prime factors (small to large) of n, will be allocated
int prime_factorization(int n, int **factors);

// Compute the SpMV communication sizes for each row block
// Input parameters:
//   nrow     : number of rows in the matrix
//   ncol     : number of columns in the matrix
//   row_ptr  : Size nrow + 1, CSR row pointer array
//   col_idx  : Size row_ptr[nrow], CSR column index array
//   nblk     : number of row blocks
//   rblk_ptr : Size nblk + 1, row block pointer array
//   x_displs : Size nblk + 1, x displacement array, process i owns x[x_displs[i] : x_displs[i+1] - 1]
// Output parameters:
//   comm_sizes  : Size nblk, communication size array
//   *total_size : total communication size
void csr_mat_row_part_comm_size(
    const int nrow, const int ncol, const int *row_ptr, const int *col_idx,
    const int nblk, const int *rblk_ptr, const int *x_displs, 
    int *comm_sizes, int *total_size
);

// Calculate a 2D process grid dimensions and matrix partitioning 
// for SpMM from a 1D row partitioning
// Input paratemers:
//   nproc      : number of processes
//   m, n, k    : size of matrix A (m * k), B (k * n), and C (m * n)
//   rb_displs0 : Size nproc + 1, row block displacements of A
//   rowptr     : Size m + 1, CSR row pointer of A
//   colidx     : Size rowptr[m], CSR column index of A
// Output parameters:
//   *pm, *pn   : Process grid dimensions, pn groups * pm-way row parallel SpMM
//   *comm_cost : SpMM communication cost of the partitioning
//   *A0_rowptr : Size nproc + 1, row pointer of A0, A0 is 1D row partitioned
//   *B_rowptr  : Size pm + 1, row pointer of B
//   *AC_rowptr : Size pm + 1, row pointer of replicated A and final C
//   *BC_colptr : Size pn + 1, column pointer of B and C
// Notes: 
//   Let idx_m0(i) = [A0_rowptr[i] : A0_rowptr[i+1] - 1],
//       idx_m(i)  = [AC_rowptr[i] : AC_rowptr[i+1] - 1], 
//       idx_k(i)  = [B_rowptr[i]  : B_rowptr[i+1] - 1],
//       idx_n(j)  = [BC_colptr[j] : BC_colptr[j+1] - 1].
//   Before redistributing A, process P_{i, j} owns A(idx_m0(i * pn + j), :).
//   Before redistributing B, process P_{i, j} owns B(idx_k(i), idx_n(j)).
//   P_{i, j} computes C(idx_m(i), idx_n(j)) = A(idx_m(i), :) * B(:, idx_n(j)).
//   {A0, B, AC, BC}_rowptr will be allocated in this function.
void calc_spmm_part2d_from_1d(
    const int nproc, const int m, const int n, const int k, const int *rb_displs0,
    const int *rowptr, const int *colidx, int *pm, int *pn, size_t *comm_cost, 
    int **A0_rowptr, int **B_rowptr, int **AC_rowptr, int **BC_colptr, int dbg_print
);

#ifdef __cplusplus
}
#endif

#endif
