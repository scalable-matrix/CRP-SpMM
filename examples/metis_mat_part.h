#ifndef __METIS_MAT_PART_H__
#define __METIS_MAT_PART_H__

#include "metis.h"

#ifdef __cplusplus
extern "C" {
#endif

// Find a 1D partitioning using METIS, return a new sparse matrix with symmetic permutation
// Input parameters:
//   nrow   : Number of rows of the symmetric sparse matrix
//   nproc  : Number of processors
//   rowptr : Size nrow + 1, CSR row pointer array
//   colidx : Size rowptr[nrow], CSR column index array
//   val    : Size rowptr[nrow], CSR value array
// Output parameters:
//   rowptr : Size nrow + 1, CSR row pointer array, updated with the permuted matrix
//   colidx : Size rowptr[nrow], CSR column index array, updated with the permuted matrix
//   val    : Size rowptr[nrow], CSR value array, updated with the permuted matrix
//   perm   : Size nrow, the original i-th row/col is permuted to the perm[i]-th row/col
//   row_displs : Size nproc + 1, the row displacement of each processor on the permuted matrix
// Note: this function assumes that the input matrix is symmetric, and has both upper and lower
//       triangular parts. The output matrix is also symmetric, and has both upper and lower parts.
void METIS_row_partition(
    const int nrow, const int nproc, int *rowptr, int *colidx, double *val,
    int *perm, int *row_displs
);

// Permute a symmetric sparse matrix with METIS, then compute a 2D process
// grid dimention and matrix partitioning for SpMM
// Input paratemers:
//   nproc   : number of processes
//   m, n    : size of matrix A (m * m), B (m * n), and C (m * n)
//   rowptr  : Size m + 1, CSR row pointer of A
//   colidx  : Size rowptr[m], CSR column index of A
//   val     : Size rowptr[m], CSR value of A
// Output parameters:
//   perm       : Size nrow, the original i-th row/col is permuted to the perm[i]-th row/col
//   rowptr     : Size m + 1, CSR row pointer of permuted pA
//   colidx     : Size rowptr[m], CSR column index of permuted pA
//   val        : Size rowptr[m], CSR value of permuted pA
//   *pm, *pn   : Process grid dimensions, pn groups * pm-way row parallel SpMM
//   *comm_cost : SpMM communication cost of the partitioning
//   *A0_rowptr : Size nproc + 1, row pointer of pA0, pA0 is 1D row partitioned
//   *B_rowptr  : Size pm + 1, row pointer of B
//   *AC_rowptr : Size pm + 1, row pointer of replicated pA and final C
//   *BC_colptr : Size pn + 1, column pointer of B and C
// Notes: 
//   Let idx_m0(i) = [A0_rowptr[i] : A0_rowptr[i+1] - 1],
//       idx_m(i)  = [AC_rowptr[i] : AC_rowptr[i+1] - 1], 
//       idx_k(i)  = [B_rowptr[i]  : B_rowptr[i+1] - 1],
//       idx_n(j)  = [BC_colptr[j] : BC_colptr[j+1] - 1].
//   Before redistributing pA, process P_{i, j} owns pA(idx_m0(i * pn + j), :).
//   Before redistributing B, process P_{i, j} owns B(idx_k(i), idx_n(j)).
//   P_{i, j} computes C(idx_m(i), idx_n(j)) = pA(idx_m(i), :) * B(:, idx_n(j)).
//   {A0, B, AC, BC}_rowptr will be allocated in this function.
void METIS_spmm_2dpg(
    const int nproc, const int m, const int n, int *perm, 
    int *rowptr, int *colidx, double *val, int *pm, int *pn, size_t *comm_cost,
    int **A0_rowptr, int **B_rowptr, int **AC_rowptr, int **BC_colptr, int dbg_print
);

#ifdef __cplusplus
}
#endif

#endif
