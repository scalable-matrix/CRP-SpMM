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
//   rowptr     : Size nrow + 1, CSR row pointer array, updated with the permuted matrix
//   colidx     : Size rowptr[nrow], CSR column index array, updated with the permuted matrix
//   val        : Size rowptr[nrow], CSR value array, updated with the permuted matrix
//   perm_idx   : Size nrow, the original i-th row/col is permuted to the perm_idx[i]-th row/col
//   row_displs : Size nproc + 1, the row displacement of each processor on the permuted matrix
// Note: this function assumes that the input matrix is symmetric, and has both upper and lower
//       triangular parts. The output matrix is also symmetric, and has both upper and lower parts.
void METIS_row_partition(
    const int nrow, const int nproc, int *rowptr, int *colidx, double *val,
    int *perm_idx, int *row_displs
);

#ifdef __cplusplus
}
#endif

#endif
