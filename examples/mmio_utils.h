#ifndef __MMIO_UTILS_H__
#define __MMIO_UTILS_H__

#include "mmio.h"

#ifdef __cplusplus
extern "C" {
#endif

// Read a sparse {Real, Pattern, Integer} matrix in Matrix Market format
// Input parameter:
//   fname     : Input Matrix Market file name
//   need_symm : Whether the matrix must be symmetric
// Output parameters:
//   *nrow_, *ncol_, *nnz_  : Number of rows, columns, and nonzeros
//   **row_, **col_, **val_ : Row indices, column indices, and values
int mm_read_sparse_RPI(
    const char *fname, const int need_symm, int *nrow_, int *ncol_, int *nnz_, 
    int **row_, int **col_, double **val_
);

// Convert a COO matrix to a sorted CSR matrix
// Input parameters:
//   nrow, ncol, nnz : Number of rows, columns, and nonzeros
//   row, col, val   : Row indices, column indices, and values
// Output parameters:
//   **row_ptr_, **col_idx_, **csr_val_ : CSR row pointers, column indices, and values
void coo2csr(
    const int nrow, const int ncol, const int nnz, 
    const int *row, const int *col, const double *val, 
    int **row_ptr_, int **col_idx_, double **csr_val_
);

#ifdef __cplusplus
}
#endif

#endif
