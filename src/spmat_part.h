#ifndef __SPMAT_PART_H__
#define __SPMAT_PART_H__

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

#ifdef __cplusplus
}
#endif

#endif
