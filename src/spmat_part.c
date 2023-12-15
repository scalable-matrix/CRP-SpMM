#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "spmat_part.h"

// Partition a CSR matrix into multiple row blocks s.t. each row block
// has contiguous rows and roughly the same number of non-zero elements
void csr_mat_row_partition(const int nrow, const int *row_ptr, const int nblk, int *rblk_ptr)
{
    int nnz = row_ptr[nrow];
    int srow = 0, erow = 0;
    rblk_ptr[0] = 0;
    for (int i = 0; i < nblk; i++)
    {
        int i_max_nnz = (nnz / nblk) * (i + 1);
        if (i == nblk - 1) i_max_nnz = nnz;
        erow = srow + 1;
        while (row_ptr[erow] < i_max_nnz) erow++;
        rblk_ptr[i + 1] = erow;
    }
}

// Compute the SpMV communication sizes for each row block
void csr_mat_row_part_comm_size(
    const int nrow, const int ncol, const int *row_ptr, const int *col_idx,
    const int nblk, const int *rblk_ptr, const int *x_displs, 
    int *comm_sizes, int *total_size
)
{
    int n_thread = omp_get_max_threads();
    int *thread_flags = (int *) malloc(sizeof(int) * n_thread * ncol);
    memset(comm_sizes, 0, sizeof(int) * nblk);
    #pragma omp parallel
    {
        int *flag = thread_flags + omp_get_thread_num() * ncol;
        #pragma omp for schedule(dynamic)
        for (int iblk = 0; iblk < nblk; iblk++)
        {
            int srow = rblk_ptr[iblk], erow = rblk_ptr[iblk + 1], cnt = 0;
            memset(flag, 0, sizeof(int) * ncol);
            for (int j = row_ptr[srow]; j < row_ptr[erow]; j++) flag[col_idx[j]] = 1;
            for (int i = 0; i < ncol; i++) cnt += flag[i];
            for (int i = x_displs[iblk]; i < x_displs[iblk + 1]; i++) cnt -= flag[i];
            comm_sizes[iblk] = cnt;
        }  // End of iblk loop
    }  // End of "#pragma omp parallel"
    *total_size = 0;
    for (int i = 0; i < nblk; i++) *total_size += comm_sizes[i];
    free(thread_flags);
}
