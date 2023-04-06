#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "mmio.h"
#include "mmio_utils.h"

// Read a sparse {Real, Pattern, Integer | General, Symmetric} matrix in Matrix Market format
int mm_read_sparse_RPI_GS(const char *fname, int *nrow_, int *ncol_, int *nnz_, int **row_, int **col_, double **val_)
{
    FILE *f;
    MM_typecode matcode;
    int i, nrow, ncol, nnz, valid_type, idx, ival;
    int *row, *col;
    double *val;
 
    if ((f = fopen(fname, "r")) == NULL) return -1;
 
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_sparse_RPI_GS(): Could not process Matrix Market banner in file [%s]\n", fname);
        return -1;
    }

    valid_type = mm_is_matrix(matcode) && mm_is_sparse(matcode);
    valid_type = valid_type && (mm_is_real(matcode) || mm_is_pattern(matcode) || mm_is_integer(matcode));
    valid_type = valid_type && (mm_is_general(matcode) || mm_is_symmetric(matcode));
    if (!valid_type)
    {
        fprintf(stderr, "mm_read_sparse_RPI_GS(): does not support Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return -1;
    }

    if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0)
    {
        fprintf(stderr, "mm_read_sparse_RPI_GS(): could not parse matrix size.\n");
        return -1;
    }

    *nrow_ = nrow;
    *ncol_ = ncol;
    if (mm_is_general(matcode))
    {
        row = (int *) malloc(sizeof(int) * nnz);
        col = (int *) malloc(sizeof(int) * nnz);
        val = (double *) malloc(sizeof(double) * nnz);
        *nnz_ = nnz;
    }
    if (mm_is_symmetric(matcode))
    {
        row = (int *) malloc(sizeof(int) * nnz * 2);
        col = (int *) malloc(sizeof(int) * nnz * 2);
        val = (double *) malloc(sizeof(double) * nnz * 2);
    }

    // Read all matrix entries first
    if (mm_is_real(matcode))
    {
        for (i = 0; i < nnz; i++) 
            fscanf(f, "%d %d %lf\n", &row[i], &col[i], &val[i]);
    }
    if (mm_is_integer(matcode))
    {
        for (i = 0; i < nnz; i++)
        {
            fscanf(f, "%d %d %d\n", &row[i], &col[i], &ival);
            val[i] = (double) ival;
        }
    }
    if (mm_is_pattern(matcode))
    {
        for (i = 0; i < nnz; i++)
        {
            fscanf(f, "%d %d\n", &row[i], &col[i]);
            val[i] = 1.0;
        }
    }

    // Adjust row and column indices to 0-based
    for (i = 0; i < nnz; i++)
    {
        row[i]--;
        col[i]--;
    }

    // If the matrix is symmetric, need to add the symmetric entries
    idx = nnz;
    if (mm_is_symmetric(matcode))
    {
        for (i = 0; i < nnz; i++)
        {
            if (row[i] != col[i])
            {
                row[idx] = col[i];
                col[idx] = row[i];
                val[idx] = val[i];
                idx++;
            }
        }
        *nnz_ = idx;
    }

    *row_ = row;
    *col_ = col;
    *val_ = val;
    fclose(f);
 
    return 0;
}

static void qsort_ascend_int_dbl_pair(int *key, double *val, int l, int r)
{
    int i = l, j = r, tmp_key;
    int mid_key = key[(l + r) / 2];
    double tmp_val;
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) qsort_ascend_int_dbl_pair(key, val, i, r);
    if (j > l) qsort_ascend_int_dbl_pair(key, val, l, j);
}

// Convert a COO matrix to a sorted CSR matrix
void coo2csr(
    const int nrow, const int ncol, const int nnz,
    const int *row, const int *col, const double *val, 
    int **row_ptr_, int **col_idx_, double **csr_val_
)
{
    int *row_ptr = (int *) malloc(sizeof(int) * (nrow + 1));
    int *col_idx = (int *) malloc(sizeof(int) * nnz);
    double *csr_val = (double *) malloc(sizeof(double) * nnz);

    // Get the number of non-zeros in each row
    memset(row_ptr, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) row_ptr[row[i] + 1]++;

    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) row_ptr[i] += row_ptr[i - 1];

    // Use row_ptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = row_ptr[row[i]];
        col_idx[idx] = col[i];
        csr_val[idx] = val[i];
        row_ptr[row[i]]++;
    }

    // Reset row_ptr
    for (int i = nrow; i >= 1; i--) row_ptr[i] = row_ptr[i - 1];
    row_ptr[0] = 0;

    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        qsort_ascend_int_dbl_pair(col_idx, csr_val, row_ptr[i], row_ptr[i + 1] - 1);

    *row_ptr_ = row_ptr;
    *col_idx_ = col_idx;
    *csr_val_ = csr_val;
}
