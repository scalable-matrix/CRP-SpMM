#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"

#include "utils.h"
#include "metis_mat_part.h"
#include "mmio_utils.h"

static void qsort_ascend_int_int_pair(int *key, int *val, const int l, const int r)
{
    int i = l, j = r;
    const int mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            int tmp_key = key[i]; key[i] = key[j];  key[j] = tmp_key;
            int tmp_val = val[i]; val[i] = val[j];  val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) qsort_ascend_int_int_pair(key, val, i, r);
    if (j > l) qsort_ascend_int_int_pair(key, val, l, j);
}

// Find a 1D partitioning using METIS, return a new sparse matrix with symmetic permutation
void METIS_row_partition(
    const int nrow, const int nproc, int *rowptr, int *colidx, double *val,
    int *perm_idx, int *row_displs
)
{
    int nnz = rowptr[nrow];

    // idx_t == int32_t
    idx_t  nvtxs    = nrow;         // Number of vertices, == matrix size
    idx_t  ncon     = 1;            // One balance constraint
    idx_t  *xadj    = rowptr;     
    idx_t  *adjncy  = colidx;
    idx_t  *vwgt    = NULL;         // Weights of the vertices, no needed
    idx_t  *vsize   = NULL;
    idx_t  *adjwgt  = NULL;         // Weights of the edges, no needed
    idx_t  nparts   = nproc;        // Number of parts to partition
    real_t *tpwgts  = NULL;
    real_t ubvec    = 1.05;         // 5% imbalance allowed
    idx_t  objval   = 0;            // Returned object value (communication size)
    idx_t  *part    = NULL;         // The partition vector, size nrow
    part = (idx_t *) malloc(sizeof(idx_t) * nrow);

    // Use default options + METIS_OBJTYPE_VOL
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    double st = omp_get_wtime();
    METIS_PartGraphKway(
        &nvtxs, &ncon, xadj, adjncy, 
        vwgt, vsize, adjwgt, &nparts, tpwgts,
        &ubvec, options, &objval, part
    );
    double et = omp_get_wtime();
    printf("METIS_PartGraphKway done, time = %.2f s\n", et - st);

    // Compute row displacements and sort the vertices based on their partition id,
    // the original i-th vertex is permuted to the perm_idx[i]-th vertex
    memset(row_displs, 0, sizeof(int) * (nproc + 1));
    for (int i = 0; i < nrow; i++) row_displs[part[i] + 1]++;
    for (int i = 1; i <= nproc; i++) row_displs[i] += row_displs[i - 1];
    int *orig_idx = (int *) malloc(sizeof(int) * nrow);
    for (int i = 0; i < nrow; i++) orig_idx[i] = i;
    qsort_ascend_int_int_pair(part, orig_idx, 0, nrow - 1);
    for (int i = 0; i < nrow; i++) perm_idx[orig_idx[i]] = i;

    // Do a symmetric permutation
    int *row1 = (int *) malloc(sizeof(int) * nnz);
    int *col1 = (int *) malloc(sizeof(int) * nnz);
    ASSERT_PRINTF(row1 != NULL && col1 != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__ );
    int nnz1 = 0;
    for (int i = 0; i < nrow; i++)
    {
        int pi = perm_idx[i];
        for (int idx = rowptr[i]; idx < rowptr[i + 1]; idx++)
        {
            int j = colidx[idx];
            int pj = perm_idx[j];
            row1[nnz1] = pi;
            col1[nnz1] = pj;
            nnz1++;
        }
    }

    // Generate the new CSR matrix and copy back
    int *rowptr1 = NULL;
    int *colidx1 = NULL;
    double *csrval1 = NULL;
    coo2csr(nrow, nrow, nnz, row1, col1, val, &rowptr1, &colidx1, &csrval1);
    memcpy(rowptr, rowptr1, sizeof(int) * (nrow + 1));
    memcpy(colidx, colidx1, sizeof(int) * nnz);
    memcpy(val, csrval1, sizeof(double) * nnz);

    free(part);
    free(orig_idx);
    free(row1);
    free(col1);
    free(rowptr1);
    free(colidx1);
    free(csrval1);
}
