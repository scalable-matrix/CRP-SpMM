#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"

#include "utils.h"
#include "mmio_utils.h"
#include "spmat_part.h"
#include "metis_mat_part.h"

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
    const int nrow, const int nproc, int *rowptr, int *colidx, 
    double *val, int *perm, int *row_displs
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
    // the original i-th vertex is permuted to the perm[i]-th vertex
    memset(row_displs, 0, sizeof(int) * (nproc + 1));
    for (int i = 0; i < nrow; i++) row_displs[part[i] + 1]++;
    for (int i = 1; i <= nproc; i++) row_displs[i] += row_displs[i - 1];
    int *orig_idx = (int *) malloc(sizeof(int) * nrow);
    for (int i = 0; i < nrow; i++) orig_idx[i] = i;
    qsort_ascend_int_int_pair(part, orig_idx, 0, nrow - 1);
    for (int i = 0; i < nrow; i++) perm[orig_idx[i]] = i;

    // Do a symmetric permutation
    int *row1 = (int *) malloc(sizeof(int) * nnz);
    int *col1 = (int *) malloc(sizeof(int) * nnz);
    ASSERT_PRINTF(row1 != NULL && col1 != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__ );
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < nrow; i++)
    {
        int pi = perm[i];
        for (int idx = rowptr[i]; idx < rowptr[i + 1]; idx++)
        {
            int j = colidx[idx];
            int pj = perm[j];
            row1[idx] = pi;
            col1[idx] = pj;
        }
    }

    // Generate the new CSR matrix and copy back
    int *rowptr1 = NULL;
    int *colidx1 = NULL;
    double *csrval1 = NULL;
    coo2csr(nrow, nrow, nnz, row1, col1, val, &rowptr1, &colidx1, &csrval1);
    memcpy(rowptr, rowptr1, sizeof(int) * (nrow + 1));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nnz; i++)
    {
        colidx[i] = colidx1[i];
        val[i] = csrval1[i];
    }

    free(part);
    free(orig_idx);
    free(row1);
    free(col1);
    free(rowptr1);
    free(colidx1);
    free(csrval1);
}

// Permute a symmetric sparse matrix with METIS, then compute a 2D process
// grid dimention and matrix partitioning for SpMM
void METIS_spmm_2dpg(
    const int nproc, const int m, const int n, int *perm, 
    int *rowptr, int *colidx, double *val, int *pm, int *pn, size_t *comm_cost,
    int **A0_rowptr, int **B_rowptr, int **AC_rowptr, int **BC_colptr, int dbg_print
)
{
    // 1. Compute the basic 1D partitioning
    int *row_displs0 = (int *) malloc(sizeof(int) * (nproc + 1));
    double st = get_wtime_sec();
    METIS_row_partition(m, nproc, rowptr, colidx, val, perm, row_displs0);
    double et = get_wtime_sec();
    printf("METIS_row_partition done, time = %.2f s\n", et - st);

    // 2. Compute the 2D process grid dimensions
    const double nnz_cf = 1.5;  // Cost factor for each nnz (assuming CSR with int32 and double)
    int pm_ = 1, pn_ = 1;
    int *m_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *m_displs2  = (int *) malloc(sizeof(int) * (nproc + 1));
    int *k_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *comm_sizes = (int *) malloc(sizeof(int) * nproc);
    int nfac, *fac = NULL, tmp;
    nfac = prime_factorization(nproc, &fac);
    // If we never choose to split m-dim, we need to initialize m_displs1 to be [0, m]
    m_displs[0] = 0;
    m_displs[1] = m;
    const int A_nnz = rowptr[m];
    size_t curr_B_copy_cost = 0;  // Originally we do not need to copy any B matrix elements
    for (int ifac = 0; ifac < nfac; ifac++)
    {
        int p_i = fac[nfac - 1 - ifac];
        // If split n-dim, the number of B matrix elements to be copied remains unchanged,
        // the number of A matrix elements to be copied increases by a factor of p_i
        size_t A_copy_cost1 = (size_t) ((double) A_nnz * (double) (      pn_ - 1) * nnz_cf);
        size_t A_copy_cost2 = (size_t) ((double) A_nnz * (double) (p_i * pn_ - 1) * nnz_cf);
        size_t split_n_cost = A_copy_cost2 + curr_B_copy_cost;
        // If split m-dim, the number of A matrix copies remains unchanged, needs to 
        // recalculated the number of B elements to be copied. Merge every (nproc/pm2_) 
        // partitions in the basic 1D partitioning into one partition.
        int pm2_ = pm_ * p_i;
        int n_merge = nproc / pm2_;
        m_displs2[0] = 0;
        for (int i = 0; i <= pm2_; i++) m_displs2[i] = row_displs0[i * n_merge];
        // Partition the rows of B in the same way as A
        memcpy(k_displs, m_displs2, sizeof(int) * (pm2_ + 1)); 
        csr_mat_row_part_comm_size(
            m, m, rowptr, colidx, pm2_, m_displs2, 
            k_displs, comm_sizes, &tmp
        );
        size_t new_B_copy_cost = (size_t) tmp * (size_t) n;
        size_t split_m_cost = A_copy_cost1 + new_B_copy_cost;
        if (dbg_print)
        {
            printf("Step %d, factor %d\n", ifac, p_i);
            printf(
                "Split m-dim (pm = %d, pn = %d) cost: copy A = %zu, copy B = %zu, total = %zu\n", 
                pm2_, pn_, A_copy_cost1, new_B_copy_cost, split_m_cost
            );
            printf(
                "Split n-dim (pm = %d, pn = %d) cost: copy A = %zu, copy B = %zu, total = %zu\n", 
                pm_, p_i * pn_, A_copy_cost2, curr_B_copy_cost, split_n_cost
            );
        }
        // Choose to split M or N
        if (split_n_cost < split_m_cost)
        {
            pn_ *= p_i;
            *comm_cost = split_n_cost;
            if (dbg_print) printf("Split n-dim, current pm = %d, pn = %d\n\n", pm_, pn_);
        } else {
            pm_ *= p_i;
            memcpy(m_displs, m_displs2, sizeof(int) * (pm_ + 1));
            curr_B_copy_cost = new_B_copy_cost;
            *comm_cost = split_m_cost;
            if (dbg_print) printf("Split m-dim, current pm = %d, pn = %d\n\n", pm_, pn_);
        }
    }  // End of ifac loop

    // 3. Copy the partitioning results
    *pm = pm_;
    *pn = pn_;
    int *B_rowptr_  = (int *) malloc(sizeof(int) * (pm_ + 1));
    int *AC_rowptr_ = (int *) malloc(sizeof(int) * (pm_ + 1));
    int *BC_colptr_ = (int *) malloc(sizeof(int) * (pn_ + 1));
    memcpy(AC_rowptr_, m_displs, sizeof(int) * (pm_ + 1));
    memcpy(B_rowptr_,  m_displs, sizeof(int) * (pm_ + 1));
    for (int i = 0; i <= pn_; i++) 
        calc_block_spos_size(n, pn_, i, BC_colptr_ + i, &tmp);
    *B_rowptr  = B_rowptr_;
    *AC_rowptr = AC_rowptr_;
    *BC_colptr = BC_colptr_;

    // 4. Compute 1D partitioning for pA
    int *A0_rowptr_ = (int *) malloc(sizeof(int) * (nproc + 1));
    int *tmp_rowptr = (int *) malloc(sizeof(int) * (m + 1));
    for (int im = 0; im < pm_; im++)
    {
        int srow_i = m_displs[im];
        int erow_i = m_displs[im + 1];
        int nrow_i = erow_i - srow_i;
        for (int i = srow_i; i <= erow_i; i++) 
            tmp_rowptr[i - srow_i] = rowptr[i] - rowptr[srow_i];
        int *A0_rowptr_i = A0_rowptr_ + im * pn_;
        csr_mat_row_partition(nrow_i, tmp_rowptr, pn_, A0_rowptr_i);
        for (int j = 0; j <= pn_; j++) A0_rowptr_i[j] += srow_i;
    }
    *A0_rowptr = A0_rowptr_;

    free(m_displs);
    free(m_displs2);
    free(k_displs);
    free(comm_sizes);
    free(fac);
    free(tmp_rowptr);
    free(row_displs0);
}
