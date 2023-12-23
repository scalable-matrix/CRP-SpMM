#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "spmat_part.h"
#include "utils.h"

// Partition a CSR matrix into multiple row blocks s.t. each row block
// has contiguous rows and roughly the same number of non-zero elements
void csr_mat_row_partition(const int nrow, const int *row_ptr, const int nblk, int *rblk_ptr)
{
    int nnz = row_ptr[nrow];
    rblk_ptr[0] = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nblk; i++)
    {
        int i_max_nnz = (nnz / nblk) * (i + 1);
        if (i == nblk - 1) i_max_nnz = nnz;
        int st = 0, end = nrow;
        while (st < end)
        {
            int mid = (st + end) / 2;
            if (row_ptr[mid] == i_max_nnz) 
            {
                st = mid; 
                break;
            }
            if (row_ptr[mid] < i_max_nnz) st = mid + 1;
            else end = mid;
        }
        rblk_ptr[i + 1] = st;
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
    char *thread_flags = (char *) malloc(sizeof(char) * n_thread * ncol);
    memset(comm_sizes, 0, sizeof(int) * nblk);
    #pragma omp parallel
    {
        char *flag = thread_flags + omp_get_thread_num() * ncol;
        #pragma omp for schedule(dynamic)
        for (int iblk = 0; iblk < nblk; iblk++)
        {
            int srow = rblk_ptr[iblk], erow = rblk_ptr[iblk + 1], cnt = 0;
            memset(flag, 0, sizeof(char) * ncol);
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

int prime_factorization(int n, int **factors)
{
    int nfac = 0, c = 2, max_nfac = (int) ceil(log2((double) n)) + 1;
    int *fac = (int *) malloc(sizeof(int) * max_nfac);
    while (n > 1)
    {
        if (n % c == 0)
        {
            fac[nfac++] = c;
            n /= c;
        }
        else c++;
    }
    *factors = fac;
    return nfac;
}

// Calculate a 2D process grid dimensions and matrix partitioning 
// for SpMM from a 1D row partitioning
void calc_spmm_part2d_from_1d(
    const int nproc, const int m, const int n, const int k, const int *rb_displs0,
    const int *rowptr, const int *colidx, int *pm, int *pn, size_t *comm_cost, 
    int **A0_rowptr, int **B_rowptr, int **AC_rowptr, int **BC_colptr, int dbg_print
)
{
    const double nnz_cf = 1.5;  // Cost factor for each nnz (assuming CSR with int32 and double)
    int *m_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *m_displs2  = (int *) malloc(sizeof(int) * (nproc + 1));
    int *k_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *comm_sizes = (int *) malloc(sizeof(int) * nproc);
    int tmp;

    // 1. Use basic 1D partitioning as the initial partitioning
    // If A is square, we partition the rows of B in the same way as A;
    // otherwise, we evenly partition the rows of B
    if (m == k)
    {
        memcpy(k_displs, rb_displs0, sizeof(int) * (nproc + 1)); 
    } else {
        for (int i = 0; i <= nproc; i++) 
            calc_block_spos_size(k, nproc, i, k_displs + i, &tmp);
    }
    csr_mat_row_part_comm_size(
        m, k, rowptr, colidx, nproc, rb_displs0, 
        k_displs, comm_sizes, &tmp
    );
    size_t best_cost = (size_t) tmp * (size_t) n;
    memcpy(m_displs, rb_displs0, sizeof(int) * (nproc + 1));
    if (dbg_print) printf("Basic 1D row partitioning comm cost: %zu\n", best_cost);
    
    // 2. Compute the 2D process grid dimensions
    int pm_ = nproc, pn_ = 1, failed_p = -1;
    int A_nnz = rowptr[m], *fac = NULL;
    int nfac = prime_factorization(nproc, &fac);
    for (int ifac = 0; ifac < nfac; ifac++)
    {
        int p_i = fac[nfac - 1 - ifac];
        if (p_i == failed_p) continue;
        int pn2_ = pn_ * p_i;
        int pm2_ = nproc / pn2_;
        m_displs2[0] = 0;
        for (int i = 0; i <= pm2_; i++) m_displs2[i] = rb_displs0[i * pn2_];
        // If A is square, we partition the rows of B in the same way as A;
        // otherwise, we evenly partition the rows of B
        if (m == k)
        {
            memcpy(k_displs, m_displs2, sizeof(int) * (pm2_ + 1)); 
        } else {
            for (int i = 0; i <= pm2_; i++) 
                calc_block_spos_size(k, pm2_, i, k_displs + i, &tmp);
        }
        double st1 = get_wtime_sec();
        csr_mat_row_part_comm_size(
            m, k, rowptr, colidx, pm2_, m_displs2, 
            k_displs, comm_sizes, &tmp
        );
        double et1 = get_wtime_sec();
        size_t A_copy_cost = (size_t) ((double) A_nnz * (double) (pn2_ - 1) * nnz_cf);
        size_t B_copy_cost = (size_t) tmp * (size_t) n;
        size_t curr_cost   = A_copy_cost + B_copy_cost;
        if (dbg_print)
        {
            printf("Step %d, factor %d, time = %.2f\n", ifac, p_i, et1 - st1);
            printf("Evaluated: pm = %d, pn = %d, cost = %zu\n", pm2_, pn2_, curr_cost);
            if (curr_cost < best_cost) printf("Found better partitioning\n");
        }
        if (curr_cost < best_cost)
        {
            best_cost = curr_cost;
            pn_ = pn2_;
            pm_ = pm2_;
            memcpy(m_displs, m_displs2, sizeof(int) * (pm2_ + 1));
            failed_p = -1;
        } else {
            failed_p = p_i;
        }
    }  // End of ifac loop
    *comm_cost = best_cost;
    if (dbg_print) printf("Final 2D partitioning: pm = %d, pn = %d, cost = %zu\n", pm_, pn_, best_cost);

    // 3. Copy the partitioning results
    *pm = pm_;
    *pn = pn_;
    int *B_rowptr_  = (int *) malloc(sizeof(int) * (pm_ + 1));
    int *AC_rowptr_ = (int *) malloc(sizeof(int) * (pm_ + 1));
    int *BC_colptr_ = (int *) malloc(sizeof(int) * (pn_ + 1));
    memcpy(AC_rowptr_, m_displs, sizeof(int) * (pm_ + 1));
    // If A is square, we partition the rows of B in the same way as A;
    // otherwise, we evenly partition the rows of B
    if (m == k)
    {
        memcpy(B_rowptr_, AC_rowptr_, sizeof(int) * (pm_ + 1));
    } else {
        for (int i = 0; i <= pm_; i++) 
            calc_block_spos_size(k, pm_, i, B_rowptr_ + i, &tmp);
    }
    for (int i = 0; i <= pn_; i++) 
        calc_block_spos_size(n, pn_, i, BC_colptr_ + i, &tmp);
    *B_rowptr  = B_rowptr_;
    *AC_rowptr = AC_rowptr_;
    *BC_colptr = BC_colptr_;

    // 4. Compute 1D partitioning for A
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
}
