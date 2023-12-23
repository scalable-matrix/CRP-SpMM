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
        int st = 0, end = nrow - 1;
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

// Calculate a 2D process grid dimensions and matrix partitioning for SpMM
void calc_spmm_2dpg(
    const int nproc, const int m, const int n, const int k,
    const int *rowptr, const int *colidx, int *pm, int *pn, size_t *comm_cost, 
    int **A0_rowptr, int **B_rowptr, int **AC_rowptr, int **BC_colptr, int dbg_print
)
{
    // 1. Compute the basic 1D partitioning
    int *row_displs0 = (int *) malloc(sizeof(int) * (nproc + 1));
    double st = get_wtime_sec();
    csr_mat_row_partition(m, rowptr, nproc, row_displs0);
    double et = get_wtime_sec();
    if (dbg_print) printf("Compute basic 1D row partitioning time = %.2f s\n", et - st);

    // 2. Compute the 2D process grid dimensions
    const double nnz_cf = 1.5;  // Cost factor for each nnz (assuming CSR with int32 and double)
    int pm_ = 1, pn_ = 1;
    int *m_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *m_displs2  = (int *) malloc(sizeof(int) * (nproc + 1));
    int *k_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *comm_sizes = (int *) malloc(sizeof(int) * nproc);
    int nfac, *fac = NULL, tmp;

    // If A is square, we partition the rows of B in the same way as A;
    // otherwise, we evenly partition the rows of B
    if (m == k)
    {
        memcpy(k_displs, row_displs0, sizeof(int) * (nproc + 1)); 
    } else {
        for (int i = 0; i <= nproc; i++) 
            calc_block_spos_size(k, nproc, i, k_displs + i, &tmp);
    }
    csr_mat_row_part_comm_size(
        m, k, rowptr, colidx, nproc, row_displs0, 
        k_displs, comm_sizes, &tmp
    );
    size_t rowpara_cost = (size_t) tmp * (size_t) n;
    if (dbg_print) printf("Basic 1D row partitioning comm cost: %zu\n", rowpara_cost);
    
    // If we never choose to split m-dim, we need to initialize m_displs1 to be [0, m]
    m_displs[0] = 0;
    m_displs[1] = m;
    const int A_nnz = rowptr[m];
    size_t curr_B_copy_cost = 0;  // Originally we do not need to copy any B matrix elements

    nfac = prime_factorization(nproc, &fac);
    for (int ifac = 0; ifac < nfac; ifac++)
    {
        int p_i = fac[nfac - 1 - ifac];
        // If split n-dim, the number of B matrix elements to be copied remains unchanged,
        // the number of A matrix elements to be copied increases by a factor of p_i
        size_t A_copy_cost1 = (size_t) ((double) A_nnz * (double) (      pn_ - 1) * nnz_cf);
        size_t A_copy_cost2 = (size_t) ((double) A_nnz * (double) (p_i * pn_ - 1) * nnz_cf);
        size_t split_n_cost = A_copy_cost2 + curr_B_copy_cost;
        // If split m-dim, the number of A matrix copies remains unchanged, needs to 
        // recalculated the number of B elements to be copied
        int pm2_ = pm_ * p_i;
        int n_merge = nproc / pm2_;
        m_displs2[0] = 0;
        for (int i = 0; i <= pm2_; i++) m_displs2[i] = row_displs0[i * n_merge];
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
        size_t new_B_copy_cost = (size_t) tmp * (size_t) n;
        size_t split_m_cost = A_copy_cost1 + new_B_copy_cost;
        if (dbg_print)
        {
            printf("Step %d, factor %d, time = %.2f\n", ifac, p_i, et1 - st1);
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
    if (*comm_cost > rowpara_cost)
    {
        pm_ = nproc;
        pn_ = 1;
        memcpy(m_displs, row_displs0, sizeof(int) * (nproc + 1));
        *comm_cost = rowpara_cost;
        if (dbg_print) printf("Use basic 1D partitioning, pm = %d, pn = %d\n\n", pm_, pn_);
    }

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

    free(row_displs0);
    free(m_displs);
    free(m_displs2);
    free(k_displs);
    free(comm_sizes);
    free(fac);
    free(tmp_rowptr);
}
