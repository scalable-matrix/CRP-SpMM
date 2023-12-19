#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "mmio_utils.h"

static int prime_factorization(int n, int **factors)
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

int main(int argc, char **argv) 
{
    if (argc < 4)
    {
        printf("Usage: %s <mtx-file> <num-of-B-col> <num-of-processes)>\n", argv[0]);
        return 255;
    }

    int m, n, k, np, nnz, bw = 0, need_symm = 0;
    int *row, *col, *rowptr, *colidx;
    double *val, *csrval;
    n  = atoi(argv[2]);
    np = atoi(argv[3]);
    printf("Reading matrix A from file %s\n", argv[1]);
    mm_read_sparse_RPI(argv[1], need_symm, &m, &k, &nnz, &row, &col, &val);
    #pragma omp parallel for reduction(max:bw)
    for (int i = 0; i < nnz; i++)
    {
        int bw_i = abs(row[i] - col[i]);
        bw = (bw > bw_i) ? bw : bw_i;
    }
    printf("A size = %d * %d, nnz = %d, nnz/row = %d, bandwidth = %d\n\n", m, k, nnz, nnz / m, bw);
    coo2csr(m, k, nnz, row, col, val, &rowptr, &colidx, &csrval);

    double st = omp_get_wtime();

    // For CSR with <int32_t> indices and <double> values, the average memory cost
    // per nonzero is 12.x bytes, which is 1.5x times of sizeof(double)
    const double A_nnz_cost_factor = 1.5;
    int nfac, *fac;
    nfac = prime_factorization(np, &fac);
    size_t curr_copy_B_size = (size_t) k * (size_t) n;  // Originally we have one copy of B
    int A_nnz = rowptr[m];
    int m_split = 1, n_split = 1;
    for (int i = 0; i < nfac; i++)
    {
        int p_i = fac[nfac - 1 - i];
        printf("Step %d, split size = %d\n", i, p_i);
        // If split N, the number of B matrix elements to be copied remains unchanged, 
        // the number of A matrix copies is multiplied by p_i
        size_t A_copy_cost1 = (size_t) ((double) A_nnz * (double) n_split * A_nnz_cost_factor);
        size_t A_copy_cost2 = A_copy_cost1 * (size_t) p_i;
        size_t split_n_comm_cost = A_copy_cost2 + curr_copy_B_size;
        printf("Split N cost: copy A = %zu, copy B = %zu, total = %zu\n", A_copy_cost2, curr_copy_B_size, split_n_comm_cost);
        if (n_split * p_i > n) split_n_comm_cost = SIZE_MAX;
        // If split M, the number of A matrix copies remains unchanged, needs to 
        // recalculate the number of B matrix elements to be copied
        size_t split_m_comm_cost = A_copy_cost1;
        size_t curr_copy_B_size2 = 0;
        m_split *= p_i;
        int curr_srow = 0;
        printf("Split M: \n");
        for (int j = 0; j < m_split; j++)
        {
            // Find the last row assigned to the j-th row panel
            int j_max_nnz = A_nnz / m_split * (j + 1);
            if (j == m_split - 1) j_max_nnz = A_nnz;
            int curr_erow = curr_srow + 1;
            int min_col_idx = colidx[rowptr[curr_srow]];
            int max_col_idx = colidx[rowptr[curr_srow + 1] - 1]; 
            while (rowptr[curr_erow] < j_max_nnz)
            {
                // Calculate the upper bound of number of B rows required by the j-th row panel
                int min_col_idx2 = colidx[rowptr[curr_erow]];
                int max_col_idx2 = colidx[rowptr[curr_erow + 1] - 1]; 
                if (min_col_idx2 < min_col_idx) min_col_idx = min_col_idx2;
                if (max_col_idx2 > max_col_idx) max_col_idx = max_col_idx2;
                curr_erow++;
            }
            int copy_B_rows = max_col_idx - min_col_idx + 1;
            printf(
                "Row block %d: [%d, %d), B rows to copy: [%d, %d) (%d)\n", 
                j, curr_srow, curr_erow, min_col_idx, max_col_idx + 1, copy_B_rows);
            curr_copy_B_size2 += (size_t) copy_B_rows * (size_t) n;
            curr_srow = curr_erow;
        }
        split_m_comm_cost += curr_copy_B_size2;
        m_split /= p_i;
        printf("Split M cost: copy A = %zu, copy B = %zu, total = %zu\n", A_copy_cost1, curr_copy_B_size2, split_m_comm_cost);
        // Choose to split M or N
        if (split_m_comm_cost < split_n_comm_cost)
        {
            m_split *= p_i;
            curr_copy_B_size = curr_copy_B_size2;
            printf("Split M, current m_split = %d, n_split = %d\n\n", m_split, n_split);
        } else {
            n_split *= p_i;
            printf("Split N, current m_split = %d, n_split = %d\n\n", m_split, n_split);
        }
    }  // End of i loop

    double et = omp_get_wtime();
    printf("Calculate partitioning time = %.2f s\n", et - st);

    free(fac);
    free(row);
    free(col);
    free(val);
    free(rowptr);
    free(colidx);
    free(csrval);
    return 0;
}