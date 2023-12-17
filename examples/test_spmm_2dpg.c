#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "mmio_utils.h"
#include "spmat_part.h"
#include "utils.h"

int main(int argc, char **argv) 
{
    if (argc < 4)
    {
        printf("Usage: %s <mtx-file> <num-of-B-col> <num-of-processes)>\n", argv[0]);
        return 255;
    }

    int m, n, k, np, nnz, bw = 0;
    int *row, *col, *rowptr, *colidx;
    double *val, *csrval;
    n  = atoi(argv[2]);
    np = atoi(argv[3]);
    printf("Reading matrix A from file %s\n", argv[1]);
    mm_read_sparse_RPI_GS(argv[1], &m, &k, &nnz, &row, &col, &val);
    #pragma omp parallel for reduction(max:bw)
    for (int i = 0; i < nnz; i++)
    {
        int bw_i = abs(row[i] - col[i]);
        bw = (bw > bw_i) ? bw : bw_i;
    }
    printf("A size = %d * %d, nnz = %d, nnz/row = %d, bandwidth = %d\n\n", m, k, nnz, nnz / m, bw);
    coo2csr(m, k, nnz, row, col, val, &rowptr, &colidx, &csrval);

    int pm = 0, pn = 0;
    size_t comm_cost = 0;
    int *A0_rowptr = NULL, *B_rowptr = NULL, *AC_rowptr = NULL, *BC_colptr = NULL;
    double st = get_wtime_sec();
    calc_spmm_2dpg(
        np, m, n, k, rowptr, colidx, &pm, &pn, &comm_cost, 
        &A0_rowptr, &B_rowptr, &AC_rowptr, &BC_colptr, 1
    );
    double et = get_wtime_sec();

    printf("============================================================\n");
    printf("Calculate partitioning time = %.2f s\n", et - st);
    printf("Calculated 2D grid: pm, pn = %d, %d, comm cost = %zu\n\n", pm, pn, comm_cost);

    printf("1D row partitioning of A:\n");
    for (int i = 0; i < pm; i++)
    {
        for (int j = 0; j < pn; j++)
        {
            int rank = i * pn + j;
            printf("Rank %3d: [%d, %d]\n", rank, A0_rowptr[rank], A0_rowptr[rank + 1] - 1);
        }
        int rank_s = i * pn, rank_e = (i + 1) * pn - 1;
        printf("Ranks [%d, %d] all own A rows [%d, %d] after replicating A\n", rank_s, rank_e, A0_rowptr[rank_s], A0_rowptr[rank_e + 1] - 1);
    }
    printf("\n");

    printf("1D row partitioning of B:\n");
    for (int i = 0; i < pm; i++)
        printf("Block %d: [%d, %d]\n", i, B_rowptr[i], B_rowptr[i + 1] - 1);
    printf("\n");

    printf("1D row partitioning of C:\n");
    for (int i = 0; i < pm; i++)
        printf("Block %d: [%d, %d]\n", i, AC_rowptr[i], AC_rowptr[i + 1] - 1);
    printf("\n");

    printf("1D column partitioning of B and C:\n");
    for (int i = 0; i < pn; i++)
        printf("Block %d: [%d, %d]\n", i, BC_colptr[i], BC_colptr[i + 1] - 1);
    printf("\n");

    free(row);
    free(col);
    free(val);
    free(rowptr);
    free(colidx);
    free(csrval);
    free(A0_rowptr);
    free(B_rowptr);
    free(AC_rowptr);
    free(BC_colptr);
    return 0;
}