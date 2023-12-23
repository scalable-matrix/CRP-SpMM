#include "test_utils.h"
#include "spmat_part.h"
#include "metis_mat_part.h"

int main(int argc, char **argv) 
{
    if (argc < 5)
    {
        printf("Usage: %s <mtx-file> <num-of-B-col> <num-of-processes> <part-method>\n", argv[0]);
        printf("<part-method>: 0 for native 1D partition, 1 for METIS 1D partition (symmetric matrix only)\n");
        return 255;
    }
    int n         = atoi(argv[2]);
    int nproc     = atoi(argv[3]);
    int method    = atoi(argv[4]);
    int need_symm = (method == 0) ? 0 : 1;

    int m, k;
    int *rowptr = NULL, *colidx = NULL;
    double *val;
    read_mtx_csr(argv[1], need_symm, &m, &k, n, &rowptr, &colidx, &val);

    int pm = 0, pn = 0, dbg_print = 1;
    size_t comm_cost = 0;
    int *A_rb_displs = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A0_rowptr = NULL, *B_rowptr = NULL, *AC_rowptr = NULL, *BC_colptr = NULL;
    double st, et, t1, t2;
    printf("============================================================\n");
    st = get_wtime_sec();
    if (method == 0)
    {
        csr_mat_row_partition(m, rowptr, nproc, A_rb_displs);
    } else {
        int *perm = (int *) malloc(sizeof(int) * m);
        METIS_row_partition(m, nproc, rowptr, colidx, val, perm, A_rb_displs);
        free(perm);
    }
    et = get_wtime_sec();
    t1 = et - st;
    printf("Calculate 1D row partitioning time = %.2f s\n", t1);
    st = get_wtime_sec();
    calc_spmm_part2d_from_1d(
        nproc, m, n, k, A_rb_displs, rowptr, colidx,
        &pm, &pn, &comm_cost, &A0_rowptr, &B_rowptr, &AC_rowptr, &BC_colptr, dbg_print
    );
    et = get_wtime_sec();
    t2 = et - st;
    printf("Calculate 2D partitioning from 1D partitioning time = %.2f s\n", t2);
    printf("Total partitioning time = %.2f s\n", t1 + t2);
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

    free(val);
    free(rowptr);
    free(colidx);
    free(A_rb_displs);
    free(A0_rowptr);
    free(B_rowptr);
    free(AC_rowptr);
    free(BC_colptr);
    return 0;
}