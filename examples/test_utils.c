#include "test_utils.h"

int can_check_res(int my_rank, int m, int n, int k)
{
    int max_n1 = INT_MAX / m;
    int max_n2 = INT_MAX / k;
    int chk_res = 1;
    if ((n > max_n1) || (n > max_n2))
    {
        if (my_rank == 0) 
        {
            printf("The complete B or C matrix is too large to be stored in a dense matrix\n");
            printf("Result validation check will be skipped\n");
            fflush(stdout);
        }
        chk_res = 0;
    }
    return chk_res;
}

void read_mtx_csr(
    const char *fname, const int need_symm, int *glb_m, int *glb_k, int glb_n, 
    int **glb_A_rowptr, int **glb_A_colidx, double **glb_A_csrval
)
{
    int glb_A_nnz = 0, glb_m_, glb_k_, bandwidth = 0;
    int *glb_A_row = NULL, *glb_A_col = NULL;
    double *glb_A_val = NULL;

    printf("B has %d columns\n", glb_n);
    printf("Rank 0 read matrix A from file %s", fname);
    fflush(stdout);
    double st = omp_get_wtime();
    mm_read_sparse_RPI(fname, need_symm, &glb_m_, &glb_k_, &glb_A_nnz, &glb_A_row, &glb_A_col, &glb_A_val);
    coo2csr(glb_m_, glb_k_, glb_A_nnz, glb_A_row, glb_A_col, glb_A_val, glb_A_rowptr, glb_A_colidx, glb_A_csrval);
    double et = omp_get_wtime();
    #pragma omp parallel for reduction(max:bandwidth)
    for (int i = 0; i < glb_A_nnz; i++)
    {
        int bw_i = abs(glb_A_row[i] - glb_A_col[i]);
        if (bw_i > bandwidth) bandwidth = bw_i;
    }
    printf(" used %.2f s\n", et - st);
    printf(
        "A size = %d * %d, nnz = %d, nnz/row = %d, bandwidth = %d\n\n", 
        glb_m_, glb_k_, glb_A_nnz, glb_A_nnz / glb_m_, bandwidth
    );
    fflush(stdout);
    *glb_m = glb_m_;
    *glb_k = glb_k_;

    free(glb_A_row);
    free(glb_A_col);
    free(glb_A_val);
}

void scatter_csr_rows(
    MPI_Comm comm, int nproc, int my_rank, int *A_m_displs, int *A_nnz_displs, int *A_m_scnts, 
    int *A_nnz_scnts, int *glb_A_rowptr, int *glb_A_colidx, double *glb_A_csrval, 
    int **loc_A_rowptr_, int **loc_A_colidx_, double **loc_A_csrval_, int dbg_print
)
{
    MPI_Request reqs[3];
    MPI_Ibcast(A_m_displs,   nproc + 1, MPI_INT, 0, comm, &reqs[0]);
    MPI_Ibcast(A_nnz_displs, nproc + 1, MPI_INT, 0, comm, &reqs[1]);
    MPI_Waitall(2, &reqs[0], MPI_STATUSES_IGNORE);
    for (int i = 0; i < nproc; i++)
    {
        A_m_scnts[i] = A_m_displs[i + 1] - A_m_displs[i];
        A_nnz_scnts[i] = A_nnz_displs[i + 1] - A_nnz_displs[i];
    }
    int loc_A_srow = A_m_displs[my_rank];
    int loc_A_nrow = A_m_displs[my_rank + 1] - loc_A_srow;
    int loc_A_nnz  = A_nnz_displs[my_rank + 1] - A_nnz_displs[my_rank];
    int *loc_A_rowptr = (int *) malloc(sizeof(int) * (loc_A_nrow + 1));
    int *loc_A_colidx = (int *) malloc(sizeof(int) * loc_A_nnz);
    double *loc_A_csrval = (double *) malloc(sizeof(double) * loc_A_nnz);
    MPI_Iscatterv(
        glb_A_rowptr, A_m_scnts, A_m_displs, MPI_INT, 
        loc_A_rowptr, loc_A_nrow, MPI_INT, 0, comm, &reqs[0]
    );
    MPI_Iscatterv(
        glb_A_colidx, A_nnz_scnts, A_nnz_displs, MPI_INT, 
        loc_A_colidx, loc_A_nnz, MPI_INT, 0, comm, &reqs[1]
    );
    MPI_Iscatterv(
        glb_A_csrval, A_nnz_scnts, A_nnz_displs, MPI_DOUBLE, 
        loc_A_csrval, loc_A_nnz, MPI_DOUBLE, 0, comm, &reqs[2]
    );
    MPI_Waitall(3, &reqs[0], MPI_STATUSES_IGNORE);
    loc_A_rowptr[loc_A_nrow] = A_nnz_displs[my_rank + 1];
    MPI_Barrier(comm);
    *loc_A_rowptr_ = loc_A_rowptr;
    *loc_A_colidx_ = loc_A_colidx;
    *loc_A_csrval_ = loc_A_csrval;

    if (!dbg_print) return;
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        if (my_rank == iproc)
        {
            printf("\nRank %d: loc_A_srow, loc_A_nrow = %d, %d\n", my_rank, loc_A_srow, loc_A_nrow);
            printf("loc_A_rowptr = ");
            for (int i = 0; i <= loc_A_nrow; i++)
                printf("%d ", loc_A_rowptr[i]);
            printf("\n");
            printf("loc_A_colidx = ");
            for (int i = 0; i < loc_A_nnz; i++)
                printf("%d ", loc_A_colidx[i]);
            printf("\n");
            printf("loc_A_csrval = ");
            for (int i = 0; i < loc_A_nnz; i++)
                printf("%.2f ", loc_A_csrval[i]);
            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(comm);
    }
}

void fill_B(
    int layout, double *B, int ldB, int srow, int nrow, int scol, int ncol,
    double factor_i, double factor_j
)
{
    if (layout == 0)
    {
        // Row major
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nrow; i++)
        {
            int glb_i = srow + i;
            double *Bi = B + i * ldB;
            for (int j = 0; j < ncol; j++)
            {
                int glb_j = scol + j;
                Bi[j] = glb_i * factor_i + glb_j * factor_j;
            }
        }
    } else {
        // Column major
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ncol; j++)
        {
            int glb_j = scol + j;
            double *Bj = B + j * ldB;
            for (int i = 0; i < nrow; i++)
            {
                int glb_i = srow + i;
                Bj[i] = glb_i * factor_i + glb_j * factor_j;
            }
        }
    }
}

#ifdef USE_MKL
void mkl_csr_spmm(
    int m, int k, int n, int *rowptr, int *colidx, double *val,
    double *B, int ldB, double *C, int ldC
)
{
    double alpha = 1.0, beta = 0.0;
    sparse_matrix_t mkl_spA;
    struct matrix_descr mkl_descA;
    mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_descA.mode = SPARSE_FILL_MODE_FULL;
    mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
    sparse_layout_t sp_layout = SPARSE_LAYOUT_ROW_MAJOR;
    mkl_sparse_d_create_csr(
        &mkl_spA, SPARSE_INDEX_BASE_ZERO, m, k, 
        rowptr, rowptr + 1, colidx, val
    );
    mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_spA, mkl_descA, 
        sp_layout, B, n, ldB, beta, C, ldC
    );
    mkl_sparse_destroy(mkl_spA);
}
#endif
