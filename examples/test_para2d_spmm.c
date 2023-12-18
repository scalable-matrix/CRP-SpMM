#include "test_utils.h"
#include "para2d_spmm.h"
#include "spmat_part.h"
#include "mat_redist.h"

int main(int argc, char **argv) 
{
    if (argc < 4)
    {
        printf("Usage: %s <mtx-file> <num-of-B-col> <num-of-tests> <check-correct>\n", argv[0]);
        printf("<check-correct>: 0 or 1, optional, default value is 0\n");
        return 255;
    }
    int glb_n  = atoi(argv[2]);
    int n_test = atoi(argv[3]);
    int chk_res = 0;
    if (argc >= 5) chk_res = atoi(argv[4]);
    int need_symm = 0;

    int nproc, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double st, et, ut;

    // 1. Rank 0 read sparse A from mtx file
    int glb_m, glb_k;
    int *glb_A_rowptr = NULL, *glb_A_colidx = NULL;
    double *glb_A_csrval = NULL;
    if (my_rank == 0) read_mtx_csr(argv[1], need_symm, &glb_m, &glb_k, glb_n, &glb_A_rowptr, &glb_A_colidx, &glb_A_csrval);
    int glb_mk[2] = {glb_m, glb_k};
    MPI_Bcast(&glb_mk[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
    glb_m = glb_mk[0];
    glb_k = glb_mk[1];
    if (chk_res) chk_res = can_check_res(my_rank, glb_m, glb_n, glb_k);

    // 2. Rank 0 compute 2D process grid and broadcast
    int pm = 0, pn = 0;
    size_t comm_cost = 0;
    int *A0_rowptr = NULL, *B_rowptr = NULL, *AC_rowptr = NULL, *BC_colptr = NULL;
    if (my_rank == 0)
    {
        st = get_wtime_sec();
        calc_spmm_2dpg(
            nproc, glb_m, glb_n, glb_k, glb_A_rowptr, glb_A_colidx, &pm, &pn, 
            &comm_cost, &A0_rowptr, &B_rowptr, &AC_rowptr, &BC_colptr, 0
        );
        et = get_wtime_sec();
        printf("Rank 0 calculate 2D partitioning time = %.2f s\n", et - st);
        printf("2D process grid: pm, pn = %d, %d\n", pm, pn);
    }
    int pmn[2] = {pm, pn};
    MPI_Bcast(&pmn[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0)
    {
        pm = pmn[0], pn = pmn[1];
        A0_rowptr = (int *) malloc(sizeof(int) * (nproc + 1));
        B_rowptr  = (int *) malloc(sizeof(int) * (pm + 1));
        AC_rowptr = (int *) malloc(sizeof(int) * (pm + 1));
        BC_colptr = (int *) malloc(sizeof(int) * (pn + 1));
    }
    int pi = my_rank / pn, pj = my_rank % pn;
    MPI_Bcast(A0_rowptr, nproc + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_rowptr,  pm + 1,    MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(AC_rowptr, pm + 1,    MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(BC_colptr, pn + 1,    MPI_INT, 0, MPI_COMM_WORLD);

    // 3. Rank 0 scatter A according to A0_rowptr
    st = get_wtime_sec();
    int *A_nnz_displs = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_m_scnts    = (int *) malloc(sizeof(int) * nproc);
    int *A_nnz_scnts  = (int *) malloc(sizeof(int) * nproc);
    int *loc_A_rowptr = NULL, *loc_A_colidx = NULL;
    double *loc_A_csrval = NULL;
    if (my_rank == 0)
    {
        for (int i = 0; i <= nproc; i++)
            A_nnz_displs[i] = glb_A_rowptr[A0_rowptr[i]];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    scatter_csr_rows(
        MPI_COMM_WORLD, nproc, my_rank, A0_rowptr, A_nnz_displs, A_m_scnts, 
        A_nnz_scnts, glb_A_rowptr, glb_A_colidx, glb_A_csrval,
        &loc_A_rowptr, &loc_A_colidx, &loc_A_csrval, 0
    );
    int loc_A_srow = A0_rowptr[my_rank];
    int loc_A_nrow = A0_rowptr[my_rank + 1] - loc_A_srow;
    et = get_wtime_sec();
    ut = et - st;
    if (my_rank == 0)
    {
        printf("1D distribution of A used %.2f s\n", ut);
        fflush(stdout);
    }

    // 4. 2D partition input dense matrix B and output dense matrix C
    int loc_B_srow  = B_rowptr[pi];
    int loc_B_nrow  = B_rowptr[pi + 1] - loc_B_srow;
    int loc_C_srow  = AC_rowptr[pi];
    int loc_C_nrow  = AC_rowptr[pi + 1] - loc_C_srow;
    int loc_BC_scol = BC_colptr[pj];
    int loc_BC_ncol = BC_colptr[pj + 1] - loc_BC_scol;
    double *loc_B = (double *) malloc(sizeof(double) * loc_B_nrow * loc_BC_ncol);
    double *loc_C = (double *) malloc(sizeof(double) * loc_C_nrow * loc_BC_ncol);
    int layout = 0;  // 0 for row major, 1 for column major
    int loc_B_ld = 0, loc_C_ld = 0;
    double factor_i = 0.19, factor_j = 0.24;
    if (layout == 0)
    {
        loc_B_ld = loc_BC_ncol;
        loc_C_ld = loc_BC_ncol;
    } else {
        loc_B_ld = loc_B_nrow;
        loc_C_ld = loc_C_nrow;
    }
    fill_B(
        layout, loc_B, loc_B_ld, loc_B_srow, loc_B_nrow, 
        loc_BC_scol, loc_BC_ncol, factor_i, factor_j
    );

    // 5. Compute C := A * B
    para2d_spmm_p para2d_spmm = NULL;
    para2d_spmm_init(
        MPI_COMM_WORLD, pm, pn, A0_rowptr, B_rowptr, AC_rowptr, BC_colptr, 
        loc_A_rowptr, loc_A_colidx, loc_A_csrval, &para2d_spmm
    );
    // Warm up
    para2d_spmm_exec(para2d_spmm, layout, loc_B, loc_B_ld, loc_C, loc_C_ld);
    para2d_spmm_clear_stat(para2d_spmm);
    for (int i = 0; i < n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = get_wtime_sec();
        para2d_spmm_exec(para2d_spmm, layout, loc_B, loc_B_ld, loc_C, loc_C_ld);
        MPI_Barrier(MPI_COMM_WORLD);
        et = get_wtime_sec();
        if (my_rank == 0) 
        {
            printf("%.2f\n", et - st);
            fflush(stdout);
        }
    }
    para2d_spmm_print_stat(para2d_spmm);
    para2d_spmm_free(&para2d_spmm);

    // 6. Validate results
    if (chk_res)
    {
        double *sbuf = (double *) malloc(sizeof(double) * loc_C_nrow * loc_BC_ncol);
        if (layout == 0)
        {
            memcpy(sbuf, loc_C, sizeof(double) * loc_C_nrow * loc_BC_ncol);
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < loc_C_nrow; i++)
                for (int j = 0; j < loc_BC_ncol; j++)
                    sbuf[i * loc_BC_ncol + j] = loc_C[i + j * loc_C_nrow];
        }
        double *glb_B = NULL, *ref_C = NULL, *recv_C = NULL;
        int req_C_srow = 0, req_C_nrow = 0, req_C_scol = 0, req_C_ncol = 0;
        if (my_rank == 0)
        {
            req_C_nrow = glb_k;
            req_C_ncol = glb_n;
            glb_B  = (double *) malloc(sizeof(double) * glb_k * glb_n);
            ref_C  = (double *) malloc(sizeof(double) * glb_m * glb_n);
            recv_C = (double *) malloc(sizeof(double) * glb_m * glb_n);
            fill_B(0, glb_B, glb_n, 0, glb_k, 0, glb_n, factor_i, factor_j);
        }
        mat_redist_engine_p rd_C = NULL;
        mat_redist_engine_init(
            loc_C_srow, loc_BC_scol, loc_C_nrow, loc_BC_ncol,
            req_C_srow, req_C_scol, req_C_nrow, req_C_ncol,
            MPI_COMM_WORLD, MPI_DOUBLE, sizeof(double), DEV_TYPE_HOST, 
            &rd_C, NULL
        );
        mat_redist_engine_exec(rd_C, sbuf, loc_BC_ncol, recv_C, glb_n);

        if (my_rank == 0)
        {
            #ifdef USE_MKL
            mkl_csr_spmm(
                glb_m, glb_k, glb_n, glb_A_rowptr, glb_A_colidx, glb_A_csrval,
                glb_B, glb_n, ref_C, glb_n
            );
            #else
            #error No CPU SpMM implementation
            #endif
            double C_fnorm, err_fnorm;
            calc_err_2norm(glb_m * glb_n, ref_C, recv_C, &C_fnorm, &err_fnorm);
            printf("||C_ref - C||_f / ||C_ref||_f = %e\n", err_fnorm / C_fnorm);
            fflush(stdout);
        }
        free(sbuf);
        free(glb_B);
        free(ref_C);
        free(recv_C);
    }  // End of "if (chk_res)"

    // Finally, it's done!
    free(glb_A_rowptr);
    free(glb_A_colidx);
    free(glb_A_csrval);
    free(A0_rowptr);
    free(B_rowptr);
    free(AC_rowptr);
    free(BC_colptr);
    free(A_nnz_displs);
    free(A_m_scnts);
    free(A_nnz_scnts);
    free(loc_A_rowptr);
    free(loc_A_colidx);
    free(loc_A_csrval);
    MPI_Finalize();
    return 0;
}