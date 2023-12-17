#include "test_utils.h"
#include "rowpara_spmm.h"
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

    int nproc, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double st, et, ut;

    // 1. Rank 0 read sparse A from mtx file
    int glb_m, glb_k;
    int *glb_A_rowptr = NULL, *glb_A_colidx = NULL;
    double *glb_A_csrval = NULL;
    if (my_rank == 0) read_mtx_csr(argv[1], &glb_m, &glb_k, glb_n, &glb_A_rowptr, &glb_A_colidx, &glb_A_csrval);
    int glb_mk[2] = {glb_m, glb_k};
    MPI_Bcast(&glb_mk[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
    glb_m = glb_mk[0];
    glb_k = glb_mk[1];
    if (chk_res) chk_res = can_check_res(my_rank, glb_m, glb_n, glb_k);

    // 2. 1D partition and distribute A s.t. each process has a contiguous  
    //    block of rows and nearly the same number of nonzeros
    st = get_wtime_sec();
    int *A_m_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_nnz_displs = (int *) malloc(sizeof(int) * (nproc + 1));
    int *x_displs     = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_m_scnts    = (int *) malloc(sizeof(int) * nproc);
    int *A_nnz_scnts  = (int *) malloc(sizeof(int) * nproc);
    if (my_rank == 0)
    {
        csr_mat_row_partition(glb_m, glb_A_rowptr, nproc, A_m_displs);
        for (int i = 0; i <= nproc; i++)
            A_nnz_displs[i] = glb_A_rowptr[A_m_displs[i]];
        
        if (glb_m == glb_k)
        {
            memcpy(x_displs, A_m_displs, sizeof(int) * (nproc + 1));
        } else {
            int tmp;
            for (int i = 0; i <= nproc; i++) 
                calc_block_spos_size(glb_k, nproc, i, x_displs + i, &tmp);
        }
    }
    int *loc_A_rowptr = NULL, *loc_A_colidx = NULL;
    double *loc_A_csrval = NULL;
    MPI_Bcast(x_displs, nproc + 1, MPI_INT, 0, MPI_COMM_WORLD);
    scatter_csr_rows(
        MPI_COMM_WORLD, nproc, my_rank, A_m_displs, A_nnz_displs, A_m_scnts, 
        A_nnz_scnts, glb_A_rowptr, glb_A_colidx, glb_A_csrval,
        &loc_A_rowptr, &loc_A_colidx, &loc_A_csrval, 0
    );
    int loc_A_srow = A_m_displs[my_rank];
    int loc_A_nrow = A_m_displs[my_rank + 1] - loc_A_srow;
    et = get_wtime_sec();
    ut = et - st;
    if (my_rank == 0)
    {
        printf("1D partition and distribution of A used %.2f s\n", ut);
        fflush(stdout);
    }
    if (my_rank == 0)
    {
        int total_size = 0;
        int *comm_sizes = (int *) malloc(sizeof(int) * nproc);
        csr_mat_row_part_comm_size(
            glb_m, glb_k, glb_A_rowptr, glb_A_colidx, 
            nproc, A_m_displs, x_displs, comm_sizes, &total_size
        );
        free(comm_sizes);
        printf("Total SpMV comm size = %d\n", total_size);
        fflush(stdout);
    }

    // 3. 1D partition input dense matrix B and output dense matrix C
    int loc_B_srow = x_displs[my_rank];
    int loc_B_nrow = x_displs[my_rank + 1] - loc_B_srow;
    int loc_C_srow = loc_A_srow;
    int loc_C_nrow = loc_A_nrow;
    double *loc_B = (double *) malloc(sizeof(double) * loc_B_nrow * glb_n);
    double *loc_C = (double *) malloc(sizeof(double) * loc_C_nrow * glb_n);
    int layout = 0;  // 0 for row major, 1 for column major
    int loc_B_ld = 0, loc_C_ld = 0;
    double factor_i = 0.19, factor_j = 0.24;
    if (layout == 0)
    {
        loc_B_ld = glb_n;
        loc_C_ld = glb_n;
    } else {
        loc_B_ld = loc_B_nrow;
        loc_C_ld = loc_C_nrow;
    }
    fill_B(layout, loc_B, loc_B_ld, loc_B_srow, loc_B_nrow, 0, glb_n, factor_i, factor_j);

    // 4. Compute C := A * B
    rp_spmm_p rp_spmm = NULL;
    rp_spmm_init(
        loc_A_srow, loc_A_nrow, loc_A_rowptr, loc_A_colidx, loc_A_csrval, 
        x_displs, glb_n, MPI_COMM_WORLD, &rp_spmm
    );
    // Warm up
    rp_spmm_exec(rp_spmm, layout, loc_B, loc_B_ld, loc_C, loc_C_ld);
    rp_spmm_clear_stat(rp_spmm);
    for (int i = 0; i < n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        st = get_wtime_sec();
        rp_spmm_exec(rp_spmm, layout, loc_B, loc_B_ld, loc_C, loc_C_ld);
        MPI_Barrier(MPI_COMM_WORLD);
        et = get_wtime_sec();
        if (my_rank == 0) 
        {
            printf("%.2f\n", et - st);
            fflush(stdout);
        }
    }
    rp_spmm_print_stat(rp_spmm);
    rp_spmm_free(&rp_spmm);

    // 5. Validate the result
    if (chk_res)
    {
        double *sbuf = (double *) malloc(sizeof(double) * loc_C_nrow * glb_n);
        if (layout == 0)
        {
            memcpy(sbuf, loc_C, sizeof(double) * loc_C_nrow * glb_n);
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < loc_C_nrow; i++)
                for (int j = 0; j < glb_n; j++)
                    sbuf[i * glb_n + j] = loc_C[i + j * loc_C_nrow];
        }
        double *glb_B = NULL, *ref_C = NULL, *recv_C = NULL;
        if (my_rank == 0)
        {
            glb_B  = (double *) malloc(sizeof(double) * glb_k * glb_n);
            ref_C  = (double *) malloc(sizeof(double) * glb_m * glb_n);
            recv_C = (double *) malloc(sizeof(double) * glb_m * glb_n);
            fill_B(layout, glb_B, glb_n, 0, glb_k, 0, glb_n, factor_i, factor_j);
        }
        int *C_rcnts   = (int *) malloc(sizeof(int) * nproc);
        int *C_rdispls = (int *) malloc(sizeof(int) * (nproc + 1));
        C_rdispls[0] = 0;
        for (int i = 0; i < nproc; i++)
        {
            C_rcnts[i] = A_m_scnts[i] * glb_n;
            C_rdispls[i + 1] = C_rdispls[i] + C_rcnts[i];
        }
        MPI_Gatherv(
            sbuf, loc_C_nrow * glb_n, MPI_DOUBLE, 
            recv_C, C_rcnts, C_rdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD
        );

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
    free(A_m_displs);
    free(A_m_scnts);
    free(A_nnz_scnts);
    free(A_nnz_displs);
    free(x_displs);
    free(loc_A_rowptr);
    free(loc_A_colidx);
    free(loc_A_csrval);
    free(loc_B);
    free(loc_C);
    MPI_Finalize();
    return 0;
}