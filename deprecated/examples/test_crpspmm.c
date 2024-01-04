#include "test_utils.h"
#include "crpspmm.h"
#include "spmat_part.h"

int main(int argc, char **argv) 
{
    if (argc < 4)
    {
        printf(
            "Usage: %s <mtx-file> <num-of-B-col> <num-of-tests> <check-correct> <use-CUDA>\n"
            "<check-correct> and <use-CUDA>: 0 or 1, optional, default values are 0", argv[0]
        );
        return 255;
    }

    int glb_n, n_test, chk_res = 0, use_CUDA = 0;
    glb_n = atoi(argv[2]);
    n_test = atoi(argv[3]);
    if (argc >= 5) chk_res = atoi(argv[4]);
    if (argc >= 6) use_CUDA = atoi(argv[5]);
    #ifdef USE_CUDA
    if (use_CUDA == 1) select_cuda_device_by_mpi_local_rank();
    #else
    use_CUDA = 0;
    #endif

    int nproc, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    double st, et, ut;
    int need_symm = 0;

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

    // 2. 1D partition and distribute A s.t. each process has a contiguous  
    //    block of rows and nearly the same number of nonzeros
    st = get_wtime_sec();
    int *A_m_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_nnz_displs = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_m_scnts    = (int *) malloc(sizeof(int) * nproc);
    int *A_nnz_scnts  = (int *) malloc(sizeof(int) * nproc);
    if (my_rank == 0)
    {
        csr_mat_row_partition(glb_m, glb_A_rowptr, nproc, A_m_displs);
        for (int i = 0; i <= nproc; i++)
            A_nnz_displs[i] = glb_A_rowptr[A_m_displs[i]];
    }
    int *loc_A_rowptr = NULL, *loc_A_colidx = NULL;
    double *loc_A_csrval = NULL;
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
    int loc_A_m_start = A_m_displs[my_rank];
    int loc_A_m = A_m_displs[my_rank + 1] - A_m_displs[my_rank];

    // 3. 2D partition input dense matrix B and output dense matrix C
    int dims[2] = {0, 0};
    int loc_B_k, loc_B_n, loc_B_k_start, loc_B_n_start;
    int loc_C_m, loc_C_n, loc_C_m_start, loc_C_n_start;
    MPI_Dims_create(nproc, 2, dims);  // Ask MPI to find a balanced 2D process grid for us
    int np_row = dims[0], np_col = dims[1];
    int rank_row = my_rank / np_col, rank_col = my_rank % np_col;
    calc_block_spos_size(glb_k, np_row, rank_row, &loc_B_k_start, &loc_B_k);
    calc_block_spos_size(glb_n, np_col, rank_col, &loc_B_n_start, &loc_B_n);
    calc_block_spos_size(glb_m, np_row, rank_row, &loc_C_m_start, &loc_C_m);
    calc_block_spos_size(glb_n, np_col, rank_col, &loc_C_n_start, &loc_C_n);
    if (chk_res > 0)
    {
        loc_C_m = 0;
        loc_C_n = 0;
        loc_C_m_start = 0;
        loc_C_n_start = 0;
        if (my_rank == 0)
        {
            loc_C_m = glb_m;
            loc_C_n = glb_n;
        }
    }
    double *loc_B = (double *) malloc(sizeof(double) * loc_B_k * loc_B_n);
    double *loc_C = (double *) malloc(sizeof(double) * loc_C_m * loc_C_n);
    int layout = 0;
    double factor_i = 0.19, factor_j = 0.24;
    fill_B(layout, loc_B, loc_B_n, loc_B_k_start, loc_B_k, loc_B_n_start, loc_B_n, factor_i, factor_j);

    // 4. Compute C := A * B
    crpspmm_engine_p crpspmm;
    crpspmm_engine_init(
        glb_m, glb_n, glb_k,
        loc_A_m_start, loc_A_m, loc_A_rowptr, loc_A_colidx, 
        loc_B_k_start, loc_B_k, loc_B_n_start, loc_B_n,
        loc_C_m_start, loc_C_m, loc_C_n_start, loc_C_n,
        MPI_COMM_WORLD, use_CUDA, &crpspmm, NULL
    );
    if (my_rank == 0)
    {
        printf("CRP-SpMM 2D partition: %d * %d\n", crpspmm->np_row, crpspmm->np_col);
        fflush(stdout);
    }
    // Warm up running
    crpspmm_engine_exec(
        crpspmm, loc_A_rowptr, loc_A_colidx, loc_A_csrval,
        loc_B, loc_B_n, loc_C, loc_C_n
    );
    crpspmm_engine_clear_stat(crpspmm);
    for (int i = 0; i < n_test; i++)
    {
        double st = MPI_Wtime();
        crpspmm_engine_exec(
            crpspmm, loc_A_rowptr, loc_A_colidx, loc_A_csrval,
            loc_B, loc_B_n, loc_C, loc_C_n
        );
        double et = MPI_Wtime();
        if (my_rank == 0) 
        {
            printf("%.2f\n", et - st);
            fflush(stdout);
        }
    }
    crpspmm_engine_print_stat(crpspmm);
    crpspmm_engine_free(&crpspmm);

    // 5. Check the correctness of the result
    if ((chk_res == 1) && (my_rank == 0))
    {
        double *glb_B = (double *) malloc(sizeof(double) * glb_k * glb_n);
        double *ref_C = (double *) malloc(sizeof(double) * glb_m * glb_n);
        fill_B(layout, glb_B, glb_n, 0, glb_k, 0, glb_n, factor_i, factor_j);
        if (use_CUDA == 0)
        {
            #ifdef USE_MKL
            mkl_csr_spmm(
                glb_m, glb_k, glb_n, glb_A_rowptr, glb_A_colidx, glb_A_csrval,
                glb_B, glb_n, ref_C, glb_n
            );
            #endif
        }
        #ifdef USE_CUDA
        if (use_CUDA == 1)
        {
            cuda_cusparse_csr_spmm(
                glb_m, glb_n, glb_k, 1.0, 
                glb_A_nnz, glb_A_rowptr, glb_A_colidx, glb_A_csrval,
                glb_B, glb_n, 0.0, ref_C, glb_n
            );
        }
        #endif
        double C_fnorm, err_fnorm;
        calc_err_2norm(glb_m * glb_n, ref_C, loc_C, &C_fnorm, &err_fnorm);
        printf("||C_ref - C||_f / ||C_ref||_f = %e\n", err_fnorm / C_fnorm);
        fflush(stdout);
        free(glb_B);
        free(ref_C);
    }
    
    free(glb_A_rowptr);
    free(glb_A_colidx);
    free(glb_A_csrval);
    free(A_m_displs);
    free(A_m_scnts);
    free(A_nnz_scnts);
    free(A_nnz_displs);
    free(loc_B);
    free(loc_C);
    MPI_Finalize();
    return 0;
}
