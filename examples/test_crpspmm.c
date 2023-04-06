#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

#include "utils.h"
#include "mmio_utils.h"
#include "crpspmm.h"

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

    int nproc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double st, et, ut;

    // 1. Rank 0 read sparse A from mtx file
    int glb_m, glb_k, glb_A_nnz, bandwidth = 0;
    int *glb_A_row = NULL, *glb_A_col = NULL, *glb_A_rowptr = NULL, *glb_A_colidx = NULL;
    double *glb_A_val = NULL, *glb_A_csrval = NULL;
    if (rank == 0)
    {
        printf("B has %d columns\n", glb_n);
        printf("Rank 0 read matrix A from file %s", argv[1]);
        fflush(stdout);
        st = MPI_Wtime();
        mm_read_sparse_RPI_GS(argv[1], &glb_m, &glb_k, &glb_A_nnz, &glb_A_row, &glb_A_col, &glb_A_val);
        coo2csr(glb_m, glb_k, glb_A_nnz, glb_A_row, glb_A_col, glb_A_val, &glb_A_rowptr, &glb_A_colidx, &glb_A_csrval);
        et = MPI_Wtime();
        ut = et - st;
        #pragma omp parallel for reduction(max:bandwidth)
        for (int i = 0; i < glb_A_nnz; i++)
        {
            int bw_i = abs(glb_A_row[i] - glb_A_col[i]);
            if (bw_i > bandwidth) bandwidth = bw_i;
        }
        printf(" used %.2f s\n", ut);
        printf(
            "A size = %d * %d, nnz = %d, nnz/row = %d, bandwidth = %d\n\n", 
            glb_m, glb_k, glb_A_nnz, glb_A_nnz / glb_m, bandwidth
        );
        fflush(stdout);
    }
    free(glb_A_row);
    free(glb_A_col);
    free(glb_A_val);
    int glb_mk[2] = {glb_m, glb_k};
    MPI_Bcast(&glb_mk[0], 2, MPI_INT, 0, MPI_COMM_WORLD);
    glb_m = glb_mk[0];
    glb_k = glb_mk[1];

    // 2. 1D partition and distribute A s.t. each process has a contiguous  
    //    block of rows and nearly the same number of nonzeros
    st = MPI_Wtime();
    int *A_m_displs     = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_nnz_displs   = (int *) malloc(sizeof(int) * (nproc + 1));
    int *A_m_sendcnts   = (int *) malloc(sizeof(int) * nproc);
    int *A_nnz_sendcnts = (int *) malloc(sizeof(int) * nproc);
    if (rank == 0)
    {
        A_m_displs[0] = 0;
        A_nnz_displs[0] = 0;
        int curr_srow = 0;
        for (int i = 0; i < nproc; i++)
        {
            int i_max_nnz = (glb_A_nnz / nproc) * (i + 1);
            if (i == nproc - 1) i_max_nnz = glb_A_nnz;
            int curr_erow = curr_srow + 1;
            while (glb_A_rowptr[curr_erow] < i_max_nnz) curr_erow++;
            A_m_displs[i + 1] = curr_erow;
            A_nnz_displs[i] = glb_A_rowptr[curr_srow];
            curr_srow = curr_erow;
        }
    }
    A_nnz_displs[nproc] = glb_A_nnz;
    MPI_Request reqs[3];
    MPI_Ibcast(A_m_displs,   nproc + 1, MPI_INT, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Ibcast(A_nnz_displs, nproc + 1, MPI_INT, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, &reqs[0], MPI_STATUSES_IGNORE);
    for (int i = 0; i < nproc; i++)
    {
        A_m_sendcnts[i] = A_m_displs[i + 1] - A_m_displs[i];
        A_nnz_sendcnts[i] = A_nnz_displs[i + 1] - A_nnz_displs[i];
    }
    int loc_A_m_start = A_m_displs[rank];
    int loc_A_m = A_m_displs[rank + 1] - A_m_displs[rank];
    int loc_A_nnz = A_nnz_displs[rank + 1] - A_nnz_displs[rank];
    int *loc_A_rowptr = (int *) malloc(sizeof(int) * (loc_A_m + 1));
    int *loc_A_colidx = (int *) malloc(sizeof(int) * loc_A_nnz);
    double *loc_A_csrval = (double *) malloc(sizeof(double) * loc_A_nnz);
    MPI_Iscatterv(
        glb_A_rowptr, A_m_sendcnts, A_m_displs, MPI_INT, 
        loc_A_rowptr, loc_A_m, MPI_INT, 0, MPI_COMM_WORLD, &reqs[0]
    );
    MPI_Iscatterv(
        glb_A_colidx, A_nnz_sendcnts, A_nnz_displs, MPI_INT, 
        loc_A_colidx, loc_A_nnz, MPI_INT, 0, MPI_COMM_WORLD, &reqs[1]
    );
    MPI_Iscatterv(
        glb_A_csrval, A_nnz_sendcnts, A_nnz_displs, MPI_DOUBLE, 
        loc_A_csrval, loc_A_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD, &reqs[2]
    );
    MPI_Waitall(3, &reqs[0], MPI_STATUSES_IGNORE);
    loc_A_rowptr[loc_A_m] = A_nnz_displs[rank + 1];
    MPI_Barrier(MPI_COMM_WORLD);
    et = MPI_Wtime();
    ut = et - st;
    if (rank == 0)
    {
        printf("1D partition and distribution of A used %.2f s\n", ut);
        fflush(stdout);
    }

    // 3. 2D partition input dense matrix B and output dense matrix C
    int dims[2] = {0, 0};
    int loc_B_k, loc_B_n, loc_B_k_start, loc_B_n_start;
    int loc_C_m, loc_C_n, loc_C_m_start, loc_C_n_start;
    MPI_Dims_create(nproc, 2, dims);  // Ask MPI to find a balanced 2D process grid for us
    int np_row = dims[0], np_col = dims[1];
    int rank_row = rank / np_col, rank_col = rank % np_col;
    calc_block_spos_size(glb_k, np_row, rank_row, &loc_B_k_start, &loc_B_k);
    calc_block_spos_size(glb_n, np_col, rank_col, &loc_B_n_start, &loc_B_n);
    calc_block_spos_size(glb_m, np_row, rank_row, &loc_C_m_start, &loc_C_m);
    calc_block_spos_size(glb_n, np_col, rank_col, &loc_C_n_start, &loc_C_n);
    if (chk_res > 0)
    {
        if (rank == 0)
        {
            loc_C_m = glb_m;
            loc_C_n = glb_n;
            loc_C_m_start = 0;
            loc_C_n_start = 0;
        } else {
            loc_C_m = 0;
            loc_C_n = 0;
            loc_C_m_start = 0;
            loc_C_n_start = 0;
        }
    }
    double *loc_B = (double *) malloc(sizeof(double) * loc_B_k * loc_B_n);
    double *loc_C = (double *) malloc(sizeof(double) * loc_C_m * loc_C_n);
    for (int i = 0; i < loc_B_k; i++)
    {
        int glb_i = loc_B_k_start + i;
        for (int j = 0; j < loc_B_n; j++)
        {
            int glb_j = loc_B_n_start + j;
            loc_B[i * loc_B_n + j] = glb_i * 0.19 + glb_j * 0.24;
        }
    }

    // 4. Compute C := A * B
    crpspmm_engine_p crpspmm;
    crpspmm_engine_init(
        glb_m, glb_n, glb_k,
        loc_A_m_start, loc_A_m, loc_A_rowptr, loc_A_colidx, 
        loc_B_k_start, loc_B_k, loc_B_n_start, loc_B_n,
        loc_C_m_start, loc_C_m, loc_C_n_start, loc_C_n,
        MPI_COMM_WORLD, use_CUDA, &crpspmm, NULL
    );
    if (rank == 0)
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
        if (rank == 0) 
        {
            printf("%.2f\n", et - st);
            fflush(stdout);
        }
    }
    crpspmm_engine_print_stat(crpspmm);
    crpspmm_engine_free(&crpspmm);

    // 5. Check the correctness of the result
    if ((chk_res == 1) && (rank == 0))
    {
        double *glb_B = (double *) malloc(sizeof(double) * glb_k * glb_n);
        double *glb_C = (double *) malloc(sizeof(double) * glb_m * glb_n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < glb_k; i++)
        {
            double *glb_B_i = glb_B + i * glb_n;
            #pragma omp simd
            for (int j = 0; j < glb_n; j++)
                glb_B_i[j] = i * 0.19 + j * 0.24;
        }
        if (use_CUDA == 0)
        {
            #ifdef USE_MKL
            double alpha = 1.0, beta = 0.0;
            sparse_matrix_t mkl_spA;
            struct matrix_descr mkl_descA;
            mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
            mkl_descA.mode = SPARSE_FILL_MODE_FULL;
            mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
            mkl_sparse_d_create_csr(
                &mkl_spA, SPARSE_INDEX_BASE_ZERO, glb_m, glb_k, 
                glb_A_rowptr, glb_A_rowptr + 1, glb_A_colidx, glb_A_csrval
            );
            mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_spA, mkl_descA, 
                SPARSE_LAYOUT_ROW_MAJOR, glb_B, glb_n, glb_n, beta, glb_C, glb_n
            );
            mkl_sparse_destroy(mkl_spA);
            #endif
        }
        #ifdef USE_CUDA
        if (use_CUDA == 1)
        {
            cuda_cusparse_csr_spmm(
                glb_m, glb_n, glb_k, 1.0, 
                glb_A_nnz, glb_A_rowptr, glb_A_colidx, glb_A_csrval,
                glb_B, glb_n, 0.0, glb_C, glb_n
            );
        }
        #endif
        double C_fnorm, err_fnorm;
        calc_err_2norm(glb_m * glb_n, glb_C, loc_C, &C_fnorm, &err_fnorm);
        printf("||C_ref - C||_f / ||C_ref||_f = %e\n", err_fnorm / C_fnorm);
        fflush(stdout);
        free(glb_B);
        free(glb_C);
    }
    
    free(glb_A_rowptr);
    free(glb_A_colidx);
    free(glb_A_csrval);
    free(A_m_displs);
    free(A_m_sendcnts);
    free(A_nnz_sendcnts);
    free(A_nnz_displs);
    free(loc_B);
    free(loc_C);
    MPI_Finalize();
    return 0;
}
