#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "utils.h"
#include "mmio_utils.h"
#include "rowpara_spmm.h"
#include "spmat_part.h"

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
    int glb_m, glb_k, glb_A_nnz, bandwidth = 0;
    int *glb_A_row = NULL, *glb_A_col = NULL, *glb_A_rowptr = NULL, *glb_A_colidx = NULL;
    double *glb_A_val = NULL, *glb_A_csrval = NULL;
    if (my_rank == 0)
    {   
        printf("B has %d columns\n", glb_n);
        printf("Rank 0 read matrix A from file %s", argv[1]);
        fflush(stdout);
        st = get_wtime_sec();
        mm_read_sparse_RPI_GS(argv[1], &glb_m, &glb_k, &glb_A_nnz, &glb_A_row, &glb_A_col, &glb_A_val);
        coo2csr(glb_m, glb_k, glb_A_nnz, glb_A_row, glb_A_col, glb_A_val, &glb_A_rowptr, &glb_A_colidx, &glb_A_csrval);
        et = get_wtime_sec();
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
    MPI_Request reqs[3];
    MPI_Ibcast(A_m_displs,   nproc + 1, MPI_INT, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Ibcast(A_nnz_displs, nproc + 1, MPI_INT, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Ibcast(x_displs,     nproc + 1, MPI_INT, 0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Waitall(3, &reqs[0], MPI_STATUSES_IGNORE);
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
        loc_A_rowptr, loc_A_nrow, MPI_INT, 0, MPI_COMM_WORLD, &reqs[0]
    );
    MPI_Iscatterv(
        glb_A_colidx, A_nnz_scnts, A_nnz_displs, MPI_INT, 
        loc_A_colidx, loc_A_nnz, MPI_INT, 0, MPI_COMM_WORLD, &reqs[1]
    );
    MPI_Iscatterv(
        glb_A_csrval, A_nnz_scnts, A_nnz_displs, MPI_DOUBLE, 
        loc_A_csrval, loc_A_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD, &reqs[2]
    );
    MPI_Waitall(3, &reqs[0], MPI_STATUSES_IGNORE);
    loc_A_rowptr[loc_A_nrow] = A_nnz_displs[my_rank + 1];
    MPI_Barrier(MPI_COMM_WORLD);
    et = get_wtime_sec();
    ut = et - st;
    if (my_rank == 0)
    {
        printf("1D partition and distribution of A used %.2f s\n", ut);
        fflush(stdout);
    }

    // 3. 1D partition input dense matrix B and output dense matrix C
    int loc_B_srow = x_displs[my_rank];
    int loc_B_nrow = x_displs[my_rank + 1] - loc_B_srow;
    int loc_C_srow = loc_A_srow;
    int loc_C_nrow = loc_A_nrow;
    double *loc_B = (double *) malloc(sizeof(double) * loc_B_nrow * glb_n);
    double *loc_C = (double *) malloc(sizeof(double) * loc_C_nrow * glb_n);
    int layout = 0;
    int loc_B_ld = 0, loc_C_ld = 0;
    if (layout == 0)
    {
        // Row major
        loc_B_ld = glb_n;
        loc_C_ld = glb_n;
        for (int i = 0; i < loc_B_nrow; i++)
        {
            int glb_i = loc_B_srow + i;
            for (int j = 0; j < glb_n; j++)
                loc_B[i * loc_B_ld + j] = glb_i * 1.0 + j * 0.01;
        }
    } else {
        // Column major
        loc_B_ld = loc_B_nrow;
        loc_C_ld = loc_C_nrow;
        for (int j = 0; j < glb_n; j++)
        {
            for (int i = 0; i < loc_B_nrow; i++)
            {
                int glb_i = loc_B_srow + i;
                loc_B[i + j * loc_B_ld] = glb_i * 1.0 + j * 0.01;
            }
        }
    }

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
        st = get_wtime_sec();
        rp_spmm_exec(rp_spmm, layout, loc_B, loc_B_ld, loc_C, loc_C_ld);
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
        double *glb_B = NULL, *ref_C = NULL, *recv_C = NULL;
        if (my_rank == 0)
        {
            glb_B  = (double *) malloc(sizeof(double) * glb_k * glb_n);
            ref_C  = (double *) malloc(sizeof(double) * glb_m * glb_n);
            recv_C = (double *) malloc(sizeof(double) * glb_m * glb_n);
        }
        if (layout == 0)
        {
            memcpy(sbuf, loc_C, sizeof(double) * loc_C_nrow * glb_n);
        } else {
            for (int i = 0; i < loc_C_nrow; i++)
                for (int j = 0; j < glb_n; j++)
                    sbuf[i * glb_n + j] = loc_C[i + j * loc_C_nrow];
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
            for (int i = 0; i < glb_k; i++)
                for (int j = 0; j < glb_n; j++)
                    glb_B[i * glb_n + j] = i * 1.0 + j * 0.01;

            #ifdef USE_MKL
            double alpha = 1.0, beta = 0.0;
            sparse_matrix_t mkl_spA;
            struct matrix_descr mkl_descA;
            mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
            mkl_descA.mode = SPARSE_FILL_MODE_FULL;
            mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
            sparse_layout_t sp_layout = SPARSE_LAYOUT_ROW_MAJOR;
            mkl_sparse_d_create_csr(
                &mkl_spA, SPARSE_INDEX_BASE_ZERO, glb_m, glb_k, 
                glb_A_rowptr, glb_A_rowptr + 1, glb_A_colidx, glb_A_csrval
            );
            mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_spA, mkl_descA, 
                sp_layout, glb_B, glb_n, glb_n, beta, ref_C, glb_n
            );
            mkl_sparse_destroy(mkl_spA);
            #else
            #error No CPU SpMM implementation
            #endif

            double C_fnorm, err_fnorm;
            calc_err_2norm(glb_m * glb_n, ref_C, recv_C, &C_fnorm, &err_fnorm);
            printf("||C_ref - C||_f / ||C_ref||_f = %e\n", err_fnorm / C_fnorm);
            fflush(stdout);
        }

        free(glb_B);
        free(ref_C);
        free(recv_C);
        free(sbuf);
    }  // End of "if (chk_res)"

    free(glb_A_rowptr);
    free(glb_A_colidx);
    free(glb_A_csrval);
    free(A_m_displs);
    free(A_m_scnts);
    free(A_nnz_scnts);
    free(A_nnz_displs);
    free(x_displs);
    free(loc_B);
    free(loc_C);
    MPI_Finalize();
    return 0;
}