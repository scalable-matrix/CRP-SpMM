#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include <omp.h>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "utils.h"
#include "rowpara_spmm.h"

// Initialize a rowpara_spmm struct for a 1D row-parallel SpMM
void rp_spmm_init(
    const int A_srow, const int A_nrow, const int *A_rowptr, const int *A_colidx, 
    const double *A_val, const int *B_row_displs, const int glb_n, MPI_Comm comm, 
    rp_spmm_p *rp_spmm
)
{
    rp_spmm_p rp_spmm_ = (rp_spmm_p) malloc(sizeof(rp_spmm_s));
    memset(rp_spmm_, 0, sizeof(rp_spmm_s));

    double st = get_wtime_sec();

    int nproc, my_rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &my_rank);
    rp_spmm_->nproc   = nproc;
    rp_spmm_->my_rank = my_rank;
    rp_spmm_->A_srow = A_srow;
    rp_spmm_->A_nrow  = A_nrow;
    rp_spmm_->B_srow  = B_row_displs[my_rank];
    rp_spmm_->B_nrow  = B_row_displs[my_rank + 1] - rp_spmm_->B_srow;
    rp_spmm_->glb_n   = glb_n;
    rp_spmm_->comm    = comm;

    // 1. Shrink the row and column ranges of A and count the nonzero rows of B
    int A_nnz = A_rowptr[A_nrow] - A_rowptr[0];
    int A_nnz_sidx = A_rowptr[0];
    int    *A1_rowptr = (int *)    malloc(sizeof(int)    * (A_nrow + 1));
    int    *A1_colidx = (int *)    malloc(sizeof(int)    * A_nnz);
    double *A1_val    = (double *) malloc(sizeof(double) * A_nnz);
    ASSERT_PRINTF(
        A1_rowptr != NULL && A1_colidx != NULL && A1_val != NULL, 
        "Failed to allocate work memory for rp_spmm\n"
    );
    int rB_srow = INT_MAX, rB_erow = 0;
    for (int i = 0; i < A_nnz; i++)
    {
        int icol = A_colidx[i];
        if (icol < rB_srow) rB_srow = icol;
        if (icol > rB_erow) rB_erow = icol;
    }
    for (int i = 0; i <= A_nrow; i++) A1_rowptr[i] = A_rowptr[i] - A_nnz_sidx;
    for (int i = 0; i < A_nnz; i++)
    {
        A1_colidx[i] = A_colidx[i] - rB_srow;
        A1_val[i] = A_val[i];
    }
    int glb_k = B_row_displs[nproc];
    int *B_rowflag = (int *) malloc(sizeof(int) * glb_k);
    ASSERT_PRINTF(B_rowflag != NULL, "Failed to allocate work memory for rp_spmm\n");
    memset(B_rowflag, 0, sizeof(int) * glb_k);
    for (int i = 0; i < A_nnz; i++)
        B_rowflag[A_colidx[i]] = 1;
    rp_spmm_->A_rowptr = A1_rowptr;
    rp_spmm_->A_colidx = A1_colidx;
    rp_spmm_->A_val    = A1_val;
    rp_spmm_->rB_srow  = rB_srow;
    rp_spmm_->rB_nrow  = rB_erow - rB_srow + 1;

    // 2. Find the rows of B that are needed by A on each process
    int *rB_rcnts   = (int *) malloc(sizeof(int) * nproc);
    int *rB_rdispls = (int *) malloc(sizeof(int) * (nproc + 1));
    int *rB_rridxs  = (int *) malloc(sizeof(int) * rp_spmm_->rB_nrow);
    memset(rB_rcnts, 0, sizeof(int) * nproc);
    for (int iproc = 0; iproc < nproc; iproc++)
        for (int irow = B_row_displs[iproc]; irow < B_row_displs[iproc + 1]; irow++)
            if (B_rowflag[irow]) rB_rcnts[iproc]++;
    rB_rdispls[0] = 0;
    for (int iproc = 0; iproc < nproc; iproc++)
        rB_rdispls[iproc + 1] = rB_rdispls[iproc] + rB_rcnts[iproc];
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        for (int irow = B_row_displs[iproc]; irow < B_row_displs[iproc + 1]; irow++)
        {
            if (B_rowflag[irow])
            {
                rB_rridxs[rB_rdispls[iproc]] = irow;
                rB_rdispls[iproc]++;
            }
        }
    }
    rB_rdispls[0] = 0;
    for (int iproc = 0; iproc < nproc; iproc++)
        rB_rdispls[iproc + 1] = rB_rdispls[iproc] + rB_rcnts[iproc];
    rp_spmm_->rB_rcnts   = rB_rcnts;
    rp_spmm_->rB_rdispls = rB_rdispls;
    rp_spmm_->rB_rridxs  = rB_rridxs;
    free(B_rowflag);
    int rB_self_size = rB_rcnts[my_rank];
    rp_spmm_->rB_recv_size = (size_t) (rB_rdispls[nproc] - rB_self_size);

    // 3. Get the indices of B rows this process needs to send
    int *rB_scnts   = (int *) malloc(sizeof(int) * nproc);
    int *rB_sdispls = (int *) malloc(sizeof(int) * (nproc + 1));
    MPI_Alltoall(rB_rcnts, 1, MPI_INT, rB_scnts, 1, MPI_INT, comm);
    rB_sdispls[0] = 0;
    for (int iproc = 0; iproc < nproc; iproc++)
        rB_sdispls[iproc + 1] = rB_sdispls[iproc] + rB_scnts[iproc];
    int *rB_sridxs = (int *) malloc(sizeof(int) * rB_sdispls[nproc]);
    MPI_Alltoallv(
        rB_rridxs, rB_rcnts, rB_rdispls, MPI_INT,
        rB_sridxs, rB_scnts, rB_sdispls, MPI_INT, comm
    );
    rp_spmm_->rB_scnts   = rB_scnts;
    rp_spmm_->rB_sdispls = rB_sdispls;
    rp_spmm_->rB_sridxs  = rB_sridxs;

    // 4. Some post-processing of counts and indices
    for (int i = 0; i < rB_rdispls[nproc]; i++)
        rB_rridxs[i] -= rp_spmm_->rB_srow;
    for (int i = 0; i < rB_sdispls[nproc]; i++)
        rB_sridxs[i] -= B_row_displs[my_rank];
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        rB_rcnts[iproc]   *= glb_n;
        rB_rdispls[iproc] *= glb_n;
        rB_scnts[iproc]   *= glb_n;
        rB_sdispls[iproc] *= glb_n;
    }
    rB_rdispls[nproc] *= glb_n;
    rB_sdispls[nproc] *= glb_n;

    double et = get_wtime_sec();
    rp_spmm_->t_init = et - st;

    *rp_spmm = rp_spmm_;
}

// Free a rowpara_spmm struct
void rp_spmm_free(rp_spmm_p *rp_spmm)
{
    rp_spmm_p rp_spmm_ = *rp_spmm;
    free(rp_spmm_->A_rowptr);
    free(rp_spmm_->A_colidx);
    free(rp_spmm_->A_val);
    free(rp_spmm_->rB_rcnts);
    free(rp_spmm_->rB_rdispls);
    free(rp_spmm_->rB_rridxs);
    free(rp_spmm_->rB_scnts);
    free(rp_spmm_->rB_sdispls);
    free(rp_spmm_->rB_sridxs);
    free(rp_spmm_);
    *rp_spmm = NULL;
}

// Compute C := A * B
void rp_spmm_exec(
    rp_spmm_p rp_spmm, const int BC_layout, const double *B, const int ldB,
    double *C, const int ldC
)
{
    int nproc = rp_spmm->nproc;
    int glb_n = rp_spmm->glb_n;
    double st, et;
    
    // 1. Pack B send buffer for redistribution
    int *rB_scnts   = rp_spmm->rB_scnts;
    int *rB_sdispls = rp_spmm->rB_sdispls;
    int *rB_sridxs  = rp_spmm->rB_sridxs;
    double *rB_sendbuf = (double *) malloc(sizeof(double) * rB_sdispls[nproc]);
    ASSERT_PRINTF(rB_sendbuf != NULL, "Failed to allocate work memory for rp_spmm_exec\n");
    st = get_wtime_sec();
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *rB_sridxs_i = rB_sridxs + rB_sdispls[iproc] / glb_n;
        int rB_send_nrow = rB_scnts[iproc] / glb_n;
        double *rB_sendbuf_i = rB_sendbuf + rB_sdispls[iproc];
        if (BC_layout == 0)
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rB_send_nrow; i++)
            {
                size_t src_offset = (size_t) rB_sridxs_i[i] * (size_t) ldB;
                size_t dst_offset = (size_t) i * (size_t) glb_n;
                const double *src = B + src_offset;
                double *dst = rB_sendbuf_i + dst_offset;
                memcpy(dst, src, sizeof(double) * glb_n);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < glb_n; j++)
            {
                size_t src_offset = (size_t) j * (size_t) ldB;
                size_t dst_offset = (size_t) j * (size_t) rB_send_nrow;
                const double *src_j = B + src_offset;
                double *dst_j = rB_sendbuf_i + dst_offset;
                for (int i = 0; i < rB_send_nrow; i++)
                    dst_j[i] = src_j[rB_sridxs_i[i]];
            }
        }
    }  // End of iproc loop
    et = get_wtime_sec();
    rp_spmm->t_pack += et - st;

    // 2. Redistribute B and unpack B receive buffer
    int rB_nrow     = rp_spmm->rB_nrow;
    int *rB_rcnts   = rp_spmm->rB_rcnts;
    int *rB_rdispls = rp_spmm->rB_rdispls;
    int *rB_rridxs  = rp_spmm->rB_rridxs;
    double *rB_recvbuf = (double *) malloc(sizeof(double) * rB_rdispls[nproc]);
    double *rB         = (double *) malloc(sizeof(double) * rB_nrow * glb_n);
    ASSERT_PRINTF(rB_recvbuf != NULL, "Failed to allocate work memory for rp_spmm_exec\n");
    int ldrB = (BC_layout == 0) ? glb_n : rB_nrow;
    st = get_wtime_sec();
    MPI_Alltoallv(
        rB_sendbuf, rB_scnts, rB_sdispls, MPI_DOUBLE, 
        rB_recvbuf, rB_rcnts, rB_rdispls, MPI_DOUBLE, rp_spmm->comm
    );
    et = get_wtime_sec();
    rp_spmm->t_a2a += et - st;
    st = get_wtime_sec();
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *rB_rridxs_i = rB_rridxs + rB_rdispls[iproc] / glb_n;
        int rB_recv_nrow = rB_rcnts[iproc] / glb_n;
        double *rB_recvbuf_i = rB_recvbuf + rB_rdispls[iproc];
        if (BC_layout == 0)
        {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rB_recv_nrow; i++)
            {
                size_t src_offset = (size_t) i * (size_t) glb_n;
                size_t dst_offset = (size_t) rB_rridxs_i[i] * (size_t) glb_n;
                double *src = rB_recvbuf_i + src_offset;
                double *dst = rB + dst_offset;
                memcpy(dst, src, sizeof(double) * glb_n);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < glb_n; j++)
            {
                size_t src_offset = (size_t) j * (size_t) rB_recv_nrow;
                size_t dst_offset = (size_t) j * (size_t) rB_nrow;
                double *src_j = rB_recvbuf_i + src_offset;
                double *dst_j = rB + dst_offset;
                for (int i = 0; i < rB_recv_nrow; i++)
                    dst_j[rB_rridxs_i[i]] = src_j[i];
            }
        }
    }  // End of iproc loop
    et = get_wtime_sec();
    rp_spmm->t_unpack += et - st;

    // 3. Local SpMM
    st = get_wtime_sec();
    #ifdef USE_MKL
    sparse_matrix_t mkl_spA;
    struct matrix_descr mkl_descA;
    mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_descA.mode = SPARSE_FILL_MODE_FULL;
    mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
    int A_nrow = rp_spmm->A_nrow;
    int *A_rowptr = rp_spmm->A_rowptr;
    int *A_colidx = rp_spmm->A_colidx;
    double *A_val = rp_spmm->A_val;
    mkl_sparse_d_create_csr(
        &mkl_spA, SPARSE_INDEX_BASE_ZERO, A_nrow, rB_nrow, 
        A_rowptr, A_rowptr + 1, A_colidx, A_val
    );
    double alpha = 1.0, beta = 0.0;
    sparse_layout_t layout = (BC_layout == 0) ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
    mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_spA, mkl_descA, 
        layout, rB, glb_n, ldrB, beta, C, ldC
    );
    mkl_sparse_destroy(mkl_spA);
    #else
    #error No CPU SpMM implementation
    #endif
    et = get_wtime_sec();
    rp_spmm->t_spmm += et - st;
    rp_spmm->n_exec++;

    free(rB_sendbuf);
    free(rB_recvbuf);
    free(rB);
}

// Print statistic info of rp_spmm_p
void rp_spmm_print_stat(rp_spmm_p rp_spmm)
{
    if (rp_spmm == NULL) return;
    int my_rank = rp_spmm->my_rank;
    if (my_rank == 0) printf("rp_spmm init time: %.3f s\n", rp_spmm->t_init);
    int n_exec = rp_spmm->n_exec;
    if (n_exec == 0) return;
    size_t rB_recv_max = 0, rB_recv_total = 0;
    double t_raw[5], t_max[5], t_avg[5];
    t_raw[0] = rp_spmm->t_init;
    t_raw[1] = rp_spmm->t_pack;
    t_raw[2] = rp_spmm->t_a2a;
    t_raw[3] = rp_spmm->t_unpack;
    t_raw[4] = rp_spmm->t_spmm;
    MPI_Reduce(&rp_spmm->rB_recv_size, &rB_recv_max,   1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, rp_spmm->comm);
    MPI_Reduce(&rp_spmm->rB_recv_size, &rB_recv_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, rp_spmm->comm);
    MPI_Reduce(&t_raw[0], &t_max[0], 5, MPI_DOUBLE, MPI_MAX, 0, rp_spmm->comm);
    MPI_Reduce(&t_raw[0], &t_avg[0], 5, MPI_DOUBLE, MPI_SUM, 0, rp_spmm->comm);
    for (int i = 0; i < 5; i++)
    {
        t_max[i] = t_max[i] / n_exec;
        t_avg[i] = t_avg[i] / (n_exec * rp_spmm->nproc);
    }
    if (my_rank == 0)
    {
        printf("B matrix receive rows (elements) max, total = %zu, %zu", rB_recv_max, rB_recv_total);
        printf(" (%zu, %zu)\n", rB_recv_max * rp_spmm->glb_n, rB_recv_total * rp_spmm->glb_n);
        printf("-------------------- Runtime (s) --------------------\n");
        printf("                                     avg         max\n");
        printf("Pack B matrix for redistribution  %6.3f      %6.3f\n", t_avg[1], t_max[1]);
        printf("Redistribute B matrix             %6.3f      %6.3f\n", t_avg[2], t_max[2]);
        printf("Unpack received B matrix data     %6.3f      %6.3f\n", t_avg[3], t_max[3]);
        printf("Local SpMM                        %6.3f      %6.3f\n", t_avg[4], t_max[4]);
        printf("\n");
        fflush(stdout);
    }
}

// Clear statistic info of rp_spmm_p
void rp_spmm_clear_stat(rp_spmm_p rp_spmm)
{
    rp_spmm->n_exec   = 0;
    rp_spmm->t_init   = 0.0;
    rp_spmm->t_pack   = 0.0;
    rp_spmm->t_a2a    = 0.0;
    rp_spmm->t_unpack = 0.0;
    rp_spmm->t_spmm   = 0.0;
}
