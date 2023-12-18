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
#include "para2d_spmm.h"

// Initialize a para2d_spmm struct
void para2d_spmm_init(
    MPI_Comm comm, const int pm, const int pn, const int *A0_rowptr, 
    const int *B_rowptr, const int *AC_rowptr, const int *BC_colptr, 
    const int *A_rowptr, const int *A_colidx, const double *A_val,
    para2d_spmm_p *para2d_spmm
)
{
    para2d_spmm_p para2d_spmm_ = (para2d_spmm_p) malloc(sizeof(para2d_spmm_s));
    memset(para2d_spmm_, 0, sizeof(para2d_spmm_s));
    para2d_spmm_->comm_glb = comm;

    double st, et;
    para2d_spmm_->t_init = 0.0;

    // 1. Get the global rank, process grid coordinate, and split communicator
    int glb_rank, pi, pj;
    MPI_Comm comm_row;
    st = get_wtime_sec();
    MPI_Comm_rank(comm, &glb_rank);
    pi = glb_rank / pn;
    pj = glb_rank % pn;
    MPI_Comm_split(comm, pi, pj, &comm_row);
    MPI_Comm_split(comm, pj, pi, &para2d_spmm_->comm_col);
    et = get_wtime_sec();
    para2d_spmm_->t_init += et - st;

    // 2. Allgather A for rp_spmm_init()
    st = get_wtime_sec();
    int A0_nrow    = A0_rowptr[glb_rank + 1] - A0_rowptr[glb_rank];
    int A0_nnz     = A_rowptr[A0_nrow] - A_rowptr[0];
    int loc_A_srow = A0_rowptr[pi * pn];
    int loc_A_nrow = A0_rowptr[(pi + 1) * pn] - loc_A_srow;
    int *loc_A_rowptr = (int *) malloc(sizeof(int) * (loc_A_nrow + 1));
    int *loc_A_colidx = NULL;
    double *loc_A_val = NULL;
    if (pn > 1)
    {
        int *loc_A_rcnts   = (int *) malloc(sizeof(int) * pn);
        int *loc_A_rdispls = (int *) malloc(sizeof(int) * (pn + 1));
        int *A0_nnzs       = (int *) malloc(sizeof(int) * pn);
        MPI_Allgather(&A0_nnz, 1, MPI_INT, A0_nnzs, 1, MPI_INT, comm_row);
        loc_A_rdispls[0] = 0;
        for (int i = 0; i < pn; i++)
        {
            int rank = pi * pn + i;
            loc_A_rcnts[i] = A0_rowptr[rank + 1] - A0_rowptr[rank];
            loc_A_rdispls[i + 1] = loc_A_rdispls[i] + loc_A_rcnts[i];
        }
        MPI_Allgatherv(A_rowptr, loc_A_rcnts[pj], MPI_INT, loc_A_rowptr, loc_A_rcnts, loc_A_rdispls, MPI_INT, comm_row);
        loc_A_rdispls[0] = 0;
        for (int i = 0; i < pn; i++)
        {
            loc_A_rcnts[i] = A0_nnzs[i];
            loc_A_rdispls[i + 1] = loc_A_rdispls[i] + loc_A_rcnts[i];
        }
        int loc_A_nnz = loc_A_rdispls[pn];
        loc_A_rowptr[loc_A_nrow] = loc_A_rowptr[0] + loc_A_nnz;
        loc_A_colidx = (int *)    malloc(sizeof(int)    * loc_A_nnz);
        loc_A_val    = (double *) malloc(sizeof(double) * loc_A_nnz);
        MPI_Allgatherv(A_colidx, A0_nnz, MPI_INT,    loc_A_colidx, loc_A_rcnts, loc_A_rdispls, MPI_INT,    comm_row);
        MPI_Allgatherv(A_val,    A0_nnz, MPI_DOUBLE, loc_A_val,    loc_A_rcnts, loc_A_rdispls, MPI_DOUBLE, comm_row);
        free(loc_A_rcnts);
        free(loc_A_rdispls);
        free(A0_nnzs);
    } else {
        int loc_A_nnz = A0_nnz;
        loc_A_colidx = (int *)    malloc(sizeof(int)    * loc_A_nnz);
        loc_A_val    = (double *) malloc(sizeof(double) * loc_A_nnz);
        memcpy(loc_A_rowptr, A_rowptr, sizeof(int) * (loc_A_nrow + 1));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < loc_A_nnz; i++)
        {
            loc_A_colidx[i] = A_colidx[i];
            loc_A_val[i]    = A_val[i];
        }
    }
    et = get_wtime_sec();
    para2d_spmm_->t_ag_A += et - st;

    // 3. Initialize a rp_spmm struct
    st = get_wtime_sec();
    int loc_BC_n = BC_colptr[pj + 1] - BC_colptr[pj];
    rp_spmm_init(
        loc_A_srow, loc_A_nrow, loc_A_rowptr, loc_A_colidx, loc_A_val,
        B_rowptr, loc_BC_n, para2d_spmm_->comm_col, &para2d_spmm_->rp_spmm
    );
    et = get_wtime_sec();
    para2d_spmm_->t_init += et - st;

    free(loc_A_rowptr);
    free(loc_A_colidx);
    free(loc_A_val);
    *para2d_spmm = para2d_spmm_;
}

// Free a para2d_spmm struct
void para2d_spmm_free(para2d_spmm_p *para2d_spmm)
{
    para2d_spmm_p para2d_spmm_ = *para2d_spmm;
    if (para2d_spmm_ == NULL) return;
    rp_spmm_free(&para2d_spmm_->rp_spmm);
    MPI_Comm_free(&para2d_spmm_->comm_col);
    free(para2d_spmm_);
    *para2d_spmm = NULL;
}

// Compute C := A * B using para2d_spmm
void para2d_spmm_exec(
    para2d_spmm_p para2d_spmm, const int BC_layout, const double *B, const int ldB,
    double *C, const int ldC
)
{
    // All matrices are ready, directly call rp_spmm_exec()
    rp_spmm_exec(para2d_spmm->rp_spmm, BC_layout, B, ldB, C, ldC);
}

// Print statistic info of a para2d_spmm struct
void para2d_spmm_print_stat(para2d_spmm_p para2d_spmm)
{
    if (para2d_spmm == NULL) return;
    rp_spmm_p rp_spmm = para2d_spmm->rp_spmm;
    int glb_rank, glb_nproc;
    MPI_Comm_rank(para2d_spmm->comm_glb, &glb_rank);
    MPI_Comm_size(para2d_spmm->comm_glb, &glb_nproc);
    int n_exec = rp_spmm->n_exec;
    if (n_exec == 0) return;
    size_t rB_recv = rp_spmm->rB_recv_size * rp_spmm->glb_n;
    size_t rB_recv_max = 0, rB_recv_sum = 0;
    double t_raw[7], t_max[7], t_avg[7];
    t_raw[0] = para2d_spmm->t_init;
    t_raw[1] = para2d_spmm->t_ag_A;
    t_raw[2] = rp_spmm->t_pack;
    t_raw[3] = rp_spmm->t_a2a;
    t_raw[4] = rp_spmm->t_unpack;
    t_raw[5] = rp_spmm->t_spmm;
    t_raw[6] = rp_spmm->t_exec;
    MPI_Reduce(&rB_recv, &rB_recv_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, para2d_spmm->comm_glb);
    MPI_Reduce(&rB_recv, &rB_recv_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, para2d_spmm->comm_glb);
    MPI_Reduce(&t_raw[0], &t_max[0], 7, MPI_DOUBLE, MPI_MAX, 0, para2d_spmm->comm_glb);
    MPI_Reduce(&t_raw[0], &t_avg[0], 7, MPI_DOUBLE, MPI_SUM, 0, para2d_spmm->comm_glb);
    for (int i = 2; i <= 6; i++)
    {
        t_max[i] = t_max[i] / n_exec;
        t_avg[i] = t_avg[i] / (n_exec * glb_nproc);
    }
    t_avg[1] /= glb_nproc;
    if (glb_rank == 0)
    {
        printf("para2d_spmm_init() time = %.2f s\n", t_max[0]);
        printf("B matrix receive rows (elements) max, total = %zu, %zu", rB_recv_max, rB_recv_sum);
        printf(" (%zu, %zu)\n", rB_recv_max, rB_recv_sum);
        printf("-------------------- Runtime (s) --------------------\n");
        printf("                                     avg         max\n");
        printf("Replicate A matrix (once)         %6.3f      %6.3f\n", t_avg[1], t_max[1]);
        printf("Pack B matrix for redistribution  %6.3f      %6.3f\n", t_avg[2], t_max[2]);
        printf("Redistribute B matrix             %6.3f      %6.3f\n", t_avg[3], t_max[3]);
        printf("Unpack received B matrix data     %6.3f      %6.3f\n", t_avg[4], t_max[4]);
        printf("Local SpMM                        %6.3f      %6.3f\n", t_avg[5], t_max[5]);
        printf("Total para2d_spmm_exec()          %6.3f      %6.3f\n", t_avg[6], t_max[6]);
        printf("Replicate A + para2d_spmm_exec()  %6.3f      %6.3f\n", t_avg[1] + t_avg[6], t_max[1] + t_max[6]);
        printf("\n");
        fflush(stdout);
    }
}

// Clear statistic info of a para2d_spmm struct
void para2d_spmm_clear_stat(para2d_spmm_p para2d_spmm)
{
    if (para2d_spmm == NULL) return;
    rp_spmm_clear_stat(para2d_spmm->rp_spmm);
}
