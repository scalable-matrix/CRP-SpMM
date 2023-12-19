#ifndef __CRPSPMM_H__
#define __CRPSPMM_H__

#include <mpi.h>
#include "dev_type.h"
#include "mat_redist.h"

struct crpspmm_engine
{
    int    np_glb, rank_glb;            // Number of processes and rank of this process in the global communicator
    int    np_row, np_col;              // Number of processes on the row / column direction (size of comm_{col, row})
    int    rank_row, rank_col;          // Rank of this process on the row / column (rank in comm_{col, row})
    int    glb_m, glb_n, glb_k;         // A, B, C matrices dimensions
    int    loc_A_srow, loc_A_erow;      // After allgatherv A, this process has A rows [A_srow_loc, A_erow_loc) 
    int    loc_A_nrow, loc_A_nnz;       // After allgatherv A, number of rows and nonzeros of A on this process
    int    loc_A_nnz_s;                 // After allgatherv A, global index of A's first nonzero on this process
    int    rd_B_srow, rd_B_erow;        // This process has B rows [rd_B_srow, rd_B_erow) after redistribution
    int    loc_B_srow, loc_B_erow;      // This process has B columns [loc_B_srow, loc_B_erow) after replication
    int    loc_B_scol, loc_B_ecol;      // This process has B rows [loc_B_scol, loc_B_ecol) after replication
    int    loc_B_nrow, loc_B_ncol;      // Number of rows and columns of B on this process after replication
    int    a2a_B_finegrain;             // If alltoallv B is done in a fine-grained way
    int    alloc_workbuf;               // If work_buf is allocated by crpspmm_engine
    int    use_CUDA;                    // If local SpMM is computed using CUDA
    size_t self_workbuf_bytes;          // Self work buffer size in bytes
    size_t rd_workbuf_bytes;            // max(rd_{Ai, Av, B, C}->workbuf_bytes)
    int    *agv_A_recvcnts;             // Size np_col, number of A nonzeros received from each process in allgatherv A
    int    *agv_A_displs;               // Size np_col, displacements of A nonzeros received from each process in allgatherv A
    int    *a2a_B_sendcnts;             // Size np_row, number of B elements sent to each process in alltoallv B
    int    *a2a_B_sdispls;              // Size np_row, displacements of B elements sent to each process in alltoallv B
    int    *a2a_B_recvcnts;             // Size np_col, number of B elements received from each process in alltoallv B
    int    *a2a_B_rdispls;              // Size np_col, displacements of B elements received from each process in alltoallv B
    int    *a2a_B_send_ridx;            // Size loc_B_nrow, row indices of B elements sent to each process in alltoallv B
    int    *a2a_B_recv_ridx;            // Size loc_B_nrow, row indices of B elements received from each process in alltoallv B
    int    *loc_A_rowptr;               // Size loc_A_nrow+1, local CSR row pointers after allgatherv A, on host
    int    *loc_A_colidx;               // Size loc_A_nnz, local CSR column indices after allgatherv A, on host
    double *loc_A_val;                  // Size loc_A_nnz, local CSR nonzeroes after allgatherv A, on host
    double *a2a_B_sbuf;                 // Size a2a_B_sendcnts[rank_row] * loc_B_ncol, row-major, on host, alltoallv B send buffer
    double *a2a_B_rbuf;                 // Size a2a_B_recvcnts[rank_col] * loc_B_ncol, row-major, on host, alltoallv B recv buffer
    double *red_B;                      // Size (rd_B_erow - rd_B_srow) * loc_B_ncol, row-major, on host, B blocks after redistribution
    double *loc_B;                      // Size loc_B_nrow * loc_B_ncol, row-major, on host, B blocks after replication
    double *loc_C;                      // Size loc_A_nrow * loc_B_ncol, row-major, on host
    double *workbuf;                    // Work buffer, on host, all double* above + local_colidx are aliases to workbuf
    MPI_Comm comm_row, comm_col;        // MPI communicators for all processes in the same row and column
    MPI_Comm comm_glb;                  // MPI communicator for all processes
    mat_redist_engine_p rd_Ai, rd_Av;   // Matrix redistribution engines for sparse matrix A's column indices and nonzeros
    mat_redist_engine_p rd_B, rd_C;     // Matrix redistribution engines for dense matrices B and C

    // Statistic info
    int n_exec;                         // Number of times crpspmm_engine_exec() is called
    double t_init, t_exec;              // Time (s) for initialization and executions
    double t_rd_A, t_agv_A;             // Time (s) for initialization, redistribution of A, and allgatherv of A
    double t_rd_B, t_a2a_B;             // Time (s) for redistribution and alltoallv of B
    double t_spmm, t_rd_C;              // Time (s) for local SpMM, and redistribution of C
    double t_exec_nr;                   // Time (s) for local SpMM without redistribution
    size_t nelem_A_rd, nelem_A_agv;     // Number of A matrix elements communicated in redistribution and allgatherv
    size_t nelem_B_rd, nelem_B_a2av;    // Number of B matrix elements communicated in redistribution and alltoallv
    size_t nelem_B_a2av_min;            // Minimum number of B matrix elements required to be communicated
};
typedef struct crpspmm_engine  crpspmm_engine_s;
typedef struct crpspmm_engine* crpspmm_engine_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a crpspmm_engine structure for C := A * B, where A is sparse, B and C are dense
// Input parameters:
//   m, n, k         : Size of matrix A (m * k), B (k * n), and C (m * n)
//   src_{A, B}_srow : First row         of input A/B matrix on this MPI process
//   src_{A, B}_nrow : Number of rows    of input A/B matrix on this MPI process
//   src_A_rowptr    : Global CSR rowptr of input  A  matrix on this MPI process
//   src_A_colidx    : Global CSR colidx of input  A  matrix on this MPI process
//   src_B_scol      : First column      of input  B  matrix on this MPI process
//   src_B_ncol      : Number of columns of input  B  matrix on this MPI process
//   dst_C_srow      : First row         of output C  matrix on this MPI process
//   dst_C_nrow      : Number of rows    of output C  matrix on this MPI process
//   dst_C_scol      : First column      of output C  matrix on this MPI process
//   dst_C_ncol      : Number of columns of output C  matrix on this MPI process
//   comm            : MPI communicator of all MPI processes participating CA3DMM
//   use_CUDA        : If local SpMM is computed using CUDA
// Output parameters:
//   *engine_       : Pointer to an initialized crpspmm_engine structure
//   *workbuf_bytes : Optional. If pointer is not NULL, the returning value is the size 
//                    of work buffer, and crpspmm_engine will not allocate work buffer.
//                    If pointer is NULL, crpspmm_engine will allocate and free work buffer.
// Note: 
//   1. crpspmm_engine does not check the correctness of src_{A, B}_{s, n}{row, col}
//   2. src_A_rowptr and src_A_colidx are always on host, no matter whether use_CUDA is true or false
void crpspmm_engine_init(
    const int m, const int n, const int k, 
    const int src_A_srow, const int src_A_nrow,
    const int *src_A_rowptr, const int *src_A_colidx, 
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    MPI_Comm comm, int use_CUDA, crpspmm_engine_p *engine_, size_t *workbuf_bytes
);

// Attach an external work buffer for crpspmm_engine
// Input parameters:
//   engine  : Initialized crpspmm_engine_p
//   workbuf : Work buffer on host, size >= *workbuf_bytes returned by crpspmm_engine_init()
void crpspmm_engine_attach_workbuf(crpspmm_engine_p engine, double *workbuf);

// Calculate SpMM C := A * B
// Input parameters:
//   engine        : Initialized crpspmm_engine_p
//   src_A_rowptr  : Same as that in crpspmm_engine_init()
//   src_A_colidx  : Same as that in crpspmm_engine_init()
//   src_A_val     : CSR nonzeroes of input A matrix on this MPI process
//   src_B         : Size >= src_B_nrow * ldB, row-major, dense matrix B on this MPI process
//   ldB, ldC      : Leading dimensions of src_B and dst_C
// Output parameter:
//   dst_C : Size >= dst_C_nrow * ldC, row-major, dense matrix C on this MPI process
// Note: All input and output matrices are on host, no matter whether use_CUDA is true or false
void crpspmm_engine_exec(
    crpspmm_engine_p engine, 
    const int *src_A_rowptr, const int *src_A_colidx, const double *src_A_val,
    const double *src_B, const int ldB, double *dst_C, const int ldC
);

// Free a crpspmm_engine_s
void crpspmm_engine_free(crpspmm_engine_p *engine_);

// Print statistic info of crpspmm_engine
void crpspmm_engine_print_stat(crpspmm_engine_p engine);

// Clear statistic info of crpspmm_engine
void crpspmm_engine_clear_stat(crpspmm_engine_p engine);

#ifdef __cplusplus
}
#endif


#endif
