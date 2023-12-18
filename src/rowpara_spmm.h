#ifndef __ROWPARA_SPMM_H__
#define __ROWPARA_SPMM_H__

#include <stddef.h>
#include <stdlib.h>
#include <mpi.h>

struct rowpara_spmm
{
    int    nproc, my_rank;      // Number of processes and rank of this process in comm
    int    glb_n;               // Global number of columns of B and C
    int    A_nrow;              // Number of rows of local A
    int    rB_nrow;             // Number of rows of local redist B
    int    rB_self_src_offset;  // Redist B self-to-self source offset
    int    rB_self_dst_offset;  // Redist B self-to-self destination offset
    int    rB_self_nrow;        // Redist B self-to-self number of rows
    int    rB_p2p;              // Use p2p or alltoallv for redist B
    int    *A_rowptr;           // Size A_nrow + 1, local A matrix CSR row pointer
    int    *A_colidx;           // Size A_nnz, local A matrix CSR column index
    int    *rB_self_src_ridxs;  // Size rB_self_nrow, redist B self-to-self source row indices
    int    *rB_scnts;           // Size nproc, send counts of redist B matrix
    int    *rB_sridxs;          // Size unknown, send row indices of redist B matrix
    int    *rB_sdispls;         // Size nproc, send buffer displacements of redist B matrix
    int    *rB_rcnts;           // Size nproc, receive counts of redist B matrix
    int    *rB_rridxs;          // Size unknown, receive row indices of redist B matrix
    int    *rB_rdispls;         // Size nproc, receive buffer displacements of redist B matrix
    double *A_val;              // Size A_nnz, local A matrix CSR value
    MPI_Comm comm;

    // Statistic info
    size_t rB_recv_size;        // Number of elements communicated in redistribution of B
    int    n_exec;              // Number of times rp_spmm_exec() is called
    double t_init;              // Time (s) for rp_spmm_init()
    double t_pack;              // Time (s) for packing B send buffer
    double t_a2a;               // Time (s) for alltoallv of B
    double t_unpack;            // Time (s) for unpacking B recv buffer
    double t_spmm;              // Time (s) for local SpMM
    double t_exec;              // Time (s) for rp_spmm_exec()
};
typedef struct rowpara_spmm  rp_spmm_s;
typedef struct rowpara_spmm *rp_spmm_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a rowpara_spmm struct for a 1D row-parallel SpMM
// Input parameters:
//   A_{srow, nrow} : Starting row and number of rows of local A
//   A_rowptr       : Size A_nrow + 1, local A matrix CSR row pointer
//   A_colidx       : Local A matrix CSR column index
//   A_val          : Local A matrix CSR value
//   B_row_displs   : Size nproc + 1, indices of the first row of B on each process
//   glb_n          : Global number of columns of B and C
//   comm           : MPI communicator for all processes, will not be duplicated, do not free it
// Output parameter:
//   rp_spmm : Pointer to an initialized rowpara_spmm struct
// Note: A_colidx[i] and A_val[i] will be accessed for 0 <= i < A_rowptr[A_srow + A_nrow] - A_rowptr[A_srow]
void rp_spmm_init(
    const int A_srow, const int A_nrow, const int *A_rowptr, const int *A_colidx, 
    const double *A_val, const int *B_row_displs, const int glb_n, MPI_Comm comm, 
    rp_spmm_p *rp_spmm
);

// Free a rowpara_spmm struct
void rp_spmm_free(rp_spmm_p *rp_spmm);

// Compute C := A * B
// Input parameters:
//   rp_spmm   : Pointer to an initialized rowpara_spmm struct
//   BC_layout : Layout of B and C, 0 for row-major, 1 for column-major
//   B         : Size >= ldB * glb_n (col-major) or rp_spmm->B_nrow * ldB (row-major), local B matrix
//   ldB       : Leading dimension of B, >= rp_spmm->rB_nrow (col-major) or glb_n (row-major)
//   ldC       : Leading dimension of C, >= rp_spmm->A_nrow (col-major) or glb_n (row-major)
// Output parameter:
//   C : Size >= ldC * glb_n (col-major) or rp_spmm->A_nrow * ldC (row-major), local C matrix
void rp_spmm_exec(
    rp_spmm_p rp_spmm, const int BC_layout, const double *B, const int ldB,
    double *C, const int ldC
);

// Print statistic info of a rowpara_spmm struct
void rp_spmm_print_stat(rp_spmm_p rp_spmm);

// Clear statistic info of a rowpara_spmm struct
void rp_spmm_clear_stat(rp_spmm_p rp_spmm);

#ifdef __cplusplus
}
#endif

#endif
