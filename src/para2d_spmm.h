#ifndef __PARA2D_SPMM_H__
#define __PARA2D_SPMM_H__

#include "rowpara_spmm.h"

struct para2d_spmm
{
    rp_spmm_p rp_spmm;      // 1D row-parallel spmm
    MPI_Comm  comm_glb;     // Global communicator, shallow copy, do not free it
    MPI_Comm  comm_col;     // Communicator for processes in the same column (for rp_spmm), need to be freed
    size_t    rA_cost;      // Communication cost of replicating A
    double    t_init;       // Time (s) for para2d_spmm_init()
    double    t_ag_A;       // Time (s) for allgather A
};
typedef struct para2d_spmm  para2d_spmm_s;
typedef struct para2d_spmm *para2d_spmm_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a para2d_spmm struct
// Input parameters:
//   comm      : MPI communicator for all processes, will not be duplicated, do not free it
//   pm, pn    : Process grid dimensions, pn groups * pm-way row parallel SpMM
//   A0_rowptr : Size nproc + 1, row pointer of A0, A0 is 1D row partitioned
//   B_rowptr  : Size pm + 1, row pointer of B
//   AC_rowptr : Size pm + 1, row pointer of replicated A and final C
//   BC_colptr : Size pn + 1, column pointer of B and C
//   A_rowptr  : Size A0_rowptr[i + 1] - A0_rowptr[i] for rank i, row pointer of local A
//   A_colidx  : Size A_rowptr[end], column index of local A
//   A_val     : Size A_rowptr[end], value of local A
// Output parameter:
//   para2d_spmm: Initialized para2d_spmm struct
// Notes: 
//   1. See calc_spmm_2dpg() for the meanings and requirements of {A0, B, AC}_rowptr and BC_colptr.
//   2. For rank r, it's process grid coordinate will be (r / pn, r % pn).
//   3. For process P_{i, j}, it has:
//      (1) loc_A_nrow = loc_C_nrow = AC_rowptr[i + 1] - AC_rowptr[i],
//      (2) loc_B_nrow = B_rowptr[i + 1] - B_rowptr[i], 
//      (3) loc_B_ncol = loc_C_ncol = BC_colptr[j + 1] - BC_colptr[j].
void para2d_spmm_init(
    MPI_Comm comm, const int pm, const int pn, const int *A0_rowptr, 
    const int *B_rowptr, const int *AC_rowptr, const int *BC_colptr, 
    const int *A_rowptr, const int *A_colidx, const double *A_val,
    para2d_spmm_p *para2d_spmm
);

// Free a para2d_spmm struct
void para2d_spmm_free(para2d_spmm_p *para2d_spmm);

// Compute C := A * B using para2d_spmm
// Input parameters:
//   para2d_spmm : Initialized para2d_spmm struct
//   BC_layout   : Layout of B and C, 0 for row-major, 1 for column-major
//   B           : Size >= ldB * loc_B_ncol (col-major) or loc_B_nrow * ldB (row-major), local B matrix
//   ldB         : Leading dimension of B, >= loc_B_nrow (col-major) or loc_B_ncol (row-major)
//   ldC         : Leading dimension of C, >= loc_C_nrow (col-major) or loc_C_ncol (row-major)
// Output parameter:
//   C : Size >= ldC * loc_B_ncol (col-major) or loc_C_nrow * ldC (row-major), local C matrix
// Notes: 
//   For process P_{i, j}, it's:
//   1. loc_B_nrow = B_rowptr[i + 1] - B_rowptr[i],
//   2. loc_C_nrow = AC_rowptr[i + 1] - AC_rowptr[i],
//   3. loc_B_ncol = loc_C_ncol = BC_colptr[j + 1] - BC_colptr[j].
void para2d_spmm_exec(
    para2d_spmm_p para2d_spmm, const int BC_layout, const double *B, const int ldB,
    double *C, const int ldC
);

// Print statistic info of a para2d_spmm struct
void para2d_spmm_print_stat(para2d_spmm_p para2d_spmm);

// Clear statistic info of a para2d_spmm struct
void para2d_spmm_clear_stat(para2d_spmm_p para2d_spmm);

#ifdef __cplusplus
}
#endif

#endif
