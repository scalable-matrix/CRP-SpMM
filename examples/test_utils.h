#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include <omp.h>
#include <mpi.h>

#include "mmio_utils.h"
#include "utils.h"

#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

int can_check_res(int my_rank, int m, int n, int k);

void read_mtx_csr(
    const char *fname, const int need_symm, int *glb_m, int *glb_k, int glb_n, 
    int **glb_A_rowptr, int **glb_A_colidx, double **glb_A_csrval
);

void scatter_csr_rows(
    MPI_Comm comm, int nproc, int my_rank, int *A_m_displs, int *A_nnz_displs, int *A_m_scnts, 
    int *A_nnz_scnts, int *glb_A_rowptr, int *glb_A_colidx, double *glb_A_csrval, 
    int **loc_A_rowptr_, int **loc_A_colidx_, double **loc_A_csrval_, int dbg_print
);

void fill_B(
    int layout, double *B, int ldB, int srow, int nrow, int scol, int ncol,
    double factor_i, double factor_j
);

#ifdef USE_MKL
void mkl_csr_spmm(
    int m, int k, int n, int *rowptr, int *colidx, double *val,
    double *B, int ldB, double *C, int ldC
);
#endif

#ifdef __cplusplus
}
#endif

#endif
