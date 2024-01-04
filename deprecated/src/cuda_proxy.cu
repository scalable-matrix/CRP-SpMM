#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "cuda_utils.h"
#include "cuda_proxy.h"
#include <cusparse_v2.h>

static int get_mpi_local_rank_env()
{
    int local_rank = -1;
    char *env_p;

    // MPICH
    env_p = getenv("MPI_LOCALRANKID");
    if (env_p != NULL) return atoi(env_p);

    // MVAPICH2
    env_p = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (env_p != NULL) return atoi(env_p);

    // OpenMPI
    env_p = getenv("OMPI_COMM_WORLD_NODE_RANK");
    if (env_p != NULL) return atoi(env_p);

    // SLURM or PBS/Torque
    env_p = getenv("SLURM_LOCALID");
    if (env_p != NULL) return atoi(env_p);

    env_p = getenv("PBS_O_VNODENUM");
    if (env_p != NULL) return atoi(env_p);

    return local_rank;
}

void select_cuda_device_by_mpi_local_rank()
{
    int local_rank = get_mpi_local_rank_env();
    if (local_rank == -1) local_rank = 0;
    int num_gpu, dev_id;
    CUDA_RUNTIME_CHECK( cudaGetDeviceCount(&num_gpu) );
    dev_id = local_rank % num_gpu;
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
}

void cuda_set_rt_dev_id(const int dev_id)
{
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
}

void cuda_memcpy_h2d(const void *hptr, void *dptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr, hptr, bytes, cudaMemcpyHostToDevice) );
}

void cuda_memcpy_d2h(const void *dptr, void *hptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(hptr, dptr, bytes, cudaMemcpyDeviceToHost) );
}

void cuda_memcpy_d2d(const void *dptr_src, void *dptr_dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr_dst, dptr_src, bytes, cudaMemcpyDeviceToDevice) );
}

void cuda_memcpy_auto(const void *src, void *dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) );
}

void cuda_malloc_dev(void **dptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMalloc(dptr_, bytes) );
}

void cuda_malloc_host(void **hptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMallocHost(hptr_, bytes) );
}

void cuda_memset_dev(void *dptr, const int value, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemset(dptr, value, bytes) );    
}

void cuda_free_dev(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaFree(dptr) );
}

void cuda_free_host(void *hptr)
{
    CUDA_RUNTIME_CHECK( cudaFreeHost(hptr) );
}

void cuda_device_sync()
{
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
}

void cuda_stream_sync(void *stream_p)
{
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(*((cudaStream_t *) stream_p)) );
}

void cuda_copy_matrix(
    size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd
)
{
    size_t dpitch = dt_size * ldd;
    size_t spitch = dt_size * lds;
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(dst, dpitch, src, spitch, dt_size * ncol, nrow, cudaMemcpyDeviceToDevice) );
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
    CUDA_RUNTIME_CHECK( cudaPeekAtLastError() );
}

static cusparseHandle_t cusparse_handle = NULL;

void cuda_cusparse_csr_spmm(
    const int m, const int n, const int k, const double alpha, 
    const int A_nnz, const int *A_rowptr_h, const int *A_colidx_h, const double *A_val_h,
    const double *B_h, const int ldB, const double beta, double *C_h, const int ldC
)
{
    int *A_rowptr_d, *A_colidx_d;
    double *A_val_d, *B_d, *C_d;

    // Allocate device memory and copy host data to device
    CUDA_RUNTIME_CHECK( cudaMalloc((void **) &A_rowptr_d, sizeof(int)    * (m + 1)) );
    CUDA_RUNTIME_CHECK( cudaMalloc((void **) &A_colidx_d, sizeof(int)    * A_nnz) );
    CUDA_RUNTIME_CHECK( cudaMalloc((void **) &A_val_d,    sizeof(double) * A_nnz) );
    CUDA_RUNTIME_CHECK( cudaMalloc((void **) &B_d,        sizeof(double) * k * n) );
    CUDA_RUNTIME_CHECK( cudaMalloc((void **) &C_d,        sizeof(double) * m * n) );
    CUDA_RUNTIME_CHECK( cudaMemcpy(A_rowptr_d, A_rowptr_h, sizeof(int)    * (m + 1), cudaMemcpyHostToDevice) );
    CUDA_RUNTIME_CHECK( cudaMemcpy(A_colidx_d, A_colidx_h, sizeof(int)    * A_nnz,   cudaMemcpyHostToDevice) );
    CUDA_RUNTIME_CHECK( cudaMemcpy(A_val_d,    A_val_h,    sizeof(double) * A_nnz,   cudaMemcpyHostToDevice) );
    size_t B_host_pitch = sizeof(double) * ldB;
    size_t B_dev_pitch  = sizeof(double) * n;
    size_t B_rowbytes   = sizeof(double) * n;
    size_t C_host_pitch = sizeof(double) * ldC;
    size_t C_dev_pitch  = sizeof(double) * n;
    size_t C_rowbytes   = sizeof(double) * n;
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(B_d, B_dev_pitch, B_h, B_host_pitch, B_rowbytes, k, cudaMemcpyHostToDevice) );
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(C_d, C_dev_pitch, C_h, C_host_pitch, C_rowbytes, m, cudaMemcpyHostToDevice) );

    // Create cusparse handle and allocate SpMM buffer
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *buff_d = NULL;
    size_t buff_bytes = 0;
    if (cusparse_handle == NULL) CUSPARSE_CHECK( cusparseCreate(&cusparse_handle) );
    CUSPARSE_CHECK( cusparseCreateCsr(
        &matA, m, k, A_nnz, A_rowptr_d, A_colidx_d, A_val_d, 
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
    ) );
    CUSPARSE_CHECK( cusparseCreateDnMat(&matB, k, n, n, B_d, CUDA_R_64F, CUSPARSE_ORDER_ROW) );
    CUSPARSE_CHECK( cusparseCreateDnMat(&matC, m, n, n, C_d, CUDA_R_64F, CUSPARSE_ORDER_ROW) );
    CUSPARSE_CHECK( cusparseSpMM_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &buff_bytes
    ) );
    CUDA_RUNTIME_CHECK( cudaMalloc(&buff_d, buff_bytes) );

    // Execute SpMM, copy result back to host, and free device memory
    CUSPARSE_CHECK( cusparseSpMM(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, buff_d
    ) );
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(C_h, C_host_pitch, C_d, C_dev_pitch, C_rowbytes, m, cudaMemcpyDeviceToHost) );
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) );
    CUSPARSE_CHECK( cusparseDestroyDnMat(matB) );
    CUSPARSE_CHECK( cusparseDestroyDnMat(matC) );
    CUDA_RUNTIME_CHECK( cudaFree(A_rowptr_d) );
    CUDA_RUNTIME_CHECK( cudaFree(A_colidx_d) );
    CUDA_RUNTIME_CHECK( cudaFree(A_val_d) );
    CUDA_RUNTIME_CHECK( cudaFree(B_d) );
    CUDA_RUNTIME_CHECK( cudaFree(C_d) );
    CUDA_RUNTIME_CHECK( cudaFree(buff_d) );
}