#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <mpi.h>

#include "utils.h"
#include "dev_type.h"
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

int is_dev_type_valid(dev_type_t dev_type)
{
    int valid_type = 0;
    if (dev_type == DEV_TYPE_HOST) valid_type = 1;
    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)) 
        valid_type = 1;
    #endif
    return valid_type;
}

// Allocate memory on specified device 
void *dev_type_malloc(size_t bytes, dev_type_t dev_type)
{
    void *mem = NULL;
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return mem;
    }

    #if !defined(USE_CUDA)
    if (dev_type == DEV_TYPE_HOST) mem = malloc(bytes);
    #endif
    
    #ifdef USE_CUDA
    // When using CUDA, host memory is usually used as a mirror or 
    // a copy buffer of a device memory, so just use pinned memory
    if (dev_type == DEV_TYPE_HOST) 
        cuda_malloc_host(&mem, bytes);
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        cuda_malloc_dev(&mem, bytes);
    #endif

    if ((bytes > 0) && (mem == NULL))
        ERROR_PRINTF("Failed to malloc %zu bytes on device type %d\n", bytes, dev_type);

    return mem;
}

// Free memory on specified device
void dev_type_free(void *mem, dev_type_t dev_type)
{
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    if (mem == NULL) return;

    #if !defined(USE_CUDA)
    if (dev_type == DEV_TYPE_HOST) free(mem);
    #endif

    #ifdef USE_CUDA
    if (dev_type == DEV_TYPE_HOST) 
        cuda_free_host(mem);
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        cuda_free_dev(mem);
    #endif
}

// Re-allocate memory on specified device if required size larger than current size
void dev_type_realloc(size_t *curr_bytes, size_t req_bytes, dev_type_t dev_type, void **mem)
{
    if (*curr_bytes >= req_bytes) return;
    dev_type_free(*mem, dev_type);
    *curr_bytes = 0;
    *mem = dev_type_malloc(req_bytes, dev_type);
    if (*mem != NULL) *curr_bytes = req_bytes;
}

// Memset on specified device
void dev_type_memset(void *mem, int value, size_t bytes, dev_type_t dev_type)
{
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    if (dev_type == DEV_TYPE_HOST) memset(mem, value, bytes);

    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        cuda_memset_dev(mem, value, bytes);
    #endif
}

// Memcpy on specified device
void dev_type_memcpy(
    void *dst, const void *src, size_t bytes, 
    dev_type_t dst_dev_type, dev_type_t src_dev_type
)
{
    if ((is_dev_type_valid(dst_dev_type) == 0) || (is_dev_type_valid(src_dev_type) == 0))
    {
        ERROR_PRINTF("Invalid dst device type %d or src device type %d\n", dst_dev_type, src_dev_type);
        return;
    }

    if ((dst_dev_type == DEV_TYPE_HOST) && (src_dev_type == DEV_TYPE_HOST)) memcpy(dst, src, bytes);

    #ifdef USE_CUDA
    int cuda_dst = ((dst_dev_type == DEV_TYPE_CUDA) || (dst_dev_type == DEV_TYPE_CUDA_MPI_DIRECT));
    int cuda_src = ((src_dev_type == DEV_TYPE_CUDA) || (src_dev_type == DEV_TYPE_CUDA_MPI_DIRECT));
    if ((cuda_dst == 1) && (cuda_src == 0)) cuda_memcpy_h2d(src, dst, bytes);
    if ((cuda_dst == 0) && (cuda_src == 1)) cuda_memcpy_d2h(src, dst, bytes);
    if ((cuda_dst == 1) && (cuda_src == 1)) cuda_memcpy_d2d(src, dst, bytes);
    #endif
}

// Copy a row-major matrix block to another row-major matrix
void dev_type_copy_matrix(
    size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd,
    dev_type_t dev_type
)
{
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    if (dev_type == DEV_TYPE_HOST) 
        copy_matrix(dt_size, nrow, ncol, src, lds, dst, ldd, 1);
    
    #ifdef USE_CUDA
    ASSERT_PRINTF(
        dt_size == 4 || dt_size == 8,
        "dt_size == 4 or 8 required for CUDA memory"
    );
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        cuda_copy_matrix(dt_size, nrow, ncol, src, lds, dst, ldd);
    #endif
}
