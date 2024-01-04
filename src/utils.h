// @brief    : Some helper functions I use here and there
// @author   : Hua Huang <huangh223@gatech.edu>
// @modified : 2023-12-17

#ifndef __HUANGH223_UTILS_H__
#define __HUANGH223_UTILS_H__

#include <stdio.h>

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define INT_MSIZE sizeof(int)
#define DBL_MSIZE sizeof(double)

#define MIN(a, b)  ((a) < (b) ? (a) : (b))
#define MAX(a, b)  ((a) > (b) ? (a) : (b))

#define INFO_PRINTF(fmt, ...)                       \
    do                                              \
    {                                               \
        fprintf(stdout, "[INFO] %s, %d: " fmt,      \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(stdout);                             \
    } while (0)

#define DEBUG_PRINTF(fmt, ...)                      \
    do                                              \
    {                                               \
        fprintf(stderr, "[DEBUG] %s, %d: " fmt,     \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(stderr);                             \
    } while (0)

#define WARNING_PRINTF(fmt, ...)                    \
    do                                              \
    {                                               \
        fprintf(stderr, "[WARNING] %s, %d: " fmt,   \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(stderr);                             \
    } while (0)

#define ERROR_PRINTF(fmt, ...)                      \
    do                                              \
    {                                               \
        fprintf(stderr, "[ERROR] %s, %d: " fmt,     \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(stderr);                             \
    } while (0)

#define ASSERT_PRINTF(expr, fmt, ...)                   \
    do                                                  \
    {                                                   \
        if (!(expr))                                    \
        {                                               \
            fprintf(stderr, "[FATAL] %s, %d: " fmt,     \
                    __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(stderr);                             \
            assert(expr);                               \
        }                                               \
    } while (0)


#define GET_ENV_INT_VAR(var, env_str, var_str, default_val, min_val, max_val, print_info) \
    do                                          \
    {                                           \
        char *env_str_p = getenv(env_str);      \
        if (env_str_p != NULL)                  \
        {                                       \
            var = atoi(env_str_p);              \
            if (var < min_val || var > max_val) var = default_val;  \
            if (print_info && var != default_val)                   \
            {                                   \
                INFO_PRINTF("Overriding parameter %s: %d (default) --> %d (runtime)\n",  \
                var_str, default_val, var);     \
            }                                   \
        } else {                                \
            var = default_val;                  \
        }                                       \
    } while (0)

#define REALLOC_BUFFER(ptr, ptr_type, curr_bytes, new_bytes)    \
    do                                                          \
    {                                                           \
        if ((new_bytes) > curr_bytes)                           \
        {                                                       \
            free(ptr);                                          \
            ptr = NULL;                                         \
            curr_bytes = (new_bytes);                           \
            ptr = (ptr_type) malloc(curr_bytes);                \
        }                                                       \
    } while (0)

// Get wall-clock time in seconds
// Output parameter:
//   <return> : Wall-clock time in second
double get_wtime_sec();

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
// Input parameters:
//   len  : Length of the array
//   nblk : Total number of blocks to be partitioned
//   iblk : Index of the block whose start position we need.
//          0 <= iblk <= nblk, iblk == 0/nblk return 0/len.
// Output parameters:
//   *blk_spos : Index of the first element in the iblk-th block, -1 means invalid parameters
//   *blk_size : Size of the iblk-th block
void calc_block_spos_size(
    const int len, const int nblk, const int iblk,
    int *blk_spos, int *blk_size
);

// Allocate a piece of aligned memory 
// Input parameters:
//   size      : Size of the memory to be allocated, in bytes
//   alignment : Size of the alignment, in bytes, must be a power of 8
// Output parameter:
//   <return>  : Pointer to the allocated aligned memory
void *malloc_aligned(size_t size, size_t alignment);

// Free a piece of aligned memory allocated by malloc_aligned()
// Input parameter:
//   mem : Pointer to the memory to be free
void free_aligned(void *mem);

// Calculate the 2-norm of a vector
// Warning: this is a naive implementation, not numerically stable
// Input parameters:
//   len : Length of the vector
//   x   : Size >= len, vector
// Output parameter:
//   <return> : 2-norm of vector x
double calc_2norm(const int len, const double *x);

// Calculate the 2-norm of the difference between two vectors 
// and the 2-norm of the reference vector 
// Input parameters:
//   len : Length of the vector
//   x0  : Size >= len, reference vector
//   x1  : Size >= len, target vector to be compared 
// Output parameters:
//   *x0_2norm_  : 2-norm of vector x0
//   *err_2norm_ : 2-norm of vector (x0 - x1)
void calc_err_2norm(
    const int len, const double *x0, const double *x1, 
    double *x0_2norm_, double *err_2norm_
);

// Copy a row-major matrix to another row-major matrix
// Input parameters:
//   dt_size : Size of matrix element data type, in bytes
//   nrow    : Number of rows to be copied
//   ncol    : Number of columns to be copied
//   src     : Size >= lds * nrow, source matrix
//   lds     : Leading dimension of src, >= ncol
//   ldd     : Leading dimension of dst, >= ncol
//   use_omp : 0 or 1, if this function should use OpenMP parallelization
// Output parameter:
//   dst : Size >= ldd * nrow, destination matrix
void copy_matrix(
    const size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd, const int use_omp
);

// ========== For debugging ==========

// Print a matrix to standard output
// Input parameters:
//   dtype : 0 -> int, 1 -> double
//   stype : 0 -> row-major, 1 -> column-major
//   mat   : Size >= ldm * nrow or ncol * ldm, matrix to print
//   ldm   : Leading dimension of mat
//   nrow  : Number of rows to print
//   ncol  : Number of columns to print
//   fmt   : Output format string
//   name  : Name of the matrix
void print_matrix(
    const int dtype, const int stype, const void *mat, const int ldm, 
    const int nrow, const int ncol, const char *fmt, const char *name
);

// Dump binary to file
void dump_binary(const char *fname, void *data, const size_t bytes);

#ifdef __cplusplus
}
#endif

#endif

