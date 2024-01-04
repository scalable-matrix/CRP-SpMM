#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <omp.h>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

#include "utils.h"
#include "crpspmm.h"

static int prime_factorization(int n, int **factors)
{
    int nfac = 0, c = 2, max_nfac = (int) ceil(log2((double) n)) + 1;
    int *fac = (int *) malloc(sizeof(int) * max_nfac);
    while (n > 1)
    {
        if (n % c == 0)
        {
            fac[nfac++] = c;
            n /= c;
        }
        else c++;
    }
    *factors = fac;
    return nfac;
}

// Calculate the intersection of two segments [s0, e0] and [s1, e1], not [s0, e0) and [s1, e1) !
static void calc_seg_intersection(
    int s0, int e0, int s1, int e1, 
    int *is_intersect, int *is, int *ie
)
{
    if (s0 > s1)
    {
        int swap;
        swap = s0; s0 = s1; s1 = swap;
        swap = e0; e0 = e1; e1 = swap;
    }
    if (s1 > e0 || s1 > e1 || s0 > e0)
    {
        *is_intersect = 0;
        *is = -1;
        *ie = -1;
        return;
    }
    *is_intersect = 1;
    *is = s1;
    *ie = (e0 < e1) ? e0 : e1;
}

// Initialize a crpspmm_engine structure for C := A * B, where A is sparse, B and C are dense
void crpspmm_engine_init(
    const int m, const int n, const int k, 
    const int src_A_srow, const int src_A_nrow,
    const int *src_A_rowptr, const int *src_A_colidx, 
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    MPI_Comm comm, int use_CUDA, crpspmm_engine_p *engine_, size_t *workbuf_bytes
)
{
    *engine_ = NULL;

    crpspmm_engine_p engine = (crpspmm_engine_p) malloc(sizeof(crpspmm_engine_s));
    memset(engine, 0, sizeof(crpspmm_engine_s));

    double st = MPI_Wtime();

    engine->glb_m = m;
    engine->glb_n = n;
    engine->glb_k = k;
    engine->use_CUDA = use_CUDA;
    MPI_Comm_size(comm, &engine->np_glb);
    MPI_Comm_rank(comm, &engine->rank_glb);
    int rank_glb = engine->rank_glb;
    int np_glb = engine->np_glb;

    // 1. Allgather A_rowptr
    int *proc_A_nrow    = (int *) malloc(sizeof(int) * np_glb);
    int *proc_A_rdispls = (int *) malloc(sizeof(int) * (np_glb + 1));
    int *A_rowptr_glb   = (int *) malloc(sizeof(int) * (m + 1));
    proc_A_nrow[rank_glb] = src_A_nrow;
    memcpy(A_rowptr_glb + src_A_srow, src_A_rowptr, sizeof(int) * src_A_nrow);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, proc_A_nrow, 1, MPI_INT, comm);
    proc_A_rdispls[0] = 0;
    for (int i = 0; i < np_glb; i++)
        proc_A_rdispls[i + 1] = proc_A_rdispls[i] + proc_A_nrow[i];
    MPI_Allgatherv(
        MPI_IN_PLACE, src_A_nrow, MPI_INT, A_rowptr_glb, 
        proc_A_nrow, proc_A_rdispls, MPI_INT, comm
    );
    if (rank_glb == np_glb - 1) A_rowptr_glb[m] = src_A_rowptr[src_A_nrow];  // A_nnz_glb
    MPI_Bcast(&A_rowptr_glb[m], 1, MPI_INT, np_glb - 1, comm);

    // 2. Calculate the column index ranges of each row of A
    //    Assume that the column indices in src_A_colidx are sorted in ascending order
    int *A_cidx_se_glb = (int *) malloc(sizeof(int) * m * 2);
    for (int i = 0; i < src_A_nrow; i++)
    {
        int row_i_1st_loc_nnz_idx  = src_A_rowptr[i]           - src_A_rowptr[0];
        int row_i_last_loc_nnz_idx = (src_A_rowptr[i + 1] - 1) - src_A_rowptr[0];
        A_cidx_se_glb[2 * (src_A_srow + i) + 0] = src_A_colidx[row_i_1st_loc_nnz_idx];
        A_cidx_se_glb[2 * (src_A_srow + i) + 1] = src_A_colidx[row_i_last_loc_nnz_idx];
    }
    proc_A_rdispls[0] = 0;
    for (int i = 0; i < np_glb; i++)
    {
        proc_A_nrow[i] *= 2;
        proc_A_rdispls[i + 1] = proc_A_rdispls[i] + proc_A_nrow[i];
    }
    MPI_Allgatherv(
        MPI_IN_PLACE, src_A_nrow * 2, MPI_INT, A_cidx_se_glb, 
        proc_A_nrow, proc_A_rdispls, MPI_INT, comm
    );
    for (int i = 0; i < np_glb; i++)
    {
        proc_A_nrow[i] /= 2;
        proc_A_rdispls[i + 1] = proc_A_rdispls[i] + proc_A_nrow[i];
    }

    // 3. Calculate the 2D partitioning of C
    //    For CSR with <int32_t> indices and <double> values, the average memory cost
    //    per nonzero is 12.x bytes, which is 1.5x times of sizeof(double)
    const double A_nnz_cost_factor = 1.5;
    int m_split = 1, n_split = 1;
    int *m_split_idx = (int *) malloc(sizeof(int) * (np_glb + 1));
    int *m_split_idx2 = (int *) malloc(sizeof(int) * (np_glb + 1));
    int nfac, *fac;
    nfac = prime_factorization(np_glb, &fac);
    // If we never choose to split M, we need to initialize m_split_idx_ to be [0, nrow]
    m_split_idx[0] = 0;
    m_split_idx[1] = m;
    size_t curr_copy_B_size = (size_t) k * (size_t) n;  // Originally we have one copy of B
    int A_nnz = A_rowptr_glb[m];
    for (int i = 0; i < nfac; i++)
    {
        int p_i = fac[nfac - 1 - i];
        // If split N, the number of B matrix elements to be copied remains unchanged, 
        // the number of A matrix copies is multiplied by p_i
        size_t A_copy_cost1 = (size_t) ((double) A_nnz * (double) n_split * A_nnz_cost_factor);
        size_t A_copy_cost2 = A_copy_cost1 * (size_t) p_i;
        size_t split_n_comm_cost = A_copy_cost2 + curr_copy_B_size;
        if (n_split * p_i > n) split_n_comm_cost = SIZE_MAX;
        // If split M, the number of A matrix copies remains unchanged, needs to 
        // recalculate the number of B matrix elements to be copied
        size_t split_m_comm_cost = A_copy_cost1;
        size_t curr_copy_B_size2 = 0;
        m_split *= p_i;
        int curr_srow = 0;
        m_split_idx2[0] = 0;
        for (int j = 0; j < m_split; j++)
        {
            // Find the last row assigned to the j-th row panel
            int j_max_nnz = A_nnz / m_split * (j + 1);
            if (j == m_split - 1) j_max_nnz = A_nnz;
            int curr_erow = curr_srow + 1;
            int min_col_idx = A_cidx_se_glb[2 * curr_srow];
            int max_col_idx = A_cidx_se_glb[2 * curr_srow + 1];
            while (A_rowptr_glb[curr_erow] < j_max_nnz)
            {
                // Calculate the upper bound of number of B rows required by the j-th row panel
                int min_col_idx2 = A_cidx_se_glb[2 * curr_erow];
                int max_col_idx2 = A_cidx_se_glb[2 * curr_erow + 1];
                if (min_col_idx2 < min_col_idx) min_col_idx = min_col_idx2;
                if (max_col_idx2 > max_col_idx) max_col_idx = max_col_idx2;
                curr_erow++;
            }
            curr_copy_B_size2 += (size_t) (max_col_idx - min_col_idx + 1) * (size_t) n;
            m_split_idx2[j + 1] = curr_erow;
            curr_srow = curr_erow;
        }
        split_m_comm_cost += curr_copy_B_size2;
        m_split /= p_i;
        // Choose to split M or N
        if (split_m_comm_cost < split_n_comm_cost)
        {
            m_split *= p_i;
            curr_copy_B_size = curr_copy_B_size2;
            memcpy(m_split_idx, m_split_idx2, sizeof(int) * (m_split + 1));
        } else {
            n_split *= p_i;
        }
    }  // End of i loop

    // 4. Create 2D row-major process grid of size m_split * n_split
    int np_row = m_split, np_col = n_split;
    int rank_row = rank_glb / np_col, rank_col = rank_glb % np_col;
    int loc_B_scol, loc_B_ecol, loc_B_ncol;
    calc_block_spos_size(n, np_col, rank_col, &loc_B_scol, &loc_B_ncol);
    loc_B_ecol = loc_B_scol + loc_B_ncol;
    engine->np_col      = np_col;
    engine->np_row      = np_row;
    engine->rank_col    = rank_col;
    engine->rank_row    = rank_row;
    engine->loc_A_srow  = m_split_idx[rank_row];
    engine->loc_A_erow  = m_split_idx[rank_row + 1];
    engine->loc_A_nrow  = engine->loc_A_erow - engine->loc_A_srow;
    engine->loc_A_nnz_s = A_rowptr_glb[engine->loc_A_srow];
    engine->loc_A_nnz   = A_rowptr_glb[engine->loc_A_erow] - A_rowptr_glb[engine->loc_A_srow];
    engine->loc_B_scol  = loc_B_scol;
    engine->loc_B_ecol  = loc_B_ecol;
    engine->loc_B_ncol  = loc_B_ecol - loc_B_scol;
    engine->loc_B_srow  = k;
    engine->loc_B_erow  = 0;
    for (int i = m_split_idx[rank_row]; i < m_split_idx[rank_row + 1]; i++)
    {
        int min_col_idx = A_cidx_se_glb[2 * i];
        int max_col_idx = A_cidx_se_glb[2 * i + 1];
        if (min_col_idx < engine->loc_B_srow) engine->loc_B_srow = min_col_idx;
        if (max_col_idx > engine->loc_B_erow) engine->loc_B_erow = max_col_idx;
    }
    engine->loc_B_erow++;
    engine->loc_B_nrow = engine->loc_B_erow - engine->loc_B_srow;
    MPI_Comm_split(comm, rank_col, rank_glb, &engine->comm_col);
    MPI_Comm_split(comm, rank_row, rank_glb, &engine->comm_row);
    engine->comm_glb = comm;
    int loc_A_srow  = engine->loc_A_srow;
    int loc_A_nrow  = engine->loc_A_nrow;
    int loc_B_srow  = engine->loc_B_srow;
    int loc_B_erow  = engine->loc_B_erow;
    int loc_B_nrow  = engine->loc_B_nrow;
    int loc_A_nnz   = engine->loc_A_nnz;
    int loc_A_nnz_s = engine->loc_A_nnz_s;

    // 5. All processes in comm_row requires the same A row panel, 
    //    the first redistribution step is tp redistribute the A row panel to 
    //    all processes in comm_row, the second step is to replicate the A row panel
    int src_A_nnz_start = A_rowptr_glb[src_A_srow];
    int src_A_nnz = A_rowptr_glb[src_A_srow + src_A_nrow] - src_A_nnz_start;
    int *agv_A_recvcnts = (int *) malloc(sizeof(int) * np_col);
    int *agv_A_displs = (int *) malloc(sizeof(int) * (np_col + 1));
    for (int i = 0; i < np_col; i++)
        calc_block_spos_size(loc_A_nnz, np_col, i, &agv_A_displs[i], &agv_A_recvcnts[i]);        
    agv_A_displs[np_col]   = loc_A_nnz;
    engine->agv_A_displs   = agv_A_displs;
    engine->agv_A_recvcnts = agv_A_recvcnts;
    int rd_A_nnz_start = loc_A_nnz_s + agv_A_displs[rank_col];
    int rd_A_nnz = agv_A_recvcnts[rank_col];
    mat_redist_engine_p rd_Ai, rd_Av;
    size_t rd_Ai_workbuf_bytes = 0, rd_Av_workbuf_bytes = 0;
    mat_redist_engine_init(
        0, src_A_nnz_start, 1, src_A_nnz,
        0, rd_A_nnz_start,  1, rd_A_nnz,
        comm, MPI_INT, sizeof(int), DEV_TYPE_HOST,
        &rd_Ai, &rd_Ai_workbuf_bytes
    );
    mat_redist_engine_init(
        0, src_A_nnz_start, 1, src_A_nnz,
        0, rd_A_nnz_start,  1, rd_A_nnz,
        comm, MPI_DOUBLE, sizeof(double), DEV_TYPE_HOST,
        &rd_Av, &rd_Av_workbuf_bytes
    );
    engine->rd_Ai = rd_Ai;
    engine->rd_Av = rd_Av;
    
    int *loc_A_rowptr = (int *) malloc(sizeof(int) * (loc_A_nrow + 1));
    for (int i = 0; i <= loc_A_nrow; i++)
        loc_A_rowptr[i] = A_rowptr_glb[loc_A_srow + i] - loc_A_nnz_s;
    engine->loc_A_rowptr = loc_A_rowptr;

    // 6. All processes in comm_col requires some rows of the same B column panel,
    //    the first redistribution step is to redistribute the B column panel to
    //    all processes in comm_col, the second step is to replicate B rows
    int *B_rd_row_displs = (int *) malloc(sizeof(int) * (np_row + 1));
    int *B_rd_row_blksizes = (int *) malloc(sizeof(int) * np_row);
    for (int i = 0; i < np_row; i++)
        calc_block_spos_size(k, np_row, i, &B_rd_row_displs[i], &B_rd_row_blksizes[i]);
    B_rd_row_displs[np_row] = k;
    int B_rd_srow = B_rd_row_displs[rank_row];
    int B_rd_erow = B_rd_row_displs[rank_row + 1];
    engine->rd_B_srow = B_rd_srow;
    engine->rd_B_erow = B_rd_erow;
    mat_redist_engine_p rd_B;
    size_t rd_B_workbuf_bytes = 0;
    mat_redist_engine_init(
        src_B_srow, src_B_scol, src_B_nrow, src_B_ncol,
        B_rd_srow, loc_B_scol, B_rd_erow - B_rd_srow, loc_B_ncol,
        comm, MPI_DOUBLE, sizeof(double), DEV_TYPE_HOST,
        &rd_B, &rd_B_workbuf_bytes
    );
    engine->rd_B = rd_B;
    GET_ENV_INT_VAR(engine->a2a_B_finegrain, "A2A_B_FINEGRAIN", "a2a_B_finegrain", 0, 0, 1, rank_glb == 0);
    if (engine->a2a_B_finegrain == 0)
    {
        int *proc_B_rows = (int *) malloc(sizeof(int) * np_row * 2);
        proc_B_rows[2 * rank_row] = loc_B_srow;
        proc_B_rows[2 * rank_row + 1] = loc_B_erow;
        MPI_Allgather(MPI_IN_PLACE, 2, MPI_INT, proc_B_rows, 2, MPI_INT, engine->comm_col);
        int *a2a_B_sendcnts = (int *) malloc(sizeof(int) * np_row);
        int *a2a_B_recvcnts = (int *) malloc(sizeof(int) * np_row);
        int *a2a_B_sdispls  = (int *) malloc(sizeof(int) * np_row);
        int *a2a_B_rdispls  = (int *) malloc(sizeof(int) * np_row);
        memset(a2a_B_sendcnts, 0, sizeof(int) * np_row);
        memset(a2a_B_recvcnts, 0, sizeof(int) * np_row);
        int is_intersect, is, ie;
        for (int i = 0; i < np_row; i++)
        {
            // Note: intersection segment is [is, ie] not [is, ie)
            int proc_i_req_srow = proc_B_rows[2 * i];
            int proc_i_req_erow = proc_B_rows[2 * i + 1];
            calc_seg_intersection(
                B_rd_srow, B_rd_erow - 1, proc_i_req_srow, proc_i_req_erow - 1, 
                &is_intersect, &is, &ie
            );
            if (is_intersect)
            {
                a2a_B_sdispls[i]  = (is - B_rd_srow) * loc_B_ncol;
                a2a_B_sendcnts[i] = (ie - is + 1) * loc_B_ncol;
            }
            int proc_i_src_srow = B_rd_row_displs[i];
            int proc_i_src_erow = B_rd_row_displs[i + 1];
            calc_seg_intersection(
                proc_i_src_srow, proc_i_src_erow - 1, loc_B_srow, loc_B_erow - 1, 
                &is_intersect, &is, &ie
            );
            if (is_intersect)
            {
                a2a_B_rdispls[i]  = (is - loc_B_srow) * loc_B_ncol;
                a2a_B_recvcnts[i] = (ie - is + 1) * loc_B_ncol;
            }
        }
        engine->a2a_B_sendcnts = a2a_B_sendcnts;
        engine->a2a_B_recvcnts = a2a_B_recvcnts;
        engine->a2a_B_sdispls  = a2a_B_sdispls;
        engine->a2a_B_rdispls  = a2a_B_rdispls;
        free(proc_B_rows);
    } else {  // else of "if (engine->a2a_B_finegrain == 0)"
        int *loc_A_colidx = (int *) malloc(sizeof(int) * loc_A_nnz);
        void *rd_Ai_workbuf = malloc(rd_Ai_workbuf_bytes);
        int *flag = (int *) malloc(sizeof(int) * loc_B_nrow);
        int *agv_colidx_ptr = loc_A_colidx + agv_A_displs[rank_col];
        mat_redist_engine_attach_workbuf(rd_Ai, rd_Ai_workbuf, NULL);
        mat_redist_engine_exec(engine->rd_Ai, src_A_colidx, loc_A_nnz, agv_colidx_ptr, loc_A_nnz);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, loc_A_colidx, agv_A_recvcnts, agv_A_displs, MPI_INT, engine->comm_row);
        for (int i = 0; i < loc_A_nnz; i++) loc_A_colidx[i] -= loc_B_srow;
        int *a2a_B_recvcnts  = (int *) malloc(sizeof(int) * np_row);
        int *a2a_B_sendcnts  = (int *) malloc(sizeof(int) * np_row);
        int *a2a_B_rdispls   = (int *) malloc(sizeof(int) * (np_row + 1));
        int *a2a_B_sdispls   = (int *) malloc(sizeof(int) * (np_row + 1));
        int *a2a_B_recv_ridx = (int *) malloc(sizeof(int) * loc_B_nrow);
        int *a2a_B_send_ridx = (int *) malloc(sizeof(int) * loc_B_nrow);
        int B_recv_row_cnt = 0, B_recv_src_rank = 0;
        memset(flag, 0, sizeof(int) * loc_B_nrow);
        memset(a2a_B_recvcnts, 0, sizeof(int) * np_row);
        for (int i = 0; i < loc_A_nnz; i++) flag[loc_A_colidx[i]] = 1;
        for (int i = 0; i < loc_B_nrow; i++)
        {
            if (flag[i] == 0) continue; 
            int B_row_idx = i + loc_B_srow;
            a2a_B_recv_ridx[B_recv_row_cnt++] = B_row_idx;
            while (B_rd_row_displs[B_recv_src_rank + 1] <= B_row_idx) B_recv_src_rank++;
            a2a_B_recvcnts[B_recv_src_rank]++;
        }
        MPI_Alltoall(a2a_B_recvcnts, 1, MPI_INT, a2a_B_sendcnts, 1, MPI_INT, engine->comm_col);
        // At this moment, a2a_B_{send, recv}cnts are the number of rows, not the number of elements
        a2a_B_rdispls[0] = 0;
        a2a_B_sdispls[0] = 0;
        for (int i = 0; i < np_row; i++) 
        {
            a2a_B_rdispls[i + 1] = a2a_B_rdispls[i] + a2a_B_recvcnts[i];
            a2a_B_sdispls[i + 1] = a2a_B_sdispls[i] + a2a_B_sendcnts[i];
        }
        MPI_Alltoallv(
            a2a_B_recv_ridx, a2a_B_recvcnts, a2a_B_rdispls, MPI_INT,
            a2a_B_send_ridx, a2a_B_sendcnts, a2a_B_sdispls, MPI_INT, engine->comm_col
        );
        // Now we need to make a2a_B_{send, recv}cnts to be the number of elements
        for (int i = 0; i < np_row; i++)
        {
            a2a_B_recvcnts[i] *= loc_B_ncol;
            a2a_B_sendcnts[i] *= loc_B_ncol;
            a2a_B_rdispls[i + 1] = a2a_B_rdispls[i] + a2a_B_recvcnts[i];
            a2a_B_sdispls[i + 1] = a2a_B_sdispls[i] + a2a_B_sendcnts[i];
        }
        engine->a2a_B_sendcnts  = a2a_B_sendcnts;
        engine->a2a_B_recvcnts  = a2a_B_recvcnts;
        engine->a2a_B_sdispls   = a2a_B_sdispls;
        engine->a2a_B_rdispls   = a2a_B_rdispls;
        engine->a2a_B_send_ridx = a2a_B_send_ridx;
        engine->a2a_B_recv_ridx = a2a_B_recv_ridx;
        free(flag);
        free(rd_Ai_workbuf);
        free(loc_A_colidx);
    }  // End of "if (engine->a2a_B_finegrain == 0)"
    
    
    // 7. Calculate the redistribution of output C
    mat_redist_engine_p rd_C;
    size_t rd_C_workbuf_bytes = 0;
    mat_redist_engine_init(
        loc_A_srow, loc_B_scol, loc_A_nrow, loc_B_ncol,
        dst_C_srow, dst_C_scol, dst_C_nrow, dst_C_ncol,
        comm, MPI_DOUBLE, sizeof(double), DEV_TYPE_HOST,
        &rd_C, &rd_C_workbuf_bytes
    );
    engine->rd_C = rd_C;

    // 8. Allocate and attach work buffer if needed
    size_t rd_workbuf_bytes = 0;
    if (rd_Ai_workbuf_bytes > rd_workbuf_bytes) rd_workbuf_bytes = rd_Ai_workbuf_bytes;
    if (rd_Av_workbuf_bytes > rd_workbuf_bytes) rd_workbuf_bytes = rd_Av_workbuf_bytes;
    if (rd_B_workbuf_bytes  > rd_workbuf_bytes) rd_workbuf_bytes = rd_B_workbuf_bytes;
    if (rd_C_workbuf_bytes  > rd_workbuf_bytes) rd_workbuf_bytes = rd_C_workbuf_bytes;
    rd_workbuf_bytes = (rd_workbuf_bytes / sizeof(double) + 1) * sizeof(double);
    size_t self_workbuf_bytes = 0;
    size_t rd_B_size  = (size_t) (B_rd_erow - B_rd_srow) * (size_t) loc_B_ncol;
    size_t loc_B_size = (size_t) loc_B_nrow * (size_t) loc_B_ncol;
    size_t loc_C_size = (size_t) loc_A_nrow * (size_t) loc_B_ncol;
    self_workbuf_bytes += sizeof(double) * (rd_B_size + loc_B_size + loc_C_size);  // rd_B, loc_B, loc_C
    self_workbuf_bytes += (sizeof(double) + sizeof(int)) * (size_t) loc_A_nnz;  // loc_A_colidx, loc_A_val
    if (engine->a2a_B_finegrain == 1)
    {
        size_t B_sendrecv_buf_size = (size_t) (engine->a2a_B_rdispls[np_row] + engine->a2a_B_sdispls[np_row]);
        self_workbuf_bytes += sizeof(double) * B_sendrecv_buf_size;
    }
    self_workbuf_bytes = (self_workbuf_bytes / sizeof(double) + 1) * sizeof(double);
    size_t total_workbuf_bytes = rd_workbuf_bytes + self_workbuf_bytes;
    engine->rd_workbuf_bytes = rd_workbuf_bytes;
    engine->self_workbuf_bytes = self_workbuf_bytes;
    if (workbuf_bytes != NULL)
    {
        engine->alloc_workbuf = 0;
        *workbuf_bytes = total_workbuf_bytes;
    } else {
        engine->alloc_workbuf = 1;
        double *workbuf = (double *) malloc(total_workbuf_bytes);
        if (workbuf == NULL)
        {
            ERROR_PRINTF("Allocate workbuf failed\n");
            crpspmm_engine_free(&engine);
            return;
        }
        crpspmm_engine_attach_workbuf(engine, workbuf);
    }

    // 9. Calculate communication sizes
    engine->nelem_A_rd   = (size_t) rd_A_nnz;
    engine->nelem_A_agv  = (size_t) loc_A_nnz;
    engine->nelem_B_rd   = rd_B_size;
    engine->nelem_B_a2av = (size_t) engine->a2a_B_rdispls[np_row];
    if (engine->a2a_B_finegrain == 0) engine->nelem_B_a2av = loc_B_size;
    engine->nelem_B_a2av_min = 0;  // Needs to be calculated in crpspmm_engine_exec()
    if (np_col == 1) engine->nelem_A_agv = 0;
    if (np_row == 1) engine->nelem_B_a2av = 0;

    // 10. Free work arrays and return
    free(proc_A_nrow);
    free(proc_A_rdispls);
    free(A_rowptr_glb);
    free(A_cidx_se_glb);
    free(m_split_idx);
    free(m_split_idx2);
    free(fac);
    free(B_rd_row_displs);
    free(B_rd_row_blksizes);

    double et = MPI_Wtime();
    engine->t_init = et - st;
    *engine_ = engine;
}

// Attach an external work buffer for crpspmm_engine
void crpspmm_engine_attach_workbuf(crpspmm_engine_p engine, double *workbuf)
{
    engine->workbuf = workbuf;

    size_t rd_workbuf_bytes = engine->rd_workbuf_bytes;
    mat_redist_engine_attach_workbuf(engine->rd_Ai, workbuf, NULL);
    mat_redist_engine_attach_workbuf(engine->rd_Av, workbuf, NULL);
    mat_redist_engine_attach_workbuf(engine->rd_B,  workbuf, NULL);
    mat_redist_engine_attach_workbuf(engine->rd_C,  workbuf, NULL);

    size_t loc_B_ncol = (size_t) engine->loc_B_ncol;
    double *self_workbuf = workbuf + (rd_workbuf_bytes / sizeof(double));
    size_t rd_B_nrow  = engine->rd_B_erow - engine->rd_B_srow;
    size_t rd_B_size  = rd_B_nrow * loc_B_ncol;
    size_t loc_B_size = (size_t) engine->loc_B_nrow * loc_B_ncol;
    size_t loc_C_size = (size_t) engine->loc_A_nrow * loc_B_ncol;
    size_t B_rbuf_size = 0, B_sbuf_size = 0;
    if (engine->a2a_B_finegrain)
    {
        int np_row = engine->np_row;
        B_rbuf_size = (size_t) engine->a2a_B_rdispls[np_row];
        B_sbuf_size = (size_t) engine->a2a_B_sdispls[np_row];
    }
    engine->red_B = self_workbuf;
    self_workbuf += rd_B_size;

    engine->loc_B = self_workbuf;
    self_workbuf += loc_B_size;

    engine->loc_C = self_workbuf;
    self_workbuf += loc_C_size;

    engine->loc_A_val = self_workbuf;
    self_workbuf += engine->loc_A_nnz;

    if (engine->a2a_B_finegrain)
    {
        engine->a2a_B_sbuf = self_workbuf;
        self_workbuf += B_sbuf_size;

        engine->a2a_B_rbuf = self_workbuf;
        self_workbuf += B_rbuf_size;
    }

    engine->loc_A_colidx = (int *) self_workbuf;
}

void crpspmm_engine_exec(
    crpspmm_engine_p engine, 
    const int *src_A_rowptr, const int *src_A_colidx, const double *src_A_val,
    const double *src_B, const int ldB, double *dst_C, const int ldC
)
{
    if (engine == NULL) return;

    int rank_col         = engine->rank_col;
    int np_row           = engine->np_row;
    int np_col           = engine->np_col;
    int loc_A_nrow       = engine->loc_A_nrow;
    int loc_B_srow       = engine->loc_B_srow;
    int loc_B_nrow       = engine->loc_B_nrow;
    int loc_B_ncol       = engine->loc_B_ncol;
    int loc_A_nnz        = engine->loc_A_nnz;
    int *agv_A_recvcnts  = engine->agv_A_recvcnts;
    int *agv_A_displs    = engine->agv_A_displs;
    int *a2a_B_sendcnts  = engine->a2a_B_sendcnts;
    int *a2a_B_sdispls   = engine->a2a_B_sdispls;
    int *a2a_B_recvcnts  = engine->a2a_B_recvcnts;
    int *a2a_B_rdispls   = engine->a2a_B_rdispls;
    int *a2a_B_send_ridx = engine->a2a_B_send_ridx;
    int *a2a_B_recv_ridx = engine->a2a_B_recv_ridx;
    int *loc_A_rowptr    = engine->loc_A_rowptr;
    int *loc_A_colidx    = engine->loc_A_colidx;
    double *loc_A_val    = engine->loc_A_val;
    double *a2a_B_sbuf   = engine->a2a_B_sbuf;
    double *a2a_B_rbuf   = engine->a2a_B_rbuf;
    double *red_B        = engine->red_B;
    double *loc_B        = engine->loc_B;
    double *loc_C        = engine->loc_C;

    double st, et, st0, et0;
    engine->n_exec++;
    st0 = MPI_Wtime();

    // 1. Redistribute and allgatherv A
    int ldA = loc_A_nnz;   // This is actually meaningless, since colidx and val are 1D row vector
    int *agv_colidx_ptr = loc_A_colidx + agv_A_displs[rank_col];
    double *agv_val_ptr = loc_A_val + agv_A_displs[rank_col];
    st = MPI_Wtime();
    mat_redist_engine_exec(engine->rd_Ai, src_A_colidx, ldA, agv_colidx_ptr, ldA);
    mat_redist_engine_exec(engine->rd_Av, src_A_val,    ldA, agv_val_ptr,    ldA);
    et = MPI_Wtime();
    engine->t_rd_A += (et - st);
    st = MPI_Wtime();
    if (np_col > 1)
    {
        MPI_Allgatherv(
            MPI_IN_PLACE, agv_A_displs[np_col], MPI_INT, 
            loc_A_colidx, agv_A_recvcnts, agv_A_displs, MPI_INT, engine->comm_row
        );
        MPI_Allgatherv(
            MPI_IN_PLACE, agv_A_displs[np_col], MPI_DOUBLE, 
            loc_A_val, agv_A_recvcnts, agv_A_displs, MPI_DOUBLE, engine->comm_row
        );
    }
    // Shift the column indices in loc_A_colidx
    for (int i = 0; i < loc_A_nnz; i++) loc_A_colidx[i] -= loc_B_srow;
    et = MPI_Wtime();
    engine->t_agv_A   += (et - st);
    engine->t_exec_nr += (et - st);

    // 1.5 Use loc_A_colidx to compute B_min_comm_size
    if (engine->nelem_B_a2av_min == 0)
    {
        double st2 = MPI_Wtime();
        int recvcnt = 0;
        int *flag = (int *) malloc(sizeof(int) * loc_B_nrow);
        // Count the number of B rows this process needs to receive
        memset(flag, 0, sizeof(int) * loc_B_nrow);
        for (int i = 0; i < loc_A_nnz; i++) flag[loc_A_colidx[i]] = 1;
        for (int i = 0; i < loc_B_nrow; i++) recvcnt += flag[i];
        engine->nelem_B_a2av_min = (size_t) recvcnt * (size_t) loc_B_ncol;
        free(flag);
        double et2 = MPI_Wtime();
        st0 += et2 - st2;
    }

    // 2. Redistribute B and replicate B rows
    st = MPI_Wtime();
    mat_redist_engine_exec(engine->rd_B, src_B, ldB, red_B, loc_B_ncol);
    et = MPI_Wtime();
    engine->t_rd_B += (et - st);
    st = MPI_Wtime();
    if (engine->a2a_B_finegrain == 0)
    {
        if (np_row > 1)
        {
            MPI_Alltoallv(
                red_B, a2a_B_sendcnts, a2a_B_sdispls, MPI_DOUBLE,
                loc_B, a2a_B_recvcnts, a2a_B_rdispls, MPI_DOUBLE, engine->comm_col
            );
        } else {
            loc_B = red_B;
        }
    } else {
        int B_a2a_nrow_send = a2a_B_sdispls[np_row] / loc_B_ncol;
        int B_a2a_nrow_recv = a2a_B_rdispls[np_row] / loc_B_ncol;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < B_a2a_nrow_send; i++)
        {
            double *src_ptr = red_B + (a2a_B_send_ridx[i] - engine->rd_B_srow) * loc_B_ncol;
            double *dst_ptr = a2a_B_sbuf + i * loc_B_ncol;
            memcpy(dst_ptr, src_ptr, sizeof(double) * loc_B_ncol);
        }
        MPI_Alltoallv(
            a2a_B_sbuf, a2a_B_sendcnts, a2a_B_sdispls, MPI_DOUBLE,
            a2a_B_rbuf, a2a_B_recvcnts, a2a_B_rdispls, MPI_DOUBLE, engine->comm_col
        );
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < B_a2a_nrow_recv; i++)
        {
            double *src_ptr = a2a_B_rbuf + i * loc_B_ncol;
            double *dst_ptr = loc_B + (a2a_B_recv_ridx[i] - loc_B_srow) * loc_B_ncol;
            memcpy(dst_ptr, src_ptr, sizeof(double) * loc_B_ncol);
        }
    }
    et = MPI_Wtime();
    engine->t_a2a_B   += (et - st);
    engine->t_exec_nr += (et - st);

    // 3. Local SpMM computation
    st = MPI_Wtime();
    if (engine->use_CUDA == 0)
    {
        #ifdef USE_MKL
        sparse_matrix_t mkl_spA;
        struct matrix_descr mkl_descA;
        mkl_descA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_descA.mode = SPARSE_FILL_MODE_FULL;
        mkl_descA.diag = SPARSE_DIAG_NON_UNIT;
        mkl_sparse_d_create_csr(
            &mkl_spA, SPARSE_INDEX_BASE_ZERO, loc_A_nrow, loc_B_nrow, 
            loc_A_rowptr, loc_A_rowptr + 1, loc_A_colidx, loc_A_val
        );
        double alpha = 1.0, beta = 0.0;
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, mkl_spA, mkl_descA, 
            SPARSE_LAYOUT_ROW_MAJOR, loc_B, loc_B_ncol, loc_B_ncol, beta, loc_C, loc_B_ncol
        );
        mkl_sparse_destroy(mkl_spA);
        #endif
    }
    #ifdef USE_CUDA
    if (engine->use_CUDA == 1)
    {
        cuda_cusparse_csr_spmm(
            loc_A_nrow, loc_B_ncol, loc_B_nrow, 1.0, 
            loc_A_nnz, loc_A_rowptr, loc_A_colidx, loc_A_val,
            loc_B, loc_B_ncol, 0.0, loc_C, loc_B_ncol
        );
    }
    #endif
    et = MPI_Wtime();
    engine->t_spmm    += (et - st);
    engine->t_exec_nr += (et - st);

    // 4. Redistribute C
    st = MPI_Wtime();
    mat_redist_engine_exec(engine->rd_C, loc_C, loc_B_ncol, dst_C, ldC);
    et = MPI_Wtime();
    engine->t_rd_C += (et - st);

    et0 = MPI_Wtime();
    engine->t_exec += (et0 - st0);
}

// Free a crpspmm_engine_s
void crpspmm_engine_free(crpspmm_engine_p *engine_)
{
    crpspmm_engine_p engine = *engine_;
    if (engine == NULL) return;
    free(engine->agv_A_recvcnts);
    free(engine->agv_A_displs);
    free(engine->a2a_B_sendcnts);
    free(engine->a2a_B_sdispls);
    free(engine->a2a_B_recvcnts);
    free(engine->a2a_B_rdispls);
    free(engine->a2a_B_send_ridx);
    free(engine->a2a_B_recv_ridx);
    if (engine->alloc_workbuf) free(engine->workbuf);
    MPI_Comm_free(&engine->comm_row);
    MPI_Comm_free(&engine->comm_col);
    mat_redist_engine_free(&engine->rd_Ai);
    mat_redist_engine_free(&engine->rd_Av);
    mat_redist_engine_free(&engine->rd_B);
    mat_redist_engine_free(&engine->rd_C);
    free(engine);
    *engine_ = NULL;
}

void crpspmm_engine_print_stat(crpspmm_engine_p engine)
{
    if (engine == NULL) return;
    if (engine->rank_glb == 0) printf("crpspmm_engine init time: %.3f s\n", engine->t_init);
    int n_exec = engine->n_exec;
    if (n_exec == 0) return;
    double t_raw[8], t_min[8], t_max[8], t_avg[8];
    size_t cs_raw[5], cs_min[5], cs_max[5], cs_sum[5];
    t_raw[0]  = engine->t_rd_A;
    t_raw[1]  = engine->t_rd_B;
    t_raw[2]  = engine->t_agv_A;
    t_raw[3]  = engine->t_a2a_B;
    t_raw[4]  = engine->t_spmm;
    t_raw[5]  = engine->t_exec_nr;
    t_raw[6]  = engine->t_rd_C;
    t_raw[7]  = engine->t_exec;
    cs_raw[0] = engine->nelem_A_rd;
    cs_raw[1] = engine->nelem_A_agv;
    cs_raw[2] = engine->nelem_B_rd;
    cs_raw[3] = engine->nelem_B_a2av;
    cs_raw[4] = engine->nelem_B_a2av_min;
    MPI_Reduce(&t_raw[0], &t_min[0], 8, MPI_DOUBLE, MPI_MIN, 0, engine->comm_glb);
    MPI_Reduce(&t_raw[0], &t_max[0], 8, MPI_DOUBLE, MPI_MAX, 0, engine->comm_glb);
    MPI_Reduce(&t_raw[0], &t_avg[0], 8, MPI_DOUBLE, MPI_SUM, 0, engine->comm_glb);
    MPI_Reduce(&cs_raw[0], &cs_min[0], 5, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, engine->comm_glb);
    MPI_Reduce(&cs_raw[0], &cs_max[0], 5, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, engine->comm_glb);
    MPI_Reduce(&cs_raw[0], &cs_sum[0], 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, engine->comm_glb);
    for (int i = 0; i < 8; i++)
    {
        t_min[i] /= (double) n_exec;
        t_max[i] /= (double) n_exec;
        t_avg[i] /= (double) (engine->np_glb * n_exec);
    }
    if (engine->rank_glb == 0)
    {
        printf("-------------------------- Runtime (s) -------------------------\n");
        printf("                                   min         avg         max\n");
        printf("Redist A to internal 1D layout  %6.3f      %6.3f      %6.3f\n", t_min[0], t_avg[0], t_max[0]);
        printf("Redist B to internal 2D layout  %6.3f      %6.3f      %6.3f\n", t_min[1], t_avg[1], t_max[1]);
        printf("Replicate A with allgatherv     %6.3f      %6.3f      %6.3f\n", t_min[2], t_avg[2], t_max[2]);
        printf("Replicate B with alltoallv      %6.3f      %6.3f      %6.3f\n", t_min[3], t_avg[3], t_max[3]);
        printf("Local SpMM                      %6.3f      %6.3f      %6.3f\n", t_min[4], t_avg[4], t_max[4]);
        printf("SpMM w/o Redist                 %6.3f      %6.3f      %6.3f\n", t_min[5], t_avg[5], t_max[5]);
        printf("Redist C to user's 2D layout    %6.3f      %6.3f      %6.3f\n", t_min[6], t_avg[6], t_max[6]);
        printf("SpMM total (avg of %3d runs)    %6.3f      %6.3f      %6.3f\n", n_exec, t_min[7], t_avg[7], t_max[7]);
        printf("----------------------------------------------------------------\n");
        printf("------------------ Communicated Matrix Elements -----------------\n");
        printf("                               min           max            sum\n");
        printf("Redist A                %10zu    %10zu    %11zu\n", cs_min[0], cs_max[0], cs_sum[0]);
        printf("Allgatherv A            %10zu    %10zu    %11zu\n", cs_min[1], cs_max[1], cs_sum[1]);
        printf("Redist B                %10zu    %10zu    %11zu\n", cs_min[2], cs_max[2], cs_sum[2]);
        printf("Alltoallv B             %10zu    %10zu    %11zu\n", cs_min[3], cs_max[3], cs_sum[3]);
        printf("Alltoallv B necessary   %10zu    %10zu    %11zu\n", cs_min[4], cs_max[4], cs_sum[4]);
        printf("----------------------------------------------------------------\n");
        printf("\n");
        fflush(stdout);
    }
}

void crpspmm_engine_clear_stat(crpspmm_engine_p engine)
{
    if (engine == NULL) return;
    engine->n_exec    = 0;
    engine->t_exec    = 0.0;
    engine->t_rd_A    = 0.0;
    engine->t_agv_A   = 0.0;
    engine->t_rd_B    = 0.0;
    engine->t_a2a_B   = 0.0;
    engine->t_spmm    = 0.0;
    engine->t_rd_C    = 0.0;
    engine->t_exec_nr = 0.0;
}
