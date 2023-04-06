# CRP-SpMM

Communication-Reduced Parallel SpMM

## 1. Compilation

Clone the CRP-SpMM library from GitHub:
```shell
git clone https://github.com/scalable-matrix/CRP-SpMM.git
```
Enter directory `CRP-SpMM/src`. CRP-SpMM provides the following example makefiles:
* `icc-mkl-impi.make`: Use Intel C compiler, Intel MKL sparse BLAS, and Intel MPI library (the C compiler is `mpiicc`)
* `icc-mkl-anympi.make`: Use Intel C compiler, Intel MKL sparse BLAS, and any MPI library (the C compiler is `mpicc`)

On the Georgia Tech PACE-Phoenix cluster, we use the `icc-mkl-anympi.make`. You may need to modify `common.make` and create your own make file for your cluster.

Run the following command to compile the CRP-SpMM library:
```shell
make -f icc-mkl-anympi.make
```

After compilation, the dynamic and static library files are copied to directory `CRP-SpMM/lib`, and the C header files are copied to directory `CRP-SpMM/include`.

Enter directory `CRP-SpMM/examples` to compile the example program. This directory also contains two example make files as those in `CRP-SpMM/src`. Run the following command to compile the CRP-SpMM library:
```shell
make -f icc-mkl-anympi.make
```

## 2. Example Programs

For single node execution or launch on clusters without a job scheduling system, the following command should work on most platforms (assuming that you are in directory `CRP-SpMM/examples`):
```shell
mpirun -np <nprocs> ./test_crpspmm.exe <mtx-file> <n> <ntest> <chkres>
```
Where:
* `nprocs`: Number of MPI processes
* `<mtx-file>`: Path to the Matrix Market file that stores the sparse matrix A
* `n`: Number of B matrix columns
* `ntest`: Number of tests to run, should be a non-negative integer
* `chkres`: 0 or 1, 0 for skipping result correctness check, 1 for result correctness check

To explicitly control MPI + OpenMP hybrid parallelization, you need to specify OpenMP environment variables, and process affinity environment variables for some MPI libraries. In the paper, we use the following environment variables for MPI + OpenMP parallel tests on the Georgia Tech PACE-Phoenix cluster:
```shell
export OMP_PLACES=cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ $SLURM_CPUS_PER_TASK -gt 1 ]
then
    echo Set MV2 ENV for SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
    export MV2_CPU_BINDING_POLICY=hybrid
    export MV2_THREADS_PER_PROCESS=$SLURM_CPUS_PER_TASK
    export MV2_USE_THREAD_WARNING=0
fi
```
For clusters and supercomputers with job scheduling systems like slurm, you need to write job scripts for launching the example program on multiple nodes.

## 3. Expected Result

The example program prints timing results and other information to the screen output. Here is an example running output on a single node with an Intel Xeon E5-2670 8-core processor using two cores per MPI process (the mtx file pwtk.mtx can be downloaded from the SuiteSparse Matrix Collection, [link](http://sparse.tamu.edu/Boeing/pwtk)):
```shell
$ mpirun -np 4 ./test_crpspmm.exe pwtk.mtx 256 5 1 0
B has 256 columns
Rank 0 read matrix A from file pwtk.mtx used 3.19 s
A size = 217918 * 217918, nnz = 11634424, nnz/row = 53, bandwidth = 189331

1D partition and distribution of A used 0.07 s
CRP-SpMM 2D partition: 1 * 4
1.05
1.06
1.06
1.06
1.06
crpspmm_engine init time: 0.009 s
-------------------------- Runtime (s) -------------------------
                                   min         avg         max
Redist A to internal 1D layout   0.074       0.074       0.074
Replicate A with allgatherv      0.068       0.069       0.069
Redist B to internal 2D layout   0.278       0.280       0.283
Replicate B with alltoallv       0.000       0.000       0.000
Local SpMM                       0.297       0.305       0.309
Redist C to user s 2D layout     0.306       0.311       0.318
SpMM total (avg of   5 runs)     1.059       1.060       1.060
----------------------------------------------------------------
------------------ Communicated Matrix Elements -----------------
                               min           max            sum
Redist A                   2908606       2908606       11634424
Allgatherv A              11634424      11634424       46537696
Redist B                  13946752      13946752       55787008
Alltoallv B                      0             0              0
Alltoallv B necessary     13946752      13946752       55787008
----------------------------------------------------------------

||C_ref - C||_f / ||C_ref||_f = 0.000000e+00
```