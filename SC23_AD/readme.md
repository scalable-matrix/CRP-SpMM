This file descripts the operations for reproducing all results in "Section V. Numerical Experiments" of the SC23 paper "Communication-Reduced Sparse-Dense Matrix Multiplication with Adaptive Parallelization". The results in the paper were obtained on the Georgia Tech PACE-Phoenix cluster (see [https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/](https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/) for detailed information of this cluster).

## Preparation

This directory contains two children folders: `job_scripts` and `figures`. Directory `job_scripts` contains all SLURM and shell scripts for reproducing all test results. Directory `figures` contains all MATLAB scripts for plotting the figures. All these MATLAB scripts work with MATLAB R2020a.

Create a working directory, for example, `$HOME/SpMM-tests`. Copy files in this directory to `$WORKDIR` and create a `mats` directory for mtx files:
```shell
export WORKDIR=$HOME/SpMM-tests
mkdir -p $WORKDIR/mats
cp -r CRP-SpMM/SC23_AD/figures        $WORKDIR/
cp    CRP-SpMM/SC23_AD/job_scripts/*  $WORKDIR/
```

Here are the download links for the mtx files we will use and the file name in `$WORKDIR/mats`:
* com-Orkut: [https://portal.nersc.gov/project/m1982/GNN/unused/com-Orkut-permuted.mtx](https://portal.nersc.gov/project/m1982/GNN/unused/com-Orkut-permuted.mtx), renamed as `$WORKDIR/mats/com-Orkut.mtx`;
* nm7: [https://portal.nersc.gov/project/m1982/GNN/unused/Nm7.mtx](https://portal.nersc.gov/project/m1982/GNN/unused/Nm7.mtx), renamed as `$WORKDIR/mats/nm7.mtx`;
* cage15: [https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage15.tar.gz](https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage15.tar.gz), extracted and moved to `$WORKDIR/mats/cage15.mtx`;
* Amazon: [https://portal.nersc.gov/project/m1982/GNN/amazon_large_randomized.mtx](https://portal.nersc.gov/project/m1982/GNN/amazon_large_randomized.mtx), renamed as `$WORKDIR/mats/amazon.mtx`.


## Figure 2

First, read the mtx files in MATLAB. Multiple approaches are available, for example, using the ``mmread.m`` provided by the NIST: [https://math.nist.gov/MatrixMarket/mmio/matlab/mmiomatlab.html](https://math.nist.gov/MatrixMarket/mmio/matlab/mmiomatlab.html) (note: this is extremely slow when reading large matrices). Then, use the function `plot_block_sparsity(A, bs)` provided in `figures/plot_block_sparsity.m` to plot the figures in Figure 2. The block sizes used for each matrix are listed below:
* com-Orkut: 8192;
* nm7: 8192;
* cage15: 16384;
* Amazon: 32768.

## Figure 3

First, compile CRP-SpMM and copy the example test file to `$WORKDIR`:
```shell
cp CRP-SpMM/examples/test_crpspmm.exe $WORKDIR/crpspmm-cpu
```

Then, clone and compile the CombBLAS library modified branch for numerical experiments used in this paper. We use Intel C compiler, Intel MKL, and MVAPICH2 for compiling CombBLAS.
```shell
# Clone and check out test branch
cd $WORKDIR
git clone https://github.com/huanghua1994/CombBLAS-SpMM-test.git
cd CombBLAS-SpMM-test
git checkout combblas-gpu

# Compile
mkdir build && cd build
CC=mpicc CXX=mpicxx cmake .. -DCMAKE_INSTALL_PREFIX=./install
make install

# Copy the CombBLAS test executable file to $WORKDIR
cp ReleaseTests/SpMM-cpu $WORKDIR/combblas-spmm-cpu
```

Modify `$WORKDIR/fig3.pbs` based on the configuration of your cluster. Modify the number of nodes required in this PBS file and submit multiple times to get all results.

Test results should be copied from the script output and fill in the file `$WORKDIR/figures/plot_all_scaling.m`. 

For each CRP-SpMM running output, the line starting with "SpMM total (avg of  10 runs)" contains the result we need for `{amazon, orkut, nm7, cage15}_crp` arrays. The last numerical value in this line (the "max" column) is the value we need.

For CombBLAS, the output of 1.5D stationary-A (sA-1.5D) / 2D stationary-A (sA-2D) / 2D stationary-C (sC-2D) algorithms starts with "Algorithm: stationary-A 1.5D" / "Algorithm: stationary-A   2D" / "Algorithm: stationary-C 1.5D". In the running output, the line starting with "total time" contains the result we need. The last numerical value in this line (the "MAX" column) is the value we need. The values should be copied to `{amazon, orkut, nm7, cage15}_comb_{1da, 2da, 2dc}` for sA-1.5D, sA-2D, and sC-2D, respectively.

## Figure 4

Modify the script `$WORKDIR/fig4.pbs` based on the configuration of your cluster. Then submit it once. 

Test results should be copied from the script output and fill in the file `$WORKDIR/figures/plot_runtime_breakdown.m`. 

For CRP-SpMM results, the 1st/2nd/3rd/4th elements in arrays `crp_{amazon, orkut, nm7, cage15}` corresponds to the 2nd numerical value (the "avg" column) values in the line starting with:
* 1st element: "Replicate A";
* 2nd element: "Replicate B";
* 3rd element: "Local SpMM";
* 4th element: "Redist A" + "Redist B" + "Redist C".

For CombBLAS, the 1st/2nd/3rd elements in arrays `comb_{amazon, orkut, nm7, cage15}` corresponds to the 2nd numerical value (the "AVG" column) in the line starting with:
* 1st element: "sC-comm-bcastA";
* 2nd element: "sC-comm-bcastX";
* 3rd element: "local comp".

## Figure 5

Reuse the script `$WORKDIR/fig3.pbs`. Modify the `N` variable and the number of nodes required in this PBS file and submit multiple times to get all results. 

Test results should be copied from the script output and fill in the file `$WORKDIR/figures/plot_vary_n.m` in the same way as Figure 3. 

## Figure 6

First, read the cage15 file in MATLAB. Assuming that the matrix is stored in `A`. Then use the following MATLAB commands to obtain Figure 6:
```matlab
p = symrcm(A);
A2 = A(p, p);
plot_block_sparsity(A2, 16384);
```

Then, store `A2` to `$WORKDIR/mats/reordered-cage15.mtx`. You can use the `mmwrite.m` provided by NIST (again it is very slow). We will use this mtx file in Figure 7 and Table 3. 

## Figure 7

Modify the script `$WORKDIR/fig7.pbs` based on the configuration of your cluster. Modify the `N` variable in this PBS file and submit multiple times to get all results. 

Test results should be copied from the script output and fill in the file `$WORKDIR/figures/plot_vary_n2.m`. 

For the original and reordered cage15 matrix, the results should be filled in arrays with prefix `cage15_` and `cage15cm_`, respectively. 

The runtime values are obtained in the same way as Figure 3. They should be filled in arrays with surfix `_time`.

The process grid sizes are in the line starting with "CRP-SpMM 2D partition". The printed grid sizes are `p_m * p_n`. The second value should be filled in arrays with suffix `_pn`.

## Table 3

Modify the script `$WORKDIR/table3.pbs` based on the configuration of your cluster. Then submit it once. 

For CRP-SpMM:
* "Grid Size" are in the line starting with "CRP-SpMM 2D partition", the printed grid sizes are `p_m * p_n`;
* "# A elements" are in the line starting with "Allgatherv A";
* "# B elements actual" are in the line starting with "Alltoallv B";
* "# B elements minimal" are in the line starting with "Alltoallv B necessary".

For "# A elements", "# B elements actual", and "# B elements minimal", the 3rd integer value (the "sum" column) in the output line is the value we need. 

For CombBLAS sc-2D, two rows after the row "Number of matrix elements communicated (MIN, MAX, SUM)" contain the values for "# A elements" and "# B elements", respectively. The 3rd integer value in each line (the "SUM" column) is the value we need. 

