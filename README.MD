## Overview
This repo contains the source code of DeepSparse , a novel task-parallel sparse solver framework which adopts a fully integrated task parallel approach. 
Deepsparse targets all computational steps in a sparse solver rather than optimizing only one computational kernel like sparse matrix matrix multiplication(SpMM) or sparse matrix vector multiplication(SpMV).
In comparison with the solvers using optimized library function calls, Deepsparse achieves an improvement in cache misses in two different eigensolver algorithms that we tried - Locally Optimal Blocked Preconditioned 
Conjugate Gradient and Lanczos algorithm. 

Our work has been published in IEEE INTERNATIONAL CONFERENCE ON HIGH PERFORMANCE COMPUTING, DATA, AND ANALYTICS(HiPC 2019). The paper can be found here - https://ieeexplore.ieee.org/abstract/document/8990579

This repo contains the source folder. Each algorithm contains four different versions - 
    1) libcsr - Solver implemented using library function calls and matrix stored in csr format.
    2) LIBCSB - only the matrix multiplication is implemented in a blocked implementation and the matrix is stored in CSB format. 
    3) global - DeepSparse task parallel implementation
    

## WACCPD 2019 Paper
We are porting our DeepSparse framework on GPU using directive-based programming models. Our initial findings have been published in [WACCPD 2019](https://waccpd.org) workshop.

Our WACCPD 2019 Paper can be found here: https://www.cse.msu.edu/~rabbimd/papers/waccpd2019_lobpcg.pdf

WACCPD 2019 Paper's Artifact (source code) is available here: https://github.com/fazlay-rabbi/WACCPD_2019_Artifact

## ICPP 2021 Paper
We evaluate the OpenMP, HPX and Regent runtime systems in terms of performance and ease of implementation, and compare them against the traditional BSP model for two popular eigensolvers, Lanczos and LOBPCG. Note that OpenMP's task parallel model is utilized through DeepSparse's framework in this evaluation.

Our paper can be found here: https://dl.acm.org/doi/abs/10.1145/3472456.3472476

We refer those who want to run our HPX codes to here: https://hpx-docs.stellar-group.org/latest/html/quickstart.html

We refer those who want to run our Regent codes to here: https://regent-lang.org/install
