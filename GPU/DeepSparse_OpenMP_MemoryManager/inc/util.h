#ifndef UTIL_H
#define UTIL_H

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <fstream>
#include <vector>
using namespace std;

#if defined(USE_MKL)
#include <mkl.h>
#endif

#if defined(USE_LAPACK)
#include <lapacke.h>
#include <cblas.h>
#endif

#ifdef USE_CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

#define DGEMM_RESTRICT __restrict__


#include <omp.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
//#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}

extern long position, maxIterations;
extern int *colptrs, *irem;
extern int  *ia , *ja;
extern double *d_ja;
extern double *acsr;
extern int numcols, numrows, nnonzero, nthrds, blocksize, block_width;
extern int nrows, ncols, nnz;
extern int wblk, nrowblks, ncolblks;

//CSB structure
template<typename T>
struct block
{
	int nnz;
    int roffset, coffset;
    unsigned short int *rloc, *cloc;
    T *val;
};

struct TaskInfo
{
	int opCode;
  	int numParamsCount;
  	int *numParamsList;
  	int strParamsCount;
  	char **strParamsList;
  	int taskID;
  	int partitionNo;
  	int priority;
};

extern struct TaskInfo *taskInfo_nonLoop, *taskInfo_firstLoop, *taskInfo_secondLoop;

extern int numTasks;
extern vector<string> function_name; //{"LOOPS", "X*Y", "Xt*Y", "ADD", "SUB", "MULT", "SPMM", "GET", "UPDATE", "dsygv", "DLACPY", "INVERSE", "TRANSPOSE", "mat_copy", "dpotrf", "memset", "SUMSQRT", "diag"};
extern double ***taskTiming;

void initialize_timing_variables();
void summarize_timing_variables();
int omp_thread_count();
double get_seconds();
int split (const char *str, char c, char ***arr);
void str_rev(char *str);
void myitoa(int x, char* dest);
int buildTaskInfoStruct(struct TaskInfo *&taskInfo, char *partFile);
void structToString(struct TaskInfo taskInfo, char *structToStr);
int readPartBoundary(int *&partBoundary, char *partBoundaryFile);

template<typename T>
void read_custom(char* filename, T *&xrem);

void read_csr(char *filename);
void read_csr_DJA(char *filename);
void read_csr_DJA_2(char* filename);
template<typename T>
void csc2blkcoord(block<T> *&matrixBlock, T *xrem);

void print_mat(double *arr, const int row, const int col);
void print_eigenvalues(int n, double* wr, double* wi);



#endif 
