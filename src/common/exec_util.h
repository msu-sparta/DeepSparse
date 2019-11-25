#ifndef EXEC_UTIL_H
#define EXEC_UTIL_H
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
using namespace std;

#include <mkl.h>
#include <omp.h>

extern long position;
extern int *colptrs, *irem;
extern int  *ia , *ja;

extern int numcols, numrows, nnonzero, nthrds;
extern int nrows, ncols, nnz;
extern int wblk, nrowblks, ncolblks;

//CSB structure
//template<typename T>


struct block
{
    int nnz;
    int roffset, coffset;
    //unsigned short int *rloc, *cloc;
    int *rloc , *cloc ; 
    double *val;
};


////taskinfo struct
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
} ;



///////libcsb and libcsr functions//////////////////////

void transpose(double *src, double *dst, const int N, const int M);
void inverse(double *arr, int m, int n);
void print_mat(double *arr, const int row, const int col);
void make_identity_mat(double *arr, const int row, const int col);
void diag(double *src, double *dst, const int size);

void sum_sqrt(double *src, double *dst, const int row, const int col);
void update_activeMask(int *activeMask, double *residualNorms, double residualTolerance, int blocksize);
void getActiveBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void updateBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void mat_copy(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst);
void print_eigenvalues( MKL_INT n, double* wr, double* wi );



/////common for all/////////////////////
void read_custom(char* filename, double *&xrem);

void csc2blkcoord(block *&matrixBlock, double *xrem);



void custom_dlacpy(double *src, double *dst, int m, int n);


bool checkEquals( double* a, double* b, size_t outterSize, size_t innerSize);




////////taskinfo related functions///////////
int split(const char *str, char c, char ***arr);
//void buildTaskInfoStruct_main(int nodeCount, char **graph , const char* loopType, int blksize , const char *matrixName);
//void buildTaskInfoStruct(struct TaskInfo *taskInfo, char *partFile);
void reverse(char str[], int length);
//void structToString(struct TaskInfo taskInfo, char *structToStr);




///////////////global version util functions///////////

void print_summary(double **timingStat, int iterationNumber);
void str_rev(char *str);
void myitoa(int x, char* dest);
int buildTaskInfoStruct(struct TaskInfo *&taskInfo, char *partFile);
void structToString(struct TaskInfo taskInfo, char *structToStr);
int readPartBoundary(int *&partBoundary, char *partBoundaryFile);



#endif