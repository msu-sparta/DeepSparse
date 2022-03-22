#ifndef MATRIX_OPS_CPU_H
#define MATRIX_OPS_CPU_H

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
using namespace std;

#include "../inc/util.h"


void spmm_csr(int row, int col, int nvec, int *row_ptr, int *col_index, double *value, double *Y, double *result);
void transpose(double *src, double *dst, const int N, const int M);
void inverse(double *arr, int m, int n);
void make_identity_mat(double *arr, const int row, const int col);
void diag(double *src, double *dst, const int size);
void mat_sub(double *src1, double *src2, double *dst, const int row, const int col);
void mat_addition(double *src1, double *src2, double *dst, const int row, const int col);
void mat_mult(double *src1, double *src2, double *dst, const int row, const int col);
void sum_sqrt(double *src, double *dst, const int row, const int col);
void update_activeMask(int *activeMask, double *residualNorms, double residualTolerance, int blocksize);
void getActiveBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void updateBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize);
void mat_copy(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst);
void custom_dlacpy(double *src, double *dst, int m, int n);

void spmm_csr_DJA(int row, int col, int nvec, int *row_ptr, double *col_index, 
                  double *value, double *Y, double *result);
#endif
