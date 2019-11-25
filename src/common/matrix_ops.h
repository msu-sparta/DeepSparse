#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

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
#include "exec_util.h"

void spmm_blkcoord_loop(int R, int C, int blocksize, int nthrds, double *X,  double *Y, block *H);
void mat_sub(double *src1, double *src2, double *dst, const int row, const int col);
void mat_addition(double *src1, double *src2, double *dst, const int row, const int col);
void mat_mult(double *src1, double *src2, double *dst, const int row, const int col);

void _XY_exe(double *X, double *Y, double *result ,int M, int N, int P, int block_width, int block_id);

void _XTY_v1_exe(double *X, double *Y, double *buf ,int M, int N, int P, int block_width, int block_id, int buf_id);

void _XTY_v1_RED(double *buf, double *result, int N, int P, int block_width);

void mat_addition_task_exe(double *src1, double *src2, double *dst, const int row,
                       const int col, int block_width, int block_id);

void mat_sub_task_exe(double *src1, double *src2, double *dst, const int row,
                       const int col, int block_width, int block_id);

void mat_mult_task_exe(double *src1, double *src2, double *dst, const int row,
                       const int col, int block_width, int block_id);

void sum_sqrt_task_COL(double *src, double *dst, const int row, const int col, int block_width, int block_id, int buf_id, double *buf);

void sum_sqrt_task_RNRED(double *buf, double *dst, const int col);

void sum_sqrt_task_SQRT(double *dst, const int col);

void update_activeMask_task_exe(int *activeMask, double *residualNorms, double residualTolerance, int blocksize);

void getActiveBlockVector_task_exe(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                               int M, int blocksize, int currentBlockSize, int block_width, int block_id);

void updateBlockVector_task_exe(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                             int M, int blocksize, int currentBlockSize, int block_width, int block_id);

void custom_dlacpy_task_exe(double *src, double *dst, int row, int col, int block_width, int block_id);

void dot_mm_exe(double *src1, double *src2, double *result, const int row, const int col, int block_width, int block_id, int buf_id);



void spmm_blkcoord(int R, int C, int M, int nthrds, double *X,  double *Y, block *H);


void spmm_blkcoord_finegrained_exe_fixed_buf(int R, int C, int M, int nbuf, double *X,  double *Y, block *H, int row_id, int col_id, int buf_id, int block_width);


void spmm_blkcoord_finegrained_SPMMRED_fixed_buf(int R, int M, int nbuf, double *Y, double *spmmBUF, int block_id, int block_width);


void spmm_blkcoord_finegrained_exe_fixed_buf_unrolled32(int R, int C, int M, int nbuf, double *X,  double *Y, block *H, int row_id, int col_id, int buf_id, int block_width);


#endif