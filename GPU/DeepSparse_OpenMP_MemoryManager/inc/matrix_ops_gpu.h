#ifndef MATRIX_OPS_GPU_H
#define MATRIX_OPS_GPU_H
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
#include <queue>
using namespace std;

#include <omp.h>
#include "../inc/util.h"
// #include "../inc/matrix_ops_gpu.h"
#include "../inc/memory_manager_v6.h"

#if defined(USE_CUBLAS) || defined(USE_DEVICE)

void update_activeMask_GPU(int *activeMask, int *d_activeMask, double *d_residualNorms, double residualTolerance, int blocksize);

void SPMM_GPU_TILE(int *row_ptr, int *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile);

void XTY_OPENMP_GPU(double *X, double *Y, double *result_buf ,int M, int N, int P, int block_width, int block_id, int buf_id);

void XTY_GPU_RED(double *buf, double *result, int N, int P, int block_width);

void cublasDgemm_xy_block(cublasHandle_t handle, double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id);

void cublasDgemm_xy_block_betaOne(cublasHandle_t handle, double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id);

void mat_sub_GPU_block(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id);
void mat_sub_GPU_block_v1(double *src1, double *src2, int row, int col, int block_width, int block_id);
void mat_sub_GPU_block_v2(double *src1, double *src2, int row, int col, int block_width, int block_id);

void mat_mult_GPU_block(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id);

void sum_sqrt_GPU_COL(double *src, double *dst, int row, int col, int block_width, int block_id, int buf_id, double *buf);

void sum_sqrt_GPU_SQRT(double *dst, int col);



void getActiveBlockVector_GPU_block(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                                    int M, int blocksize, int currentBlockSize, int block_width, int block_id);

void cublasDgemm_xty_tiling(cublasHandle_t handle, double *devPtrA, double *devPtrB, double *devPtrC, 
                            int N, int blocksize, int currentBlockSize, int block_width, int block_id);

void updateBlockVector_GPU_block(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                            int M, int blocksize, int currentBlockSize, int block_width, int block_id);

void custom_dlacpy_GPU_block(double *src, double *dst, int row, int col, int block_width, int block_id);

void custom_dlacpy_task_CPU(double *src, double *dst, int row, int col, int block_width, int block_id);

void transpose_GPU_block(double *src, double *d_dst, 
                            int row, int col, int block_width);

void transpose_GPU(double *src, double *dst, int N, int M);

void transpose_GPU_deviceptr(double *src, double *dst, int N, int M);

void cusparseDcsrmm_tile(int *ia, int *ja, double *acsr, 
                        double *d_activeBlockVectorR, double *activeBlockVectorAR, double *d_activeBlockVectorAR_tile, double *d_temp_actAR,
                        int *rowPtrTile, int *colIndexTile, double *coolValTile, 
                        int *nnz_per_tile, int numrows, int numcols, int nrowblk, int currentBlockSize, int block_width, int block_id, int h, int t,
                        cusparseHandle_t handle, cusparseMatDescr_t descr);

void mat_copy_GPU(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst);

void mat_addition_GPU_block(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id);

// dst += src1
void mat_addition_GPU_block_v1(double *src1, double *dst, int row, int col, int block_width, int block_id);

void XY_OPENMP_GPU_tiled(double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id);

void clear_buffer(double *buf, int row, int col);

void sum_sqrt_task_RNRED(double *buf, double *dst, const int col);

void sum_sqrt_task_SQRT(double *dst, const int col);

// Memory Managed Managed
void XY_GPU_MM(double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, double alpha, double beta,
                                bool A_on_device, bool B_on_device, bool C_on_device);

void mat_sub_GPU_MM(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber,
                    bool src1_deviceptr, bool src2_deviceptr, bool dst_deviceptr);

void mat_mult_GPU_MM(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id, double *d_memory, long iterationNumber);

void sum_sqrt_GPU_COL_MM(double *src, double *dst, int row, int col, int block_width, int block_id, int buf_id, double *buf, double *d_memory, long iterationNumber);

void getActiveBlockVector_GPU_MM(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                                    int M, int blocksize, int currentBlockSize, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, bool actR_deviceptr);

void XTY_GPU_MM(double *X, double *Y, double *result_buf ,int M, int N, int P, int block_width, int block_id, int buf_id, double *d_memory, int task_id, long iterationNumber,
                double alpha, double beta, bool X_deviceptr, bool Y_deviceptr);

void XTY_GPU_RED_MM(double *buf, double *result, int N, int P, int block_width, double *d_memory, long iterationNumber);

void custom_dlacpy_GPU_MM(double *src, double *dst, int row, int col, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, 
                            bool src_devicptr, bool dst_deviceptr);

void SPMM_GPU_MM(int *row_ptr, int *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, long iterationNumber);

void updateBlockVector_GPU_MM(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                                int M, int blocksize, int currentBlockSize, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, bool actR_deviceptr);

void mat_addition_GPU_MM(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber,
                        bool src1_deviceptr, bool src2_deviceptr, bool dst_deviceptr);

void SPMM_GPU_MM_v1(int *row_ptr, int *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, int *ia, int *ja, long iterationNumber);

void SPMM_GPU_MM_v2(int *row_ptr, int *col_index, double *d_xrem, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, int *ia, int *ja, double *acsr, long iterationNumber);

void SPMM_GPU_MM_v3(int *row_ptr, double *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, int *ia, int *ja, long iterationNumber);

void XY_GPU_MM_v1(double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, cublasHandle_t handle, double alpha, double beta);


#endif

#endif
