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

// #if defined(USE_MKL)
// #include <mkl.h>
// #endif
// #if defined(USE_LAPACK)
// #include <lapacke.h>
// #include <cblas.h>
// #endif

// #ifdef USE_CUBLAS
// #include <cuda_runtime.h>
// #include "cublas_v2.h"
// #include "cusparse.h"
// #endif

#include <omp.h>
#include "../inc/util.h"
#include "../inc/matrix_ops_cpu.h"

void transpose(double *src, double *dst, const int N, const int M)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

void inverse(double *arr, int m, int n)
{   
    /**************
    input: arr[m*n] in row major format.
    **************/

    int lda_t = m;
    int lda = n;
    int info;
    int lwork = -1;
    double* work = NULL;
    double work_query;
    int *ipiv = new int[n+1]();

    double *arr_t = new double[m*n]();
    transpose(arr, arr_t, n, m);
    dgetrf_( &n, &m, arr_t, &lda_t, ipiv, &info );
    if(info < 0)
    {
       cout << "dgetrf_: Transpose error!!" << endl;
       exit(1);
    }
   //transpose(arr_t, arr, m, n);
   //LAPACKE_dgetri(LAPACK_ROW_MAJOR, n,arr,n,ipiv);

   /* Query optimal working array(s) size */
   dgetri_( &m, arr_t, &lda_t, ipiv, &work_query, &lwork, &info );
   if(info<0)
   {
       cout<<"dgetri_ 1: Transpose error!!"<<endl;
       //exit(1);
   }
   lwork = (int)work_query;
   //cout<<"lwork: "<<lwork<<endl;
   work = new double[lwork]();
   dgetri_( &m, arr_t, &lda_t, ipiv, work, &lwork, &info );
   if(info<0)
   {
       cout<<"dgetri_ 2: Transpose error!!"<<endl;
       //exit(1);
   }
   transpose(arr_t, arr, m, n);
   delete []arr_t;
   delete []ipiv;
}

void make_identity_mat(double *arr, const int row, const int col)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            if(i == j)
                arr[i * row + j] = 1.00;
            else
                arr[i * row + j] = 0.00;
        }
    }
}

void diag(double *src, double *dst, const int size)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0; i < size ; i++)
    {
        for(j = 0 ; j < size ; j++)
        {
            if(i == j)
            {
                dst[i * size + j] = src[i];
            }
            else
                dst[i * size + j] =0.0;
        }
    }
}

void mat_sub(double *src1, double *src2, double *dst, const int row, const int col)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[i * col + j] = src1[i * col + j] - src2[i * col + j];
        }
    }
}

void mat_addition(double *src1, double *src2, double *dst, const int row, const int col)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[i * col + j] = src1[i * col + j] + src2[i * col + j];
        }
    }
}

void mat_mult(double *src1, double *src2, double *dst, const int row, const int col)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[i * col + j] = src1[i * col + j] * src2[i * col + j];
        }
    }
}

void sum_sqrt(double *src, double *dst, const int row, const int col)
{
    int i, j;
    
    #pragma omp parallel for default(shared) private(j)
    for(i = 0 ; i < col ; i++) //i->col
    {
        for(j = 0 ; j < row ; j++) //j->row
        {
            dst[i] += src[j * col + i];
        }
    }

    #pragma omp parallel for default(shared)
    for(i = 0; i < col ; i++) //i->col
    {
        dst[i] = sqrt(dst[i]);
    }
}

void update_activeMask(int *activeMask, double *residualNorms, double residualTolerance, int blocksize)
{
    int i;
    #pragma omp parallel for
    for(i=0; i<blocksize; i++)
    {
        if((residualNorms[i]>residualTolerance) && activeMask[i]==1)
            activeMask[i]=1;
        else
            activeMask[i]=0;
    }
}

void getActiveBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize)
{
    //activeBlockVectorR -> M*currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize

    int i, j, k=0;
    #pragma omp parallel for firstprivate(k) private(j) default(shared)
    for(i=0; i<M; i++)
    {
        k=0;
        for(j=0; j<blocksize; j++)
        {
             if(activeMask[j] == 1)
             {
                activeBlockVectorR[i*currentBlockSize+k] = blockVectorR[i*blocksize+j];
                k++;
             }
        }
    }
}
void updateBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize)
{
    //activeBlockVectorR -> M*currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize
    int i, j, k = 0;
    #pragma omp parallel for firstprivate(k) private(j) default(shared)
    for(i=0; i<M; i++)
    {
        k=0;
        for(j=0; j<blocksize; j++)
        {
             if(activeMask[j] == 1)
             {
                blockVectorR[i*blocksize+j]= activeBlockVectorR[i*currentBlockSize+k];
                k++;
             }
        }
    }
}
void mat_copy(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst)
{
    int i,j;
    #pragma omp parallel for private(j) default(shared)
    for(i=0; i<row; i++)
    {
        for(j=0; j<col; j++)
        {
            dst[(start_row+i)*ld_dst+(start_col+j)]=src[i*col+j];
        }

    }
}

void custom_dlacpy(double *src, double *dst, int m, int n)
{
    //src[m*n] and dst[m*n]
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < m ; i++) //each row
    {
        for(j = 0 ; j < n ; j++) //each column
        {
            dst[i * n + j] = src[i * n + j];
        }
    }
}

void spmm_csr(int row, int col, int nvec, int *row_ptr, int *col_index, double *value, double *Y, double *result)
{
    int i, j, k, start, end;
    int r, c, xcoef;

    #pragma omp parallel for default(shared) private(start, end, r, c, xcoef, j, k)
    for(i = 0 ; i < row ; i++)
    {
        start = row_ptr[i];
        end = row_ptr[i + 1];
        //printf("%d %d %d\n", i, start, end);
        //printf("col_index[%d]: %d col[%d]: %d\n", start, col_index[start], end, col_index[end]);
        for(j = start ; j < end ; j++)
        {
            r = i;
            c = col_index[j];
            xcoef = value[j];  
            //printf("row: %d col: %d value: %lf\n", r, c, xcoef);
            #pragma omp simd
            for(k = 0 ; k < nvec ; k++)
            {
                result[r * nvec + k] = result[r * nvec + k] + xcoef * Y[c * nvec + k];
            }
        }
    }
}


void spmm_csr_DJA(int row, int col, int nvec, int *row_ptr, double *col_index, 
                  double *value, double *Y, double *result)
{
    int i, j, k, start, end;
    int r, c, xcoef;

    #pragma omp parallel for default(shared) private(start, end, r, c, xcoef, j, k)
    for(i = 0 ; i < row ; i++)
    {
        start = row_ptr[i];
        end = row_ptr[i + 1];
        //printf("%d %d %d\n", i, start, end);
        //printf("col_index[%d]: %d col[%d]: %d\n", start, col_index[start], end, col_index[end]);
        for(j = start ; j < end ; j++)
        {
            r = i;
            c = (int) col_index[j];
            xcoef = value[j];  
            //printf("row: %d col: %d value: %lf\n", r, c, xcoef);
            #pragma omp simd
            for(k = 0 ; k < nvec ; k++)
            {
                result[r * nvec + k] = result[r * nvec + k] + xcoef * Y[c * nvec + k];
            }
        }
    }
}
#endif
