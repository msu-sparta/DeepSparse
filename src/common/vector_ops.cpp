#include "vector_ops.h"
#include "exec_util.h"

/* reduce on partial squared norms */
void norm_task(double *Y, double *save_norm, int iterationNumber)
{
    int i;
    
    /* TODO: make sure the buffer count matches
     * the thread count each time */
    #pragma omp task firstprivate(Y, save_norm, iterationNumber) private(i)\
    depend(in: Y[0:1], Y[1:1], Y[2:1], Y[3:1], Y[4:1], Y[5:1], Y[6:1],\
    Y[7:1], Y[8:1], Y[9:1], Y[10:1], Y[11:1], Y[12:1], Y[13:1]/*, Y[14:1], Y[15:1],\
    Y[16:1], Y[17:1], Y[18:1], Y[19:1]*/)\
    depend(out: save_norm[iterationNumber: 1])
    {
        save_norm[iterationNumber] = 0;
        for(i = 0 ; i < nthrds; i++)
        {
            save_norm[iterationNumber] += Y[i];
        }
        save_norm[iterationNumber] = sqrt(save_norm[iterationNumber]);
    }
}

void normalize_task(double *Y, int n, int block_id, int block_width, double *save_norm, int iterationNumber)
{
    int i, j, blksz, start_index;

    i = block_id;
    blksz = block_width;
    start_index = i * block_width;

    if(i * block_width + blksz > n)
        blksz = n - i * block_width;
        
    #pragma omp task firstprivate(i, blksz, start_index, Y, n, block_id)\
    private(j) depend(in: save_norm[iterationNumber : 1])\
    depend(out: Y[start_index : blksz])
    {
        for(j = start_index ; j < start_index + blksz ; j++)
        {
            Y[j] = Y[j] / save_norm[iterationNumber];
        }
    }
}

/* single column, for power iteration in particular */
void custom_dlacpy_vector(double *src, double *dst, int n, int block_id, int block_width)
{
    int i, k, blksz;
    
    k = block_id * block_width; //starting point of the block 
    blksz = block_width;
        
    if(k + blksz > n)
        blksz = n - k;

    #pragma omp task private(i)\
    firstprivate(blksz, n, src, dst, block_width, k)\
    depend(in: src[k : blksz], n)\
    depend(out: dst[k : blksz])
    {
        for(i = k; i < k + blksz ; i++)
        {
            dst[i] = src[i];
        }
    }
}

/* added with power iteration, calculates
 * squared norm of a vector block*/
void dotV(double *Y, int n, double *squared_norm, int block_id, int buf_id, int block_width)
{
    int i, j, blksz, start_index;
    int tid;
    
    i = block_id;
    blksz = block_width;
    start_index = i * block_width;

    if(i * block_width + blksz > n)
    {
        blksz = n - i * block_width;
    }

    #pragma omp task firstprivate(i, Y, squared_norm, nthrds, start_index, blksz, buf_id)\
    private(j, tid)\
    depend(in: Y[start_index : blksz])\
    depend(out: squared_norm[buf_id : 1])
    {
        tid = omp_get_thread_num();
        for(j = start_index ; j < start_index + blksz ; j++)
        {
            squared_norm[tid] += Y[j] * Y[j];
        }
    }
}

void _XTY(double *X, double *Y, double *result ,int M, int N, int P, int blocksize)
{
    /*******************
    Input: X[M*N], Y[M*P]
    Output: result[N*P]
    ********************/

    int i, j, k, blksz, tid, nthreads;
    double sum;
    
    #pragma omp parallel shared(nthreads)
    nthreads=omp_get_num_threads();
    
    double *buf = new double[nthreads * N * P]();

    //--- task based implementation
    #pragma omp parallel num_threads(nthreads)\
    shared(nthreads)
    {
        #pragma omp single
        {
            for(k=0;k<M;k=k+blocksize)
            {
                blksz=blocksize;
                if(k+blksz>M)
                    blksz=M-k;
                #pragma omp task firstprivate(k, blksz) private(tid) shared(X, Y, buf, M, N, P)
                {
                    tid=omp_get_thread_num();
                    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,N,P,blksz,1.0,X+(k * N),N,Y+(k * P),P,1.0,buf+(tid * N * P),P);
                }
            }
        }
    }
    #pragma omp taskwait

  //--------task based summation of arrays

  #pragma omp parallel num_threads(nthreads)\
  shared(nthreads, result)
  {
    #pragma omp single
    {
      for(i=0;i<N;i++)
      {
        #pragma omp task firstprivate(i) private(sum, k) shared(nthreads, result, N, P)
        {
          for(k=0; k<P; k++)
          {
            sum=0.0;
            for(j=0;j<nthreads;j++) //for each thread access corresponding N*N matrix
            {
              sum+=buf[j*N*P+i*P+k];
            }
            result[i*P+k]=sum;
          }
        }
      }
    }
  }

  #pragma omp taskwait

  delete[] buf;
}
