#include "vector_ops.h"
#include "exec_util.h"


void _XTY(double *X, double *Y, double *result ,int M, int N, int P, int blocksize)
{
    /*******************
    Input: X[M*N], Y[M*P]
    Output: result[N*P]
    ********************/

    int i, j, k, blksz, tid, nthreads;
    double sum,tstart,tend;
    
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
                //tid=omp_get_thread_num();
                blksz=blocksize;
                if(k+blksz>M)
                    blksz=M-k;
                #pragma omp task firstprivate(k, blksz, tid) shared(X, Y, buf, M, N, P)
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
        #pragma omp task firstprivate(sum, i) private(k) shared(nthreads, result, N, P)
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