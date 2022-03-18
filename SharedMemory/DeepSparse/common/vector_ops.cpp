#include "vector_ops.h"
#include "exec_util.h"


void init_vector(double *Y, int len)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < len; i++)
    {
        Y[i] = 0.0;
    }
}


/* reduce on partial squared norms */
void norm_task(double *Y, double *save_norm, int iterationNumber)
{
    int i;
    
#ifdef EPYC
    #pragma omp task firstprivate(Y, save_norm, iterationNumber) private(i)\
    depend(in: Y[0:1], Y[1:1], Y[2:1], Y[3:1], Y[4:1], Y[5:1], Y[6:1],\
            Y[7:1], Y[8:1], Y[9:1], Y[10:1], Y[11:1], Y[12:1], Y[13:1],\
            Y[14:1], Y[15:1], Y[16:1], Y[17:1], Y[18:1], Y[19:1], Y[20:1],\
            Y[21:1], Y[22:1], Y[23:1], Y[24:1], Y[25:1], Y[26:1], Y[27:1],\
            Y[28:1], Y[29:1], Y[30:1], Y[31:1], Y[32:1], Y[33:1], Y[34:1],\
            Y[35:1], Y[36:1], Y[37:1], Y[38:1], Y[39:1], Y[40:1], Y[41:1],\
            Y[42:1], Y[43:1], Y[44:1], Y[45:1], Y[46:1], Y[47:1], Y[48:1],\
            Y[49:1], Y[50:1], Y[51:1], Y[52:1], Y[53:1], Y[54:1], Y[55:1],\
            Y[56:1], Y[57:1], Y[58:1], Y[59:1], Y[60:1], Y[61:1], Y[62:1],\
            Y[63:1], Y[64:1], Y[65:1], Y[66:1], Y[67:1], Y[68:1], Y[69:1],\
            Y[70:1], Y[71:1], Y[72:1], Y[73:1], Y[74:1], Y[75:1], Y[76:1],\
            Y[77:1], Y[78:1], Y[79:1], Y[80:1], Y[81:1], Y[82:1], Y[83:1],\
            Y[84:1], Y[85:1], Y[86:1], Y[87:1], Y[88:1], Y[89:1], Y[90:1],\
            Y[91:1], Y[92:1], Y[93:1], Y[94:1], Y[95:1], Y[96:1], Y[97:1],\
            Y[98:1], Y[99:1], Y[100:1], Y[101:1], Y[102:1], Y[103:1], Y[104:1],\
            Y[105:1], Y[106:1], Y[107:1], Y[108:1], Y[109:1], Y[110:1], Y[111:1],\
            Y[112:1], Y[113:1], Y[114:1], Y[115:1], Y[116:1], Y[117:1], Y[118:1],\
            Y[119:1], Y[120:1], Y[121:1], Y[122:1], Y[123:1], Y[124:1], Y[125:1],\
            Y[126:1], Y[127:1])\
    depend(out: save_norm[iterationNumber: 1])
#else
    #pragma omp task firstprivate(Y, save_norm, iterationNumber) private(i)\
    depend(in: Y[0:1], Y[1:1], Y[2:1], Y[3:1], Y[4:1], Y[5:1], Y[6:1],\
    Y[7:1], Y[8:1], Y[9:1], Y[10:1], Y[11:1], Y[12:1], Y[13:1])\
    depend(out: save_norm[iterationNumber: 1])
#endif
    {
        save_norm[iterationNumber] = 0;
        for(i = 0 ; i < nthrds; i++)
        {
            save_norm[iterationNumber] += Y[i];
        }
        save_norm[iterationNumber] = sqrt(save_norm[iterationNumber]);
    }
}

/* reduce on partial values as in dot product */
void reduce_task(double *Y, double *local_res, int iterationNumber)
{
    int i;

#ifdef EPYC
    #pragma omp task firstprivate(Y, local_res, iterationNumber) private(i)\
    depend(in: Y[0:1], Y[1:1], Y[2:1], Y[3:1], Y[4:1], Y[5:1], Y[6:1],\
            Y[7:1], Y[8:1], Y[9:1], Y[10:1], Y[11:1], Y[12:1], Y[13:1],\
            Y[14:1], Y[15:1], Y[16:1], Y[17:1], Y[18:1], Y[19:1], Y[20:1],\
            Y[21:1], Y[22:1], Y[23:1], Y[24:1], Y[25:1], Y[26:1], Y[27:1],\
            Y[28:1], Y[29:1], Y[30:1], Y[31:1], Y[32:1], Y[33:1], Y[34:1],\
            Y[35:1], Y[36:1], Y[37:1], Y[38:1], Y[39:1], Y[40:1], Y[41:1],\
            Y[42:1], Y[43:1], Y[44:1], Y[45:1], Y[46:1], Y[47:1], Y[48:1],\
            Y[49:1], Y[50:1], Y[51:1], Y[52:1], Y[53:1], Y[54:1], Y[55:1],\
            Y[56:1], Y[57:1], Y[58:1], Y[59:1], Y[60:1], Y[61:1], Y[62:1],\
            Y[63:1], Y[64:1], Y[65:1], Y[66:1], Y[67:1], Y[68:1], Y[69:1],\
            Y[70:1], Y[71:1], Y[72:1], Y[73:1], Y[74:1], Y[75:1], Y[76:1],\
            Y[77:1], Y[78:1], Y[79:1], Y[80:1], Y[81:1], Y[82:1], Y[83:1],\
            Y[84:1], Y[85:1], Y[86:1], Y[87:1], Y[88:1], Y[89:1], Y[90:1],\
            Y[91:1], Y[92:1], Y[93:1], Y[94:1], Y[95:1], Y[96:1], Y[97:1],\
            Y[98:1], Y[99:1], Y[100:1], Y[101:1], Y[102:1], Y[103:1], Y[104:1],\
            Y[105:1], Y[106:1], Y[107:1], Y[108:1], Y[109:1], Y[110:1], Y[111:1],\
            Y[112:1], Y[113:1], Y[114:1], Y[115:1], Y[116:1], Y[117:1], Y[118:1],\
            Y[119:1], Y[120:1], Y[121:1], Y[122:1], Y[123:1], Y[124:1], Y[125:1],\
            Y[126:1], Y[127:1])\
    depend(out: local_res[iterationNumber: 1])
#else
    #pragma omp task firstprivate(Y, local_res, iterationNumber) private(i)\
    depend(in: Y[0:1], Y[1:1], Y[2:1], Y[3:1], Y[4:1], Y[5:1], Y[6:1],\
    Y[7:1], Y[8:1], Y[9:1], Y[10:1], Y[11:1], Y[12:1], Y[13:1])\
    depend(out: local_res[iterationNumber: 1])
#endif
    {
        local_res[iterationNumber] = 0;
        for(i = 0 ; i < nthrds; i++)
        {
            local_res[iterationNumber] += Y[i];
        }
    }
}
void normalize_task(double *Y, int n, int block_id, int block_width, double *save_norm, int iterationNumber)
{
    int i, j, blksz, start_index;
    double factor;

    i = block_id;
    blksz = block_width;
    start_index = i * block_width;

    if(i * block_width + blksz > n)
        blksz = n - i * block_width;

    #pragma omp task firstprivate(i, blksz, start_index, Y, n, block_id)\
    private(factor, j) depend(in: save_norm[iterationNumber : 1])\
    depend(out: Y[start_index : blksz])
    {
        factor = 1.0 / save_norm[iterationNumber];
        for(j = start_index ; j < start_index + blksz ; j++)
        {
            Y[j] = Y[j] * factor;
        }
    }
}

/* src = dst / scalar beta[iterationNumber] */
void divide_task(double *src, double *dst, double *beta, int n, int block_id, int block_width, int iterationNumber)
{
    int i, k, blksz;
    double factor;

    k = block_id * block_width;
    blksz = block_width;

    if(k + blksz > n)
        blksz = n - k;
    
    #pragma omp task private(i, factor)\
    firstprivate(blksz, n, src, dst, block_width, k, beta,iterationNumber)\
    depend(in: src[k : blksz], n,beta[iterationNumber : 1])\
    depend(out: dst[k : blksz])
    {
        factor = 1.0 / beta[iterationNumber];
        for(i = k; i < k + blksz; i++)
        {
            dst[i] = src[i] * factor;
        }
    }
}

/* dst = src1 - src2 */
void sub_task(double *src1, double *src2, double *dst, int n, int block_id, int block_width)
{
    int i, k, blksz;

    k = block_id * block_width;
    blksz = block_width;

    if(k + blksz > n)
        blksz = n - k;
    
    #pragma omp task private(i)\
    firstprivate(blksz, n, src1, src2, dst, block_width, k)\
    depend(in: src1[k : blksz], n)\
    depend(in: src2[k : blksz])\
    depend(out: dst[k : blksz])
    {
        for(i = k; i < k + blksz; i++)
        {
            dst[i] = src1[i] - src2[i];
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
