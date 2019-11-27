#include "matrix_ops.h"
#include "exec_util.h"

void spmm_blkcoord_loop(int R, int C, int blocksize, int nthrds, double *X,  double *Y, block *H){


	int k, k1, k2;
    int i, j, l, rbase, cbase, r, c;
    float xcoef;
    int length;
    double tstart;

    #pragma omp parallel for default(shared) private(tstart, rbase, cbase, j, k, l, r, c, xcoef)
    for(i = 0; i < nrowblks ; i++)
    {
        //tstart = omp_get_wtime();
        rbase = H[i * ncolblks + 0].roffset;
        
        for(j = 0 ; j < ncolblks ; j++)
        {
            cbase = H[i * ncolblks + j].coffset;
            if(H[i * ncolblks + j].nnz > 0)
            {
                for(k = 0 ; k < H[i * ncolblks + j].nnz ; k++)
                {
                    r = rbase + H[i * ncolblks + j].rloc[k] - 1;
                    c = cbase + H[i * ncolblks + j].cloc[k] - 1;
                    xcoef = H[i * ncolblks + j].val[k];
                    //#pragma omp simd 
                    for(l = 0 ;l < blocksize ; l++)
                    {
                        Y[r * blocksize + l] = Y[r * blocksize + l] + xcoef * X[c * blocksize + l];
                    }
                }
            }
        }
        //taskTiming[omp_get_thread_num()][1] += (omp_get_wtime() - tstart);
    } //end for
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




void spmm_blkcoord_finegrained_exe_fixed_buf(int R, int C, int M, int nbuf, double *X,  double *Y, block *H, int row_id, int col_id, int buf_id, int block_width)
{
    //code: 15

    int k, k1, k2, tid, blksz, offset;
    int spmm_offset, spmm_blksz;
    int i, j, l, rbase, cbase, r, c, nthreads;
    double tstart, tend;

    i = row_id;
    j = col_id;

    blksz = block_width;
    offset = j * block_width; //starting row of RHS matrix
    
    if(offset + block_width > R)
        blksz = R - offset;
    
    spmm_offset = i * block_width;
    spmm_blksz = block_width;
    
    if(spmm_offset + block_width > R)
        spmm_blksz = R - spmm_offset;
    
    double xcoef, sum;

    if(H[i * ncolblks + j].nnz > 0)
    {
        #pragma omp task firstprivate(i, j, buf_id, X, Y, H, R, C, M, nbuf, block_width, offset, blksz, spmm_blksz, spmm_offset)\
        private(k, l, r, c, xcoef, rbase, cbase, tid, tstart, tend)\
        depend(inout: Y[spmm_offset * M : spmm_blksz * M])\
        depend(in: X[offset * M : blksz * M])
        {
            //printf("i: %d j: %d SPMM inout: %p[%d : %d]\nSPMM in: %p[%d : %d]\n", i, j, Y, spmm_offset * M, spmm_blksz * M, X, offset * M, blksz * M);
            tid = omp_get_thread_num();
            tstart = omp_get_wtime();

            rbase = H[i * ncolblks + 0].roffset;
            cbase = H[i * ncolblks + j].coffset;
            
            for(k = 0 ; k < H[i * ncolblks + j].nnz ; k++)
            {
                r = rbase + H[i * ncolblks + j].rloc[k] - 1;
                c = cbase + H[i * ncolblks + j].cloc[k] - 1;
                xcoef = H[i * ncolblks + j].val[k];
                
                #pragma omp simd    
                for(l = 0 ; l < M ; l++)
                {
                    Y[r * M + l] = Y[r * M + l] + xcoef * X[c * M + l];
                }
            }
            tend = omp_get_wtime();
            //taskTiming_exec[15][tid] += (tend - tstart);
        }// end task
    } 
}


void spmm_blkcoord_finegrained_SPMMRED_fixed_buf(int R, int M, int nbuf, double *Y, double *spmmBUF, int block_id, int block_width)
{
    int k, k1, k2, tid, blksz, offset;
    int i, j, l, rbase, cbase, r, c, nthreads;
    double tstart, tend;

    double sum;
    blksz = block_width;
    offset = block_id * block_width;
    
    if(offset + blksz > R)
        blksz = R - offset;
    
    

    #pragma omp task firstprivate(offset, blksz, R, M, nbuf, spmmBUF, Y, block_width, block_id)\
    private(i, j, k, sum, tid, tstart, tend)\
    depend(inout: Y[offset * M : blksz * M])\
    depend(in: spmmBUF[0 * R * M + (block_id * block_width * M) : blksz * M]) //for one buffer only
    //depend(in: spmmBUF[0 * R * M + (block_id * block_width * M) : blksz * M], spmmBUF[1 * R * M + (block_id * block_width * M) : blksz * M],\
    spmmBUF[2 * R * M + (block_id * block_width * M) : blksz * M], spmmBUF[3 * R * M + (block_id * block_width * M) : blksz * M],\
    spmmBUF[4 * R * M + (block_id * block_width * M) : blksz * M], spmmBUF[5 * R * M + (block_id * block_width * M) : blksz * M],\
    spmmBUF[6 * R * M + (block_id * block_width * M) : blksz * M])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(i = offset ; i < offset + blksz ; i++)
        {
                
            for(k = 0 ; k < M ; k++)
            {
                sum = 0.0;
                for(j = 0 ; j < nbuf; j++) //for each thread access corresponding N * N matrix
                {
                    sum += spmmBUF[j * R * M + i * M + k];
                }
                Y[i * M + k] = sum;
            }
        }

        tend = omp_get_wtime();
        //taskTiming_exec[16][tid] += (tend - tstart);
    } //end task
}


void spmm_blkcoord_exec(int R, int C, int M, int nthrds, double *X,  double *Y, block *H)
{
    int k, k1, k2;
    int i, j, l, rbase, cbase, r, c;
    double xcoef;

    #pragma omp parallel num_threads(nthrds)
    {
        #pragma omp single
        {
            for(i = 0 ; i < nrowblks ; i++)
            {
                #pragma omp task default(shared) firstprivate(i) private(j, k, l, rbase, cbase, r, c, xcoef)
                {
                    rbase = H[i * ncolblks + 0].roffset;
                    for(j = 0 ; j < ncolblks ; j++)
                    {
                        cbase = H[i * ncolblks + j].coffset;
                        if(H[i * ncolblks + j].nnz > 0)
                        {
                            for(k = 0 ; k < H[i * ncolblks + j].nnz ; k++)
                            {
                                r = rbase + H[i * ncolblks + j].rloc[k] - 1;
                                c = cbase + H[i * ncolblks + j].cloc[k] - 1;
                                xcoef = H[i * ncolblks + j].val[k];
                                #pragma omp simd 
                                for(l = 0 ; l < M ; l++)
                                {
                                    Y[r * M + l] = Y[r * M + l] + xcoef * X[c * M +l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #pragma omp taskwait
}

void spmm_blkcoord_finegrained_exe_fixed_buf_unrolled32(int R, int C, int M, int nbuf, double *X,  double *Y, block *H, int row_id, int col_id, int buf_id, int block_width)
{
    //code: 15
    int k, k1, k2, tid, blksz, offset;
    int spmm_offset, spmm_blksz;
    int offset1, offset2, offset3;
    int i, j, l, rbase, cbase, r, c, nthreads;
    double tstart, tend;

    i = row_id;
    j = col_id;

    blksz = block_width;
    offset = j * block_width; //starting row of RHS matrix
    
    if(offset + block_width > R)
        blksz = R - offset;
    
    spmm_offset = i * block_width;
    spmm_blksz = block_width;
    
    if(spmm_offset + block_width > R)
        spmm_blksz = R - spmm_offset;
    
    double xcoef, sum;

    if(H[i * ncolblks + j].nnz > 0)
    {
        #pragma omp task firstprivate(i, j, buf_id, X, Y, H, R, C, M, nbuf, block_width, offset, blksz, spmm_blksz, spmm_offset)\
        private(k, l, r, c, xcoef, rbase, cbase, tid, tstart, tend, offset1, offset2, offset3)\
        depend(in: Y[i * block_width * nbuf * M : spmm_blksz * nbuf * M])\
        depend(in: X[offset * M : blksz * M])\
        depend(out: Y[(buf_id * R * M) + (i * block_width * M) : spmm_blksz * M])
        {
            tid = omp_get_thread_num();
            tstart = omp_get_wtime();

            rbase = H[i * ncolblks + 0].roffset;
            cbase = H[i * ncolblks + j].coffset;

            for(k = 0 ; k < H[i * ncolblks + j].nnz ; k++)
            {
                    r = rbase + H[i * ncolblks + j].rloc[k] - 1;
                    c = cbase + H[i * ncolblks + j].cloc[k] - 1;
                    xcoef = H[i * ncolblks + j].val[k];

                    offset1 = buf_id * R * 32 + r * 32;
                    offset2 = c * 32;

                    #pragma omp simd
                    for(l = 0 ; l < 32 ; l++)
                    {
                        Y[offset1 + l] = Y[offset1 + l] + xcoef * X[offset2 + l];
                    }
            }
            tend = omp_get_wtime();
            //taskTiming_exec[15][tid] += (tend - tstart);
        }// end task
    } 
}




void _XTY_v1_exe(double *X, double *Y, double *buf ,int M, int N, int P, int block_width, int block_id, int buf_id)
{
    //code: 10
    /*
    _XTY_v1: adding partial sums block by block, not row by row
    Input: X[M*N], Y[M*P]
    Output: result[N*P]
    nthrds : global variable, total # of threads
    buf : how to free/deallocate corresponding memory location
    block_width: each chunk
    */

    int i, j, k, l, blksz, tid, nthreads, length;
    double sum, tstart, tend;
    
    k = block_id * block_width; //start row # of block
    blksz = block_width;
    if(k + blksz > M)
        blksz = M - k;
    length = blksz * N ;
    
    i = buf_id;
    
    #pragma omp task private(tid, tstart, tend)\
    firstprivate(k, M, N, P, blksz, length, X, Y, nthrds, buf, block_id, block_width, buf_id)\
    depend(in: X[k * N : blksz * N], M, N, P)\
    depend(in: Y[k * P : blksz * P])\
    depend(out: buf[buf_id * N * P : N * P])
    {   
        //printf("block_id: %d XTY in: %p[%d : %d]     %p[%d : %d]\n", block_id, X, k * N , blksz * N, Y, k * N , blksz * N);
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, P, blksz, 1.0, X+(k*N), N, Y+(k*P), P, 1.0, buf+(tid * N * P), P);
        
        tend = omp_get_wtime();
        //taskTiming_exec[10][tid] += (tend - tstart);
        
    } //task end
}

void _XTY_v1_RED(double *buf, double *result, int N, int P, int block_width)
{
    //code: 11
    /*
    _XTY_v1_RED: adding partial sums block by block, not row by row
    Input: buf: nthds * [N * P]
    Output: result[N * P]
    nthrds : global variable, total # of threads
    buf : how to free/deallocate corresponding memory location
    */
    
    int i, j, k, l, blksz, tid, nthreads, length;
    double sum, tstart, tend;
    
    int nbuf = nthrds;

    for(i = 0 ; i < N ; i = i + block_width)
    {
        //reduce

        blksz = block_width;
        if(i + blksz > N)
            blksz = N - i;

        #pragma omp task private(sum, k, l, tid, tstart, tend)\
        firstprivate(i, nthrds, blksz, buf, N, P, result, block_width)\
        depend(in: N, P) depend(out: result[i * P : blksz * P])\
        depend(in: buf[0*N*P:N*P], buf[1*N*P:N*P], buf[2*N*P:N*P], buf[3*N*P:nthrds*N*P], buf[4*N*P:N*P], buf[5*N*P:N*P], buf[6*N*P:N*P],\
        buf[7*N*P:N*P], buf[8*N*P:N*P], buf[9*N*P:N*P], buf[10*N*P:nthrds*N*P], buf[11*N*P:N*P], buf[12*N*P:N*P], buf[13*N*P:N*P],\
        buf[14*N*P:N*P], buf[15*N*P:N*P])
        {
            tid = omp_get_thread_num();
            tstart = omp_get_wtime();
            
            for( l = i ; l < (i + blksz) ; l++) //for each row in the block
            {
                for(k  = 0 ; k < P ; k++) //each col
                {
                    sum = 0.0;
                    for(j = 0 ; j < nthrds ; j++) //for each thread access corresponding N*N matrix
                    {
                        sum += buf[j * N * P + l * P + k];
                    }
                    result[l * P + k] = sum;
                }
                
            } //end inner for 1

            tend = omp_get_wtime();
            //taskTiming_exec[11][tid] += (tend - tstart);

        }// end of task 
    }//end outer for
}

void _XY_exe(double *X, double *Y, double *result ,int M, int N, int P, int block_width, int block_id)
{
    //code: 1
    /*
    Input: X[M * N], Y[N * P] (RHS matrix)
    Output: result[M * P] (result = X * Y)
    nthrds : global variable, total # of threads
    */

    int i, j, k, blksz, tid;
    double tstart;
    
    k = block_id * block_width; //starting row # of the current block

    blksz = block_width;
        
    if(k + blksz > M)
        blksz = M - k;

    #pragma omp task private(tid, tstart) firstprivate(k, blksz, X, Y, result, M, N, P)\
    depend(in: X[k * N : blksz * N], Y[0 : N * P], M, N, P)\
    depend(out: result[k * P : blksz * P]) //should be P here, check other version if those have N here. 
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blksz, P, N, 1.0, X + (k * N), N, Y, P, 0.0, result + (k * P), P);
        
        //taskTiming_exec[1][tid] += (omp_get_wtime() - tstart);
    }//end task
}

void mat_addition_task_exe(double *src1, double *src2, double *dst, const int row,
                       const int col, int block_width, int block_id)
{
    //code: 21
    int i, j, k, tid;
    int blksz;
    double tstart, tend;
    
    i = block_id * block_width; //starting point of the block
    blksz = block_width;
    if(i + block_width > row)
        blksz = row - i;
    #pragma omp task private(j, k, tid, tstart, tend)\
    firstprivate(i, blksz, src1, src2, row, col, dst, block_width)\
    depend(in: src1[i * col : blksz * col])\
    depend(in: src2[i * col : blksz * col])\
    depend(out: dst[i * col : blksz * col])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();
        
        for(k = i ; k < i + blksz ; k++)
        {
            for(j = 0 ; j < col ; j++)
            {
                dst[k * col + j] = src1[k * col + j] + src2[k * col + j];
            }
        }

        tend = omp_get_wtime();
        //taskTiming_exec[21][tid] += (tend - tstart);
    } //task end
}

void mat_sub_task_exe(double *src1, double *src2, double *dst, const int row,
                       const int col, int block_width, int block_id)
{
    //code: 2
    int i, j, k, tid;
    int blksz;
    double tstart;
    
    i = block_id * block_width; //starting point of the block
    blksz = block_width;
    if(i + block_width > row)
        blksz = row - i;
    #pragma omp task firstprivate(i, blksz, src1, src2, row, col, dst)\
    private(j, k, tid, tstart)\
    depend(in: src1[i * col : blksz * col], row, col)\
    depend(in: src2[i * col : blksz * col])\
    depend(out: dst[i * col : blksz * col])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();
        
        for(k = i ; k < i + blksz ; k++)
        {
            for(j = 0 ; j < col ; j++)
            {
                dst[k * col + j] = src1[k * col + j] - src2[k * col + j];
            }
        }
        //taskTiming_exec[2][tid] += (omp_get_wtime() - tstart);
    }//task end
}

void mat_mult_task_exe(double *src1, double *src2, double *dst, const int row,
                       const int col, int block_width, int block_id)
{
    //code: 3
    int i, j, k, tid;
    int blksz;
    double tstart;
    i = block_id * block_width; //starting point of the block
    blksz = block_width;
    if(i + block_width > row)
        blksz = row - i;
    #pragma omp task firstprivate(i, blksz, src1, src2, row, col, dst)\
    private(j, k, tstart, tid)\
    depend(in: src1[i * col : blksz * col])\
    depend(in: src2[i * col : blksz * col])\
    depend(out: dst[i * col : blksz * col])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();
        
        for(k = i ; k < i + blksz ; k++)
        {
            for(j = 0 ; j < col ; j++)
            {
                dst[k * col + j] = src1[k * col + j] * src2[k * col + j];
            }
        }
        //taskTiming_exec[3][tid] += (omp_get_wtime() - tstart);
    }//end task
}

void sum_sqrt_task_COL(double *src, double *dst, const int row, const int col, int block_width, int block_id, int buf_id, double *buf)
{
    //code: 4
    int i, j, k, l, length, blksz, tid;
    int nbuf = nthrds;
    double tstart, tend;
    
    k = block_id * block_width; //starting point of the block
    blksz = block_width;
    i = buf_id; //pseudo_tid;

    if(k + blksz > row)
        blksz = row - k;
    
    #pragma omp task private(tid, j, l, tstart, tend)\
    firstprivate(k, blksz, src, nthrds, buf, block_width, buf_id, col)\
    depend(in: src[k * col : blksz * col])\
    depend(out: buf[buf_id * col : col])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();
        //printf("k: %d blksz: %d pseudo_tid: %d tid: %d\n", k, blksz, pseudo_tid, tid);

        for(j = k ; j < (k + blksz) ; j++) //row
        {
            for(l = 0 ; l < col ; l++) //col
            {
                buf[tid * col + l] += src[j * col + l];
                //printf("tid: %d col: %d j: %d l: %d\n", tid, col, j, l);
            }
        }

        tend = omp_get_wtime();
        //taskTiming_exec[4][tid] += (tend - tstart); 
    } //end task
}

void sum_sqrt_task_RNRED(double *buf, double *dst, const int col)
{
    //code: 6
    int i, j, k, l, length, blksz, tid;
    int nbuf = nthrds;
    double tstart, tend;
    //printf("nthrds: %d \n", nthrds);
    //adding partial sums
    #pragma omp task private(i, j, tid, tstart, tend)\
    firstprivate(nthrds, buf, col, dst)\
    depend(in: buf[0 * col : col], buf[1 * col : col], buf[2 * col : col],\
    buf[3 * col : col], buf[4 * col : col], buf[5 * col : col], buf[6 * col : col])\
    depend(inout: dst[0 : col])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(i = 0 ; i < nthrds ; i++) //threads
        {
            for(j = 0 ; j < col ; j++) //each col
            {
                dst[j] += buf[i * col + j]; 
            }
        } //end for

        tend = omp_get_wtime();
        //taskTiming_exec[6][tid] += (tend - tstart); 
    } //end task

}

void sum_sqrt_task_SQRT(double *dst, const int col)
{
    //code: 7
    int i, j, k, l, length, blksz, tid;
    int nbuf = nthrds;
    double tstart, tend;

    #pragma omp task private(i, tid, tstart, tend)\
    firstprivate(dst, col)\
    depend(inout: dst[0 : col])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(i = 0; i < col; i++) //i->col
        {
            dst[i] = sqrt(dst[i]);
        }

        tend = omp_get_wtime();
        //taskTiming_exec[7][tid] += (tend - tstart);         
    }
}

void update_activeMask_task_exe(int *activeMask, double *residualNorms, double residualTolerance, int blocksize)
{
    //code: 13
    int i, tid;
    double tstart, tend;

    #pragma omp task private(i, tid, tstart, tend)\
    firstprivate(activeMask, residualNorms, blocksize, residualTolerance)\
    depend(in: residualNorms[0 : blocksize], activeMask[0 : blocksize])\
    depend(out: activeMask[0 : blocksize]) 
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(i = 0; i < blocksize; i++)
        {
            if((residualNorms[i] > residualTolerance) && activeMask[i] == 1)
                activeMask[i] = 1;
            else
                activeMask[i] = 0;
            //printf("residualTolerance: %lf residualNorms[%d]: %lf activeMask[%d] : %d\n", residualTolerance, i, residualNorms[i], i, activeMask[i]);
        }   

        tend = omp_get_wtime();
        //taskTiming_exec[13][tid] += (tend - tstart); 
    }//end task 

}

void getActiveBlockVector_task_exe(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                               int M, int blocksize, int currentBlockSize, int block_width, int block_id)
{
    //code: 9
    /*  
        activeBlockVectorR dimension: M * currentBlockSize 
        blockVectorR dimension: M * blocksize 
        activeMask tells which columns are active now 
    */

    int i, j, k, l, blksz, tid;
    double tstart, tend;

    i = block_id * block_width; //starting point of the block
    blksz = block_width;
    if(i + blksz > M)
        blksz = M - i;

    #pragma omp task private(j, k, l, tstart, tend, tid)\
    firstprivate(activeBlockVectorR, blockVectorR, activeMask, blksz, block_width, currentBlockSize, blocksize, i)\
    depend(in : activeMask[0 : blocksize], activeBlockVectorR[i * currentBlockSize : blksz * currentBlockSize], currentBlockSize)\
    depend(in : blockVectorR[i * blocksize : blksz * blocksize], M, blocksize)\
    depend(out : activeBlockVectorR[i * currentBlockSize : blksz * currentBlockSize])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(j = i ; j < i + blksz ; j++) //rows
        {
            l = 0;
            for(k = 0 ; k < blocksize ; k++) //cols
            {
                if(activeMask[k] == 1)
                {
                   activeBlockVectorR[j * currentBlockSize + l] = blockVectorR[j * blocksize + k];
                   l++;
                }   
            }
            
        }
        tend = omp_get_wtime();
        //taskTiming_exec[9][tid] += (tend - tstart);
    } //end task
}

void updateBlockVector_task_exe(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                             int M, int blocksize, int currentBlockSize, int block_width, int block_id)
{
    //code: 13
    /*  
        activeBlockVectorR dimension: M * currentBlockSize 
        blockVectorR dimension: M * blocksize 
        activeMask tells which columns are active now 
    */
    
    int i, j, k, l, blksz, tid;
    double tstart, tend;
    i = block_id * block_width; //starting point of the block
    blksz = block_width;
    
    if(i + blksz > M)
        blksz = M - i;

    #pragma omp task private(j, k, tstart, tend)\
    firstprivate(activeBlockVectorR, blockVectorR, activeMask, blksz, block_width, currentBlockSize, blocksize, i)\
    depend(in : activeMask[0 : blocksize],  currentBlockSize, M, blocksize)\
    depend(in : activeBlockVectorR[i * currentBlockSize : blksz * currentBlockSize])\
    depend(inout : blockVectorR[i * blocksize : blksz * blocksize])
    {
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(j = i ; j < i + blksz ; j++) //rows
        {
            l = 0;
            for(k = 0 ; k < blocksize ; k++) //cols
            {
                if(activeMask[k] == 1)
                {
                    blockVectorR[j * blocksize + k] = activeBlockVectorR[j * currentBlockSize + l];
                    l++;
                } 
            } 
        }

        tend = omp_get_wtime();
        //taskTiming_exec[13][tid] += (tend - tstart);
    } //end task
}


void custom_dlacpy_task_exe(double *src, double *dst, int row, int col, int block_width, int block_id)
{
    //code: 14
    /* src and dst dimension: row * col */

    int i, j, k, blksz, tid;
    double tstart, tend;
    
    k = block_id * block_width; //starting point of the block 
    blksz = block_width;
        
    if(k + blksz > row)
        blksz = row - k;

    #pragma omp task private(i, j, tid, tstart, tend)\
    firstprivate(blksz, row, col, src, dst, block_width, k)\
    depend(in: src[k * col : blksz * col], row, col)\
    depend(out: dst[k * col : blksz * col])
    {
        //printf("DLACPY dst[%d * %d : %d * %d]\n", k, col, blksz, col);
        //printf("dst k: %d blksz: %d\n", k, blksz);
        
        tid = omp_get_thread_num();
        tstart = omp_get_wtime();

        for(i = k; i < k + blksz ; i++) //each row
        {
            for(j = 0 ; j < col ; j++) //each column
            {
                dst[i * col + j] = src[i * col + j];
            }
        }

        tend = omp_get_wtime();
        //taskTiming_exec[14][tid] += (tend - tstart);
    }  //end task
}



void dot_mm_exe(double *src1, double *src2, double *buf, const int row,
                       const int col, int block_width, int block_id, int buf_id)
{
    //code: 3
    int i, j, k, tid;
    int buf_base;
    int blksz;
    double tstart;
    i = block_id * block_width; //starting point of the block
    blksz = block_width;
    if(i + block_width > row)
        blksz = row - i;
        
    #pragma omp task firstprivate(i, blksz, src1, src2, buf, row, col, block_id, buf_id)\
    private(j, k, tstart, tid, buf_base)\
    depend(in: src1[i * col : blksz * col])\
    depend(in: src2[i * col : blksz * col])\
    depend(out: buf[buf_id * col : col])
    {
        tid = omp_get_thread_num();
        //tstart = omp_get_wtime();
        buf_base = tid * col;
        
        //printf("tid: %d col: %d block_id: %d buf_id: %d\n", tid, col, block_id, buf_id);

        for(k = i ; k < i + blksz ; k++) //row
        {
            for(j = 0 ; j < col ; j++) //col
            {
                buf[buf_base + j] += src1[k * col + j] * src2[k * col + j];
                //printf("%lf\n", buf[buf_base + j]);
            }
        }
        //taskTiming[3][tid] += (omp_get_wtime() - tstart);
    }//end task
}

