#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    (void)argc;

    int i, j;
    int blocksize, block_width;
    int iterationNumber, eig_wanted;
    char *filename;

    double *xrem;
    block *matrixBlock;

    double *q, *qq;
    double norm_q;
    double *Q;
    double *z;
    double *alpha, *beta, *QpZ, *QQpZ;

    double *QpZBUFF, *AlphaBUFF, *normBUFF;

    struct TaskInfo *taskInfo;
    int structIterator, taskCount;
    int block_id, row_id, col_id, buf_id, task_id, blksz;
    
    double tstart, tend;
    double total_time;

    filename = argv[1];
    block_width = atoi(argv[2]);
    eig_wanted = atoi(argv[3]);
    taskCount = buildTaskInfoStruct(taskInfo, argv[4]);
    printf("taskCount: %d\n", taskCount);
    
    wblk = block_width; 
    
    read_custom(filename, xrem);
    csc2blkcoord(matrixBlock, xrem);

    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    q = (double *) malloc(numcols * sizeof(double));
    qq = (double *) malloc(numcols * sizeof(double));
    z = (double *) malloc(numcols * sizeof(double));
    alpha = (double *) malloc(eig_wanted * sizeof(double));
    beta = (double *) malloc(eig_wanted * sizeof(double));
    QpZ = (double *) malloc(eig_wanted * sizeof(double));
    QQpZ = (double *) malloc(numcols * sizeof(double));
    Q = (double*) calloc(numcols*(eig_wanted + 1),sizeof(double));
    
    QpZBUFF = (double *) malloc (nthrds * eig_wanted * sizeof(double));
    AlphaBUFF = (double *) malloc (nthrds * sizeof(double));
    normBUFF = (double *) malloc (nthrds * sizeof(double));

    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        q[i] = 1.0;
    }

    norm_q = sqrt(numcols);

    #pragma omp parallel for default(shared)
    for(i = 0 ; i < numcols; ++i)
    {
        qq[i] = q[i]/norm_q;
    }

    #pragma omp parallel for default(shared)
    for(i = 0 ; i < numcols; ++i)
    {
        Q[i*(eig_wanted)] = qq[i];
    }


    blocksize = 1;

    #pragma omp parallel
    {
        #pragma omp master
        {
            for(iterationNumber = 0; iterationNumber < eig_wanted; ++iterationNumber)
            {
                tstart = omp_get_wtime();

                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, AlphaBUFF)\
                depend(in: nthrds)\
                depend(inout: AlphaBUFF[0 * blocksize : blocksize], AlphaBUFF[1 * blocksize : blocksize], AlphaBUFF[2 * blocksize : blocksize],\
                        AlphaBUFF[3 * blocksize : blocksize], AlphaBUFF[4 * blocksize : blocksize], AlphaBUFF[5 * blocksize : blocksize], AlphaBUFF[6 * blocksize : blocksize],\
                        AlphaBUFF[7 * blocksize : blocksize], AlphaBUFF[8 * blocksize : blocksize], AlphaBUFF[9 * blocksize : blocksize], AlphaBUFF[10 * blocksize : blocksize],\
                        AlphaBUFF[11 * blocksize : blocksize], AlphaBUFF[12 * blocksize : blocksize], AlphaBUFF[13 * blocksize : blocksize])
                {
                    tstart = omp_get_wtime();
                    for(i = 0 ; i < nthrds * blocksize; ++i)
                    {
                        AlphaBUFF[i] = 0.0;
                    }
                }
                
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, normBUFF)\
                depend(in: nthrds)\
                depend(inout: normBUFF[0 * blocksize : blocksize], normBUFF[1 * blocksize : blocksize], normBUFF[2 * blocksize : blocksize],\
                        normBUFF[3 * blocksize : blocksize], normBUFF[4 * blocksize : blocksize], normBUFF[5 * blocksize : blocksize], normBUFF[6 * blocksize : blocksize],\
                        normBUFF[7 * blocksize : blocksize], normBUFF[8 * blocksize : blocksize], normBUFF[9 * blocksize : blocksize], normBUFF[10 * blocksize : blocksize],\
                        normBUFF[11 * blocksize : blocksize], normBUFF[12 * blocksize : blocksize], normBUFF[13 * blocksize : blocksize])
                {
                    tstart = omp_get_wtime();
                    for(i = 0 ; i < nthrds * blocksize; ++i)
                    {
                        normBUFF[i] = 0.0;
                    }
                }
                
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, QpZBUFF)\
                depend(in: nthrds)\
                depend(inout: QpZBUFF[0 * blocksize : eig_wanted], QpZBUFF[1 * blocksize : eig_wanted], QpZBUFF[2 * blocksize : eig_wanted],\
                        QpZBUFF[3 * blocksize : eig_wanted], QpZBUFF[4 * blocksize : eig_wanted], QpZBUFF[5 * blocksize : eig_wanted], QpZBUFF[6 * blocksize : eig_wanted],\
                        QpZBUFF[7 * blocksize : eig_wanted], QpZBUFF[8 * blocksize : eig_wanted], QpZBUFF[9 * blocksize : eig_wanted], QpZBUFF[10 * blocksize : eig_wanted],\
                        QpZBUFF[11 * blocksize : eig_wanted], QpZBUFF[12 * blocksize : eig_wanted], QpZBUFF[13 * blocksize : eig_wanted])
                {
                    tstart = omp_get_wtime();
                    for(i = 0; i < nthrds * eig_wanted; ++i)
                    {
                        QpZBUFF[i] = 0.0;
                    }
                }
                for(structIterator = 0 ; structIterator < taskCount; ++structIterator) 
                {
                    if(taskInfo[structIterator].opCode == 19) // taskName starts SETZERO 
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        if(task_id == 1)
                        {
                            i = block_id * block_width; // starting point of the block
                            blksz = block_width;
                            if(i + blksz > numcols)
                            {
                                blksz = numcols - i;
                            }
                                
                            #pragma omp task default(shared) private(j, tstart, tend)\
                            firstprivate(z, blksz, i, numrows, block_width)\
                            depend(inout : z[i : blksz])
                            {
                                for(j = i; j < i + blksz; ++j)
                                {
                                    z[j] = 0.0;
                                }
                            }
                        }
                    }
                    else if(taskInfo[structIterator].opCode == 7) // taskName starts DLACPY
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        i = block_id * block_width; // starting point of the block
                        blksz = block_width;
                        if(i + blksz > numcols)
                        {
                            blksz = numcols - i;
                        }
                        #pragma omp task default(shared) private(j, tstart, tend)\
                        firstprivate( qq,Q,blksz, i, numrows, block_width)\
                        depend(in : qq[i:blksz])\
                        depend(out : Q[i*eig_wanted : blksz*eig_wanted])
                        {
                            for(j = i; j < i + blksz; ++j)
                            {
                                Q[j*eig_wanted+iterationNumber+1] = qq[j];
                            }
                        } //end task
                    }
                    else if(taskInfo[structIterator].opCode == 23) //SPMV
                    {
                        row_id = taskInfo[structIterator].numParamsList[0]; 
                        col_id = taskInfo[structIterator].numParamsList[1]; 
                        buf_id = taskInfo[structIterator].numParamsList[2]; 
                        spmv_blkcoord_task(numcols, z, matrixBlock, qq, row_id, col_id, block_width);
                    }
                        
                    else if(taskInfo[structIterator].opCode == 21) // task name without , in it ==> opCode = 21 
                    {
                        if(!strcmp(taskInfo[structIterator].strParamsList[0], "NORM")) 
                        {
                            //norm_task_kk(normBUFF, numcols, beta, iterationNumber, eig_wanted);
                            norm_task(normBUFF, beta, iterationNumber);
                        }
                    }
                    
                    else if(taskInfo[structIterator].opCode == 27) // taskName DAXPY
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        divide_task(z, qq, beta, numcols, block_id, block_width, iterationNumber);
                    }

                    else if(taskInfo[structIterator].opCode == 9) // taskName SUB
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        sub_task(z, QQpZ, z, numcols, block_id, block_width);
                    }

                    else if(taskInfo[structIterator].opCode == 28) // taskName DGEMV
                    {
                        block_id = taskInfo[structIterator].numParamsList[0];
                        buf_id = taskInfo[structIterator].numParamsList[1];
                        task_id = taskInfo[structIterator].taskID;
                        if (task_id == 1 )
                        {
                            //dgemv_task_xty(Q, z, QpZBUFF, numcols, eig_wanted, 1, block_width, block_id, buf_id);
                            _XTY_v1_exe(Q, z, QpZBUFF, numcols, eig_wanted, 1, block_width, block_id, buf_id);
                        }
                        else if(task_id == 2)
                        {
                            //dgemv_task_xy(Q, QpZ, QQpZ ,numcols, eig_wanted, 1, block_width, block_id);
                            _XY_exe(Q, QpZ, QQpZ ,numcols, eig_wanted, 1, block_width, block_id);
                        }
                    }

                    else if(taskInfo[structIterator].opCode == 29) // taskName DOTV
                    {
                        block_id = taskInfo[structIterator].numParamsList[0];
                        buf_id = taskInfo[structIterator].numParamsList[1];
                        task_id = taskInfo[structIterator].taskID;
                        dotV(z, numcols , normBUFF, block_id, buf_id, block_width);			
                    }

                    else if(taskInfo[structIterator].opCode == 3) // taskName starts XTY 
                    {
                        block_id = taskInfo[structIterator].numParamsList[0];
                        buf_id = taskInfo[structIterator].numParamsList[1];
                        task_id = taskInfo[structIterator].taskID;
                        if(task_id == 1)
                        {
                            _XTY_v1_exe(qq, z, AlphaBUFF,numcols, 1, 1, block_width, block_id, buf_id);
                        }
                    }

                    else if(taskInfo[structIterator].opCode == 4) // RED
                    {
                        if(!strcmp(taskInfo[structIterator].strParamsList[0], "alpha")) //xty 1 reduction
                        {
                            //_XTY_v1_RED(AlphaBUFF, (alpha+iterationNumber), 1, 1, block_width);
                            reduce_task(AlphaBUFF, alpha, iterationNumber);
                        }
                        else if(!strcmp(taskInfo[structIterator].strParamsList[0], "QpZ")) //xty 2 reduction
                        {
                            //RED_QpZ(QpZBUFF, QpZ, eig_wanted, 1, block_width);
                            _XTY_v1_RED(QpZBUFF, QpZ, eig_wanted, 1, block_width);
                        }
                    }
                    else if(taskInfo[structIterator].opCode != 22) // undefined taskName
                    {
                        printf("NANI? WRONG opCOde %d\n", taskInfo[structIterator].opCode);
                        exit(1);
                    }
                }
                #pragma omp taskwait
                tend = omp_get_wtime();
                printf("%.4lf,",tend-tstart);
                total_time += (tend - tstart);
            }
        }
    }

    printf("%.4lf\n",total_time/eig_wanted);

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");
    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", beta[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");

    LAPACKE_dsterf(eig_wanted,alpha,beta);

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");

    //deallocation
    for(i = 0; i < nrowblks; ++i)
    {
        for(j = 0; j < ncolblks; ++j)
        {
            if(matrixBlock[i * ncolblks + j].nnz > 0)
            {
                delete [] matrixBlock[i * ncolblks + j].rloc;
                delete [] matrixBlock[i * ncolblks + j].cloc;
                delete [] matrixBlock[i * ncolblks + j].val;
            }
        }
    }
    delete [] matrixBlock;


    free(q);
    free(qq);
    free(z);
    free(alpha);
    free(beta);
    free(QpZ);
    free(QQpZ);
    free(Q);

    free(QpZBUFF);
    free(AlphaBUFF);
    free(normBUFF);

    return 0;
}
