#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    int i, j;
    int block_width;
    int iterationNumber, maxIterations;
    char *filename;

    double *xrem;
    block *matrixBlock;

    struct TaskInfo *taskInfo;
    int structIterator, taskCount;
    int block_id, row_id, col_id, buf_id, task_id, blksz;
    double *squared_norm;

    double *loopTime, *save_norm;
    double tstart, tend;

    filename = argv[1];
    block_width = atoi(argv[2]);
    maxIterations = atoi(argv[3]);
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

    /* deleting csc storage memory*/
    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    double *Y = (double *) malloc(numcols * sizeof(double));
    double *Y_temp = (double *) malloc(numcols * sizeof(double));

    #pragma omp parallel for default(shared)
    for(i = 0 ; i < numcols ; i++)
    {
        Y[i] = 0.5;
    }

    /* mkl_dcsrmv params */
    loopTime = (double *) malloc(maxIterations * sizeof(double));
    save_norm = (double *) malloc(maxIterations * sizeof(double));
    for(i = 0 ; i < maxIterations; i++)
    {
        loopTime[i] = 0.0;
        save_norm[i] = 0.0;
    }

    squared_norm = (double *) malloc(nthrds * sizeof(double));

    #pragma omp parallel for default(shared)
    for(i = 0; i < nthrds; i++)
    {
        squared_norm[i]= 0;
    }


    #pragma omp parallel
    {
        #pragma omp master
        {
            for(iterationNumber = 0; iterationNumber < maxIterations; iterationNumber++)
            {
                tstart = omp_get_wtime();
                for(i = 0 ; i < nthrds; i++) 
                {
                    squared_norm[i] = 0;
                }

                for(structIterator = 0 ; structIterator < taskCount; structIterator++)
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
                            firstprivate(Y_temp, blksz, i, numrows, block_width)\
                            depend(inout : Y_temp[i : blksz])
                            {
                                for(j = i ; j < i + blksz ; j++)
                                {
                                    Y_temp[j] = 0.0;
                                }
                            }
                            
                        }
                    }
                    else if(taskInfo[structIterator].opCode == 7) // taskName starts DLACPY
                    {   
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        
                        custom_dlacpy_vector(Y_temp, Y, numcols, block_id, block_width);
                    }
                    else if(taskInfo[structIterator].opCode == 23) //SPMV
                    {
                        row_id = taskInfo[structIterator].numParamsList[0]; 
                        col_id = taskInfo[structIterator].numParamsList[1]; 
                        buf_id = taskInfo[structIterator].numParamsList[2]; 
                    
                        spmv_blkcoord_task(numrows, numcols, nthrds, Y_temp, matrixBlock, Y, row_id, col_id, buf_id, block_width);
                    }
                    else if(taskInfo[structIterator].opCode == 24) //DOT PRODUCT, changed from SUBMAX
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        buf_id = taskInfo[structIterator].numParamsList[1]; 
                        task_id = taskInfo[structIterator].taskID;
                        dotV(Y_temp, numcols, squared_norm,  block_id, buf_id, block_width);
                    }
                    else if(taskInfo[structIterator].opCode == 21) // task name without , in it ==> opCode = 21 
                    {
                        if(!strcmp(taskInfo[structIterator].strParamsList[0], "MAX")) 
                        {
                            norm_task(squared_norm, save_norm, iterationNumber);
                        }
                    }
                    else if(taskInfo[structIterator].opCode == 25) // taskName starts NORM 
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                    
                        normalize_task(Y_temp, numcols, block_id, block_width, save_norm, iterationNumber);
                    }
                }
                #pragma omp taskwait
                
                tend = omp_get_wtime();
                loopTime[iterationNumber] = tend - tstart;
            }
        }
    }

    double totalSum = 0;
    for(i = 0 ; i < maxIterations; i++)
    {
        totalSum += loopTime[i];
        printf("%.4lf,", loopTime[i]);
    }
    printf("%.4lf\n", totalSum/maxIterations);

    for(i = 0 ; i < maxIterations; i++)
    {
        printf("%.4lf", save_norm[i]);
        if(i != maxIterations - 1)
            printf(",");
    }
    printf("\n");

    free(Y);
    free(Y_temp);
    
    free(loopTime);
    free(save_norm);

    for(i = 0; i < nrowblks; i++)
    {
        for(j = 0; j < ncolblks; j++)
        {
            if(matrixBlock[i * ncolblks + j].nnz)
            {
                delete [] matrixBlock[i * ncolblks + j].rloc;
                delete [] matrixBlock[i * ncolblks + j].cloc;
                delete [] matrixBlock[i * ncolblks + j].val;
            }
        }
    }
    delete [] matrixBlock;

    return 0;
}
