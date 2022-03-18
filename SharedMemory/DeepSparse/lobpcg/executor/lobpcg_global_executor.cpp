#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    int M, N, index = 0, blocksize;
    int block_width;
    double *A, *blockVectorX;
    double *at, *bt;
    double residualTolerance = 0.0001;
    long maxIterations = 10;
    int constraintStyle = 0; //operatorB
    long iterationNumber;
    int info, i, j, k, priority;

    stringstream s(argv[3]);
    s >> blocksize;
    stringstream s1(argv[2]);
    s1 >> block_width;
    if(argc == 8)
    {
        stringstream sIter(argv[7]);
        sIter >> maxIterations;
    }
    
    int currentBlockSize = blocksize, prevCurrentBlockSize = blocksize;

    const char jobvl = 'N';
    const char jobvr = 'V';
    const char jobz = 'V';
    const char uplo = 'U';
    const char uplo_dlacpy = ' ';
    const int itype = 1;
    double work_query;
    int lwork;
    double *work;

	struct TaskInfo *taskInfo_nonLoop, *taskInfo_firstLoop, *taskInfo_secondLoop;



    //file read
    double *xrem;
    block *matrixBlock;
    char *filename = argv[1];

    //printf("filename: %s\n", filename);
    
    wblk = block_width; 

    read_custom(filename, xrem);
    csc2blkcoord(matrixBlock, xrem);
    
    #pragma omp parallel shared(nthrds)
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    cout << "Block Size: " << blocksize << " Block Width: " << block_width << " nthrds: " << nthrds << endl;
    
    free(colptrs);
    free(irem); 
    free(xrem); 

    // M -> row and N -> col
    M = numrows;
    N = numcols;

    // tastTiming variables 
    //numOperation = 24;
    /*taskTiming = (double **) malloc(numOperation * sizeof(double *));
    for(i = 0 ; i < numOperation ; i++)
        taskTiming[i] = (double *) malloc(nthrds * sizeof(double));
    
    // reseting taskTiming
    for(i = 0 ; i < numOperation ; i++)
        for(j = 0 ; j < nthrds ; j++)
            taskTiming[i][j] = 0.0;

    // saving timing stat
    // timingStat[task * maxIterations]
    double **timingStat = (double **) malloc(12 * maxIterations * sizeof(double *));
    for(i = 0 ; i < 12 ; i++)
        timingStat[i] = (double *) malloc(maxIterations * sizeof(double));
    
    for(i = 0 ; i < 12 ; i++)
        for(j = 0 ; j < maxIterations ; j++)
            timingStat[i][j] = 0.0;*/

    // saving lamda history
    // saveLamda[blocksize * maxIterations]

    printf("blockSize = %d maxIterations = %d\n",blocksize,maxIterations);
    double **saveLamda = (double **) malloc(blocksize * maxIterations * sizeof(double *));
    for(i = 0 ; i < blocksize ; i++)
        saveLamda[i] = (double *) malloc(maxIterations * sizeof(double));
    
    for(i = 0 ; i < blocksize ; i++)
        for(j = 0 ; j < maxIterations ; j++)
            saveLamda[i][j] = 0.0;


    double loop_start_time = 0, loop_finish_time = 0;
    //double before_loop_start_time = 0, before_loop_finish_time = 0;
    double iteraton_time = 0, iteraton_start_time = 0;
    
    double *loopTime = (double *) malloc(maxIterations * sizeof(double));
    for(j = 0 ; j < maxIterations ; j++)
        loopTime[j] = 0.0;

    //Matrix X
    blockVectorX = (double *) malloc(M * blocksize * sizeof(double));
    //std::fstream blockVectorXfile("../mtx/MatX100.txt", std::ios_base::in);
    
    srand(0);
    //unsigned int seed = 0;

    //#pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < blocksize ; j++)
        {
            //blockVectorXfile >> blockVectorX[i * blocksize + j];
            blockVectorX[i * blocksize + j] = (double)rand()/(double)RAND_MAX;
            //blockVectorX[i * blocksize + j] = -1.00 + rand() % 2 ;
        }
    }

    //printf("Printing blockVectorX: \n");
    //print_mat(blockVectorX, 9, blocksize);

    //cout << "finished reading X" << endl;

    //memory allocation for matrices

    double *blockVectorAX = (double *) malloc(M * blocksize * sizeof(double)); 
    double *blockVectorR = (double *) malloc(M * blocksize * sizeof(double)); 
    double *blockVectorAR = (double *) malloc(M * blocksize * sizeof(double)); 
    double *blockVectorP = (double *) malloc(M * blocksize * sizeof(double)); 
    double *blockVectorAP = (double *) malloc(M * blocksize * sizeof(double)); 
     

    double *activeBlockVectorR = (double *) malloc(M * currentBlockSize * sizeof(double)); 
    double *activeBlockVectorAR = (double *) malloc(M * currentBlockSize * sizeof(double));  
    double *activeBlockVectorP = (double *) malloc(M * currentBlockSize * sizeof(double));  
    double *activeBlockVectorAP = (double *) malloc(M * currentBlockSize * sizeof(double));  

    double *newX = (double *) malloc(M * blocksize * sizeof(double)); 
    //double *temp3 = (double *) malloc(M * currentBlockSize * sizeof(double));  
    double *temp3_R = (double *) malloc(M * currentBlockSize * sizeof(double)); 
    double *temp3_P = (double *) malloc(M * currentBlockSize * sizeof(double)); 
    double *temp3_AP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *newP = (double *) malloc(M * blocksize * sizeof(double)); 
    double *newAP = (double *) malloc(M * blocksize * sizeof(double)); 
    double *newAX = (double *) malloc(M * blocksize * sizeof(double)); 

    double *gramA, *gramB;
    double *eigen_value;
    double *residualNorms = (double *) malloc(blocksize * 1 * sizeof(double));
    double *gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *trans_gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *temp2 = (double *)malloc(blocksize * currentBlockSize * sizeof(double));
    double *gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *trans_gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *gramXAR = (double *) malloc(blocksize * currentBlockSize * sizeof(double));
    double *transGramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *gramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *transGramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *gramXAP = (double *) malloc(blocksize * currentBlockSize * sizeof(double)); 
    double *transGramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *gramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *gramPAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *identity_PAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *gramXBP = (double *) malloc(blocksize * currentBlockSize * sizeof(double)); 
    double *transGramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 
    double *identity_BB = (double *) malloc(blocksize * blocksize * sizeof(double)); 
    double *gramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double)); 

    double *zeros_B_CB = (double *) malloc(blocksize * currentBlockSize * sizeof(double)); 
    double *zeros_CB_B =  (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 

    //non loop
    double *gramXBX = (double *) malloc(blocksize * blocksize * sizeof(double)); 
    double *tempGramXBX = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *gramXAX = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *lambda = (double *) malloc(blocksize * blocksize * sizeof(double)); 

    /* allocating fixed buffers for different XTY and SUM_SQRT */
    double *RNBUF  = (double *) malloc(nthrds * blocksize * sizeof(double));
    double *temp2XTYBUF  = (double *) malloc(nthrds * blocksize * currentBlockSize * sizeof(double)); //xty 1
    double *gramRBRXTYBUF  = (double *) malloc(nthrds * currentBlockSize * currentBlockSize * sizeof(double)); //xty 2
    double *gramPBPXTYBUF  = (double *) malloc(nthrds * currentBlockSize * currentBlockSize * sizeof(double)); //xty 3
    double *gramXARXTYBUF  = (double *) malloc(nthrds * blocksize * currentBlockSize * sizeof(double)); //xty 4
    double *gramRARXTYBUF  = (double *) malloc(nthrds * currentBlockSize * currentBlockSize * sizeof(double)); //xty 5
    double *gramXAPXTYBUF  = (double *) malloc(nthrds * blocksize * currentBlockSize * sizeof(double)); //xty 6
    double *gramRAPXTYBUF  = (double *) malloc(nthrds * currentBlockSize * currentBlockSize * sizeof(double)); //xty 7
    double *gramPAPXTYBUF  = (double *) malloc(nthrds * currentBlockSize * currentBlockSize * sizeof(double)); //xty 8
    double *gramXBPXTYBUF  = (double *) malloc(nthrds * blocksize * currentBlockSize * sizeof(double)); //xty 9
    double *gramRBPXTYBUF  = (double *) malloc(nthrds * currentBlockSize * currentBlockSize * sizeof(double)); //xty 10

    double *gramXBXBUF  = (double *) malloc(nthrds * blocksize * blocksize * sizeof(double)); //xty 20
    double *gramXAXBUF  = (double *) malloc(nthrds * blocksize * blocksize * sizeof(double)); //xty 21
    
    #pragma omp parallel for default(shared)
    for(i = 0 ; i < blocksize * currentBlockSize ; i++)
    {
        zeros_B_CB[i] = 0.0;
        zeros_CB_B[i] = 0.0;
    }
    

    //if 9
    //gramXBX = blockVectorX'* blockVectorX;
    //[gramXBX,cholFlag] = chol(gramXBX);
    //blockVectorX = blockVectorX/gramXBX;
    
   

    make_identity_mat(identity_BB, blocksize, blocksize);
    make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize); //--> used in loop

    //NON-LOOP starts here 

    //char taskName[20000];
    //char **splitParams;
    //int maxTaskLength = -1;
    int tid;
    double tstart, tend;
    //int partNo, tokenCount;


    int taskCounter = 0, block_id, task_id, buf_id, blksz, row_id, col_id;
    int convergeFlag = 0;
    int nbuf = 1 ; //How many buffer for SPMM tasks

    int spmmTaskCounter = 0;
    int spmmTaskCounter_2 = 0;
    int sp = 0;

    //######### Preparing taskInfo_nonLoop here #########
    int taskCount_nonLoop, taskCount_firstLoop, taskCount_secondLoop;
    int partCount_nonLoop, partCount_firstLoop, partCount_secondLoop;
    int *partBoundary_nonLoop, *partBoundary_firstLoop, *partBoundary_secondLoop;
    int structIterator, partIterator;

    taskCount_nonLoop = buildTaskInfoStruct(taskInfo_nonLoop, argv[4]);
    cout << "taskCount_nonLoop: " << taskCount_nonLoop << endl;
    taskCount_firstLoop = buildTaskInfoStruct(taskInfo_firstLoop, argv[5]);
    cout << "taskCount_firstLoop: " << taskCount_firstLoop << endl;
    taskCount_secondLoop = buildTaskInfoStruct(taskInfo_secondLoop, argv[6]);
    cout << "taskCount_secondLoop: " << taskCount_secondLoop << endl;
    


    //opening partition file
    // ifstream nonLoopFile(argv[4]);
    
    // if(nonLoopFile.fail())
    // {
    //     printf("Non-loop File doesn't exist\n");
    // }
   
    //printf("argv[4]: %s\n", argv[4]);
   
    for(i = 0 ; i < nthrds * blocksize * blocksize ; i++)
    {
        gramXBXBUF[i] = 0.0; 
        gramXAXBUF[i] = 0.0;       
    }
    //printf("Non loop part starts");
    /* nonloop part execution starts here */
    #pragma omp parallel
    {
    #pragma omp master
    {
    for(structIterator = 0 ; structIterator < taskCount_nonLoop; structIterator++) 
    {
        //printf("task opcode = %d\n",taskInfo_nonLoop[structIterator].opCode );
        if(taskInfo_nonLoop[structIterator].opCode == 22) 
        {
        }
        //strcpy(taskName_copy, taskName);
        /* if the taskName has more than or equal 1 tokens */
        //tokenCount = split(taskName, ',', &splitParams);

        //if(tokenCount > 0 && taskName[0] != '_') 
        //{
            else if(taskInfo_nonLoop[structIterator].opCode == 3) // taskName starts XTY 
            {
                // block_id = atoi(splitParams[1]);
                // buf_id = atoi(splitParams[2]);
                // task_id = atoi(splitParams[3]);
                
                block_id = taskInfo_nonLoop[structIterator].numParamsList[0];
                buf_id = taskInfo_nonLoop[structIterator].numParamsList[1];
                task_id = taskInfo_nonLoop[structIterator].taskID;

                if(task_id == 20)
                {
                    _XTY_v1_exe(blockVectorX, blockVectorX, gramXBXBUF, M, blocksize, blocksize, block_width, block_id, buf_id);
                }
                else if(task_id == 21)
                {
                    _XTY_v1_exe(blockVectorX, blockVectorAX, gramXAXBUF, M, blocksize, blocksize, block_width, block_id, buf_id);  
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 4)  //RED
            {
                if(!strcmp(taskInfo_nonLoop[structIterator].strParamsList[0], "XBXBUF")) //xty 20 reduction
                {
                    _XTY_v1_RED(gramXBXBUF, gramXBX, blocksize, blocksize, block_width);
                }
                else if(!strcmp(taskInfo_nonLoop[structIterator].strParamsList[0], "XAXBUF")) //xty 20 reduction
                {
                    _XTY_v1_RED(gramXAXBUF, gramXAX, blocksize, blocksize, block_width);
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 7) //DLAPCY
            {
                // block_id = atoi(splitParams[1]);
                // task_id = atoi(splitParams[2]);
                block_id = taskInfo_nonLoop[structIterator].numParamsList[0];
                task_id = taskInfo_nonLoop[structIterator].taskID;
                
                //printf(" out %s %d %d\n", splitParams[0], block_id, task_id);
                if(task_id == 20)
                {
                    //printf("in %s %d %d\n", splitParams[0], block_id, task_id);

                    #pragma omp task private(i, j) firstprivate(gramXBX, tempGramXBX, blocksize)\
                    depend(in: gramXBX[0 : blocksize * blocksize])\
                    depend(out: tempGramXBX[0 : blocksize * blocksize])
                    {
                        for(i = 0 ; i < blocksize ; i++)
                        {
                            for(j = 0 ; j < blocksize ; j++)
                            {
                                tempGramXBX[i * blocksize + j] = gramXBX[i * blocksize + j];
                            }
                        }
                    }

                    //custom_dlacpy_task_exe(gramXBX, tempGramXBX, blocksize, blocksize, block_width, block_id);
                }
                else if(task_id == 21)
                {
                    custom_dlacpy_task_exe(newX, blockVectorX, M, blocksize, block_width, block_id);   
                }
                else if(task_id == 22)
                {
                    custom_dlacpy_task_exe(newX, blockVectorX, M, blocksize, block_width, block_id);   
                }
                else if(task_id == 23)
                {
                    custom_dlacpy_task_exe(newAX, blockVectorAX, M, blocksize, block_width, block_id);   
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 5)  //XY
            {
                // block_id = atoi(splitParams[1]);
                // task_id = atoi(splitParams[2]);
                block_id = taskInfo_nonLoop[structIterator].numParamsList[0];
                task_id = taskInfo_nonLoop[structIterator].taskID;
                
                if(task_id == 20)
                {
                    _XY_exe(blockVectorX, tempGramXBX, newX, M, blocksize, blocksize, block_width, block_id);
                }
                else if(task_id == 21)
                {
                    _XY_exe(blockVectorX, gramXAX, newX, M, blocksize, blocksize, block_width, block_id);
                }
                else if(task_id == 22)
                {
                    _XY_exe(blockVectorAX, gramXAX, newAX, M, blocksize, blocksize, block_width, block_id);
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 2) //SPMM 
            {
                // if(tokenCount > 4)
                // {
                //     spmmTaskCounter += (tokenCount - 1)/3 ;
                    
                //         for(sp = 1 ; sp < tokenCount ; sp += 3) //SPMM,1,1,1,2,2,2,3,3,3 
                //         {
                //             row_id = atoi(splitParams[sp]);
                //             col_id = atoi(splitParams[sp + 1]); 
                //             buf_id = atoi(splitParams[sp + 2]);
                //             spmm_blkcoord_finegrained_exe_fixed_buf<double>(numrows, numcols, blocksize, nbuf, blockVectorX, blockVectorAX, matrixBlock, row_id, col_id, buf_id, block_width);
                //             spmmTaskCounter_2++;
                //         }
                // }
                // else
                // {
                        // row_id = atoi(splitParams[1]);
                        // col_id = atoi(splitParams[2]); 
                        // buf_id = atoi(splitParams[3]);
                        
                        row_id = taskInfo_nonLoop[structIterator].numParamsList[0];
                        col_id = taskInfo_nonLoop[structIterator].numParamsList[1]; 
                        buf_id = taskInfo_nonLoop[structIterator].numParamsList[2];
                        spmm_blkcoord_finegrained_exe_fixed_buf(numrows, blocksize, blockVectorX, blockVectorAX, matrixBlock, row_id, col_id, block_width);
                        //spmmTaskCounter++;
                        //spmmTaskCounter_2++;
                //}
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 19)  //SETZERO
            {
                // block_id = atoi(splitParams[1]);
                // task_id = atoi(splitParams[2]);
                
                block_id = taskInfo_nonLoop[structIterator].numParamsList[0];
                task_id = taskInfo_nonLoop[structIterator].taskID;


                if(task_id == 1)
                {
                    i = block_id * block_width; // starting point of the block
                    blksz = block_width;
                    if(i + blksz > M)
                        blksz = M - i;
                    
                    //code: 0    
                    #pragma omp task default(shared) private(j, k, tid, tstart, tend)\
                    firstprivate(blockVectorAX, blksz, i, M, block_width, blocksize)\
                    depend(in :  M, blocksize)\
                    depend(inout : blockVectorAX[i * blocksize : blksz * blocksize])
                    {
                        //tid = omp_get_thread_num();
                        //tstart = omp_get_wtime();

                        for(j = i ; j < i + blksz ; j++)
                        {
                            for(k = 0 ; k < blocksize ; k++)
                            {
                                blockVectorAX[j * blocksize + k] = 0.0;
                            }
                        }

                        //tend = omp_get_wtime();
                        //taskTiming[0][tid] += (tend - tstart);
                    } //end task
                    
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 17)  //CHOL
            {
                if(!strcmp(taskInfo_nonLoop[structIterator].strParamsList[0], "XBX"))
                {

                    #pragma omp task private(i, j) firstprivate(blocksize, gramXBX, uplo)\
                    depend(inout: gramXBX[0 : blocksize * blocksize], info)
                    {
                        double *trans_gramXBX = (double *) malloc(blocksize * blocksize * sizeof(double));

                        transpose(gramXBX, trans_gramXBX, blocksize, blocksize);
                        dpotrf_(&uplo, &blocksize, trans_gramXBX, &blocksize, &info);

                        if(info != 0)
                        {
                          cout << "dpotrf: chol error!" << endl;
                        }

                        //printf("CHOL info: %d\n", info);

                        transpose(trans_gramXBX, gramXBX, blocksize, blocksize);
                        free(trans_gramXBX); 
                        
                        for(i = 0 ; i < blocksize ; i++)
                        {
                            for(j = 0 ; j < i ; j++)
                            {
                                gramXBX[i * blocksize + j] = 0.0;
                            }
                        }
                    }
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 18)  //INV
            {
                //printf("%s %s\n", splitParams[0], splitParams[1]);

                if(!strcmp(taskInfo_nonLoop[structIterator].strParamsList[0], "XBX"))
                {
                    //printf("out %s %s\n", splitParams[0], splitParams[1]);
                    //code 12
                    #pragma omp task default(shared) private(tid, tstart, tend)\
                    firstprivate(tempGramXBX, blocksize)\
                    depend(in: info, blocksize)\
                    depend(inout: tempGramXBX[0 : blocksize * blocksize])
                    {
                        //printf("INV info: %d\n", info);
                        if(info == 0) //we don't need to check it here, because if CHOL,XBX fails, it will print an error msg
                        {
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            inverse(tempGramXBX, blocksize, blocksize);
                            //tend = omp_get_wtime();
                            //taskTiming[12][tid] += (tend - tstart);
                        }
                        else
                        {
                            printf("INV,XBX info: %d\n", info);
                        }
                    } //end task
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 16)  //SPEUPDATE
            {
                if(!strcmp(taskInfo_nonLoop[structIterator].strParamsList[0], "XAX"))
                {
                    #pragma omp task firstprivate(identity_BB, gramXAX, blocksize)\
                    depend(in: identity_BB[0 : blocksize * blocksize])\
                    depend(inout: gramXAX[0 : blocksize * blocksize])
                    {
                        double *transGramXAX = (double *)malloc(blocksize * blocksize * sizeof(double)); 
                        transpose(gramXAX, transGramXAX, blocksize, blocksize);
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blocksize, blocksize, blocksize, 0.5, transGramXAX, blocksize, identity_BB, blocksize, 0.5, gramXAX, blocksize);
                        free(transGramXAX);
                    }
                }
            }
            else if(taskInfo_nonLoop[structIterator].opCode == 21) // task name without , in it ==> opCode = 21 
            {
                if(!strcmp(taskInfo_nonLoop[structIterator].strParamsList[0], "EIGEN")) 
                {
                    #pragma omp task firstprivate(lwork, uplo, jobz, itype, work_query, work, blocksize, gramXAX, lambda, identity_BB)\
                    depend(in: identity_BB[0 : blocksize * blocksize], info)\
                    depend(inout: gramXAX[0 : blocksize * blocksize])\
                    depend(inout: lambda[0 : blocksize * blocksize])
                    {    
                        double *temp_GramXAX = (double *)malloc(blocksize * blocksize * sizeof(double)); 
                        transpose(gramXAX, temp_GramXAX, blocksize, blocksize);

                        //dummy call: work size query
                        lwork = -1;
                        
                        double *tempLambda = (double *) malloc(blocksize * sizeof(double)); 
                        dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, &work_query, &lwork, &info);
                       
                        if(info != 0)
                        {
                          cout << "Error in dummy call" << endl;
                        }

                        lwork = (int) work_query;
                        work = (double *) malloc(lwork * sizeof(double));
                        
                        dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, work, &lwork, &info);
                        
                        if( info !=0 )
                        {
                            printf( "The algorithm failed to compute eigenvalues.\n" );
                        }
                        
                        transpose(temp_GramXAX, gramXAX, blocksize, blocksize);
                        free(temp_GramXAX);
                        

                        //[coordX,gramXAX]=eig(gramXAX,eye(blockSize));
                        //lambda=diag(gramXAX);
                        
                        diag(tempLambda, lambda, blocksize);
                        free(tempLambda);
                    } //end task
                }
            } //end 21
        //}//end if (2nd)

    } //end for
    #pragma omp taskwait
    } //end master
    } //end parallel

    free(taskInfo_nonLoop);
    /*printf("After nonloop part \n");
    for(i = 0 ; i < blocksize ; i++)
    {
        //cout << lambda[i * blocksize + i] << endl;
        printf("%.4lf ", lambda[i * blocksize + i]);
    }
    printf("\n");*/


    free(gramXAX); 
    double *coordX;
    
    double *coordX1 = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *coordX2 = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *coordX3 = (double *) malloc(blocksize * blocksize * sizeof(double));

    int gramASize = -1;
    int *activeMask = (int *) malloc(blocksize * sizeof(int)); 

    
    #pragma omp parallel for default(shared)
    for(i = 0 ; i < blocksize ; i++)
        activeMask[i] = 1;


     //reseting taskTiming
    // for(i = 0 ; i < numOperation ; i++)
    //     for(j = 0 ; j < nthrds ; j++)
    //         taskTiming[i][j] = 0.0;

    //finish upto iteration loop here

    
    //iteraton_start_time = omp_get_wtime();

    int activeRSize = -1, activePSize = -1, explicitGramFlag = 0, restart = -1;
    

    //first loop task-based starts here

    /* resetting various buffer before jumping into execution of first loop*/
    /*for(i = 0 ; i < nthrds * blocksize ; i++)
    {
        RNBUF[i] = 0.0;
    }

    for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
    {
        temp2XTYBUF[i] = 0.0;
        gramXARXTYBUF[i] = 0.0;
        gramXAPXTYBUF[i] = 0.0;
        gramXBPXTYBUF[i] = 0.0;
    }
    for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
    {
        gramRBRXTYBUF[i] = 0.0;
        gramPBPXTYBUF[i] = 0.0;
        gramRBPXTYBUF[i] = 0.0;
        gramPAPXTYBUF[i] = 0.0;
        gramRARXTYBUF[i] = 0.0;
        gramRAPXTYBUF[i] = 0.0;
    }*/
    
    //opening partition file
    // ifstream firstLoopFile(argv[5]);
    
    // if(firstLoopFile.fail())
    // {
    //     printf("First-loop File doesn't exist\n");
    // }
   
    //printf("argv[5]: %s\n", argv[5]);

    iterationNumber = 1;

    //printf("iterationNumber: %d starts..\n", iterationNumber);

    #pragma omp parallel
    {
    #pragma omp master
    {
        loop_start_time = omp_get_wtime();

#ifdef EPYC
        #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, blocksize, RNBUF)\
            depend(in: nthrds, blocksize)\
            depend(inout: \
            RNBUF[0 * blocksize : blocksize], RNBUF[1 * blocksize : blocksize],\
            RNBUF[2 * blocksize : blocksize], RNBUF[3 * blocksize : blocksize],\
            RNBUF[4 * blocksize : blocksize], RNBUF[5 * blocksize : blocksize],\
            RNBUF[6 * blocksize : blocksize], RNBUF[7 * blocksize : blocksize],\
            RNBUF[8 * blocksize : blocksize], RNBUF[9 * blocksize : blocksize],\
            RNBUF[10 * blocksize : blocksize], RNBUF[11 * blocksize : blocksize],\
            RNBUF[12 * blocksize : blocksize], RNBUF[13 * blocksize : blocksize],\
            RNBUF[14 * blocksize : blocksize], RNBUF[15 * blocksize : blocksize],\
            RNBUF[16 * blocksize : blocksize], RNBUF[17 * blocksize : blocksize],\
            RNBUF[18 * blocksize : blocksize], RNBUF[19 * blocksize : blocksize],\
            RNBUF[20 * blocksize : blocksize], RNBUF[21 * blocksize : blocksize],\
            RNBUF[22 * blocksize : blocksize], RNBUF[23 * blocksize : blocksize],\
            RNBUF[24 * blocksize : blocksize], RNBUF[25 * blocksize : blocksize],\
            RNBUF[26 * blocksize : blocksize], RNBUF[27 * blocksize : blocksize],\
            RNBUF[28 * blocksize : blocksize], RNBUF[29 * blocksize : blocksize],\
            RNBUF[30 * blocksize : blocksize], RNBUF[31 * blocksize : blocksize],\
            RNBUF[32 * blocksize : blocksize], RNBUF[33 * blocksize : blocksize],\
            RNBUF[34 * blocksize : blocksize], RNBUF[35 * blocksize : blocksize],\
            RNBUF[36 * blocksize : blocksize], RNBUF[37 * blocksize : blocksize],\
            RNBUF[38 * blocksize : blocksize], RNBUF[39 * blocksize : blocksize],\
            RNBUF[40 * blocksize : blocksize], RNBUF[41 * blocksize : blocksize],\
            RNBUF[42 * blocksize : blocksize], RNBUF[43 * blocksize : blocksize],\
            RNBUF[44 * blocksize : blocksize], RNBUF[45 * blocksize : blocksize],\
            RNBUF[46 * blocksize : blocksize], RNBUF[47 * blocksize : blocksize],\
            RNBUF[48 * blocksize : blocksize], RNBUF[49 * blocksize : blocksize],\
            RNBUF[50 * blocksize : blocksize], RNBUF[51 * blocksize : blocksize],\
            RNBUF[52 * blocksize : blocksize], RNBUF[53 * blocksize : blocksize],\
            RNBUF[54 * blocksize : blocksize], RNBUF[55 * blocksize : blocksize],\
            RNBUF[56 * blocksize : blocksize], RNBUF[57 * blocksize : blocksize],\
            RNBUF[58 * blocksize : blocksize], RNBUF[59 * blocksize : blocksize],\
            RNBUF[60 * blocksize : blocksize], RNBUF[61 * blocksize : blocksize],\
            RNBUF[62 * blocksize : blocksize], RNBUF[63 * blocksize : blocksize],\
            RNBUF[64 * blocksize : blocksize], RNBUF[65 * blocksize : blocksize],\
            RNBUF[66 * blocksize : blocksize], RNBUF[67 * blocksize : blocksize],\
            RNBUF[68 * blocksize : blocksize], RNBUF[69 * blocksize : blocksize],\
            RNBUF[70 * blocksize : blocksize], RNBUF[71 * blocksize : blocksize],\
            RNBUF[72 * blocksize : blocksize], RNBUF[73 * blocksize : blocksize],\
            RNBUF[74 * blocksize : blocksize], RNBUF[75 * blocksize : blocksize],\
            RNBUF[76 * blocksize : blocksize], RNBUF[77 * blocksize : blocksize],\
            RNBUF[78 * blocksize : blocksize], RNBUF[79 * blocksize : blocksize],\
            RNBUF[80 * blocksize : blocksize], RNBUF[81 * blocksize : blocksize],\
            RNBUF[82 * blocksize : blocksize], RNBUF[83 * blocksize : blocksize],\
            RNBUF[84 * blocksize : blocksize], RNBUF[85 * blocksize : blocksize],\
            RNBUF[86 * blocksize : blocksize], RNBUF[87 * blocksize : blocksize],\
            RNBUF[88 * blocksize : blocksize], RNBUF[89 * blocksize : blocksize],\
            RNBUF[90 * blocksize : blocksize], RNBUF[91 * blocksize : blocksize],\
            RNBUF[92 * blocksize : blocksize], RNBUF[93 * blocksize : blocksize],\
            RNBUF[94 * blocksize : blocksize], RNBUF[95 * blocksize : blocksize],\
            RNBUF[96 * blocksize : blocksize], RNBUF[97 * blocksize : blocksize],\
            RNBUF[98 * blocksize : blocksize], RNBUF[99 * blocksize : blocksize],\
            RNBUF[100 * blocksize : blocksize], RNBUF[101 * blocksize : blocksize],\
            RNBUF[102 * blocksize : blocksize], RNBUF[103 * blocksize : blocksize],\
            RNBUF[104 * blocksize : blocksize], RNBUF[105 * blocksize : blocksize],\
            RNBUF[106 * blocksize : blocksize], RNBUF[107 * blocksize : blocksize],\
            RNBUF[108 * blocksize : blocksize], RNBUF[109 * blocksize : blocksize],\
            RNBUF[110 * blocksize : blocksize], RNBUF[111 * blocksize : blocksize],\
            RNBUF[112 * blocksize : blocksize], RNBUF[113 * blocksize : blocksize],\
            RNBUF[114 * blocksize : blocksize], RNBUF[115 * blocksize : blocksize],\
            RNBUF[116 * blocksize : blocksize], RNBUF[117 * blocksize : blocksize],\
            RNBUF[118 * blocksize : blocksize], RNBUF[119 * blocksize : blocksize],\
            RNBUF[120 * blocksize : blocksize], RNBUF[121 * blocksize : blocksize],\
            RNBUF[122 * blocksize : blocksize], RNBUF[123 * blocksize : blocksize],\
            RNBUF[124 * blocksize : blocksize], RNBUF[125 * blocksize : blocksize],\
            RNBUF[126 * blocksize : blocksize], RNBUF[127 * blocksize : blocksize])
#else
        #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, blocksize, RNBUF)\
            depend(in: nthrds, blocksize)\
            depend(inout: \
            RNBUF[0 * blocksize : blocksize], RNBUF[1 * blocksize : blocksize],\
            RNBUF[2 * blocksize : blocksize], RNBUF[3 * blocksize : blocksize],\
            RNBUF[4 * blocksize : blocksize], RNBUF[5 * blocksize : blocksize],\
            RNBUF[6 * blocksize : blocksize], RNBUF[7 * blocksize : blocksize],\
            RNBUF[8 * blocksize : blocksize], RNBUF[9 * blocksize : blocksize],\
            RNBUF[10 * blocksize : blocksize], RNBUF[11 * blocksize : blocksize],\
            RNBUF[12 * blocksize : blocksize], RNBUF[13 * blocksize : blocksize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize ; i++)
                {
                    RNBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }
            //b * cb
            //temp2XTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, temp2XTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            temp2XTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, temp2XTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: temp2XTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    temp2XTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramXARXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXARXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            gramXARXTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXARXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: gramXARXTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    gramXARXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramXAPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXAPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            gramXAPXTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXAPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: gramXAPXTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    gramXAPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }
            //gramXBPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXBPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            gramXBPXTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXBPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: gramXBPXTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif

            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    gramXBPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }


            //cb * cb
            //gramRBRXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBRXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRBRXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBRXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRBRXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRBRXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramPBPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramPBPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramPBPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramPBPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramRBPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRBPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRBPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRBPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramPAPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramPAPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramPAPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramPAPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramRARXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRARXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRARXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRARXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRARXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRARXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramRAPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRAPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRAPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRAPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

        ///// Execution
        for(structIterator = 0 ; structIterator < taskCount_firstLoop; structIterator++)
        {
            /* taskName starts with _ means it is a dummy task */
            if(taskInfo_firstLoop[structIterator].opCode == 22) 
            {
                //taskCounter++;
                //continue;
            }

            /* if the taskName has more than or equal 1 tokens */
            //tokenCount = split(taskName, ',', &splitParams);
            
            //all real tasks goes here
            //if(tokenCount > 0 && taskName[0] != '_') 
            //{
                else if(taskInfo_firstLoop[structIterator].opCode == 1) // taskName starts RESET 
                {
                    if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "actMask")) //reseting activeMask
                    {
                        //printf("update_activeMask_task_exe\n");
                        update_activeMask_task_exe(activeMask, residualNorms, residualTolerance, blocksize);
                    }
                    else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RN")) //reseting residualNorms
                    {
                        //printf("RESET,RN");
                        //printf("%s\n", splitParams[1]);
                        #pragma omp task private(i) firstprivate(blocksize, residualNorms)\
                        depend(inout: residualNorms[0 : blocksize])
                        {
                            for(i = 0; i < blocksize; i++)
                                residualNorms[i] = 0.0;
                        }
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 2) //SPMM
                {
                    //printf("partNo: %d tokenCount: %d\n", partNo, tokenCount);
                    // if(tokenCount > 4)
                    // {
                    //     for(sp = 1 ; sp < tokenCount ; sp += 3) //SPMM,1,1,1,2,2,2,3,3,3 
                    //     {
                    //         row_id = atoi(splitParams[sp]);
                    //         col_id = atoi(splitParams[sp + 1]); 
                    //         buf_id = atoi(splitParams[sp + 2]);
                    //         spmm_blkcoord_finegrained_exe_fixed_buf<double>(numrows, numcols, currentBlockSize, nbuf, activeBlockVectorR, activeBlockVectorAR, matrixBlock, row_id, col_id, buf_id, block_width);
                    //     }
                    // }
                    // else
                    // {
                        // row_id = atoi(splitParams[1]);
                        // col_id = atoi(splitParams[2]); 
                        // buf_id = atoi(splitParams[3]);

                        row_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                        col_id = taskInfo_firstLoop[structIterator].numParamsList[1]; 
                        buf_id = taskInfo_firstLoop[structIterator].numParamsList[2];
                        spmm_blkcoord_finegrained_exe_fixed_buf(numrows, currentBlockSize, activeBlockVectorR, activeBlockVectorAR, matrixBlock, row_id, col_id, block_width);
                    //}
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 3) // taskName starts XTY 
                {
                    // block_id = atoi(splitParams[1]);
                    // buf_id = atoi(splitParams[2]);
                    // task_id = atoi(splitParams[3]);

                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                    buf_id = taskInfo_firstLoop[structIterator].numParamsList[1]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;
                    //printf("XTY %d -> %d -> %d\n", block_id, buf_id, task_id);
                    if(task_id == 1)
                    {
                        _XTY_v1_exe(blockVectorX, activeBlockVectorR, temp2XTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id);
                    }
                    else if(task_id == 2)
                    {
                        _XTY_v1_exe(activeBlockVectorR, activeBlockVectorR, gramRBRXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                    }
                     else if(task_id == 4)
                    {
                        _XTY_v1_exe(blockVectorAX, activeBlockVectorR, gramXARXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id);
                    }
                    else if(task_id == 5)
                    {
                        _XTY_v1_exe(activeBlockVectorAR, activeBlockVectorR, gramRARXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 4) // RED
                {
                    //printf("%s %s -> %s \n", taskName, splitParams[0], splitParams[1]);
                    if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "temp2BUF")) //xty 1 reduction
                    {
                        _XTY_v1_RED(temp2XTYBUF, temp2, blocksize, currentBlockSize, block_width);
                    }
                    else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RBRBUF")) //xty 2 reduction
                    {
                        _XTY_v1_RED(gramRBRXTYBUF, gramRBR, currentBlockSize, currentBlockSize, block_width);
                    }
                    else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "XARBUF")) //xty 4 reduction
                    {
                        _XTY_v1_RED(gramXARXTYBUF, gramXAR, blocksize, currentBlockSize, block_width);
                    }
                    else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RARBUF")) //xty 5 reduction
                    {
                        _XTY_v1_RED(gramRARXTYBUF, gramRAR, currentBlockSize, currentBlockSize, block_width);
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 5) // taskName starts XY 
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                    task_id = taskInfo_firstLoop[structIterator].taskID; 
                    
                    if(task_id == 1)
                    {
                        _XY_exe(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width, block_id);
                    }
                    else if(task_id == 2)
                    {
                        _XY_exe(blockVectorX, temp2, temp3_R, M, blocksize, currentBlockSize, block_width, block_id);
                    }
                    else if(task_id == 3)
                    {
                        _XY_exe(activeBlockVectorR, gramRBR, temp3_R, M, currentBlockSize, currentBlockSize, block_width, block_id);
                    }
                    else if(task_id == 6)
                    {
                        //printf("%s\n", taskName);
                        //_XY_exe(activeBlockVectorR, coordX+(blocksize*blocksize), blockVectorP, M, currentBlockSize, blocksize, block_width, block_id);
                        _XY_exe(activeBlockVectorR, coordX2, blockVectorP, M, currentBlockSize, blocksize, block_width, block_id);
                    }
                    else if(task_id == 8)
                    {
                        //_XY_exe(activeBlockVectorAR, coordX+(blocksize * blocksize), blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id);
                        _XY_exe(activeBlockVectorAR, coordX2, blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id);
                    }
                     else if(task_id == 10)
                    {
                        //_XY_exe(blockVectorX, coordX, newX, M, blocksize, blocksize, block_width, block_id);
                        _XY_exe(blockVectorX, coordX1, newX, M, blocksize, blocksize, block_width, block_id);
                    }
                    else if(task_id == 11)
                    {
                        //_XY_exe(blockVectorAX, coordX, newAX, M, blocksize, blocksize, block_width, block_id);
                        _XY_exe(blockVectorAX, coordX1, newAX, M, blocksize, blocksize, block_width, block_id);
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 6) // ADD
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);

                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;
                    
                    if(task_id == 3)
                    {
                        mat_addition_task_exe(newX, blockVectorP, blockVectorX, M, blocksize, block_width, block_id);
                    }
                    else if(task_id == 4)
                    {
                        mat_addition_task_exe(newAX, blockVectorAP, blockVectorAX, M, blocksize, block_width, block_id);
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 9) //SUB
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;

                    if(task_id == 1)
                    {
                        mat_sub_task_exe(blockVectorAX, blockVectorR, blockVectorR , M, blocksize, block_width, block_id);
                    }
                    else if(task_id == 2)
                    {
                        mat_sub_task_exe(activeBlockVectorR, temp3_R, activeBlockVectorR, M, currentBlockSize, block_width, block_id);   
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 10) //MULT
                {
                    //block_id = atoi(splitParams[1]);
                    /* only ONE MULT op, so no task_id */
                    //printf("%s -> %d\n", taskName, block_id);
                    
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0];

                    mat_mult_task_exe(blockVectorR, blockVectorR, newX, M, blocksize, block_width, block_id);
                    
                    //#pragma omp taskwait
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 26) //DOT
                {
                    //block_id = atoi(splitParams[1]);
                    /* only ONE MULT op, so no task_id */
                    //printf("%s -> %d\n", taskName, block_id);
                    
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                    buf_id = taskInfo_firstLoop[structIterator].numParamsList[1]; 

                    //mat_mult_task_exe(blockVectorR, blockVectorR, newX, M, blocksize, block_width, block_id);
                    dot_mm_exe(blockVectorR, blockVectorR, RNBUF, M, blocksize, block_width, block_id, buf_id);
                    //#pragma omp taskwait
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 11) // COL 
                {
                    // block_id = atoi(splitParams[1]);
                    // buf_id = atoi(splitParams[2]);
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                    buf_id = taskInfo_firstLoop[structIterator].numParamsList[1];
                    sum_sqrt_task_COL(newX, M, blocksize, block_width, block_id, buf_id, RNBUF);
                    //#pragma omp taskwait
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 12) /* taskName starts RNRED */
                {
                    //printf("sum_sqrt_task_RNRED\n");
                    sum_sqrt_task_RNRED(RNBUF, residualNorms, blocksize);
                    //#pragma omp taskwait
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 13) /* taskName starts SQRT */
                {
                    //printf("sum_sqrt_task_SQRT\n");
                    sum_sqrt_task_SQRT(residualNorms, blocksize);
                    
                    //#pragma omp taskwait
                }
                 else if(taskInfo_firstLoop[structIterator].opCode == 7) /* taskName starts DLACPY */
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;

                    if(task_id == 1)
                    {
                        custom_dlacpy_task_exe(temp3_R, activeBlockVectorR, M, currentBlockSize, block_width, block_id);
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 14) /* taskName starts GET */
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;

                    if(task_id == 1)
                    {
                        getActiveBlockVector_task_exe(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id); 
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 8) //UPDATE
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);

                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;

                    if(task_id == 1)
                    {
                        updateBlockVector_task_exe(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id);
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 17) //CHOL
                {
                    //code: 22
                    if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RBR"))
                    {
                        #pragma omp task private(i, j, tid, tstart, tend)\
                        firstprivate(gramRBR, trans_gramRBR) default(shared)\
                        depend(in: currentBlockSize, trans_gramRBR[0 : currentBlockSize * currentBlockSize])\
                        depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize]) depend(out: info)
                        {
                            /* OP: [gramRBR,cholFlag] = chol(gramRBR); */
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();
                            
                            transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);

                            //tend = omp_get_wtime();
                            //taskTiming[tid][9] += (tend - tstart);
                            //tstart = omp_get_wtime();
                            
                            dpotrf_( &uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info );
                            if(info != 0)
                            {
                                cout<<"dportf_ error 2!!"<<endl;
                            }

                            //taskTiming[tid][6] += (omp_get_wtime() - tstart);
                            //tstart = omp_get_wtime();

                            transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);

                            //tend = omp_get_wtime();
                            //taskTiming[tid][9] += (tend - tstart);
                            //tstart = omp_get_wtime();
                            
                            for(i = 0 ; i < currentBlockSize ; i++)
                            {
                                for(j = 0 ; j < i ; j++)
                                {
                                    gramRBR[i * currentBlockSize + j] = 0.0;
                                }
                            }
                            
                            //tend = omp_get_wtime();
                            //taskTiming[22][tid] += (tend - tstart);
                        }//end task
                        
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 18) //INV
                {
                    //printf("INV: %s\n", splitParams[1]);
                    //string invPRB = splitParams[1];
                    //strcpy(splitParams[1], invPRB.substr(0, 3).c_str());

                    if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RBR"))
                    {
                        //code 12
                        #pragma omp task default(shared) private(tid, tstart, tend)\
                        firstprivate(gramRBR, currentBlockSize)\
                        depend(in: info, currentBlockSize)\
                        depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize])
                        {
                            if(info == 0)
                            {
                                /* OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask)/gramRBR; */
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                inverse(gramRBR, currentBlockSize, currentBlockSize);
                                
                                //tend = omp_get_wtime();
                                //taskTiming[12][tid] += (tend - tstart);
                            }
                            else
                            {
                                printf("INV,RBR info: %d\n", info);
                            }
                        } //end task
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 19) /* taskName starts SETZERO */
                {
                    // block_id = atoi(splitParams[1]);
                    // task_id = atoi(splitParams[2]);
                    block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_firstLoop[structIterator].taskID;

                    if(task_id == 1)
                    {
                        i = block_id * block_width; // starting point of the block
                        blksz = block_width;
                        if(i + blksz > M)
                            blksz = M - i;
                        
                        //code: 0    
                        #pragma omp task default(shared) private(j, k, tid, tstart, tend)\
                        firstprivate(activeBlockVectorAR, blksz, i, M, block_width, blocksize, currentBlockSize)\
                        depend(in :  M, blocksize, currentBlockSize)\
                        depend(out : activeBlockVectorAR[i * currentBlockSize : blksz * currentBlockSize])
                        {
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            for(j = i ; j < i + blksz ; j++)
                            {
                                for(k = 0 ; k < currentBlockSize ; k++)
                                {
                                    activeBlockVectorAR[j * currentBlockSize + k] = 0.0;
                                }
                            }

                            //tend = omp_get_wtime();
                            //taskTiming[0][tid] += (tend - tstart);
                        } //end task
                        
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 15) //TRANS
                {
                    #pragma omp task default(shared) private(tid, tstart, tend)\
                    firstprivate(gramRAR, transGramRAR)\
                    depend(in: gramRAR[0 : currentBlockSize * currentBlockSize], currentBlockSize)\
                    depend(inout: transGramRAR[0 : currentBlockSize * currentBlockSize])
                    {
                        //transGramRAR(0)_TRANS(gramRAR)
                        //tid = omp_get_thread_num();
                        //tstart = omp_get_wtime();

                        transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);

                        //tend = omp_get_wtime();
                        //taskTiming[23][tid] += (tend - tstart);
                    } //end task
                    //#pragma omp taskwait
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 16) //SPEUPDATE
                {
                    //code: 13 (update)
                    if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RAR"))
                    {
                        #pragma omp task default(shared) private(tid, tstart, tend)\
                        firstprivate(identity_PAP, transGramRAR, gramRAR)\
                        depend(in : currentBlockSize)\
                        depend(inout : identity_PAP[0 : currentBlockSize * currentBlockSize])\
                        depend(in : transGramRAR[0 : currentBlockSize * currentBlockSize])\
                        depend(inout : gramRAR[0 : currentBlockSize * currentBlockSize])
                        {
                            //gramRAR(0)_UPDATE(gramRAR,transGramRAR) SPEUPDATE
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            //make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize);
                            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 
                                        0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);

                            //tend = omp_get_wtime();
                            //taskTiming[13][tid] += (tend - tstart);
                        } //end task
                        //#pragma omp taskwait
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 21)
                {
                    if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "CONSTRUCTGB"))
                    {
                        gramASize = blocksize + blocksize;
                        //printf("GB 1 gramASize: %d currentBlockSize: %d\n", gramASize, currentBlockSize);
                        //printf("GB 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                        
                        gramB = (double *) malloc(gramASize * gramASize * sizeof(double));
                        
                        #pragma omp task private(tid, tstart, tend)\
                        firstprivate(gramASize, gramB, currentBlockSize)\
                        depend(in: activeRSize, activePSize, restart, gramASize, currentBlockSize)\
                        depend(in: activeMask[0 : blocksize])\
                        depend(out: gramB[0 : gramASize * gramASize])
                        {
                            //printf("GB 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);

                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            //printf("GB gramASize: %d\n", gramASize);
                            make_identity_mat(gramB, gramASize, gramASize);

                            //tend = omp_get_wtime();
                            //taskTiming[19][tid] += (tend - tstart);
                        }
                    }
                    else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "CONSTRUCTGA"))
                    {
                        gramASize = blocksize + blocksize;
                        //printf("GA 1 gramASize: %d\n", gramASize);
                        gramA = (double *) malloc(gramASize * gramASize * sizeof(double));
                        coordX = (double *) malloc(gramASize * blocksize * sizeof(double)); 

                        #pragma omp task private(tid, tstart, tend)\
                        firstprivate(gramASize, gramB, currentBlockSize)\
                        depend(in: activeRSize, activePSize, restart, gramASize, currentBlockSize)\
                        depend(in: gramXAR[0: blocksize * currentBlockSize], lambda[0 : blocksize * blocksize])\
                        depend(in: transGramXAR[0 : currentBlockSize * blocksize], gramRAR[0 : currentBlockSize * currentBlockSize])\
                        depend(in: activeMask[0 : blocksize])\
                        depend(out: gramA[0 : gramASize * gramASize])
                        {
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            //printf("GA 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                            mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                            mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                            transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                            mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                            mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);

                            //tend = omp_get_wtime();
                            //taskTiming[17][tid] += (tend - tstart);
                        }

                    }
                    else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "EIGEN"))
                    {
                        //code: 20
                        //eigen value computation here
                        #pragma omp task default(shared) private(i, j, k, tid, tstart, tend)\
                        firstprivate(gramASize, lambda, gramA, gramB, coordX, coordX1, currentBlockSize)\
                        depend(inout: gramA[0 : gramASize * gramASize])\
                        depend(in: gramB[0 : gramASize * gramASize], gramASize, currentBlockSize)\
                        depend(out: coordX1[0 : blocksize * blocksize], coordX2[0 : currentBlockSize * blocksize], lambda[0 : blocksize * blocksize], activeRSize, blocksize)
                        //depend(out: coordX[0 : gramASize * blocksize], lambda[0 : blocksize * blocksize], activeRSize, blocksize)
                        {
                            //printf("EIG 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                            
                            //EIGEN
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();
                            
                            eigen_value = (double *) malloc(gramASize * sizeof(double));
                            double *trans_gramA = (double *) malloc(gramASize * gramASize * sizeof(double)); 
                            double *trans_gramB = (double *) malloc(gramASize * gramASize * sizeof(double)); 
                            
                            transpose(gramA, trans_gramA, gramASize, gramASize);
                            transpose(gramB, trans_gramB, gramASize, gramASize);
                            
                                //dummy call for lwork
                            lwork = -1;
                            
                            dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, &work_query, &lwork, &info);
                            
                                //cout<<"work_query: "<<work_query<<endl;
                            if(info != 0)
                                cout<<"Error in dummy call"<<endl;

                            lwork = (int) work_query;
                            work = (double *) malloc(lwork * sizeof(double)); 

                            dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, work, &lwork, &info);

                            if(info != 0)
                                cout<<"Error in eigen value calculation"<<endl;
                            transpose(trans_gramA, gramA, gramASize, gramASize);
                            transpose(trans_gramB, gramB, gramASize, gramASize);

                            free(trans_gramA); 
                            free(trans_gramB); 
                            free(work); 

                                //info=LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, gramASize, gramA, gramASize, gramB, gramASize, eigen_value);

                                //std::cout << "task 32" << '\n';

                                //briging following task here
                            diag(eigen_value, lambda, blocksize);
                            
                            /*int column = 0;
                            column = 0;
                            for(j = 0 ; column < blocksize && j < gramASize; j++)
                            {
                                for(i = 0 ; i < gramASize ; i++)
                                {
                                    coordX[i * blocksize + column] = gramA[i * gramASize + j];
                                }
                                column++;
                            }*/

                            //coordX1
                            for(i = 0 ; i < blocksize ; i++)
                            {
                                for(j = 0 ; j < blocksize ; j++)
                                {
                                    coordX1[i * blocksize + j] = gramA[i * gramASize + j];
                                }
                            }

                            //coordX2
                            for(k = 0, i = blocksize ; k < currentBlockSize && i < blocksize + currentBlockSize ; i++, k++)
                            {
                                for(j = 0 ; j < blocksize ; j++)
                                {
                                    coordX2[k * blocksize + j] = gramA[i * gramASize + j];
                                }
                            }

                            /*printf("printing coordX1: \n");
                            print_mat(coordX1, blocksize, blocksize);

                            printf("printing coordX: \n");
                            print_mat(coordX, gramASize, blocksize);

                            printf("printing coordX2: \n");
                            print_mat(coordX2, currentBlockSize, blocksize);*/

                            //tend = omp_get_wtime();
                            //taskTiming[20][tid] += (tend - tstart);
                            

                        } //end task
                        //#pragma omp taskwait
                    }
                }
                else if(taskInfo_firstLoop[structIterator].opCode == 20) /* taskName starts CONV */
                {
                    //code: 8
                    #pragma omp task default(shared) private(tid, tstart, tend)\
                    firstprivate(blocksize, activeMask,residualNorms, iterationNumber)\
                    depend(inout: activeMask[0 : blocksize], currentBlockSize, convergeFlag)\
                    depend(in: residualNorms[0 : blocksize])\
                    depend(out: activeRSize, explicitGramFlag, activePSize, restart)
                    {   
                        //tid = omp_get_thread_num();
                        //tstart = omp_get_wtime();

                        //currentBlockSize = sum(activeMask);
                        //printf("* CONV => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                        int tempCB = 0;
                        for(i = 0 ; i < blocksize ; i++)
                            tempCB += activeMask[i];

                        currentBlockSize = tempCB;
                        //if 14
                        if(currentBlockSize == 0)
                        {
                            convergeFlag = 1;
                            printf("converge!! convergeFlag: %d", convergeFlag);
                        }
                        activeRSize = currentBlockSize;
                        activePSize = 0;
                        restart = 1;
                        //printf("CONV => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                        //tend = omp_get_wtime();
                        //taskTiming[8][tid] += (tend - tstart);

                        int flag = 1;
                        for(i = 0 ; i < blocksize ; i++)
                        {
                            if(residualNorms[i] < 4.0538e-10)
                            {
                                flag = 0;
                                //break;
                            }
                        }
                        if(flag == 0 )
                            explicitGramFlag = 1;
                        else
                            explicitGramFlag = 0;


                        //tend = omp_get_wtime();
                        //taskTiming[8][tid] += (tend - tstart);
                    } //end task
                }
            //} //end if
        }//endl while
        loopTime[iterationNumber - 1] = omp_get_wtime() - loop_start_time;
    }//end master
    }//end parallel region

    free(taskInfo_firstLoop);

    //printf("convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
    //print_mat(lambda, blocksize, blocksize);

    // printf("actMask\n");
    // for(i = 0; i < blocksize ; i++)
    //     printf("%d ", activeMask[i]);
    // printf("\n");
    //exit(1);

    //cout << "Eigen values at the end of iterationNumber: " << iterationNumber << endl;
    // //print_mat(lambda, blocksize, blocksize);
    for(i = 0 ; i < blocksize ; i++)
    {
        //cout << lambda[i * blocksize + i] << endl;
        saveLamda[i][iterationNumber - 1] = lambda[i * blocksize + i];
    }

    // iteraton_time = omp_get_wtime() - iteraton_start_time;
    // //cout << "Finished 1st iteration" << endl;

    //print_summary(timingStat, iterationNumber);
    
    /* executioner starts from here */

    //printf("\nargv[6]: %s\n", argv[4]);

    /*printf("After firstloop part \n");
    for(i = 0 ; i < blocksize ; i++)
    {
        //cout << lambda[i * blocksize + i] << endl;
        printf("%.4lf ", lambda[i * blocksize + i]);
    }
    printf("\n");*/

   
    
    //if(0 > 1){

    //double partitioner_execution_start = omp_get_wtime();

    iterationNumber = 2;

    //printf("2nd loop part starts");

    /* execution starts here */
    #pragma omp parallel
    {
    #pragma omp master
    {
        while(iterationNumber <= maxIterations && convergeFlag != 1)
        {
            loop_start_time = omp_get_wtime();

            //open the file here
            // ifstream inputFile(argv[6]);
    
            // if(inputFile.fail())
            // {
            //     printf("2nd loop File doesn't exist\n");
            // }

            //cout << "iterationNumber: " << iterationNumber << " starts" << endl;
            //reseting taskTiming
            // for(i = 0 ; i < numOperation ; i++)
            //     for(j = 0 ; j < nthrds ; j++)
            //         taskTiming[i][j] = 0.0;

            /* resetting various buffer before jumping into execution */

#ifdef EPYC            
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, blocksize, RNBUF)\
            depend(in: nthrds, blocksize)\
            depend(inout: \
            RNBUF[0 * blocksize : blocksize], RNBUF[1 * blocksize : blocksize],\
            RNBUF[2 * blocksize : blocksize], RNBUF[3 * blocksize : blocksize],\
            RNBUF[4 * blocksize : blocksize], RNBUF[5 * blocksize : blocksize],\
            RNBUF[6 * blocksize : blocksize], RNBUF[7 * blocksize : blocksize],\
            RNBUF[8 * blocksize : blocksize], RNBUF[9 * blocksize : blocksize],\
            RNBUF[10 * blocksize : blocksize], RNBUF[11 * blocksize : blocksize],\
            RNBUF[12 * blocksize : blocksize], RNBUF[13 * blocksize : blocksize],\
            RNBUF[14 * blocksize : blocksize], RNBUF[15 * blocksize : blocksize],\
            RNBUF[16 * blocksize : blocksize], RNBUF[17 * blocksize : blocksize],\
            RNBUF[18 * blocksize : blocksize], RNBUF[19 * blocksize : blocksize],\
            RNBUF[20 * blocksize : blocksize], RNBUF[21 * blocksize : blocksize],\
            RNBUF[22 * blocksize : blocksize], RNBUF[23 * blocksize : blocksize],\
            RNBUF[24 * blocksize : blocksize], RNBUF[25 * blocksize : blocksize],\
            RNBUF[26 * blocksize : blocksize], RNBUF[27 * blocksize : blocksize],\
            RNBUF[28 * blocksize : blocksize], RNBUF[29 * blocksize : blocksize],\
            RNBUF[30 * blocksize : blocksize], RNBUF[31 * blocksize : blocksize],\
            RNBUF[32 * blocksize : blocksize], RNBUF[33 * blocksize : blocksize],\
            RNBUF[34 * blocksize : blocksize], RNBUF[35 * blocksize : blocksize],\
            RNBUF[36 * blocksize : blocksize], RNBUF[37 * blocksize : blocksize],\
            RNBUF[38 * blocksize : blocksize], RNBUF[39 * blocksize : blocksize],\
            RNBUF[40 * blocksize : blocksize], RNBUF[41 * blocksize : blocksize],\
            RNBUF[42 * blocksize : blocksize], RNBUF[43 * blocksize : blocksize],\
            RNBUF[44 * blocksize : blocksize], RNBUF[45 * blocksize : blocksize],\
            RNBUF[46 * blocksize : blocksize], RNBUF[47 * blocksize : blocksize],\
            RNBUF[48 * blocksize : blocksize], RNBUF[49 * blocksize : blocksize],\
            RNBUF[50 * blocksize : blocksize], RNBUF[51 * blocksize : blocksize],\
            RNBUF[52 * blocksize : blocksize], RNBUF[53 * blocksize : blocksize],\
            RNBUF[54 * blocksize : blocksize], RNBUF[55 * blocksize : blocksize],\
            RNBUF[56 * blocksize : blocksize], RNBUF[57 * blocksize : blocksize],\
            RNBUF[58 * blocksize : blocksize], RNBUF[59 * blocksize : blocksize],\
            RNBUF[60 * blocksize : blocksize], RNBUF[61 * blocksize : blocksize],\
            RNBUF[62 * blocksize : blocksize], RNBUF[63 * blocksize : blocksize],\
            RNBUF[64 * blocksize : blocksize], RNBUF[65 * blocksize : blocksize],\
            RNBUF[66 * blocksize : blocksize], RNBUF[67 * blocksize : blocksize],\
            RNBUF[68 * blocksize : blocksize], RNBUF[69 * blocksize : blocksize],\
            RNBUF[70 * blocksize : blocksize], RNBUF[71 * blocksize : blocksize],\
            RNBUF[72 * blocksize : blocksize], RNBUF[73 * blocksize : blocksize],\
            RNBUF[74 * blocksize : blocksize], RNBUF[75 * blocksize : blocksize],\
            RNBUF[76 * blocksize : blocksize], RNBUF[77 * blocksize : blocksize],\
            RNBUF[78 * blocksize : blocksize], RNBUF[79 * blocksize : blocksize],\
            RNBUF[80 * blocksize : blocksize], RNBUF[81 * blocksize : blocksize],\
            RNBUF[82 * blocksize : blocksize], RNBUF[83 * blocksize : blocksize],\
            RNBUF[84 * blocksize : blocksize], RNBUF[85 * blocksize : blocksize],\
            RNBUF[86 * blocksize : blocksize], RNBUF[87 * blocksize : blocksize],\
            RNBUF[88 * blocksize : blocksize], RNBUF[89 * blocksize : blocksize],\
            RNBUF[90 * blocksize : blocksize], RNBUF[91 * blocksize : blocksize],\
            RNBUF[92 * blocksize : blocksize], RNBUF[93 * blocksize : blocksize],\
            RNBUF[94 * blocksize : blocksize], RNBUF[95 * blocksize : blocksize],\
            RNBUF[96 * blocksize : blocksize], RNBUF[97 * blocksize : blocksize],\
            RNBUF[98 * blocksize : blocksize], RNBUF[99 * blocksize : blocksize],\
            RNBUF[100 * blocksize : blocksize], RNBUF[101 * blocksize : blocksize],\
            RNBUF[102 * blocksize : blocksize], RNBUF[103 * blocksize : blocksize],\
            RNBUF[104 * blocksize : blocksize], RNBUF[105 * blocksize : blocksize],\
            RNBUF[106 * blocksize : blocksize], RNBUF[107 * blocksize : blocksize],\
            RNBUF[108 * blocksize : blocksize], RNBUF[109 * blocksize : blocksize],\
            RNBUF[110 * blocksize : blocksize], RNBUF[111 * blocksize : blocksize],\
            RNBUF[112 * blocksize : blocksize], RNBUF[113 * blocksize : blocksize],\
            RNBUF[114 * blocksize : blocksize], RNBUF[115 * blocksize : blocksize],\
            RNBUF[116 * blocksize : blocksize], RNBUF[117 * blocksize : blocksize],\
            RNBUF[118 * blocksize : blocksize], RNBUF[119 * blocksize : blocksize],\
            RNBUF[120 * blocksize : blocksize], RNBUF[121 * blocksize : blocksize],\
            RNBUF[122 * blocksize : blocksize], RNBUF[123 * blocksize : blocksize],\
            RNBUF[124 * blocksize : blocksize], RNBUF[125 * blocksize : blocksize],\
            RNBUF[126 * blocksize : blocksize], RNBUF[127 * blocksize : blocksize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, blocksize, RNBUF)\
            depend(in: nthrds, blocksize)\
            depend(inout: RNBUF[0 * blocksize : blocksize], RNBUF[1 * blocksize : blocksize], RNBUF[2 * blocksize : blocksize], RNBUF[3 * blocksize : blocksize],\
            RNBUF[4 * blocksize : blocksize], RNBUF[5 * blocksize : blocksize], RNBUF[6 * blocksize : blocksize], RNBUF[7 * blocksize : blocksize], RNBUF[8 * blocksize : blocksize],\
            RNBUF[9 * blocksize : blocksize], RNBUF[10 * blocksize : blocksize], RNBUF[11 * blocksize : blocksize], RNBUF[12 * blocksize : blocksize],\
            RNBUF[13 * blocksize : blocksize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize ; i++)
                {
                    RNBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }
            //b * cb
            //temp2XTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, temp2XTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            temp2XTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, temp2XTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: temp2XTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], temp2XTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            temp2XTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    temp2XTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramXARXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXARXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            gramXARXTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXARXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: gramXARXTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXARXTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXARXTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    gramXARXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramXAPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXAPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            gramXAPXTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXAPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: gramXAPXTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXAPXTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXAPXTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    gramXAPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }
            //gramXBPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXBPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: \
            gramXBPXTYBUF[0 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[1 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[2 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[3 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[4 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[5 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[6 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[7 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[8 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[9 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[10 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[11 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[12 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[13 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[14 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[15 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[16 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[17 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[18 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[19 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[20 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[21 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[22 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[23 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[24 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[25 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[26 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[27 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[28 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[29 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[30 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[31 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[32 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[33 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[34 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[35 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[36 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[37 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[38 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[39 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[40 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[41 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[42 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[43 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[44 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[45 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[46 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[47 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[48 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[49 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[50 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[51 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[52 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[53 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[54 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[55 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[56 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[57 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[58 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[59 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[60 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[61 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[62 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[63 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[64 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[65 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[66 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[67 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[68 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[69 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[70 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[71 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[72 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[73 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[74 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[75 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[76 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[77 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[78 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[79 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[80 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[81 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[82 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[83 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[84 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[85 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[86 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[87 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[88 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[89 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[90 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[91 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[92 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[93 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[94 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[95 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[96 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[97 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[98 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[99 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[100 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[101 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[102 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[103 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[104 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[105 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[106 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[107 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[108 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[109 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[110 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[111 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[112 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[113 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[114 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[115 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[116 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[117 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[118 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[119 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[120 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[121 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[122 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[123 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[124 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[125 * blocksize*currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[126 * blocksize*currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[127 * blocksize*currentBlockSize : blocksize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, blocksize, gramXBPXTYBUF)\
            depend(in: nthrds, currentBlockSize, blocksize)\
            depend(inout: gramXBPXTYBUF[0 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[1 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[2 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[3 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[4 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[5 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[6 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[7 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[8 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[9 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[10 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[11 * blocksize * currentBlockSize : blocksize*currentBlockSize], gramXBPXTYBUF[12 * blocksize * currentBlockSize : blocksize*currentBlockSize],\
            gramXBPXTYBUF[13 * blocksize * currentBlockSize : blocksize*currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * blocksize * currentBlockSize ; i++)
                {
                    gramXBPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }


            //cb * cb
            //gramRBRXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBRXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRBRXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBRXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBRXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBRXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRBRXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBRXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBRXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRBRXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramPBPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramPBPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPBPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPBPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramPBPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPBPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPBPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramPBPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramRBPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRBPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRBPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRBPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRBPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRBPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRBPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRBPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRBPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramPAPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramPAPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramPAPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramPAPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramPAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramPAPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramPAPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramPAPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramPAPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramRARXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRARXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRARXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRARXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRARXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRARXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRARXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRARXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRARXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRARXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }

            //gramRAPXTYBUF
#ifdef EPYC
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: \
            gramRAPXTYBUF[0 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[1 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[2 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[3 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[4 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[5 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[6 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[7 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[8 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[9 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[10 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[11 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[12 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[13 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[14 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[15 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[16 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[17 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[18 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[19 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[20 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[21 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[22 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[23 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[24 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[25 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[26 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[27 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[28 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[29 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[30 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[31 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[32 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[33 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[34 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[35 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[36 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[37 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[38 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[39 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[40 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[41 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[42 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[43 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[44 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[45 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[46 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[47 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[48 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[49 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[50 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[51 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[52 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[53 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[54 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[55 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[56 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[57 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[58 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[59 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[60 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[61 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[62 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[63 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[64 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[65 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[66 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[67 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[68 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[69 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[70 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[71 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[72 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[73 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[74 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[75 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[76 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[77 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[78 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[79 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[80 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[81 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[82 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[83 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[84 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[85 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[86 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[87 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[88 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[89 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[90 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[91 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[92 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[93 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[94 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[95 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[96 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[97 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[98 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[99 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[100 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[101 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[102 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[103 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[104 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[105 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[106 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[107 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[108 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[109 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[110 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[111 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[112 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[113 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[114 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[115 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[116 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[117 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[118 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[119 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[120 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[121 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[122 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[123 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[124 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[125 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize],\
            gramRAPXTYBUF[126 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize], gramRAPXTYBUF[127 * currentBlockSize*currentBlockSize : currentBlockSize*currentBlockSize])
#else
            #pragma omp task private(i, tid, tstart, tend)\
            firstprivate(nthrds, currentBlockSize, gramRAPXTYBUF)\
            depend(in: nthrds, currentBlockSize)\
            depend(inout: gramRAPXTYBUF[0 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[1 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[2 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[3 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[4 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[5 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[6 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[7 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[8 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[9 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[10 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[11 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize], gramRAPXTYBUF[12 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize],\
            gramRAPXTYBUF[13 * currentBlockSize * currentBlockSize : currentBlockSize * currentBlockSize])
#endif
            {
                //tid = omp_get_thread_num();
                //tstart = omp_get_wtime();

                for(i = 0 ; i < nthrds * currentBlockSize * currentBlockSize ; i++)
                {
                    gramRAPXTYBUF[i] = 0.0;
                }
                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
            }


            for(structIterator = 0 ; structIterator < taskCount_secondLoop; structIterator++)  
            {
                //if(taskInfo_secondLoop[structIterator].opCode == 26)
                //    printf("got it\n");
               
                //if(maxTaskLength < strlen(taskName))
                //    maxTaskLength = strlen(taskName);
                //printf("%s -> %d\n", taskName, strlen(taskName));
                //continue;

                /* taskName starts with _ means it is a dummy task */
                if(taskInfo_secondLoop[structIterator].opCode == 22) 
                {
                    //taskCounter++;
                    //printf("here: %s\n", taskName);
                    //continue;
                }

                /* if the taskName has more than or equal 1 tokens */
                //tokenCount = split(taskName, ',', &splitParams);
                
                //if(tokenCount > 0 && taskName[0] != '_') 
                //{
                    else if(taskInfo_secondLoop[structIterator].opCode == 1) /* taskName starts RESET */
                    {
                        if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "actMask")) //reseting activeMask
                        {
                            //printf("update_activeMask_task_exe\n");
                            update_activeMask_task_exe(activeMask, residualNorms, residualTolerance, blocksize);
                            //printf("actMask\n");
                            //for(i = 0; i < blocksize; i++)
                            //    activeMask[i] = 0;
                        }
                        else //reseting residualNorms
                        {
                            //printf("RESET,RN\n");
                            #pragma omp task private(i) private(tid, tstart, tend)\
                            firstprivate(blocksize, residualNorms)\
                            depend(out: residualNorms[0 : blocksize])
                            {
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                for(i = 0; i < blocksize; i++)
                                    residualNorms[i] = 0.0;

                                //taskTiming[0][tid] += (omp_get_wtime() - tstart);
                            }
                        }
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 2) //SPMM
                    {
                        //printf("partNo: %d tokenCount: %d\n", partNo, tokenCount);
                        // if(tokenCount > 4)
                        // {
                        //     spmmTaskCounter += (tokenCount - 1)/3 ;
                        //     //#pragma omp taskgroup
                        //     //{
                        //         for(sp = 1 ; sp < tokenCount ; sp += 3) //SPMM,1,1,1,2,2,2,3,3,3 
                        //         {
                        //             row_id = atoi(splitParams[sp]);
                        //             col_id = atoi(splitParams[sp + 1]); 
                        //             buf_id = atoi(splitParams[sp + 2]);
                        //             //printf("part: %d => %d %d %d\n", partNo, row_id, col_id, buf_id);
                        //             //spmm_blkcoord_finegrained_exe(numrows, numcols, currentBlockSize, nthrds, activeBlockVectorR, spmmBUF, matrixBlock, row_id, col_id, buf_id, block_width);
                        //             spmm_blkcoord_finegrained_exe_fixed_buf<double>(numrows, numcols, currentBlockSize, nbuf, activeBlockVectorR, activeBlockVectorAR, matrixBlock, row_id, col_id, buf_id, block_width);
                        //             spmmTaskCounter_2++;
                        //         }
                        //     //}
                        // }
                        // else
                        // {
                            //#pragma omp taskgroup
                            //{
                                // row_id = atoi(splitParams[1]);
                                // col_id = atoi(splitParams[2]); 
                                // buf_id = atoi(splitParams[3]);
                                //spmm_blkcoord_finegrained_exe(numrows, numcols, currentBlockSize, nthrds, activeBlockVectorR, spmmBUF, matrixBlock, row_id, col_id, buf_id, block_width);
                                row_id = taskInfo_secondLoop[structIterator].numParamsList[0]; //atoi(splitParams[1]);
                                col_id = taskInfo_secondLoop[structIterator].numParamsList[1]; //atoi(splitParams[2]); 
                                buf_id = taskInfo_secondLoop[structIterator].numParamsList[2]; //atoi(splitParams[3]);
                                spmm_blkcoord_finegrained_exe_fixed_buf(numrows, currentBlockSize, activeBlockVectorR, activeBlockVectorAR, matrixBlock, row_id, col_id, block_width);
                                //spmmTaskCounter++;
                                //spmmTaskCounter_2++;
                            //}
                        //}
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 3) /* taskName starts XTY */
                    {
                        // block_id = atoi(splitParams[1]);
                        // buf_id = atoi(splitParams[2]);
                        // task_id = atoi(splitParams[3]);
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; //atoi(splitParams[1]);
                        buf_id = taskInfo_secondLoop[structIterator].numParamsList[1]; //atoi(splitParams[2]);
                        task_id = taskInfo_secondLoop[structIterator].taskID; //atoi(splitParams[3]);
                        //printf("XTY %d -> %d -> %d\n", block_id, buf_id, task_id);
                        if(task_id == 1)
                        {
                            _XTY_v1_exe(blockVectorX, activeBlockVectorR, temp2XTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 2)
                        {
                            _XTY_v1_exe(activeBlockVectorR, activeBlockVectorR, gramRBRXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 3)
                        {
                            _XTY_v1_exe(activeBlockVectorP, activeBlockVectorP, gramPBPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id); 
                        }
                        else if(task_id == 4)
                        {
                            _XTY_v1_exe(blockVectorAX, activeBlockVectorR, gramXARXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 5)
                        {
                            _XTY_v1_exe(activeBlockVectorAR, activeBlockVectorR, gramRARXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 6)
                        {
                            _XTY_v1_exe(blockVectorAX, activeBlockVectorP, gramXAPXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 7)
                        {
                            _XTY_v1_exe(activeBlockVectorAR, activeBlockVectorP, gramRAPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 8)
                        {
                            _XTY_v1_exe(activeBlockVectorAP, activeBlockVectorP, gramPAPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 9)
                        {
                            _XTY_v1_exe(blockVectorX, activeBlockVectorP, gramXBPXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        else if(task_id == 10)
                        {
                            _XTY_v1_exe(activeBlockVectorR, activeBlockVectorP, gramRBPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 4) /* XTY partial sum reduction */
                    {
                        //printf("%s %s -> %s \n", taskName, splitParams[0], splitParams[1]);
                        if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "temp2BUF")) //xty 1 reduction
                        {
                            _XTY_v1_RED(temp2XTYBUF, temp2, blocksize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBRBUF")) //xty 2 reduction
                        {
                            _XTY_v1_RED(gramRBRXTYBUF, gramRBR, currentBlockSize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PBPBUF")) //xty 3 reduction
                        {
                            _XTY_v1_RED(gramPBPXTYBUF, gramPBP, currentBlockSize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "XARBUF")) //xty 4 reduction
                        {
                            _XTY_v1_RED(gramXARXTYBUF, gramXAR, blocksize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RARBUF")) //xty 5 reduction
                        {
                            _XTY_v1_RED(gramRARXTYBUF, gramRAR, currentBlockSize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "XAPBUF")) //xty 6 reduction
                        {
                            _XTY_v1_RED(gramXAPXTYBUF, gramXAP, blocksize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RAPBUF")) //xty 7 reduction
                        {
                            _XTY_v1_RED(gramRAPXTYBUF, gramRAP, currentBlockSize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PAPBUF")) //xty 8 reduction
                        {
                            _XTY_v1_RED(gramPAPXTYBUF, gramPAP, currentBlockSize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "XBPBUF")) //xty 9 reduction
                        {
                            _XTY_v1_RED(gramXBPXTYBUF, gramXBP, blocksize, currentBlockSize, block_width);
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBPBUF")) //xty 10 reduction
                        {
                            _XTY_v1_RED(gramRBPXTYBUF, gramRBP, currentBlockSize, currentBlockSize, block_width);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 5) /* taskName starts XY */
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);
                        
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;

                        if(task_id == 1)
                        {
                            _XY_exe(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width, block_id);
                        }
                        else if(task_id == 2)
                        {
                            _XY_exe(blockVectorX, temp2, temp3_R, M, blocksize, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 3)
                        {
                            _XY_exe(activeBlockVectorR, gramRBR, temp3_R, M, currentBlockSize, currentBlockSize, block_width, block_id);
                        }
                        else if (task_id == 4)
                        {
                            _XY_exe(activeBlockVectorP, gramPBP, temp3_P, M, currentBlockSize, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 5)
                        {
                            _XY_exe(activeBlockVectorAP, gramPBP, temp3_AP, M, currentBlockSize, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 6)
                        {
                            _XY_exe(activeBlockVectorR, coordX2, blockVectorP, M, currentBlockSize, blocksize, block_width, block_id);
                            //_XY_exe(activeBlockVectorR, coordX + (blocksize * blocksize), blockVectorP, M, currentBlockSize, blocksize, block_width, block_id);
                        }
                        else if(task_id == 7)
                        {
                            _XY_exe(activeBlockVectorP, coordX3, newP, M, currentBlockSize, blocksize, block_width, block_id);
                            //_XY_exe(activeBlockVectorP, coordX+((blocksize+currentBlockSize)*blocksize), newP, M, currentBlockSize, blocksize, block_width, block_id);
                        }
                        else if(task_id == 8)
                        {
                            _XY_exe(activeBlockVectorAR, coordX2, blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id);
                            //_XY_exe(activeBlockVectorAR, coordX+(blocksize*blocksize), blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id);
                        }
                        else if(task_id == 9)
                        {
                            _XY_exe(activeBlockVectorAP, coordX3, newAP, M, currentBlockSize, blocksize, block_width, block_id);
                            //_XY_exe(activeBlockVectorAP, coordX+((blocksize+currentBlockSize)*blocksize), newAP, M, currentBlockSize, blocksize, block_width, block_id);
                        }
                        else if(task_id == 10)
                        {
                            _XY_exe(blockVectorX, coordX1, newX, M, blocksize, blocksize, block_width, block_id);
                            //_XY_exe(blockVectorX, coordX, newX, M, blocksize, blocksize, block_width, block_id);
                        }
                        else if(task_id == 11)
                        {
                            _XY_exe(blockVectorAX, coordX1, newAX, M, blocksize, blocksize, block_width, block_id);
                            //_XY_exe(blockVectorAX, coordX, newAX, M, blocksize, blocksize, block_width, block_id);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 6) // ADD
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;
                        
                        if(task_id == 1)
                        {
                            mat_addition_task_exe(blockVectorP, newP, blockVectorP, M, blocksize, block_width, block_id);
                        }
                        else if(task_id == 2)
                        {
                            mat_addition_task_exe(blockVectorAP, newAP, blockVectorAP, M, blocksize, block_width, block_id);
                        }
                        else if(task_id == 3)
                        {
                            mat_addition_task_exe(newX, blockVectorP, blockVectorX, M, blocksize, block_width, block_id);
                        }
                        else if(task_id == 4)
                        {
                            mat_addition_task_exe(newAX, blockVectorAP, blockVectorAX, M, blocksize, block_width, block_id);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 7) /* taskName starts DLACPY */
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);
                        
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;

                        if(task_id == 1)
                        {
                            custom_dlacpy_task_exe(temp3_R, activeBlockVectorR, M, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 2)
                        {
                            custom_dlacpy_task_exe(temp3_P, activeBlockVectorP, M, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 3)
                        {
                            custom_dlacpy_task_exe(temp3_AP, activeBlockVectorAP, M, currentBlockSize, block_width, block_id);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 8 && 0) //UPDATE
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);

                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;

                        if(task_id == 1)
                        {
                            updateBlockVector_task_exe(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 2)
                        {
                            updateBlockVector_task_exe(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 3)
                        {
                            updateBlockVector_task_exe(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width, block_id);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 9) //SUB
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;

                        if(task_id == 1)
                        {
                            mat_sub_task_exe(blockVectorAX, blockVectorR, blockVectorR , M, blocksize, block_width, block_id);
                        }
                        else if(task_id == 2)
                        {
                            mat_sub_task_exe(activeBlockVectorR, temp3_R, activeBlockVectorR, M, currentBlockSize, block_width, block_id);
                            //#pragma omp taskwait
                            //printf("\n");
                            //print_mat(activeBlockVectorR, M, currentBlockSize);
                            
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 10) /* taskName starts MULT */
                    {
                       //block_id = atoi(splitParams[1]);
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                        mat_mult_task_exe(blockVectorR, blockVectorR, newX, M, blocksize, block_width, block_id);
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 26) //DOT
                    {
                        //block_id = atoi(splitParams[1]);
                        /* only ONE MULT op, so no task_id */
                        //printf("%s -> %d\n", taskName, block_id);
                        
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                        buf_id = taskInfo_secondLoop[structIterator].numParamsList[1]; 

                        //printf("second DOT --> block_id: %d buf_id: %d\n", block_id, buf_id);

                        //mat_mult_task_exe(blockVectorR, blockVectorR, newX, M, blocksize, block_width, block_id);
                        dot_mm_exe(blockVectorR, blockVectorR, RNBUF, M, blocksize, block_width, block_id, buf_id);
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 11) /* taskName starts COL */
                    {
                        //block_id = atoi(splitParams[1]);
                        //buf_id = atoi(splitParams[2]);
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                        buf_id = taskInfo_secondLoop[structIterator].numParamsList[1];

                        //printf("sum_sqrt_task_COL\n");

                        sum_sqrt_task_COL(newX, M, blocksize, block_width, block_id, buf_id, RNBUF);
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 12) /* taskName starts RNRED */
                    {
                        //printf("sum_sqrt_task_RNRED\n");
                        sum_sqrt_task_RNRED(RNBUF, residualNorms, blocksize);
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 13) /* taskName starts SQRT */
                    {
                        //printf("sum_sqrt_task_SQRT\n");

                        sum_sqrt_task_SQRT(residualNorms, blocksize);
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 14) /* taskName starts GET */
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);

                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;
                        
                        if(task_id == 1)
                        {
                            getActiveBlockVector_task_exe(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id); 
                        }
                        else if(task_id == 2)
                        {
                            getActiveBlockVector_task_exe(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width, block_id);
                        }
                        else if(task_id == 3)
                        {
                            getActiveBlockVector_task_exe(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width, block_id);
                        }
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 20) /* taskName starts CONV */
                    {
                        //code: 8
                        #pragma omp task default(shared) private(tid, tstart, tend)\
                        firstprivate(blocksize, activeMask,residualNorms, iterationNumber)\
                        depend(inout: activeMask[0 : blocksize], currentBlockSize, convergeFlag)\
                        depend(in: residualNorms[0 : blocksize])\
                        depend(out: activeRSize, explicitGramFlag, activePSize, restart)
                        //lastprivate(explicitGramFlag, activeRSize, activePSize, restart)
                        {   
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            //currentBlockSize = sum(activeMask);
                            //printf("* CONV 2 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                            
                            int tempCurrentBlockSize = 0;
                            for(i = 0 ; i < blocksize ; i++)
                                tempCurrentBlockSize += activeMask[i];
                            
                            currentBlockSize = tempCurrentBlockSize;
                            //if 14
                            if(currentBlockSize == 0)
                            {
                                convergeFlag = 1;
                                printf("converge!! convergeFlag: %d", convergeFlag);
                            }
                            activeRSize = currentBlockSize;
                            activePSize = currentBlockSize;
                            restart = 0;
                            //printf("CONV 2 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                            //tend = omp_get_wtime();
                            //taskTiming[8][tid] += (tend - tstart);

                            int flag = 1;
                            for(i = 0 ; i < blocksize ; i++)
                            {
                                if(residualNorms[i] < 4.0538e-10)
                                {
                                    flag = 0;
                                    //break;
                                }
                            }
                            if(flag == 0 )
                                explicitGramFlag = 1;
                            else
                                explicitGramFlag = 0;


                            //tend = omp_get_wtime();
                            //taskTiming[8][tid] += (tend - tstart);
                        } //end task
                        
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 15) //TRANS
                    {
                        #pragma omp task default(shared) private(tid, tstart, tend)\
                        firstprivate(gramRAR, transGramRAR)\
                        depend(in: gramRAR[0 : currentBlockSize * currentBlockSize], currentBlockSize)\
                        depend(inout: transGramRAR[0 : currentBlockSize * currentBlockSize])
                        {
                            //transGramRAR(0)_TRANS(gramRAR)
                            //tid = omp_get_thread_num();
                            //tstart = omp_get_wtime();

                            transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);

                            //tend = omp_get_wtime();
                            //taskTiming[23][tid] += (tend - tstart);
                        } //end task
                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 16) //SPEUPDATE
                    {
                        //code: 13 (update)
                        if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RAR"))
                        {
                            #pragma omp task default(shared) private(tid, tstart, tend)\
                            firstprivate(identity_PAP, transGramRAR, gramRAR)\
                            depend(in : currentBlockSize) depend(inout : identity_PAP[0 : currentBlockSize * currentBlockSize])\
                            depend(in : transGramRAR[0 : currentBlockSize * currentBlockSize])\
                            depend(inout : gramRAR[0 : currentBlockSize * currentBlockSize])
                            {
                                //gramRAR(0)_UPDATE(gramRAR,transGramRAR) SPEUPDATE
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize);
                                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 
                                            0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);

                                //tend = omp_get_wtime();
                                //taskTiming[13][tid] += (tend - tstart);
                            } //end task
                            //#pragma omp taskwait
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PAP"))
                        {
                            /* OP: gramPAP=(gramPAP'+gramPAP)*0.5; */
                            #pragma omp task default(shared) firstprivate(identity_PAP, gramPAP)\
                            depend(in : currentBlockSize)\
                            depend(inout: identity_PAP[0 : currentBlockSize * currentBlockSize])\
                            depend(inout: gramPAP[0 : currentBlockSize * currentBlockSize])
                            {
                                //gramPAP(0)_UPDATE(gramPAP) SPEUPDATE
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();
                                
                                make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize);
                                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 
                                    0.5, gramPAP, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramPAP, currentBlockSize);

                                //tend = omp_get_wtime();
                                //taskTiming[13][tid] += (tend - tstart);
                            } //end task
                            //#pragma omp taskwait
                        }
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 17) //CHOL
                    {
                        //code: 22
                        if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBR"))
                        {
                            #pragma omp task private(i, j, tid, tstart, tend) firstprivate(gramRBR, trans_gramRBR) default(shared)\
                            depend(in: currentBlockSize, trans_gramRBR[0 : currentBlockSize * currentBlockSize])\
                            depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize]) depend(out: info)
                            {
                                /* OP: [gramRBR,cholFlag] = chol(gramRBR); */
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();
                                
                                transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);

                                //tend = omp_get_wtime();
                                //taskTiming[tid][9] += (tend - tstart);
                                //tstart = omp_get_wtime();
                                
                                dpotrf_( &uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info );
                                if(info != 0)
                                {
                                    cout<<"dpotrf_ error: "<< info << endl;
                                }

                                //taskTiming[tid][6] += (omp_get_wtime() - tstart);
                                //tstart = omp_get_wtime();

                                transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);

                                //tend = omp_get_wtime();
                                //taskTiming[tid][9] += (tend - tstart);
                                //tstart = omp_get_wtime();
                                
                                for(i = 0 ; i < currentBlockSize ; i++)
                                {
                                    for(j = 0 ; j < i ; j++)
                                    {
                                        gramRBR[i * currentBlockSize + j] = 0.0;
                                    }
                                }
                                //tend = omp_get_wtime();
                                //taskTiming[22][tid] += (tend - tstart);
                            }//end task
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PBP"))
                        {
                            #pragma omp task default(shared) private(tid, tstart, tend)\
                            firstprivate(gramPBP, trans_gramPBP)\
                            depend(inout: gramPBP[0 : currentBlockSize * currentBlockSize])\
                            depend(inout: trans_gramPBP[0 : currentBlockSize * currentBlockSize])\
                            depend(out: info)
                            {
                                //gramPBP(0)_CHOL(gramPBP)
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                transpose(gramPBP, trans_gramPBP, currentBlockSize, currentBlockSize);

                                //tend = omp_get_wtime();
                                //taskTiming[tid][9] += (tend - tstart);
                                //tstart = omp_get_wtime();

                                dpotrf_( &uplo, &currentBlockSize, trans_gramPBP, &currentBlockSize, &info );
                                if(info != 0)
                                {
                                    //cout<<"dportf_ error 3"<<endl;
                                    cout << "BLOPEX:lobpcg:DirectionNotFullRank...The direction matrix is not full rank." << endl;
                                }

                                //tend = omp_get_wtime();
                                //taskTiming[tid][6] += (tend - tstart);
                                //tstart = omp_get_wtime();

                                transpose(trans_gramPBP, gramPBP, currentBlockSize, currentBlockSize);

                                //tend = omp_get_wtime();
                                //taskTiming[tid][9] += (tend - tstart);
                                //tstart = omp_get_wtime();

                                //making the lower part of gramPBP zero
                                for(i = 0 ; i < currentBlockSize ; i++)
                                {
                                    for(j = 0 ; j < i ; j++)
                                    {
                                        gramPBP[i * currentBlockSize + j] = 0.0;
                                    }
                                }
                        
                                //tend = omp_get_wtime();
                                //taskTiming[22][tid] += (tend - tstart);
                            } //end task
                        }

                        //#pragma omp taskwait
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 18) //INV
                    {

                        if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBR"))
                        {
                            //code 12
                            #pragma omp task default(shared) private(tid, tstart, tend)\
                            firstprivate(gramRBR, currentBlockSize)\
                            depend(in: info, currentBlockSize)\
                            depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize])
                            {
                                if(info == 0)
                                {
                                    /* OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask)/gramRBR; */
                                    //tid = omp_get_thread_num();
                                    //tstart = omp_get_wtime();

                                    inverse(gramRBR, currentBlockSize, currentBlockSize);
                                    //tend = omp_get_wtime();
                                    //taskTiming[12][tid] += (tend - tstart);
                                }
                                else
                                {
                                    printf("INV,RBR info: %d\n", info);
                                }
                            } //end task
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PBP"))
                        {   
                            #pragma omp task default(shared) private(tid, tstart, tend)\
                            firstprivate(gramPBP)\
                            depend(in : info, currentBlockSize, gramPBP)\
                            depend(inout: gramPBP[0 : currentBlockSize * currentBlockSize])
                            {
                                if(info == 0)
                                {
                                    //tid = omp_get_thread_num();
                                    //tstart = omp_get_wtime();

                                    inverse(gramPBP, currentBlockSize, currentBlockSize);
                                
                                    //tend = omp_get_wtime();
                                    //taskTiming[12][tid] += (tend - tstart);
                                }
                                else
                                {
                                    printf("INV,PBP info: %d\n", info);
                                }
                            } //end task
                        }
                    }

                    else if(taskInfo_secondLoop[structIterator].opCode == 19) /* taskName starts SETZERO */
                    {
                        // block_id = atoi(splitParams[1]);
                        // task_id = atoi(splitParams[2]);
                        block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                        task_id = taskInfo_secondLoop[structIterator].taskID;


                        if(task_id == 1)
                        {
                            i = block_id * block_width; // starting point of the block
                            blksz = block_width;
                            if(i + blksz > M)
                                blksz = M - i;
                            
                            //code: 0    
                            #pragma omp task default(shared) private(j, k, tid, tstart, tend)\
                            firstprivate(activeBlockVectorAR, blksz, i, M, block_width, blocksize, currentBlockSize)\
                            depend(in :  M, blocksize, currentBlockSize)\
                            depend(out : activeBlockVectorAR[i * currentBlockSize : blksz * currentBlockSize])
                            {
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                for(j = i ; j < i + blksz ; j++)
                                {
                                    for(k = 0 ; k < currentBlockSize ; k++)
                                    {
                                        activeBlockVectorAR[j * currentBlockSize + k] = 0.0;
                                    }
                                }

                                //tend = omp_get_wtime();
                                //taskTiming[0][tid] += (tend - tstart);
                            } //end task
                            
                        }
                    }
                    else if(taskInfo_secondLoop[structIterator].opCode == 21)
                    {
                        if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "CONSTRUCTGA1"))
                        {
                            gramASize = blocksize + currentBlockSize + currentBlockSize;
                            coordX = (double *) malloc(gramASize * blocksize * sizeof(double));
                            //declare it only once up there, fix it later
                            gramA = (double *) malloc(gramASize * gramASize * sizeof(double));

                            //code: 17
                            #pragma omp task default(shared) private(tid, tstart, tend)\
                            firstprivate(gramA)\
                            depend(in: lambda[0 : blocksize * blocksize], gramXAR[0 : blocksize * currentBlockSize], gramXAP[0 : blocksize * currentBlockSize])\
                            depend(inout: activeMask[0 : blocksize])\
                            depend(out: gramASize, gramA[0 : gramASize * gramASize])
                            {
                                //CONSTRUCTGA1
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();
                                
                                mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                                mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                                mat_copy(gramXAP, blocksize, currentBlockSize, gramA, 0, blocksize+currentBlockSize, gramASize);

                                //tend = omp_get_wtime();
                                //taskTiming[17][tid] += (tend - tstart);
                                //printf("GA1 2 gramASize: %d\n", gramASize);
                            } //end task 
                            //#pragma omp taskwait
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "CONSTRUCTGA2"))
                        {
                            //code: 18
                            #pragma omp task default(shared) private(tid, tend, tstart)\
                            firstprivate(gramASize, gramPAP, gramXAR, gramRAP, gramA)\
                            depend(in : gramASize, gramPAP[0 : currentBlockSize * currentBlockSize])\
                            depend(in : gramXAR[0 : blocksize * currentBlockSize], gramRAR[0 : currentBlockSize * currentBlockSize])\
                            depend(in : gramRAP[0 : currentBlockSize * currentBlockSize], gramXAP[0 : blocksize * currentBlockSize])\
                            depend(inout: activeMask[0 : blocksize])\
                            depend(inout : gramA[0 : gramASize * gramASize])
                            {
                                //CONSTRUCTGA2   
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                                
                                mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                                mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                                mat_copy(gramRAP, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize+currentBlockSize, gramASize);

                                transpose(gramXAP, transGramXAP, currentBlockSize, blocksize);
                                transpose(gramRAP, transGramRAP, currentBlockSize, currentBlockSize);

                                mat_copy(transGramXAP, currentBlockSize, blocksize, gramA, blocksize+currentBlockSize, 0, gramASize);
                                mat_copy(transGramRAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize, gramASize);
                                mat_copy(gramPAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);

                                //tend = omp_get_wtime();
                                //taskTiming[18][tid] += (tend - tstart);
                                //printf("GA2 2 gramASize: %d\n", gramASize);
                            } //end task
                            //#pragma omp taskwait
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "CONSTRUCTGB"))
                        {
                            //fix it later, define up there
                            gramASize = blocksize + currentBlockSize + currentBlockSize;
                            gramB = (double *) malloc(gramASize * gramASize * sizeof(double));

                            //code: 19
                            #pragma omp task  default(shared) private(tid, tstart, tend)\
                            firstprivate(gramASize, identity_PAP, gramXBP, gramRBP, gramB)\
                            depend(in: gramASize, identity_PAP[0 : currentBlockSize * currentBlockSize])\
                            depend(in : gramXBP[0 : blocksize * currentBlockSize], gramRBP[0 : currentBlockSize * currentBlockSize])\
                            depend(inout: activeMask[0 : blocksize])\
                            depend(out: gramB[0 : gramASize * gramASize]) 
                            {
                                //CONSTRUCTGB
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();

                                mat_copy(identity_BB, blocksize, blocksize, gramB, 0, 0, gramASize);
                                mat_copy(zeros_B_CB, blocksize, currentBlockSize, gramB, 0, blocksize, gramASize);
                                mat_copy(gramXBP, blocksize, currentBlockSize, gramB, 0, blocksize+currentBlockSize, gramASize);
                                mat_copy(zeros_CB_B, activeRSize, blocksize, gramB, blocksize, 0, gramASize);
                                mat_copy(identity_PAP, activeRSize, activeRSize, gramB, blocksize, blocksize, gramASize);
                                mat_copy(gramRBP, currentBlockSize, currentBlockSize, gramB, blocksize, blocksize+currentBlockSize, gramASize);

                                transpose(gramXBP, transGramXBP, currentBlockSize, blocksize);
                                transpose(gramRBP, transGramRBP, currentBlockSize, currentBlockSize);

                                mat_copy(transGramXBP, currentBlockSize, blocksize, gramB, blocksize+currentBlockSize, 0, gramASize);
                                mat_copy(transGramRBP, currentBlockSize, currentBlockSize, gramB, blocksize+currentBlockSize, blocksize, gramASize);
                                mat_copy(identity_PAP, activePSize, activePSize, gramB, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);

                                //tend = omp_get_wtime();
                                //taskTiming[19][tid] += (tend - tstart);
                                //tstart = omp_get_wtime();
                                //printf("GB 2 gramASize: %d\n", gramASize);

                            } //end task
                        }
                        else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "EIGEN"))
                        {
                            //code: 20
                            //eigen value computation here
                            //#pragma omp taskwait
                            
                            #pragma omp task default(shared) private(tid, tstart, tend, i, j, k)\
                            firstprivate(gramASize, lambda, gramA, gramB, coordX, coordX1, coordX2, coordX3)\
                            depend(inout: gramA[0 : gramASize * gramASize])\
                            depend(inout: activeMask[0 : blocksize])\
                            depend(in: gramB[0 : gramASize * gramASize], gramASize, currentBlockSize, activeRSize, activePSize)\
                            depend(out: coordX1[0 : blocksize * blocksize], coordX2[0 : currentBlockSize * blocksize], coordX3[0 : currentBlockSize * blocksize], lambda[0 : blocksize * blocksize])
                            //depend(out: coordX[0 : gramASize * blocksize], lambda[0 : blocksize * blocksize])
                            {
                                //printf("EIG 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                                //printf("EIG 2 gramASize: %d\n", gramASize);
                                    //EIGEN
                                //tid = omp_get_thread_num();
                                //tstart = omp_get_wtime();
                                
                                eigen_value = (double *) malloc(gramASize * sizeof(double));
                                double *trans_gramA = (double *) malloc(gramASize * gramASize * sizeof(double)); 
                                double *trans_gramB = (double *) malloc(gramASize * gramASize * sizeof(double)); 
                                
                                transpose(gramA, trans_gramA, gramASize, gramASize);
                                transpose(gramB, trans_gramB, gramASize, gramASize);
                                
                                //dummy call for lwork
                                lwork = -1;
                                
                                dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, &work_query, &lwork, &info);
                                
                                    
                                if(info != 0)
                                    cout << "Error in dummy call" << endl;

                                lwork = (int) work_query;
                                work = (double *) malloc(lwork * sizeof(double)); 

                                dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, work, &lwork, &info);

                                if(info != 0)
                                    cout << "** Error in eigen value calculation **" << endl;
                                transpose(trans_gramA, gramA, gramASize, gramASize);
                                transpose(trans_gramB, gramB, gramASize, gramASize);

                                free(trans_gramA); 
                                free(trans_gramB); 
                                free(work); 
                                
                                //briging following task here
                                diag(eigen_value, lambda, blocksize);
                                
                                /*int column = 0;
                                column = 0;
                                for(j = 0 ; column < blocksize && j < gramASize; j++)
                                {
                                    for(i = 0 ; i < gramASize ; i++)
                                    {
                                        coordX[i * blocksize + column] = gramA[i * gramASize + j];
                                    }
                                    column++;
                                }*/

                                //coordX1
                                for(i = 0 ; i < blocksize ; i++)
                                {
                                    for(j = 0 ; j < blocksize ; j++)
                                    {
                                        coordX1[i * blocksize + j] = gramA[i * gramASize + j];
                                    }
                                }

                                //coordX2
                                for(k = 0, i = blocksize ; k < currentBlockSize && i < blocksize + currentBlockSize ; i++, k++)
                                {
                                    for(j = 0 ; j < blocksize ; j++)
                                    {
                                        coordX2[k * blocksize + j] = gramA[i * gramASize + j];
                                    }
                                }

                                //coordX3
                                for(k = 0, i = blocksize + currentBlockSize ; k < currentBlockSize && i < blocksize + currentBlockSize + currentBlockSize ; i++, k++)
                                {
                                    for(j = 0 ; j < blocksize ; j++)
                                    {
                                        coordX3[k * blocksize + j] = gramA[i * gramASize + j];
                                    }
                                }

                                //tend = omp_get_wtime();
                                //taskTiming[20][tid] += (tend - tstart);
                            } //end task
                        }
                    } //end 21
            } //end while file read

            #pragma omp taskwait
            //cout << "iteration #" << iterationNumber << ":" << endl;
            loopTime[iterationNumber - 1] = omp_get_wtime() - loop_start_time;
            for(i = 0 ; i < blocksize ; i++)
            {
                saveLamda[i][iterationNumber - 1] = lambda[i * blocksize + i];
                //printf("%.4lf ", saveLamda[i][iterationNumber-1]);
            }
            //cout << endl;
            //print_summary(timingStat, iterationNumber);
            iterationNumber++;
            //inputFile.close();

        } //end while iterationNumber <= 10
    } //end pragma omp master
    } //end pragma omp parallel

    //}
    //double partitioner_execution_end = omp_get_wtime();
    //cout << "Executioner finished" << endl;

    /* writing timing statistics and lamda values in csv format */

    // for(i = 0 ; i < 12 ; i++)
    // {
    //     for(j = 0 ; j < maxIterations ; j++)
    //     {
    //         printf("%.6lf", timingStat[i][j]);
    //         if(j != maxIterations - 1)
    //             printf(",");
    //     }
    //     printf("\n");
    // }
    
    double totalSum = 0;
    for(j = 0 ; j < maxIterations ; j++)
    {
        totalSum += loopTime[j];
        printf("%.4lf,", loopTime[j]);
        //if(j != maxIterations - 1)
        //    printf(",");
    }
    printf("%.4lf", totalSum/maxIterations);
    printf("\n");

    for(i = 0 ; i < blocksize ; i++)
    {
        for(j = 0 ; j < maxIterations ; j++)
        {
            printf("%.4lf", saveLamda[i][j]);
            if(j != maxIterations - 1)
                printf(",");
        }
        printf("\n");
    }

    return 0;
} //end of main
