

#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"

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

    int info;


    stringstream s(argv[1]);
    s >> blocksize;
    stringstream s1(argv[2]);
    s1 >> block_width;
    cout << "Block Size: " << blocksize << " Block Width: " << block_width << endl;

    int currentBlockSize = blocksize, prevCurrentBlockSize = blocksize;

    

    //----- lib spmm params 
    char matdescra[6];
    char transA = 'n';
    matdescra[0] = 'g';
    matdescra[1] = 'l';
    matdescra[2] = 'u';
    matdescra[3] = 'c';
    
    double alpha = 1.0, beta = 0.0;

    //---- dportf_ & dsygv_ params
    const char jobvl = 'N';
    const char jobvr = 'V';
    const char jobz = 'V';
    const char uplo = 'U';
    const char uplo_dlacpy = ' ';
    const int itype = 1;
    double work_query;
    int lwork;
    double *work;

    int i,j,k;

    double *xrem;
    block *matrixBlock;
    char *filename = argv[3] ; //"/global/cscratch1/sd/rabbimd/Matrices/Nm7-original.cus"; 
    wblk = block_width; 
    read_custom(filename , xrem);
    csc2blkcoord(matrixBlock , xrem);
    
    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    //-- deleting CSC storage memory ------
    //delete []colptrs;
    //delete []irem;
    //delete []xrem;

    M = numrows;
    N = numcols;

    //timing variables
    int numTaks = 11;
    double *taskTiming = (double *) malloc(sizeof(double) * numTaks);

    double tstart, tend, temp1Time;
    double loop_start_time = 0, loop_finish_time = 0;
    double iteraton_time = 0, iteraton_start_time = 0;

    blockVectorX = (double *) malloc(M * blocksize * sizeof(double));

    //Converting CSC to CSR format
    
    /*int job_dcsrcsc[] = {1, 0, 0, 0, 0, 1}; 
    int dcsrcsc_info = -1;
    int *ja, *ia; 
    double *acsr;
    
    acsr = (double *) malloc(nnonzero * sizeof(double) ); //new double[nnonzero](); //xrem
    ja = (int *) malloc(nnonzero * sizeof(int) ); //new int[nnonzero](); //irem
    ia = (int *) malloc((numrows + 1) * sizeof(int) ); //new int[numrows + 1](); //colsptr

    tstart = omp_get_wtime();
    
    mkl_dcsrcsc(job_dcsrcsc, &numrows, acsr, ja, ia, xrem, irem, colptrs, &dcsrcsc_info);
    
    printf("mkl_dcsrcsc: %lf sec.\n", omp_get_wtime() - tstart);*/

    srand(0);
    //#pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < blocksize ; j++)
        {
            //blockVectorXfile >> blockVectorX[i * blocksize + j];
            blockVectorX[i * blocksize + j] = (double)rand()/(double)RAND_MAX;
            //blockVectorX[i*blocksize+j]= -1.00 + rand() % 2 ;
        }
    }

    //cout<<"finished reading X"<<endl;

    //******** memory allocation for matrices ********

    //double allocation_time = omp_get_wtime();

    double *blockVectorAX = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorR = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorAR = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorP = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorAP = (double *) malloc(M * blocksize * sizeof(double));
    

    //-- new here
    double *activeBlockVectorR = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorAR = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorAP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *temp3 = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *newX = (double *) malloc(M * blocksize * sizeof(double));

    double *gramA, *gramB;
    double *eigen_value;

    //--- modified new

    
    //double *newAX = new double[M*blocksize]();
    //double *temp1 = new double[M*blocksize]();

    //double *newActP = new double[M*currentBlockSize]();
    //double *tempMultResult=new double[M*currentBlockSize]();
    
    double *residualNorms = (double *) malloc(blocksize * sizeof(double));
    double *gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *trans_gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *temp2 = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *trans_gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *gramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *transGramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *transGramRAP= (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramPAP= (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *identity_PAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *transGramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *identity_BB = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *gramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));

    double *zeros_B_CB = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *zeros_CB_B = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    std::memset(zeros_B_CB, 0.0, sizeof(zeros_B_CB));
    std::memset(zeros_CB_B, 0.0, sizeof(zeros_CB_B));

    // saveLamda[blocksize * maxIterations]
    double **saveLamda = (double **) malloc(blocksize * maxIterations * sizeof(double *));
    for(i = 0 ; i < blocksize ; i++)
        saveLamda[i] = (double *) malloc(maxIterations * sizeof(double));
    
    for(i = 0 ; i < blocksize ; i++)
        for(j = 0 ; j < maxIterations ; j++)
            saveLamda[i][j] = 0.0;

    double *loopTime = (double *) malloc(maxIterations * sizeof(double));
    for(j = 0 ; j < maxIterations ; j++)
        loopTime[j] = 0.0;

    //cout<<"Allocation 8"<<endl;

    //cout<<"Total allocation time: "<<omp_get_wtime()-allocation_time<<" sec."<<endl;

    //---- if 9 ----
    //gramXBX=blockVectorX'*blockVectorX;
    //[gramXBX,cholFlag]=chol(gramXBX);
    //blockVectorX = blockVectorX/gramXBX;
    
    double *gramXBX = new double[blocksize * blocksize]();

    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,blocksize,blocksize,M,1.0,blockVectorX,blocksize,blockVectorX,blocksize,0.0,gramXBX,blocksize);
    //_XTY(blockVectorX, blockVectorX, gramXBX, M, blocksize, blocksize, block_width);

    int cholFlag;
    //making the lower part of gramXBX zero

    //-- changing LAPACKE_dpotrf to dpotrf_
    double *trans_gramXBX = new double[blocksize * blocksize]();
    
    transpose(gramXBX, trans_gramXBX, blocksize, blocksize);
    dpotrf_( &uplo, &blocksize, trans_gramXBX, &blocksize, &info );
    if(info != 0 )
    {
        cout<<"dpotrf: chol error!"<<endl;
        //exit(1);
    }
    
    transpose(trans_gramXBX, gramXBX, blocksize, blocksize);
    delete []trans_gramXBX;
    
    #pragma omp parallel for  private(j) default(shared)
    for(i = 0 ; i < blocksize ; i++)
    {
        for(j = 0 ; j < i ; j++)
        {
            gramXBX[i * blocksize + j] = 0.0;
        }
    }
    
    double *tempGramXBX = new double[blocksize * blocksize]();
    //int copyGramXBX=LAPACKE_dlacpy(LAPACK_ROW_MAJOR,' ',blocksize,blocksize,gramXBX,blocksize,tempGramXBX,blocksize);
    custom_dlacpy(gramXBX, tempGramXBX, blocksize, blocksize);
    
    inverse(tempGramXBX, blocksize, blocksize);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, M, blocksize, blocksize,1.0,blockVectorX, blocksize,tempGramXBX,blocksize,0.0, newX,blocksize);
    
    //copyResult=LAPACKE_dlacpy(LAPACK_ROW_MAJOR,' ',M,blocksize,tempResult,blocksize,blockVectorX,blocksize);
    custom_dlacpy(newX, blockVectorX, M, blocksize);
    free(tempGramXBX);
    
    // if 17 
    // blockVectorAX = operatorA*blockVectorX;
    std::memset(blockVectorAX, 0.0, sizeof(blockVectorAX));
    
    //mkl_dcscmm(&transA, &M, &blocksize, &N, &alpha, matdescra, xrem, irem, colptrs, colptrs+1, blockVectorX, &blocksize,  &beta, blockVectorAX, &blocksize);
    //mkl_dcsrmm(&transA, &M, &blocksize, &N, &alpha, matdescra, acsr, ja, ia, ia+1, blockVectorX, &blocksize, &beta, blockVectorAX, &blocksize);
    //spmm_blkcoord(numrows, numcols, blocksize, nthrds, blockVectorX, blockVectorAX, matrixBlock);
    spmm_blkcoord_loop(numrows, numcols, blocksize, nthrds, blockVectorX, blockVectorAX, matrixBlock);
    
    //gramXAX = full(blockVectorX'*blockVectorAX);
    double *gramXAX=new double[blocksize*blocksize]();
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,blocksize,blocksize,M,1.0,blockVectorX,blocksize,blockVectorAX,blocksize,0.0,gramXAX,blocksize);
    //_XTY(blockVectorX, blockVectorAX, gramXAX, M, blocksize, blocksize, block_width);
    
    //gramXAX = (gramXAX + gramXAX')*0.5;
    double *transGramXAX=new double[blocksize*blocksize]();
    transpose(gramXAX,transGramXAX, blocksize, blocksize);
    
    make_identity_mat(identity_BB,blocksize, blocksize);
    make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize); //--> used in loop
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blocksize, blocksize, blocksize, 0.5, transGramXAX, blocksize, identity_BB, blocksize, 0.5, gramXAX, blocksize);
    
    double *temp_GramXAX = new double[blocksize*blocksize]();
    transpose(gramXAX, temp_GramXAX, blocksize, blocksize);
    
    //dummy call: work size query
    lwork = -1;
    double *tempLambda = new double[blocksize]();
    
    dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, &work_query, &lwork, &info);
    if(info != 0)
    {
        cout<<"Error in dummy call"<<endl;
        //exit(1);
    }

    lwork = (int) work_query;
    work = new double[lwork]();
    
    dsygv_(&itype, &jobz, &uplo, &blocksize, temp_GramXAX, &blocksize, identity_BB, &blocksize, tempLambda, work, &lwork, &info);
    
    if( info != 0 )
    {
        printf( "The algorithm failed to compute eigenvalues.\n" );
    }
    transpose(temp_GramXAX, gramXAX, blocksize, blocksize);
    
    free(temp_GramXAX);
    free(work);
    free(transGramXAX);
    
    //[coordX,gramXAX]=eig(gramXAX,eye(blockSize));
    //lambda=diag(gramXAX);

    //double *tempLambda= new double[blocksize]();
    //info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', blocksize, gramXAX, blocksize, tempLambda );
    double *lambda = (double *) malloc(blocksize * blocksize * sizeof(double));
    diag(tempLambda, lambda, blocksize);
    
    free(tempLambda);
    
    //note: after applying dsyevd_ function gramXAX will be coordX
    //blockVectorX  =  blockVectorX*coordX;  //after this, dimension of blockVectorX will be M*blocksize
    //blockVectorAX = blockVectorAX*coordX; //blockVectorAX will remain M*blocksize
    //double *newBlockVectorX=new double[M*blocksize]();
    
    double *coordX;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,blocksize,blocksize,1.0,blockVectorX,blocksize,gramXAX,blocksize,0.0,newX,blocksize);
    
    //copyResult=LAPACKE_dlacpy(LAPACK_ROW_MAJOR,' ',M,blocksize,newBlockVectorX,blocksize,blockVectorX,blocksize); //blockVectorX=newBlockVectorX
    custom_dlacpy(newX, blockVectorX, M, blocksize);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.00, blockVectorAX, blocksize, gramXAX, blocksize, 0.00, newX, blocksize);
    custom_dlacpy(newX, blockVectorAX, M, blocksize);
    free(gramXAX);
    

    int gramASize = -1;
    int *activeMask = (int *) malloc(blocksize * sizeof(int));

    #pragma omp parallel for
    for(i = 0 ; i < blocksize ; i++)
        activeMask[i] = 1;

    iteraton_start_time = omp_get_wtime();

    int activeRSize = 0, activePSize = 0, explicitGramFlag = 0, restart = 0;
    
    //loop starts here
    for(iterationNumber = 1 ; iterationNumber <= maxIterations ; iterationNumber++)
    {
        // for(i = 0 ; i < numTaks ; i++)
        //     taskTiming[i] = 0;

        //cout << "\niterationNumber: " << iterationNumber << endl;
        loop_start_time = omp_get_wtime();

        //if 12 nested if
        //blockVectorR = blockVectorAX - blockVectorX*spdiags(lambda,0,blockSize,blockSize);
    
        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorX, blocksize, lambda,blocksize, 0.0, blockVectorR, blocksize); //XY code 1
        //taskTiming[1] += (omp_get_wtime() - tstart);

        //tstart = omp_get_wtime();
        mat_sub(blockVectorAX, blockVectorR, blockVectorR , M, blocksize); //SUB : 4
        //taskTiming[4] += (omp_get_wtime() - tstart); 

        //residualNorms=full(sqrt(sum(conj(blockVectorR).*blockVectorR)')); 
        #pragma omp parallel for default(shared)
        for(i = 0 ; i < blocksize ; i++)
            residualNorms[i] = 0.0;

        //tstart = omp_get_wtime();
        mat_mult(blockVectorR, blockVectorR, newX, M, blocksize); //MULT : 5
        //taskTiming[5] += (omp_get_wtime() - tstart);
        
        sum_sqrt(newX, residualNorms, M, blocksize);
        
        //residualNormsHistory(1:blockSize,iterationNumber)=residualNorms;
        //activeMask = full(residualNorms > residualTolerance) & activeMask;

        //tstart = omp_get_wtime();
        update_activeMask(activeMask, residualNorms, residualTolerance, blocksize);
        //taskTiming[8] += (omp_get_wtime() - tstart); //UPDATE : 8
    
        //currentBlockSize = sum(activeMask);
        currentBlockSize=0;
        for(i = 0 ; i < blocksize ; i++)
            currentBlockSize += activeMask[i];

        if(currentBlockSize == 0)
        {
            cout<<"converge!!"<<endl;
            break;
        }

        //if loop-17
        //blockVectorR(:,activeMask) = blockVectorR(:,activeMask) - ...
        //        blockVectorX*(blockVectorX'*blockVectorR(:,activeMask));
        
        if(currentBlockSize != prevCurrentBlockSize)
        {
            printf("Here 1\n");
            delete []activeBlockVectorR;
            activeBlockVectorR = new double[M * currentBlockSize]();
            delete []temp2;
            temp2 = new double[blocksize * currentBlockSize]();
        }

        //tstart = omp_get_wtime();
        getActiveBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize); //GET: 7
        //taskTiming[7] += (omp_get_wtime() - tstart);
    
        //blockVectorX'*blockVectorR(:,activeMask)  -> temp2 is the result
        std::memset(temp2, 0.0, sizeof(temp2));
        
        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorX, blocksize, activeBlockVectorR, currentBlockSize,0.0, temp2, currentBlockSize); //XTY : 2
        //_XTY(blockVectorX, activeBlockVectorR, temp2, M, blocksize, currentBlockSize, block_width);
        //taskTiming[2] += (omp_get_wtime() - tstart);
    
        //temp3 = blockVectorX * temp2
        if(currentBlockSize!=prevCurrentBlockSize)
        {
            printf("Here 2\n");
            delete []temp3;
            temp3 = new double[M*currentBlockSize]();
        }

        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, blocksize, 1.0, blockVectorX, blocksize, temp2, currentBlockSize, 0.0, temp3, currentBlockSize);
        //taskTiming[1] += (omp_get_wtime() - tstart);

        //tstart = omp_get_wtime();
        mat_sub(activeBlockVectorR, temp3, activeBlockVectorR, M, currentBlockSize);
        //taskTiming[4] += (omp_get_wtime() - tstart);

        //tstart = omp_get_wtime();
        //updateBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize); //UPDATE: 8
        //taskTiming[8] += (omp_get_wtime() - tstart);
    
        //------- if 18 ------
        //gramRBR=blockVectorR(:,activeMask)'*blockVectorR(:,activeMask);  //blockVectorR(:,activeMask) ->activeBlockVectorR

        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorR, currentBlockSize, activeBlockVectorR, currentBlockSize, 0.0,gramRBR, currentBlockSize);
        //_XTY(activeBlockVectorR, activeBlockVectorR, gramRBR, M, currentBlockSize, currentBlockSize, block_width);
        //taskTiming[2] += (omp_get_wtime() - tstart);

        //[gramRBR,cholFlag]=chol(gramRBR);
        transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);
        dpotrf_( &uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info );
        if(info != 0)
        {
            cout<<"dportf_ error 2!!"<<endl;
            break;
        }

        transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);
    
        #pragma omp parallel for private(j) default(shared)
        for(i = 0 ; i < currentBlockSize ; i++)
        {
            for(j = 0 ; j < i ; j++)
            {
                gramRBR[i * currentBlockSize + j] = 0.0;
            }
        }
        //------- if 18 nested if -----
        if(info == 0)
        {
            inverse(gramRBR, currentBlockSize, currentBlockSize);

            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorR, currentBlockSize, gramRBR, currentBlockSize, 0.0, temp3, currentBlockSize);
            //taskTiming[1] += (omp_get_wtime() - tstart);

            //tstart = omp_get_wtime();
            custom_dlacpy(temp3, activeBlockVectorR, M, currentBlockSize); //DLACPY: 11
            //taskTiming[10] += (omp_get_wtime() - tstart);
            
            //tstart = omp_get_wtime();
            updateBlockVector(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize);
            //taskTiming[8] += (omp_get_wtime() - tstart);
        } //end if

        //tstart = omp_get_wtime();
        #pragma omp parallel for private(j) default(shared)
        for(i = 0; i < M ; i++)
        {
            for(j = 0 ; j < currentBlockSize ; j++)
            {
                activeBlockVectorAR[i * currentBlockSize + j] = 0.0;
            }
        }
        //taskTiming[0] += (omp_get_wtime() - tstart); //SETZERO : 0
     
        //tstart = omp_get_wtime();
        //mkl_dcscmm(&transA, &M, &currentBlockSize, &M, &alpha, matdescra, xrem, irem, colptrs, colptrs+1, activeBlockVectorR, &currentBlockSize,  &beta, activeBlockVectorAR, &currentBlockSize);
        //mkl_dcsrmm(&transA, &M, &currentBlockSize, &M, &alpha, matdescra, acsr, ja, ia, ia+1, activeBlockVectorR, &currentBlockSize, &beta, activeBlockVectorAR, &currentBlockSize);
        //spmm_blkcoord(numrows, numcols, currentBlockSize, nthrds, activeBlockVectorR, activeBlockVectorAR, matrixBlock);
        spmm_blkcoord_loop(numrows, numcols, currentBlockSize, nthrds, activeBlockVectorR, activeBlockVectorAR, matrixBlock);
        
        //taskTiming[6] += (omp_get_wtime() - tstart); //SPMM 6

        //tstart = omp_get_wtime();
        //updateBlockVector(activeBlockVectorAR, activeMask, blockVectorAR, M, blocksize, currentBlockSize);
        //taskTiming[8] += (omp_get_wtime() - tstart);

        if(iterationNumber > 1)
        {
            //if 20 first nested if
            // gramPBP=blockVectorP(:,activeMask)'*blockVectorP(:,activeMask);

            //tstart = omp_get_wtime();
            getActiveBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
            //taskTiming[7] += (omp_get_wtime() - tstart);

            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorP, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramPBP, currentBlockSize);
            //_XTY(activeBlockVectorP, activeBlockVectorP, gramPBP, M, currentBlockSize, currentBlockSize, block_width);
            //taskTiming[2] += (omp_get_wtime() - tstart);
            
            transpose(gramPBP, trans_gramPBP, currentBlockSize, currentBlockSize);
            dpotrf_( &uplo, &currentBlockSize, trans_gramPBP, &currentBlockSize, &info );
            if(info != 0)
            {
                cout<<"dportf_ error 3"<<endl;
                break;
            }

            transpose(trans_gramPBP, gramPBP, currentBlockSize, currentBlockSize);
         
            //making the lower part of gramPBP zero
            #pragma omp parallel for private(j) default(shared)
            for(i = 0 ; i < currentBlockSize ; i++)
            {
                for(j = 0 ; j < i ; j++)
                {
                    gramPBP[i * currentBlockSize + j] = 0.0;
                }
            }

            if(info == 0)
            {
                //if 20 first nested if 2
                // blockVectorP(:,activeMask) = blockVectorP(:,activeMask)/gramPBP;
                inverse(gramPBP, currentBlockSize, currentBlockSize);

                /*if(currentBlockSize!=prevCurrentBlockSize)
                 {
                   delete []newActP;
                   newActP = new double[M*currentBlockSize]();
                   delete []activeBlockVectorAP;
                   activeBlockVectorAP= new double[M*currentBlockSize]();
                }*/

                //tstart = omp_get_wtime();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorP, currentBlockSize, gramPBP, currentBlockSize, 0.0, temp3, currentBlockSize);
                //taskTiming[1] += (omp_get_wtime() - tstart);
                
                //tstart = omp_get_wtime();
                custom_dlacpy(temp3, activeBlockVectorP, M, currentBlockSize);
                //taskTiming[10] += (omp_get_wtime() - tstart);
             
                //tstart = omp_get_wtime();
                updateBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);
                //taskTiming[8] += (omp_get_wtime() - tstart);

             
                //blockVectorAP(:,activeMask) = blockVectorAP(:,activeMask)/gramPBP;
                //temp1Time=omp_get_wtime();
                getActiveBlockVector(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize);
                //taskTiming[10] += (omp_get_wtime() - tstart);
                
                //tstart = omp_get_wtime();
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, currentBlockSize, currentBlockSize, 1.0, activeBlockVectorAP, currentBlockSize, gramPBP, currentBlockSize, 0.0, temp3, currentBlockSize);
                //taskTiming[1] += (omp_get_wtime() - tstart);
                
                //tstart = omp_get_wtime();
                custom_dlacpy(temp3, activeBlockVectorAP, M, currentBlockSize);
                //taskTiming[10] += (omp_get_wtime() - tstart);
                
                //tstart = omp_get_wtime();
                updateBlockVector(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize);
                //taskTiming[8] += (omp_get_wtime() - tstart);
            } //end if info
            else
            {
                cout<<"BLOPEX:lobpcg:DirectionNotFullRank...The direction matrix is not full rank."<<endl;
            }
        } //end outer if

        //restart=1;
        //The Raileight-Ritz method for [blockVectorX blockVectorR blockVectorP]

        //------ if 21
        int flag = 1;
        for(i = 0 ; i < blocksize ; i++)
        {
            //cout<<"residualNorms[i] :"<<residualNorms[i]<<endl;
            if(residualNorms[i] < 4.0538e-10)
            {
                flag = 0;
                break;
            }
        }
        if(flag == 0)
            explicitGramFlag = 1;
        else
            explicitGramFlag = 0;

        activeRSize = currentBlockSize;

        //---- if 22 -----
        //cout<<"if 22"<<endl;
        if(iterationNumber == 1)
        {
            activePSize = 0;
            restart = 1;
            //cout<<"restart: "<<restart<<endl;
        }
        else
        {
            activePSize = currentBlockSize;
            restart = 0;
        }

        //gramXAR=full(blockVectorAX'*blockVectorR(:,activeMask));
        //gramRAR=full(blockVectorAR(:,activeMask)'*blockVectorR(:,activeMask));
        //gramRAR=(gramRAR'+gramRAR)*0.5;
        
        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorAX, blocksize, activeBlockVectorR, currentBlockSize, 0.0, gramXAR, currentBlockSize);
        //_XTY(blockVectorAX, activeBlockVectorR, gramXAR, M, blocksize, currentBlockSize, block_width);
        //taskTiming[2] += (omp_get_wtime() - tstart);
     
        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAR, currentBlockSize, activeBlockVectorR, currentBlockSize, 0.0, gramRAR, currentBlockSize);
        //_XTY(activeBlockVectorAR, activeBlockVectorR, gramRAR, M, currentBlockSize, currentBlockSize, block_width);
        //taskTiming[2] += (omp_get_wtime() - tstart);
     
        transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);
       
        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);
        //taskTiming[1] += (omp_get_wtime() - tstart);

     
        //--- cond_try for loop -----
        for(int cond_try = 1 ; cond_try <=2 ; cond_try++)
        {
            if(restart == 0) //---- if 24 ----
            {
                if(restart == 0)
                {
                    //cout<<"if 24"<<endl;
                    //gramXAP=full(blockVectorAX'*blockVectorP(:,activeMask));

                    //activeBlockVectorP= new double[M*currentBlockSize]();
                    //getActiveBlockVector(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize);

                    //tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorAX, blocksize, activeBlockVectorP, currentBlockSize, 0.0, gramXAP, currentBlockSize);
                    //_XTY(blockVectorAX, activeBlockVectorP, gramXAP, M, blocksize, currentBlockSize, block_width);
                    //taskTiming[2] += (omp_get_wtime() - tstart);
                 
                    //tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAR, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramRAP, currentBlockSize);
                    //_XTY(activeBlockVectorAR, activeBlockVectorP, gramRAP, M, currentBlockSize, currentBlockSize, block_width);
                    //taskTiming[2] += (omp_get_wtime() - tstart);
                    
                    //tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorAP, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramPAP, currentBlockSize);
                    //_XTY(activeBlockVectorAP, activeBlockVectorP, gramPAP, M, currentBlockSize, currentBlockSize, block_width);
                    //taskTiming[2] += (omp_get_wtime() - tstart);
                 
                    //gramPAP=(gramPAP'+gramPAP)*0.5;

                    //tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, gramPAP, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramPAP, currentBlockSize);
                    //taskTiming[1] += (omp_get_wtime() - tstart);
                 
                    if(explicitGramFlag==1)
                    {
                        cout<<"nested if 24"<<endl;
                    }
                    else
                    {
                        //cout<<"if 24 nested if 1 else"<<endl;
                        //gramA = [ diag(lambda)  gramXAR  gramXAP
                        //           gramXAR'      gramRAR  gramRAP
                        //           gramXAP'      gramRAP'  gramPAP ];

                        gramASize = blocksize+currentBlockSize+currentBlockSize;

                        gramA = new double[gramASize*gramASize]();
                        
                        mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                        mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                        mat_copy(gramXAP, blocksize, currentBlockSize, gramA, 0, blocksize+currentBlockSize, gramASize);
            
                        transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                        mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                        mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                        mat_copy(gramRAP, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize+currentBlockSize, gramASize);
                        transpose(gramXAP, transGramXAP, currentBlockSize, blocksize);
                        
                        transpose(gramRAP, transGramRAP, currentBlockSize, currentBlockSize);
                         
                        mat_copy(transGramXAP, currentBlockSize, blocksize, gramA, blocksize+currentBlockSize, 0, gramASize);
                        mat_copy(transGramRAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize, gramASize);
                        mat_copy(gramPAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                    } //end else

                    //double *gramXBP = new double[blocksize*currentBlockSize]();
                    //tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, currentBlockSize, M, 1.0, blockVectorX, blocksize, activeBlockVectorP, currentBlockSize, 0.0, gramXBP, currentBlockSize);
                    //_XTY(blockVectorX, activeBlockVectorP, gramXBP, M, blocksize, currentBlockSize, block_width);
                    //taskTiming[2] += (omp_get_wtime() - tstart);

                    //tstart = omp_get_wtime();
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, M, 1.0, activeBlockVectorR, currentBlockSize, activeBlockVectorP, currentBlockSize, 0.0, gramRBP, currentBlockSize);
                    //_XTY(activeBlockVectorR, activeBlockVectorP, gramRBP, M, currentBlockSize, currentBlockSize, block_width);
                    //taskTiming[2] += (omp_get_wtime() - tstart);
                 
                    if(explicitGramFlag==1)
                    {
                        cout<<"if 24 nested if 3"<<endl;
                    }
                    else
                    {
                        //cout<<"if 24 nested if 3 else"<<endl;
                        //gramB=[eye(blockSize) zeros(blockSize,activeRSize) gramXBP
                        //       zeros(blockSize,activeRSize)' eye(activeRSize) gramRBP
                        //       gramXBP' gramRBP' eye(activePSize) ];

                        gramB = new double[gramASize*gramASize];
                        
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
                    }
                } //inner if end
            } //outer if end
            else //---- if 24 else ----
            {
                if(explicitGramFlag == 1 ) //--- if 24 else nested if --
                {
                    //cout<<"if 24 else nested if"<<endl;
                    //gramA = [ gramXAX   gramXAR
                    //          gramXAR'    gramRAR  ];
                    //gramB = [ gramXBX  gramXBR
                    //            gramXBR' eye(activeRSize)  ];
                }
                else //--- if 24 else nested else;
                {
                    //cout<<"if 24 else nested else"<<endl;
                    //gramA = [ diag(lambda)  gramXAR
                    //          gramXAR'        gramRAR  ];
                    //gramB = eye(blockSize+activeRSize);
                    gramASize = blocksize + activeRSize;
                    gramA = new double[gramASize*gramASize]();
                    
                    mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                    mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                    
                    transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                    mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                    mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                    
                    gramB = new double[gramASize*gramASize]();
                    make_identity_mat(gramB, gramASize, gramASize);
                }
            } //end else

            //--------- if 25 part of it ------
            //cout<<"if 25 part of it"<<endl;
            if(cond_try == 1 && ~restart)
            {
                //cout<<"if 25 else-> break from here"<<endl;
                //restart=1;
                break;
            }
        }//inner loop finish here


        //tstart = omp_get_wtime();

        eigen_value = new double[gramASize]();
        double *trans_gramA = new double[gramASize*gramASize]();
        double *trans_gramB = new double[gramASize*gramASize]();
      
        transpose(gramA, trans_gramA, gramASize, gramASize);
        transpose(gramB, trans_gramB, gramASize, gramASize);
        
        lwork = -1;
        dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, &work_query, &lwork, &info);
      
        if(info != 0)
            cout<<"Error in dummy call"<<endl;

        lwork = (int) work_query;

        work = new double[lwork]();
        dsygv_(&itype, &jobz, &uplo, &gramASize, trans_gramA, &gramASize, trans_gramB, &gramASize, eigen_value, work, &lwork, &info);
      
        if(info != 0)
            cout<<"Error in eigen value calculation"<<endl;

        transpose(trans_gramA, gramA, gramASize, gramASize);
        transpose(trans_gramB, gramB, gramASize, gramASize);
        delete []trans_gramA;
        delete []trans_gramB;
      
        if( info != 0 )
        { 
            printf("LAPACKE_dsygv error: The algorithm failed to compute eigenvalues.\n" );
            break;
        }

        diag(eigen_value, lambda, blocksize);
      
        int column = 0;
        coordX = new double[gramASize*blocksize]();
      
        for(j=0; column<blocksize && j<gramASize; j++)
        {   
            #pragma omp parallel for default(shared)
            for(i = 0 ; i < gramASize ; i++)
            {
                coordX[i * blocksize + column] = gramA[i * gramASize + j];
            }
            column++;
        }
        //taskTiming[9] += (omp_get_wtime() - tstart);

        if(restart == 0)
        {
            // partil result- part1:- blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:)
            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorP, blocksize);
            //taskTiming[1] += (omp_get_wtime() - tstart);
            
            /*blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize+activeRSize+activePSize,:); */
            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorP, currentBlockSize, coordX+((blocksize+currentBlockSize)*blocksize), blocksize, 1.0, blockVectorP, blocksize);
            //taskTiming[1] += (omp_get_wtime() - tstart);
         
            /*blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorAP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize + activeRSize+activePSize,:);*/

            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorAR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorAP, blocksize);
            //taskTiming[1] += (omp_get_wtime() - tstart);
            
            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, currentBlockSize, 1.0, activeBlockVectorAP, currentBlockSize, coordX+((blocksize+currentBlockSize)*blocksize), blocksize, 1.0, blockVectorAP, blocksize);
            //taskTiming[1] += (omp_get_wtime() - tstart);
        }
        else
        {
            //cout<<"if 26 else"<<endl;
            // blockVectorP =   blockVectorR(:,activeMask)*...
            //    coordX(blockSize+1:blockSize+activeRSize,:);

            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, M, blocksize, activeRSize, 1.0, activeBlockVectorR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorP, blocksize);
            //taskTiming[1] += (omp_get_wtime() - tstart);
         
            //blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:);
        
            //tstart = omp_get_wtime();
            cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, M, blocksize, activeRSize, 1.0, activeBlockVectorAR, currentBlockSize, coordX+(blocksize*blocksize), blocksize, 0.0, blockVectorAP, blocksize);
            //taskTiming[1] += (omp_get_wtime() - tstart);
        }

        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorX, blocksize, coordX, blocksize, 0.0, newX, blocksize);
        //taskTiming[1] += (omp_get_wtime() - tstart);

        //tstart = omp_get_wtime();
        mat_addition(newX, blockVectorP, blockVectorX, M, blocksize);
        //taskTiming[3] += (omp_get_wtime() - tstart);
        
        //tstart = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 1.0, blockVectorAX, blocksize, coordX, blocksize, 0.0, newX, blocksize);
        //taskTiming[1] += (omp_get_wtime() - tstart);

        //tstart = omp_get_wtime();
        mat_addition(newX, blockVectorAP, blockVectorAX, M, blocksize);
        //taskTiming[3] += (omp_get_wtime() - tstart);
            

        //temp1Time=omp_get_wtime();
        delete []eigen_value;
        delete []work;
        delete []gramA;
        delete []gramB;
        delete []coordX;
        
        prevCurrentBlockSize = currentBlockSize;
        loopTime[iterationNumber - 1] = omp_get_wtime() - loop_start_time ;

        /*printf("%10s %.6lf sec.\n", "SETZERO", taskTiming[0]);
        printf("%10s %.6lf sec.\n", "XY", taskTiming[1]);
        printf("%10s %.6lf sec.\n", "XTY", taskTiming[2]);
        printf("%10s %.6lf sec.\n", "ADD", taskTiming[3]);
        printf("%10s %.6lf sec.\n", "SUB", taskTiming[4]);
        printf("%10s %.6lf sec.\n", "MULT", taskTiming[5]);
        printf("%10s %.6lf sec.\n", "SPMM", taskTiming[6]);
        printf("%10s %.6lf sec.\n", "GET", taskTiming[7]);
        printf("%10s %.6lf sec.\n", "UPDATE", taskTiming[8]);
        printf("%10s %.6lf sec.\n", "EIGEN", taskTiming[9]);
        printf("%10s %.6lf sec.\n", "DLACPY", taskTiming[10]);

        printf("\n");*/

        /*cout << "Eigen values: " << endl;
        for(i = 0 ; i < blocksize ; i++)
        {
           cout << lambda[i * blocksize + i] << endl;
        }*/

        for(i = 0 ; i < blocksize ; i++)
        {
            //cout << lambda[i * blocksize + i] << endl;
            saveLamda[i][iterationNumber - 1] = lambda[i * blocksize + i];
        }

    } //loop ends

    //print_mat(blockVectorP, 16, blocksize);

    //iteraton_time = omp_get_wtime() - iteraton_start_time;
    
    /*cout<<"Total iterations: "<<iterationNumber-1<<endl;
    cout<<"\nTotal Execution time: "<<iteraton_time<<" sec."<<endl;*/

    //print_mat(blockVectorX, 16, blocksize);

    /*cout<<"printing activeMask:" <<endl;
    for(i = 0 ; i < blocksize ; i++)
    {
      cout << activeMask[i] << " ";
    }
    cout<<endl;*/
    
    /*cout<<"Final Eigen values: "<<endl;
    for(i = 0 ; i < blocksize ; i++)
    {
	   cout << lambda[i * blocksize + i] << endl;
    }*/

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
}
