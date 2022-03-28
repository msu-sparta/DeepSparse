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
#include <vector>
#include <string>
using namespace std;

#include <omp.h>
#include "../inc/matrix_ops_cpu.h"
#include "../inc/matrix_ops_gpu_v6.h"
#include "../inc/memory_manager_v6.h"

#if defined(NSYS)
#include <nvToolsExt.h>
#endif

int main(int argc, char *argv[])
{
    int M, N, index = 0;
    double residualTolerance = 0.0001;
    long iterationNumber; maxIterations = 5;
    int i, j, k, info, convergeFlag, constraintStyle = 0; 

    stringstream s(argv[1]);
    s >> blocksize;
    stringstream s1(argv[2]);
    s1 >> block_width;
    cout << "# of RHS Vectors: " << blocksize << endl;
    cout << "Tile size: " << block_width << endl; 
    
    num_per_blk = block_width * blocksize; 
    memGranularity = num_per_blk * sizeof(double);
    
    int currentBlockSize = blocksize, prevCurrentBlockSize = blocksize;

    char *filename = argv[3] ; 
    wblk = block_width; 
    // read_csr(filename); // reading sparse matrix in CSR format for file
  read_csr_DJA(filename);
  cout << (int) d_ja[0] << " ---> " << (int) d_ja[nnonzero - 1] << endl;
  // cout << d_ja[0] << " -- " << d_ja[nnonzero - 1] << endl;
  cout << "In double format: " << nnonzero * sizeof(double) * 1e-9
       << " GB --> In integer format: " << nnonzero * sizeof(int) * 1e-9 
       << " GB" << endl; 
  cout << "Successfully finised reading the file" << endl;

    #pragma omp parallel shared(nthrds)
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }
    
    cout << "nthrds: " << nthrds << endl;
    cout << "omp_thread_count: " << omp_thread_count() << endl;
    int nthrds_gpu = 128;    
    
    M = numrows;
    N = numcols;

    double *blockVectorX = (double *) malloc(M * blocksize * sizeof(double)); 

    setMatrixInfo(blockVectorX, M, blocksize, "blockVectorX"); 

    srand(0);
   //#pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < blocksize ; j++)
        {
            blockVectorX[i * blocksize + j] = (double)rand()/(double)RAND_MAX;
        }
    }
    // memory allocation for matrices 
    double *blockVectorAX = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorR = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorAR = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorP = (double *) malloc(M * blocksize * sizeof(double));
    double *blockVectorAP = (double *) malloc(M * blocksize * sizeof(double));
    double *newX = (double *) malloc(M * blocksize * sizeof(double));
    double *newAX = (double *) malloc(M * blocksize * sizeof(double));
    double *newP = (double *) malloc(M * blocksize * sizeof(double)); 
    double *newAP = (double *) malloc(M * blocksize * sizeof(double)); 
    double *new_COL = (double *) malloc(M * blocksize * sizeof(double)); 
    

    setMatrixInfo(blockVectorAX, M, blocksize, "blockVectorAX");
    setMatrixInfo(blockVectorR, M, blocksize, "blockVectorR");
    setMatrixInfo(blockVectorAR, M, blocksize, "blockVectorAR");
    setMatrixInfo(blockVectorP, M, blocksize, "blockVectorP");
    setMatrixInfo(blockVectorAP, M, blocksize, "blockVectorAP");
    setMatrixInfo(newX, M, blocksize, "newX");
    setMatrixInfo(newAX, M, blocksize, "newAX");
    setMatrixInfo(newP, M, blocksize, "newP");
    setMatrixInfo(newAP, M, blocksize, "newAP");
    setMatrixInfo(new_COL, M, blocksize, "new_COL");

    double *activeBlockVectorR = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorAR = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *activeBlockVectorAP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *temp3 = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *temp3_AP = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *temp3_P = (double *) malloc(M * currentBlockSize * sizeof(double));
    double *temp3_R = (double *) malloc(M * currentBlockSize * sizeof(double));

    setMatrixInfo(activeBlockVectorR, M, currentBlockSize, "activeBlockVectorR");
    setMatrixInfo(activeBlockVectorAR, M, currentBlockSize, "activeBlockVectorAR");
    setMatrixInfo(activeBlockVectorP, M, currentBlockSize, "activeBlockVectorP");
    setMatrixInfo(activeBlockVectorAP, M, currentBlockSize, "activeBlockVectorAP");
    setMatrixInfo(temp3, M, currentBlockSize, "temp3");
    setMatrixInfo(temp3_AP, M, currentBlockSize, "temp3_AP");
    setMatrixInfo(temp3_P, M, currentBlockSize, "temp3_P");
    setMatrixInfo(temp3_R, M, currentBlockSize, "temp3_R");
    
    double *gramA, *gramB, *eigen_value;

    double *gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *trans_gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *temp2 = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *trans_gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));    
    double *gramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *transGramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *transGramRAP= (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramPAP= (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *identity_PAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));

    setMatrixInfo(gramPBP, currentBlockSize, currentBlockSize, "gramPBP");
    setMatrixInfo(trans_gramPBP, currentBlockSize, currentBlockSize, "trans_gramPBP");
    setMatrixInfo(temp2, currentBlockSize, currentBlockSize, "temp2");
    setMatrixInfo(gramRBR, currentBlockSize, currentBlockSize, "gramRBR");
    setMatrixInfo(trans_gramRBR, currentBlockSize, currentBlockSize, "trans_gramRBR");
    setMatrixInfo(gramRAR, currentBlockSize, currentBlockSize, "gramRAR");
    setMatrixInfo(transGramRAR, currentBlockSize, currentBlockSize, "transGramRAR");
    setMatrixInfo(transGramRAP, currentBlockSize, currentBlockSize, "transGramRAP");
    setMatrixInfo(gramRAP, currentBlockSize, currentBlockSize, "gramRAP");
    setMatrixInfo(gramPAP, currentBlockSize, currentBlockSize, "gramPAP");
    setMatrixInfo(identity_PAP, currentBlockSize, currentBlockSize, "identity_PAP");
    setMatrixInfo(gramRBP, currentBlockSize, currentBlockSize, "gramRBP");

    double *gramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *gramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *transGramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    double *gramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double)); 
    double *transGramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double));

    setMatrixInfo(gramXAP, currentBlockSize, blocksize, "gramXAP");
    setMatrixInfo(transGramXAP, currentBlockSize, blocksize, "transGramXAP");
    setMatrixInfo(gramXBP, currentBlockSize, blocksize, "gramXBP");
    setMatrixInfo(transGramXBP, currentBlockSize, blocksize, "transGramXBP");
    setMatrixInfo(transGramRBP, currentBlockSize, blocksize, "transGramRBP");
    setMatrixInfo(gramXAR, currentBlockSize, blocksize, "gramXAR");
    setMatrixInfo(transGramXAR, currentBlockSize, blocksize, "transGramXAR");
    

    double *zeros_CB_B = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *zeros_B_CB = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    double *identity_BB = (double *) malloc(blocksize * blocksize * sizeof(double));

    setMatrixInfo(zeros_CB_B, currentBlockSize, blocksize, "zeros_CB_B");
    setMatrixInfo(zeros_B_CB, currentBlockSize, blocksize, "zeros_B_CB");
    setMatrixInfo(identity_BB, blocksize, blocksize, "identity_BB");

    double *residualNorms = (double *) malloc(blocksize * sizeof(double));

    setMatrixInfo(residualNorms, 1, blocksize, "residualNorms");

    std::memset(zeros_B_CB, 0.0, blocksize * currentBlockSize * sizeof(double));
    std::memset(zeros_CB_B, 0.0, blocksize * currentBlockSize * sizeof(double));

    initialize_timing_variables();
    
    #if defined(TIMER)
        for(i = 0 ; i < maxIterations ; i++)
        {
            h2dTime[i] = 0;
            d2hTime[i] = 0;
            h2dTransfer[i] = 0;
            d2hTransfer[i] = 0;
            mmTiming[i] = 0;
            mmLoopTime[i] = 0;
            pushPopTime[i] = 0;
            freeBlockTiming[i] = 0;
            grabRHSTiming[i] = 0;
            dsUpdateTiming[i] = 0;
            copyToHostTiming[i] = 0;
        }
    #endif
    
    //timing variables
    // int numTasks = 19;
    // vector<string> function_name{"LOOPS", "X*Y", "Xt*Y", "ADD", "SUB", "MULT", "SPMM", "GET", "UPDATE", "dsygv", "DLACPY", "INVERSE", "TRANSPOSE", "mat_copy", "dpotrf", "memset", "SUMSQRT", "diag"};
    // double **taskTiming = (double **) malloc(sizeof(double *) * maxIterations);
    // for(i = 0 ; i < maxIterations ; i++)
    //     taskTiming[i] = (double *) malloc(sizeof(double) * numTasks);
    
    // #pragma omp parallel for default(shared) private(j)
    // for(i = 0 ; i < maxIterations ; i++)
    // {
    //     for(j = 0 ; j < numTasks ; j++)
    //     {
    //         taskTiming[i][j] = 0.0;
    //     }
    // }

    double tstart, tend, temp1Time;
    double loop_start_time = 0, loop_finish_time = 0;
    double iteraton_time = 0;

    // saveLamda[blocksize * maxIterations]
    double **saveLamda = (double **) malloc(blocksize * maxIterations * sizeof(double *));
    for(i = 0 ; i < blocksize ; i++)
        saveLamda[i] = (double *) malloc(maxIterations * sizeof(double));
    
    for(i = 0 ; i < blocksize ; i++)
        for(j = 0 ; j < maxIterations ; j++)
            saveLamda[i][j] = 0.0;

    double *loopTime = (double *) malloc(maxIterations * sizeof(double));
    for(i = 0 ; i < maxIterations ; i++)
        loopTime[i] = 0;

    int taskCount_nonLoop, taskCount_firstLoop, taskCount_secondLoop;
    int partCount_nonLoop, partCount_firstLoop, partCount_secondLoop;
    int *partBoundary_nonLoop, *partBoundary_firstLoop, *partBoundary_secondLoop;
    int structIterator, partIterator;

    taskCount_firstLoop = buildTaskInfoStruct(taskInfo_firstLoop, argv[4]);
    cout << "taskCount_firstLoop: " << taskCount_firstLoop << endl;

    taskCount_secondLoop = buildTaskInfoStruct(taskInfo_secondLoop, argv[5]);
    cout << "taskCount_secondLoop: " << taskCount_secondLoop << endl;
    
    int tid, taskCounter = 0, block_id, task_id, buf_id, blksz, row_id, col_id;

    // mkl_dcsrmm parameters 
    char matdescra[6];
    char transA = 'n';
    matdescra[0] = 'g'; matdescra[1] = 'l'; matdescra[2] = 'u'; matdescra[3] = 'c';
    int job_dcsrcsc[] = {1, 0, 0, 0, 0, 1}; 
    int dcsrcsc_info = -1;
   
    double alpha = 1.0, beta = 0.0;

    // LAPACKE_dportf & LAPACKE_dsygv parameters
    char jobvl = 'N'; char jobvr = 'V'; char jobz = 'V'; char uplo = 'U';
    int itype = 1; int lwork; double work_query; double *work;

    // if 9
    // gramXBX=blockVectorX'*blockVectorX;
    // [gramXBX,cholFlag]=chol(gramXBX);
    // blockVectorX = blockVectorX/gramXBX;
    
    double *gramXBX = new double[blocksize * blocksize]();

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, blocksize, M, 
                1.0, blockVectorX, blocksize, blockVectorX, blocksize, 0.0, gramXBX, blocksize);
    //_XTY(blockVectorX, blockVectorX, gramXBX, M, blocksize, blocksize, block_width);

    int cholFlag;
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR , 'U' , blocksize , gramXBX , blocksize);

    if(info != 0)
        cout << "dpotrf: chol error!" << endl;

    
    #pragma omp parallel for  private(j) default(shared)
    for(i = 0 ; i < blocksize ; i++)
    {
        for(j = 0 ; j < i ; j++)
        {
            gramXBX[i * blocksize + j] = 0.0;
        }
    }
    
    double *tempGramXBX = new double[blocksize * blocksize]();
    custom_dlacpy(gramXBX, tempGramXBX, blocksize, blocksize);
    inverse(tempGramXBX, blocksize, blocksize);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize,
                1.0, blockVectorX, blocksize,tempGramXBX, blocksize, 0.0, newX, blocksize);
    custom_dlacpy(newX, blockVectorX, M, blocksize);
    delete []tempGramXBX;
    
    // if 17 
    // blockVectorAX = operatorA * blockVectorX
    #pragma omp parallel for private(j) default(shared)
    for(i = 0; i < M ; i++)
    {
        for(j = 0 ; j < blocksize ; j++)
        {
            blockVectorAX[i * blocksize + j] = 0.0;
        }
    }
    
    #if defined(USE_MKL)
        mkl_dcsrmm(&transA, &M, &blocksize, &N, &alpha, matdescra, acsr, ja, ia, ia+1, blockVectorX, &blocksize, &beta, blockVectorAX, &blocksize);
    #else
        spmm_csr_DJA(numrows, numcols, blocksize, ia, d_ja, acsr, blockVectorX, blockVectorAX);
    #endif    
    
    // gramXAX = full(blockVectorX'*blockVectorAX);
    double *gramXAX = new double[blocksize * blocksize]();
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, blocksize, blocksize,
                M, 1.0, blockVectorX, blocksize, blockVectorAX, blocksize, 0.0, gramXAX, blocksize);
    // _XTY(blockVectorX, blockVectorAX, gramXAX, M, blocksize, blocksize, block_width);
    
    // gramXAX = (gramXAX + gramXA  X')*0.5;
    double *transGramXAX = new double[blocksize * blocksize]();
    transpose(gramXAX,transGramXAX, blocksize, blocksize);
    
    make_identity_mat(identity_BB,blocksize, blocksize);
    make_identity_mat(identity_PAP, currentBlockSize, currentBlockSize); // used in loop
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blocksize, blocksize, blocksize, 
                0.5, transGramXAX, blocksize, identity_BB, blocksize, 0.5, gramXAX, blocksize);
    free(transGramXAX);

    double *tempLambda = new double[blocksize]();
    info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, blocksize, gramXAX, blocksize, identity_BB, blocksize, tempLambda);
    
    if(info != 0)
        cout << "The algorithm failed to compute eigenvalues." << endl;
    
    double *lambda = (double *) malloc(blocksize * blocksize * sizeof(double));
    setMatrixInfo(lambda, blocksize, blocksize, "lambda");
    diag(tempLambda, lambda, blocksize);
    free(tempLambda);
    
    // note: after applying dsyevd_ function gramXAX will be coordX
    // blockVectorX  =  blockVectorX*coordX;  //after this, dimension of blockVectorX will be M*blocksize
    // blockVectorAX = blockVectorAX*coordX; //blockVectorAX will remain M*blocksize
    
    double *coordX;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 
                1.0, blockVectorX, blocksize, gramXAX, blocksize, 0.0, newX, blocksize);
    custom_dlacpy(newX, blockVectorX, M, blocksize);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, blocksize, blocksize, 
                1.0, blockVectorAX, blocksize, gramXAX, blocksize, 0.0, newX, blocksize);
    custom_dlacpy(newX, blockVectorAX, M, blocksize);
    free(gramXAX);
    

    int gramASize = 3 * blocksize;
    int *activeMask = (int *) malloc(blocksize * sizeof(int));

    #pragma omp parallel for
    for(i = 0 ; i < blocksize ; i++)
        activeMask[i] = 1;

    int activeRSize = 0, activePSize = 0, explicitGramFlag = 0, restart = 0;
    // allocating necessary memeory on GPU
   

#if defined(USE_CUBLAS)
    int h, t;
    h = omp_get_initial_device();
    t = omp_get_default_device();
    
    device_id = t, host_id = h;

    // cout << h << " " << t << " " << host_id << " " << device_id << endl;

    // collecting status of the Sparse Matrix (like how many row bloacks there, how many nnz per row blocks etc)
    int nrowblk = ceil(1.0 * numrows/block_width);
    int *nnz_per_tile = (int *)malloc(nrowblk * sizeof(int));
    int max_nnz = 0;
    
    printf("                  nrowblk =  %d\n", nrowblk);
    int totalnnz = 0;
    
    for(i = 0 ; i < nrowblk; i++)
    {
        if(i < nrowblk - 1)
            nnz_per_tile[i] = ia[(i + 1) * block_width] - ia[i * block_width];
        else
            nnz_per_tile[i] = ia[numrows] - ia[i * block_width];

        if(max_nnz < nnz_per_tile[i])
            max_nnz = nnz_per_tile[i];
        
        totalnnz += nnz_per_tile[i];
    }
    
    printf("            max_nnz/block =  %d\n", max_nnz);
    printf("                 totalnnz =  %d\n", totalnnz);
    printf("\n");

    int status;
    const double cudaAlpha = 1.0;
    const double cudaBeta = 0.0;
    const double cudaBetaOne = 1.0;
        
    cublasStatus_t cubstat;
    cusparseStatus_t status1;
    cublasHandle_t handle;
    cudaError_t cuberror;
    cusparseMatDescr_t descr = 0;
    cusparseHandle_t cusparseHandle = 0;

    // allocating big chunk of memory on device
    // double device_capacity = 10 * 1e9; //17 * M * blocksize * sizeof(double); //4 * 1073741824; // in bytes
    double fixed_memory = (numrows * blocksize) * sizeof(double) + (numrows + 1) * sizeof(int);  
    capacity = 15.75 * 1e+9 - fixed_memory;
    cout << "capacity: " << capacity/1e+9  << " GB. Fixed_memory: " << fixed_memory/1e+9  << " GB" << endl;
    unsigned long int total_element = capacity / sizeof(double);
    available_mem = capacity = total_element * sizeof(double);
    //return 0;
    cout << "capacity: " << capacity/1e+9 << " GB. # of element: " << capacity / sizeof(double) << endl;

    totalMemBlock = (unsigned long int) floor(capacity/(sizeof(double) * num_per_blk));
    cout << "totalMemBlock: " << totalMemBlock << endl << endl;
    memView.resize(totalMemBlock); 

    // for mm_v4, we need to properly fill the memView vector
    unsigned long int ii, total_block = totalMemBlock;
    // pair<double *, int> tempPR;
    for(ii = 0 ; ii < totalMemBlock ; ii++)
    {
        memView[ii] = make_pair(nullptr, total_block);
            // memView[ii] = tempPR;
        total_block--;
    }
    setMatrixInfo(nullptr, -1, -1, "nullptr");

    d_memory = (double *) omp_target_alloc(capacity, t);
    if(d_memory == NULL){ printf("omp_target_alloc Filed on d_memory\n"); return 0; }

    // We need ot store the whole activeBlockVectorR matrix for SPMM operation
    double *d_activeBlockVectorR, *d_residualNorms;
    int *d_activeMask;

    d_activeBlockVectorR = (double *) omp_target_alloc(M * currentBlockSize * sizeof(double), t);
    if(d_activeBlockVectorR == NULL){ printf("omp_target_alloc Filed d_activeBlockVectorR\n"); return 0; }
    d_residualNorms = (double *) omp_target_alloc(blocksize * sizeof(double), t);
    if(d_residualNorms == NULL){ printf("omp_target_alloc Filed d_residualNorms\n"); return 0; }
    d_activeMask = (int *) omp_target_alloc(blocksize * sizeof(int), t);
    if(d_activeMask == NULL){ printf("omp_target_alloc Filed d_activeMask\n"); return 0; }
    status = omp_target_memcpy(d_activeMask, activeMask, blocksize * sizeof(int), 0, 0, t, h);
    if( status != 0 ){ printf("omp_target_memcpy failed d_activeMask ==> %d\n", cuberror); return 0; }
    
    double *d_colptr; int *d_rowptr;
    // d_colptr = (double *) malloc(nnonzero * sizeof(double));
    // if(d_colptrs == NULL){ printf("omp_target_alloc Filed d_colptrs\n"); return 0; }
    // d_rowptr = (double *) malloc((numrows + 1) * sizeof(double));
    d_rowptr = (int *) omp_target_alloc((numrows + 1) * sizeof(int), t);
    if(d_rowptr == NULL){ printf("omp_target_alloc failed on d_rowptr\n"); return 0; }
    // setMatrixInfo(d_colptr, 1, nnonzero, "d_colptr");
    // setMatrixInfo(d_rowptr, 1, numrows + 1, "d_rowptr");
    
    // #pragma omp parallel for
    // for(i = 0 ; i < numrows + 1 ; i++)
    //     d_rowptr[i] = (double) ia[i];
    
    // #pragma omp parallel for
    // for(i = 0 ; i < nnonzero ; i++)
    //    d_colptr[i] = (double) ja[i];
    // if(d_irem == NULL){ printf("omp_target_alloc Filed d_colptrs\n"); return 0; }

    // status = omp_target_memcpy(d_irem, ja, nnonzero * sizeof(int), 0, 0, t, h);
    // if( status != 0 ){ printf("cudaMemcpy failed irem ==> %d\n", cuberror); return 0; }
    
    status = omp_target_memcpy(d_rowptr, ia, (numrows + 1) * sizeof(int), 0, 0, t, h);
    if( status != 0 ){ printf("cudaMemcpy failed d_rowptr ==> %d\n", cuberror); return 0; }

    // status = omp_target_memcpy(d_xrem, acsr, nnonzero * sizeof(double), 0, 0, t, h);
    // if( status != 0 ){ printf("cudaMemcpy failed xrem ==> %d\n", cuberror); return 0; }

    // new buffers
    double *d_RNBUF  = (double *) omp_target_alloc(nthrds_gpu * blocksize * sizeof(double), t);
    if(d_RNBUF == NULL){ printf("omp_target_alloc Filed d_RNBUF\n"); return 0; }
    double *d_temp2XTYBUF  = (double *) omp_target_alloc(nthrds_gpu * blocksize * currentBlockSize * sizeof(double), t); //xty 1
    if(d_temp2XTYBUF == NULL){ printf("omp_target_alloc Filed d_temp2XTYBUF\n"); return 0; }
    double *d_gramRBRXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * currentBlockSize * currentBlockSize * sizeof(double), t); //xty 2
    if(d_gramRBRXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramRBRXTYBUF\n"); return 0; }
    double *d_gramPBPXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * currentBlockSize * currentBlockSize * sizeof(double), t); //xty 3
    if(d_gramPBPXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramPBPXTYBUF\n"); return 0; }
    double *d_gramXARXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * blocksize * currentBlockSize * sizeof(double), t); //xty 4
    if(d_gramXARXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramXARXTYBUF\n"); return 0; }
    double *d_gramRARXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * currentBlockSize * currentBlockSize * sizeof(double), t); //xty 5
    if(d_gramRARXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramRARXTYBUF\n"); return 0; }
    double *d_gramXAPXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * blocksize * currentBlockSize * sizeof(double), t); //xty 6
    if(d_gramXAPXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramXAPXTYBUF\n"); return 0; }
    double *d_gramRAPXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * currentBlockSize * currentBlockSize * sizeof(double), t); //xty 7
    if(d_gramRAPXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramRAPXTYBUF\n"); return 0; }
    double *d_gramPAPXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * currentBlockSize * currentBlockSize * sizeof(double), t); //xty 8
    if(d_gramPAPXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramPAPXTYBUF\n"); return 0; }
    double *d_gramXBPXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * blocksize * currentBlockSize * sizeof(double), t); //xty 9
    if(d_gramXBPXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramXBPXTYBUF\n"); return 0; }
    double *d_gramRBPXTYBUF  = (double *) omp_target_alloc(nthrds_gpu * currentBlockSize * currentBlockSize * sizeof(double), t); //xty 10
    if(d_gramRBPXTYBUF == NULL){ printf("omp_target_alloc Filed d_gramRBPXTYBUF\n"); return 0; }

    // initialize cusparse library
    status1 = cusparseCreate(&cusparseHandle);
    if (status1 != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("CUSPARSE Library initialization failed");
        return 1;
    }
    // create and setup matrix descriptor
    status1 = cusparseCreateMatDescr(&descr);
    if (status1 != CUSPARSE_STATUS_SUCCESS) 
    {
        printf("Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cubstat = cublasCreate(&handle);
    if(cubstat != CUBLAS_STATUS_SUCCESS) { 
      printf("HandleCreationFailure\n"); 
      //return 0; 
    }
    cudaDeviceSynchronize();
#endif

    // exit(1);

    // print_mat(lambda, blocksize, blocksize);
    // cout << endl;

    double *coordX1 = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *coordX2 = (double *) malloc(blocksize * blocksize * sizeof(double));
    double *coordX3 = (double *) malloc(blocksize * blocksize * sizeof(double));

    setMatrixInfo(coordX1, blocksize, blocksize, "coordX1");
    setMatrixInfo(coordX2, blocksize, blocksize, "coordX2");
    setMatrixInfo(coordX3, blocksize, blocksize, "coordX3");
    setMatrixInfo(acsr, M, N, "acsr");

    // pair<double *, int> prX1 = make_pair(coordX1, 0);
    // pair<double *, int> prX2 = make_pair(coordX2, 0);
    // pair<double *, int> prX3 = make_pair(coordX3, 0);

    // if(!isOnDevice(prX1))  
    // { 
    //     status = reserveOnDevice(coordX1, blocksize, blocksize, 0, 0, blocksize * blocksize);
    //     if(status !=0)
    //         cout << "Failed to reserve coordX1" << endl;
    // }
    // if(!isOnDevice(prX2))  
    // { 
    //     status = reserveOnDevice(coordX2, blocksize, blocksize, 0, 0, blocksize * blocksize);
    //     if(status !=0)
    //         cout << "Failed to reserve coordX2" << endl;
    // }
    // else
    // {
    //     cout << "Already in device" << endl;
    // }
    // if(!isOnDevice(prX3))  
    // { 
    //     status = reserveOnDevice(coordX3, blocksize, blocksize, 0, 0, blocksize * blocksize);
    //     if(status !=0)
    //         cout << "Failed to reserve coordX3" << endl;
    // }
    
    cout << "Offloading first loop to GPU....." << endl;
  // return 0;

    #if defined(NSYS)
        string str = "iterationNumber : " + to_string(iterationNumber);
        nvtxRangePush(str.c_str());
    #endif

    // print_mat(blockVectorAX + (5 * block_width * blocksize), 10, blocksize);
    // Offloading first  iteration to GPU
    iterationNumber = 1;
    #pragma omp parallel
    {
    #pragma omp master
    {
        cout << "iterationNumber: " << iterationNumber << endl;
        loop_start_time = get_seconds();

        clear_buffer(d_RNBUF, 1, blocksize);
        clear_buffer(d_temp2XTYBUF, blocksize, currentBlockSize);
        clear_buffer(d_gramRBRXTYBUF, blocksize, currentBlockSize);
        clear_buffer(d_gramXARXTYBUF, blocksize, currentBlockSize); // xty_id = 4
        clear_buffer(d_gramRARXTYBUF, currentBlockSize, currentBlockSize); // xty_id = 5

        for(structIterator = 0 ; structIterator < taskCount_firstLoop; structIterator++)
        {
            if(taskInfo_firstLoop[structIterator].opCode == 1) // RESET 
            {
                if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RN")) //reseting residualNorms
                {
                    #pragma omp target is_device_ptr(d_residualNorms) map(tofrom: residualNorms[0 : blocksize])\
                    depend(inout: residualNorms[0 : blocksize], d_residualNorms[0 : blocksize])
                    {
                        for(i = 0; i < blocksize; i++)
                            d_residualNorms[i] = residualNorms[i] = 0.0;
                    }
                }
                else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "actMask")) //reseting activeMask
                {
                    update_activeMask_GPU(activeMask, d_activeMask, d_residualNorms, residualTolerance, blocksize);
                }
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 2)  // SpMM
            {
                // cout << "SpMM : block_id - " << block_id << endl;

                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                tstart = omp_get_wtime();
                
                #if defined(NSYS)
                    str = "SPMM b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                SPMM_GPU_MM_v3(d_rowptr, d_ja, acsr, d_activeBlockVectorR, activeBlockVectorAR, 
                        numrows, numcols, currentBlockSize, block_width, block_id, nnz_per_tile, d_memory, ia, ja, iterationNumber);

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][1][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 3) // XTY 
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                buf_id = taskInfo_firstLoop[structIterator].numParamsList[1]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;
                
                tstart = omp_get_wtime();// cout << "XTY : task_id - " << task_id << " block_id - " << block_id << endl;

                #if defined(NSYS)
                    str = "XTY t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 1)
                {
                    XTY_GPU_MM(blockVectorX, d_activeBlockVectorR, d_temp2XTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, true);
                }
                else if(task_id == 2)
                {
                    XTY_GPU_MM(d_activeBlockVectorR, d_activeBlockVectorR, d_gramRBRXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, true);
                }
                else if(task_id == 4)
                {
                    XTY_GPU_MM(blockVectorAX, d_activeBlockVectorR, d_gramXARXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, true);
                }
                else if(task_id == 5)
                {   
                    XTY_GPU_MM(activeBlockVectorAR, d_activeBlockVectorR, d_gramRARXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, true);                                         
                }
                
                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][2][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 4) // RED
            {
                tstart = omp_get_wtime();

                #if defined(NSYS)
                    str = "XTYRED" ;
                    nvtxRangePush(str.c_str());
                #endif

                if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "temp2BUF")) //xty 1 reduction
                {
                    XTY_GPU_RED_MM(d_temp2XTYBUF, temp2, blocksize, currentBlockSize, block_width, d_memory, iterationNumber);
                }
                else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RBRBUF")) //xty 2 reduction
                {
                    XTY_GPU_RED_MM(d_gramRBRXTYBUF, gramRBR, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                }
                else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "XARBUF")) //xty 4 reduction
                {
                    XTY_GPU_RED_MM(d_gramXARXTYBUF, gramXAR, blocksize, currentBlockSize, block_width, d_memory, iterationNumber);
                }
                else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RARBUF")) //xty 5 reduction
                {
                    // cout << "RARBUF" << endl;
                    XTY_GPU_RED_MM(d_gramRARXTYBUF, gramRAR, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                }

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][2][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 5) // taskName starts XY 
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                task_id = taskInfo_firstLoop[structIterator].taskID; 
                // cout << "XY : task_id - " << task_id << " block_id - " << block_id << endl;
                tstart = omp_get_wtime();

                #if defined(NSYS)
                    str = "XY t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 1)
                {
                    // XY_OPENMP_GPU_tiled(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width, block_id);
                    XY_GPU_MM(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                }
                else if(task_id == 2)
                {
                    XY_GPU_MM(blockVectorX, temp2, temp3_R, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                }
                else if(task_id == 3)
                {
                    XY_GPU_MM(d_activeBlockVectorR, gramRBR, temp3_R, M, currentBlockSize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, false, false);
                }
                else if(task_id == 6)
                {
                    XY_GPU_MM(d_activeBlockVectorR, coordX2, blockVectorP, M, currentBlockSize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, false, false);
                }
                else if(task_id == 8)
                {
                    XY_GPU_MM(activeBlockVectorAR, coordX2, blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                }
                else if(task_id == 10)
                {
                    // XY_GPU_MM(blockVectorX, coordX1, d_activeBlockVectorR, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, true);
                    XY_GPU_MM(blockVectorX, coordX1, temp3_R, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                }
                else if(task_id == 11)
                {
                    // XY_GPU_MM(blockVectorAX, coordX1, d_activeBlockVectorR, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, true);
                    XY_GPU_MM(blockVectorAX, coordX1, temp3_R, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                }

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][3][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 6) // ADD
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;
                
                // cout << "ADD : task_id - " << task_id << " block_id - " << block_id << endl;
                tstart = omp_get_wtime();

                #if defined(NSYS)
                    str = "ADD t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 3)
                {
                    // mat_addition_GPU_MM(d_activeBlockVectorR, blockVectorP, blockVectorX, numrows, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, true, false, false);
                    mat_addition_GPU_MM(temp3_R, blockVectorP, blockVectorX, numrows, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, false, false, false); //newX creates some prb, AX finally doesn't match
                }
                else if(task_id == 4)
                {
                    // mat_addition_GPU_MM(d_activeBlockVectorR, blockVectorAP, blockVectorAX, M, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, true, false, false);
                    mat_addition_GPU_MM(temp3_R, blockVectorAP, blockVectorAX, M, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, false, false, false);
                }

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][4][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 7) // DLACPY 
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;

                // cout << "DLACPY : task_id - " << task_id << " block_id - " << block_id << endl;
                tstart = omp_get_wtime();

                #if defined(NSYS)
                    str = "CPY t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 1)
                {
                    custom_dlacpy_GPU_MM(temp3_R, d_activeBlockVectorR, M, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false, true);
                }

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][5][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 8 && 0) //UPDATE
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;

                // cout << "UPDATE : task_id - " << task_id << " block_id - " << block_id << endl;
                tstart = omp_get_wtime();
                
                 #if defined(NSYS)
                    str = "UP t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 1)
                {
                    updateBlockVector_GPU_MM(d_activeBlockVectorR, d_activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, true);
                }
                
                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][10][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 9) //SUB
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;
                
                // cout << "SUB : task_id - " << task_id << " block_id - " << block_id << endl;
                tstart = omp_get_wtime();

                #if defined(NSYS)
                    str = "SUB t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 1)
                {
                    // mat_sub_GPU_block_v2(blockVectorAX, blockVectorR, numrows, blocksize, block_width, block_id);
                    mat_sub_GPU_MM(blockVectorAX, blockVectorR, blockVectorR, numrows, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, false, false, false);
                }
                else if(task_id == 2)
                {
                    // mat_sub_GPU_block_v1(activeBlockVectorR, temp3_R, M, currentBlockSize, block_width, block_id);
                    mat_sub_GPU_MM(d_activeBlockVectorR, temp3_R, d_activeBlockVectorR, M, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, true, false, true);      
                }

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][6][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 10) //MULT
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                // cout << "MULT : block_id - " << block_id << endl;
                tstart = omp_get_wtime();
                
                #if defined(NSYS)
                    str = "MULT t: " + to_string(1) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                mat_mult_GPU_MM(blockVectorR, blockVectorR, d_activeBlockVectorR, M, blocksize, block_width, block_id, d_memory, iterationNumber);

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][7][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 11) // COL 
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0];
                buf_id = taskInfo_firstLoop[structIterator].numParamsList[1];
                // cout << "COL : block_id - " << block_id << endl;
                tstart = omp_get_wtime();

                #if defined(NSYS)
                    str = "COL t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                sum_sqrt_GPU_COL_MM(d_activeBlockVectorR, residualNorms, M, blocksize, block_width, block_id, buf_id, d_RNBUF, d_memory, iterationNumber);

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][8][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 12) // RNRED 
            {
                tstart = omp_get_wtime();
                sum_sqrt_task_RNRED(d_RNBUF, d_residualNorms, blocksize);
                taskTiming[iterationNumber - 1][8][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 13) // SQRT 
            {
                tstart = omp_get_wtime();
                sum_sqrt_task_SQRT(d_residualNorms, blocksize);
                taskTiming[iterationNumber - 1][8][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 14) // GET 
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;
                
                
                // cout << "GET : task_id - " << task_id << " block_id - " << block_id << endl;
                tstart = omp_get_wtime();
                
                #if defined(NSYS)
                    str = "GET t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                    nvtxRangePush(str.c_str());
                #endif

                if(task_id == 1)
                {
                    getActiveBlockVector_GPU_MM(d_activeBlockVectorR, d_activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, true); 
                }

                #if defined(NSYS)
                    nvtxRangePop();
                #endif

                taskTiming[iterationNumber - 1][9][1] += omp_get_wtime() - tstart;
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 15) //TRANS
            {
                #pragma omp task depend(in: gramRAR[0 : currentBlockSize * currentBlockSize], currentBlockSize)\
                depend(out: transGramRAR[0 : currentBlockSize * currentBlockSize])
                {
                    pair<double *, int> prRAR = make_pair(gramRAR, 0);
                    if(isOnDevice(prRAR))
                    {
                        cudaError_t error_rbr = cudaMemcpy(gramRAR, d_memory+(unsigned long int) mp[prRAR][0] * num_per_blk, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();
                        if(error_rbr != 0){ printf("cudaMemcpy failed gramRAR: %d\n", error_rbr);}
                    }

                    transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);
                } //end task
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 16) //SPEUPDATE
            {
                if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RAR"))
                {
                    #pragma omp task depend(in: currentBlockSize)\
                    depend(in: identity_PAP[0 : currentBlockSize * currentBlockSize])\
                    depend(inout: transGramRAR[0 : currentBlockSize * currentBlockSize])\
                    depend(out : gramRAR[0 : currentBlockSize * currentBlockSize])
                    {
                            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 
                                        0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);
                            
                            pair<double *, int> prRAR = make_pair(gramRAR, 0);
                            if(isOnDevice(prRAR))
                            {
                                cudaError_t error_rbr = cudaMemcpy(d_memory+(unsigned long int) mp[prRAR][0] * num_per_blk, gramRAR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                                cudaDeviceSynchronize();
                                if(error_rbr != 0){ printf("cudaMemcpy failed gramRBR: %d\n", error_rbr);}
                                else{ mp[prRAR][5] = 1.0;}
                            }
                            // else ==> complete later
                    } //end task
                }
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 17) //CHOL
            {
                if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RBR"))
                {
                    #pragma omp task private(i, j) firstprivate(h, t)\
                    depend(in: currentBlockSize, trans_gramRBR[0 : currentBlockSize * currentBlockSize])\
                    depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize]) depend(out: info)
                    {
                        // copying gramRBR from device to host
                        // if(isOnDevice(make_pair(gramRBR, 0)))
                        // {
                        //     // int chol_status = copyToHost(gramRBR, currentBlockSize, currentBlockSize, 0, currentBlockSize, currentBlockSize * currentBlockSize);
                        //     // if(chol_status != 0)
                        //     //     cout << "Error in copyToHost" << endl;
                        //     int stat = omp_target_memcpy(gramRBR, d_memory+(unsigned long int) mp[make_pair(gramRBR, 0)][0], currentBlockSize * currentBlockSize * sizeof(double), 0, 0, h, t);
                        // }

                        if(isOnDevice(make_pair(gramRBR, 0)))
                        {
                            cudaError_t cuberror1 = cudaMemcpy(gramRBR, d_memory+(unsigned long int) mp[make_pair(gramRBR, 0)][0] * num_per_blk, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            if(cuberror1 != 0){ printf("cudaMemcpy failed gramRBR: %d\n", cuberror1);}
                        }

                        // OP: [gramRBR,cholFlag] = chol(gramRBR);
                        transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);    
                        dpotrf_(&uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info);
                        if(info != 0)
                            cout << "dportf_ error 2!!" << endl;
                        transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);
                        for(i = 0 ; i < currentBlockSize ; i++)
                        {
                            for(j = 0 ; j < i ; j++)
                            {
                                gramRBR[i * currentBlockSize + j] = 0.0;
                            }
                        }
                    }//end task    
                } //end CHOL,RBR
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 18) //INV
            {
                if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "RBR"))
                {
                    #pragma omp task depend(in: info, currentBlockSize)\
                    depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize])
                    { 
                        // OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask)/gramRBR; 
                        if(info == 0)
                        {   
                            inverse(gramRBR, currentBlockSize, currentBlockSize);
                            
                            //copying gramRBR from host to device
                            if(isOnDevice(make_pair(gramRBR, 0)))
                            {
                                cudaError_t cuberror2 = cudaMemcpy(d_memory+(unsigned long int) mp[make_pair(gramRBR, 0)][0] * num_per_blk, gramRBR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                                cudaDeviceSynchronize();
                                if(cuberror2 != 0){ printf("cudaMemcpy failed gramRBR: %d\n", cuberror);}
                                else{mp[make_pair(gramRBR, 0)][5] = 1.0;}
                            }
                            //else ==> complete later
                        }
                        else
                            printf("INV,RBR info: %d\n", info);
                    } //end task
                } //end INV,RBR
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 19) // SETZERO
            {
                block_id = taskInfo_firstLoop[structIterator].numParamsList[0]; 
                task_id = taskInfo_firstLoop[structIterator].taskID;
                
                #if defined(DEBUG)
                    cout << "SETZERO : task_id - " << task_id << " block_id - " << block_id << " iterationNumber " << iterationNumber << endl;
                #endif

                if(task_id == 1)
                {
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "SETZERO t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    i = block_id * block_width; // starting point of the block
                    blksz = block_width;
                    if(i + blksz > M)
                        blksz = M - i;

                    pair<double *, int> prAR = make_pair(activeBlockVectorAR, block_id);
                    
                    if(!isOnDevice(prAR))
                    {
                        status = reserveOnDevice(activeBlockVectorAR, M, currentBlockSize, block_id, block_id * block_width * currentBlockSize, blksz * currentBlockSize, 1.0, iterationNumber);
                        if(status !=0)
                            cout << "dst reservation is failed on SETZERO :"  << " block_id : " << block_id << endl;
                    }
                    else
                    {
                        mp[prAR][5] = 1.0;
                    }
                    unsigned long int actAR_offset = (unsigned long int) mp[prAR][0] * num_per_blk;
                    taskTiming[iterationNumber - 1][0][0] += omp_get_wtime() - tstart;

                    #pragma omp target private(j, k, tid, tstart, tend) is_device_ptr(d_memory)\
                    firstprivate(blksz, i, block_width, actAR_offset)\
                    depend(in :  M, blocksize, currentBlockSize)\
                    depend(out : activeBlockVectorAR[block_id * block_width * currentBlockSize : blksz * currentBlockSize])
                    #pragma omp teams distribute parallel for //collapse(2) --> adding collpase() slows down with CCE compiler -- weird
                    for(j = 0 ; j < blksz ; j++)
                    {
                        #pragma omp simd
                            for(k = 0 ; k < currentBlockSize ; k++)
                            {
                                d_memory[actAR_offset + j * currentBlockSize + k] = 0.0;
                            }
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif
                    
                    taskTiming[iterationNumber - 1][0][1] += omp_get_wtime() - tstart;
                }
            }
            else if(taskInfo_firstLoop[structIterator].opCode == 20) // CONV
            {
                #pragma omp task firstprivate(blocksize, taskTiming, iterationNumber)\
                depend(inout: activeMask[0 : blocksize], currentBlockSize, convergeFlag)\
                depend(in: residualNorms[0 : blocksize])\
                depend(out: activeRSize, explicitGramFlag, activePSize, restart)
                {  
                    //currentBlockSize = sum(activeMask);
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
                    int flag = 1;
                    for(i = 0 ; i < blocksize ; i++)
                    {
                        if(residualNorms[i] < 4.0538e-10)
                            flag = 0;
                    }
                    if(flag == 0)
                        explicitGramFlag = 1;
                    else
                        explicitGramFlag = 0;
                } //end task
            } //end CONV
            else if(taskInfo_firstLoop[structIterator].opCode == 21)
            {
                if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "CONSTRUCTGB"))
                {
                    gramASize = blocksize + blocksize;
                    // printf("GB 1 gramASize: %d currentBlockSize: %d\n", gramASize, currentBlockSize);
                    // printf("GB 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                    gramB = (double *) malloc(gramASize * gramASize * sizeof(double));
                        
                    #pragma omp task firstprivate(gramASize)\
                    depend(in: activeRSize, activePSize, restart, currentBlockSize)\
                    depend(in: activeMask[0 : blocksize])\
                    depend(out: gramB[0 : gramASize * gramASize])
                    {
                        // printf("GB 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                        make_identity_mat(gramB, gramASize, gramASize);
                    }
                }
                else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "CONSTRUCTGA"))
                {
                    gramASize = blocksize + blocksize;
                    // printf("GA 1 gramASize: %d\n", gramASize);
                    gramA = (double *) malloc(gramASize * gramASize * sizeof(double));
                    // coordX = (double *) malloc(gramASize * blocksize * sizeof(double)); 

                    #pragma omp task firstprivate(gramASize)\
                    depend(in: activeRSize, activePSize, restart, currentBlockSize)\
                    depend(in: gramXAR[0: blocksize * currentBlockSize], lambda[0 : blocksize * blocksize])\
                    depend(in: transGramXAR[0 : currentBlockSize * blocksize], gramRAR[0 : currentBlockSize * currentBlockSize])\
                    depend(in: activeMask[0 : blocksize]) depend(in: transGramRAR[0 : currentBlockSize * currentBlockSize])\
                    depend(out: gramA[0 : gramASize * gramASize])
                    {     
                        //printf("GA 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                        
                        pair<double *, int> prXAR = make_pair(gramXAR, 0);
                        if(isOnDevice(prXAR))
                        {
                            cudaError_t error_rbr = cudaMemcpy(gramXAR, d_memory+(unsigned long int) mp[prXAR][0] * num_per_blk, blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            if(error_rbr != 0){ printf("cudaMemcpy failed gramRAR: %d\n", error_rbr);}
                        }

                        mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                        mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                        transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                        mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                        mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                    }
                }
                else if(!strcmp(taskInfo_firstLoop[structIterator].strParamsList[0], "EIGEN"))
                {   
                    #pragma omp task private(i, j, k)\
                    firstprivate(gramASize, d_memory)\
                    depend(in: gramA[0 : gramASize * gramASize])\
                    depend(in: gramB[0 : gramASize * gramASize], currentBlockSize)\
                    depend(out: coordX1[0 : blocksize * blocksize], coordX2[0 : currentBlockSize * blocksize], lambda[0 : blocksize * blocksize])
                    {
                        cudaError_t lambda_error;

                        //printf("EIG 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                        eigen_value = (double *) malloc(gramASize * sizeof(double));
                        
                        info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, gramASize, gramA, gramASize, gramB, gramASize, eigen_value);
                        if(info != 0)
                            printf("LAPACKE_dsygv error: The algorithm failed to compute eigenvalues.\n" );

                        diag(eigen_value, lambda, blocksize);

                        //lambda copied back to device
                        pair<double *, int> prLambda = make_pair(lambda, 0);
                        if(isOnDevice(prLambda))
                        {
                            lambda_error = cudaMemcpy(d_memory+(unsigned long int) mp[prLambda][0] * num_per_blk, lambda, blocksize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                            cudaDeviceSynchronize();
                            if(lambda_error != 0){ printf("cudaMemcpy failed gramRBR: %d\n", lambda_error);}
                            else{mp[prLambda][5] = 1.0;}
                        }

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

                        pair<double *, int> x1 = make_pair(coordX1, 0);
                        pair<double *, int> x2 = make_pair(coordX2, 0);
                        if(isOnDevice(x1))
                        {
                            lambda_error = cudaMemcpy(d_memory + (unsigned long int) mp[x1][0] * num_per_blk, coordX1, blocksize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                            cudaDeviceSynchronize();
                            if(lambda_error != 0){ printf("cudaMemcpy failed coordX1: %d\n", lambda_error);}
                            else{mp[x1][5] = 1.0;}
                        }
                        if(isOnDevice(x2))
                        {
                            lambda_error = cudaMemcpy(d_memory + (unsigned long int) mp[x2][0] * num_per_blk, coordX2, currentBlockSize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                            cudaDeviceSynchronize();
                            if(lambda_error != 0){ printf("cudaMemcpy failed coordX2: %d\n", lambda_error);}
                            else{mp[x2][5] = 1.0;}
                        }
                        cudaDeviceSynchronize();

                    } //end task
                }
            } //end of taskID 21
            else if(taskInfo_firstLoop[structIterator].opCode == 27) // MEMCPY
            {
                #pragma omp taskwait
                cudaDeviceSynchronize();
                // cout << "Memcpy" << endl;
                
                // #pragma omp task private(blksz)\
                // firstprivate(M, t, h, nrowblk, activeBlockVectorR, block_width)\
                // depend(out: currentBlockSize)\
                // depend(out: d_activeBlockVectorR[0 : M * currentBlockSize])
                // {
                //     // status = omp_target_memcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), 0, 0, t, h);
                //     // Error MSG: ACC: craylibs/libcrayacc/acc_hw_nvidia.c:1266 CRAY_ACC_ERROR -  cuCtxGetDevice returned CUDA_ERROR_INVALID_CONTEXT

                //     // if the block is on device then copy from there, else copy from host ==> two task
                //     cudaError_t cuberror3;// = cudaMemcpy(d_activeBlockVectorR, activeBlockVectorR, M * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                //     // cudaDeviceSynchronize();
                //     // if(cuberror1 != 0){ printf("cudaMemcpy failed d_activeBlockVectorR: %d\n", cuberror);}

                //     for(i = 0 ; i < nrowblk ; i++) //for each block
                //     {
                //         blksz = block_width;
                //         if(i * block_width + blksz > M)
                //             blksz = M - i * block_width;

                //         // cout << "i: " << i << " blksz: " << blksz << endl;

                //         pair<double *, int> prActR = make_pair(activeBlockVectorR, i);
                        
                //         if(isOnDevice(prActR)) //device to device copy
                //         {
                //             cuberror3 = cudaMemcpy(d_activeBlockVectorR + (i * block_width * currentBlockSize), 
                //                                     d_memory + (unsigned long int) mp[prActR][0] * num_per_blk, 
                //                                     blksz * currentBlockSize * sizeof(double), cudaMemcpyDeviceToDevice);
                //             cudaDeviceSynchronize();
                //             if(cuberror3 != 0 ) cout << "activeBlockVectorR block_id: " << block_id << " iterationNumber- " << iterationNumber << " not on device" << endl;
                //         }
                //         else //host to device copy
                //         {
                //             cuberror3 = cudaMemcpy(d_activeBlockVectorR + (i * block_width * currentBlockSize), 
                //                                     activeBlockVectorR + (i * block_width * currentBlockSize), 
                //                                     blksz * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                //             cudaDeviceSynchronize();
                //             if(cuberror3 != 0 ) cout << "activeBlockVectorR block_id: " << block_id << " iterationNumber- " << iterationNumber << " not on host" << endl;
                //             // cout << "R block_id: " << block_id << " not in device memory" << endl;
                //         }
                //     }
                //     cudaDeviceSynchronize();
                // }
            }
        }//end taskinfo loop
        #pragma omp taskwait
        loop_finish_time = get_seconds();
        loopTime[iterationNumber - 1] =  loop_finish_time - loop_start_time;
    } // omp master
    } //omp parallel

    #if defined(NSYS)
        nvtxRangePop();
    #endif

    cout << "loopTime[0]: " << loopTime[0] << endl;

    // saving lambda (eigenvalues) of first iteration
    for(i = 0 ; i < blocksize ; i++)
    {
        saveLamda[i][iterationNumber - 1] = lambda[i * blocksize + i];
    }

    printMapInfo(); printMemoryUsageStat(iterationNumber);

    // *** M * B
    // cout << "currentBlockSize: " << currentBlockSize << endl;
    // for(i = 0 ; i < nrowblk ; i++){pair<double *, int> pr = make_pair(blockVectorP, i); cout << "block_id: " << i << endl;
    // if(isOnDevice(pr)){
    //     cout << "Need to copy from device" << endl;
    // status = copyToHost(blockVectorP, M, blocksize, i, block_width, block_width * blocksize);
    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;
    // } 
    // print_mat(blockVectorP + (i * block_width * blocksize + 171), 10, blocksize); cout << endl << endl;}

   // *** M * CB
    // cout << endl << endl;
    // status = copyToHost(activeBlockVectorAR, M, currentBlockSize, 60, block_width, block_width * currentBlockSize);
    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(activeBlockVectorAR + (60 * block_width * currentBlockSize + i * 171), 10, currentBlockSize);
    // cout << endl << endl;
    


    // status = omp_target_memcpy(activeBlockVectorR + ( 5 * block_width * currentBlockSize), d_activeBlockVectorR + ( 5 * block_width * currentBlockSize), block_width * currentBlockSize * sizeof(double), 0, 0, h, t);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(activeBlockVectorR + (5 * block_width * currentBlockSize), 10, currentBlockSize);

    // cout << endl << endl;

    // *** B * CB

    // status = copyToHost(lambda, blocksize, currentBlockSize, 0, blocksize, blocksize * currentBlockSize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(lambda, blocksize, currentBlockSize);
    // cout << endl;

    // *** CB * CB

    // status = copyToHost(gramRBR, currentBlockSize, currentBlockSize, 0, currentBlockSize, currentBlockSize * currentBlockSize);

    // // status = omp_target_memcpy(gramRBR, d_memory + (unsigned long int) mp[make_pair(gramRAR, 0)][0] * num_per_blk, currentBlockSize * currentBlockSize * sizeof(double), 0, 0, h, t);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(gramRBR, currentBlockSize, currentBlockSize);
    // cout << endl;


    // ***
    // status = copyToHost(blockVectorX, M, blocksize, 5, block_width, block_width * blocksize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(blockVectorX + (5 * block_width * blocksize), 10, blocksize);


    // status = omp_target_memcpy(residualNorms, d_residualNorms, blocksize * sizeof(double), 0, 0, h, t);
    // if( status != 0 ){ printf("omp_target_memcpy failed residualNorms ==> %d\n", cuberror); return 0; }

    // print_mat(residualNorms, 1, blocksize);
    
    // status = omp_target_memcpy(activeMask, d_activeMask, blocksize * sizeof(int), 0, 0, h, t);
    // if( status != 0 ){ printf("omp_target_memcpy failed activeMask ==> %d\n", cuberror); return 0; }
    // for(i = 0 ; i < blocksize ; i++)
    //     cout << activeMask[i] << " ";
    // cout << endl;

    // print_mat(blockVectorR + (5 * block_width * blocksize), 10, blocksize);
   
    // cout << endl << endl;

    // print_mat(lambda, blocksize, blocksize);
   
    // cout << endl << endl;
    
    // status = copyToHost(lambda, blocksize, blocksize, 0, blocksize, blocksize * blocksize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(lambda, blocksize, blocksize);
    
     

    // cout << "Exiting main...." << endl << endl;

    // exit(1);
    
    // return 0;

    cout << "Offloading second and rest of the loops to GPU....." << endl;
    
    
    
    // Offloading 2nd and onward iterations from here
    double cudaAlphaHalf = 0.5, cudaBetaHalf = 0.5;

    iterationNumber = 2;
    
    #pragma omp parallel
    {
    #pragma omp master
    {
        while(iterationNumber <= maxIterations && convergeFlag != 1 && !errorFlag)
        {
            cout << "iterationNumber: " << iterationNumber << endl;
            
            #if defined(NSYS)
                str = "iterationNumber : " + to_string(iterationNumber);
                nvtxRangePush(str.c_str());
            #endif

            loop_start_time = get_seconds();

            clear_buffer(d_RNBUF, 1, blocksize);
            clear_buffer(d_temp2XTYBUF, blocksize, currentBlockSize);
            clear_buffer(d_gramRBRXTYBUF, blocksize, currentBlockSize);
            clear_buffer(d_gramPBPXTYBUF, currentBlockSize, currentBlockSize); // xty_id = 3
            clear_buffer(d_gramXARXTYBUF, blocksize, currentBlockSize); // xty_id = 4
            clear_buffer(d_gramRARXTYBUF, currentBlockSize, currentBlockSize); // xty_id = 5

            clear_buffer(d_gramXAPXTYBUF, blocksize, currentBlockSize); // xty_id = 6
            clear_buffer(d_gramRAPXTYBUF, currentBlockSize, currentBlockSize); // xty_id = 7
            clear_buffer(d_gramPAPXTYBUF, currentBlockSize, currentBlockSize); // xty_id = 8
            clear_buffer(d_gramXBPXTYBUF, blocksize, currentBlockSize); // xty_id = 9
            clear_buffer(d_gramRBPXTYBUF, currentBlockSize, currentBlockSize); // xty_id = 10

            for(structIterator = 0 ; structIterator < taskCount_secondLoop; structIterator++)
            {
                if(taskInfo_secondLoop[structIterator].opCode == 1) // taskName starts RESET 
                {
                    if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RN")) //reseting residualNorms
                    {
                        #pragma omp target is_device_ptr(d_residualNorms) map(tofrom: residualNorms[0 : blocksize])\
                        depend(inout: residualNorms[0 : blocksize], d_residualNorms[0 : blocksize])
                        {
                            for(i = 0; i < blocksize; i++)
                                d_residualNorms[i] = residualNorms[i] = 0.0;
                        }
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "actMask")) //reseting activeMask
                    {
                        update_activeMask_GPU(activeMask, d_activeMask, d_residualNorms, residualTolerance, blocksize);
                    }
                }
                
                else if(taskInfo_secondLoop[structIterator].opCode == 2)  // SpMM
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    tstart = omp_get_wtime();
                    
                    #if defined(NSYS)
                        str = "SPMM b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    SPMM_GPU_MM_v3(d_rowptr, d_ja, acsr, d_activeBlockVectorR, activeBlockVectorAR, 
                            numrows, numcols, currentBlockSize, block_width, block_id, nnz_per_tile, d_memory, ia, ja, iterationNumber);

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][1][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 3) // XTY 
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                    buf_id = taskInfo_secondLoop[structIterator].numParamsList[1]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    
                    tstart = omp_get_wtime();
                    
                    #if defined(NSYS)
                        str = "XTY t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    if(task_id == 1)
                    {
                        XTY_GPU_MM(blockVectorX, d_activeBlockVectorR, d_temp2XTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, true);
                    }
                    else if(task_id == 2)
                    {
                        XTY_GPU_MM(d_activeBlockVectorR, d_activeBlockVectorR, d_gramRBRXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, true);
                    }
                    else if(task_id == 3)
                    {
                        XTY_GPU_MM(activeBlockVectorP, activeBlockVectorP, d_gramPBPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false); 
                    }
                    else if(task_id == 4)
                    {
                        XTY_GPU_MM(blockVectorAX, d_activeBlockVectorR, d_gramXARXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, true);
                    }
                    else if(task_id == 5)
                    {   
                        XTY_GPU_MM(activeBlockVectorAR, d_activeBlockVectorR, d_gramRARXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, true);                                         
                    }
                    else if(task_id == 6)
                    {
                        XTY_GPU_MM(blockVectorAX, activeBlockVectorP, d_gramXAPXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false);
                    }
                    else if(task_id == 7)
                    {
                        XTY_GPU_MM(activeBlockVectorAR, activeBlockVectorP, d_gramRAPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false);
                    }
                    else if(task_id == 8)
                    {
                        XTY_GPU_MM(activeBlockVectorAP, activeBlockVectorP, d_gramPAPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false);
                    }
                    else if(task_id == 9)
                    {
                        XTY_GPU_MM(blockVectorX, activeBlockVectorP, d_gramXBPXTYBUF, M, blocksize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false);
                    }
                    else if(task_id == 10)
                    {
                        XTY_GPU_MM(d_activeBlockVectorR, activeBlockVectorP, d_gramRBPXTYBUF, M, currentBlockSize, currentBlockSize, block_width, block_id, buf_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, false);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][2][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 4) // RED
                {
                    // cout << "XTYRED :  " << taskInfo_secondLoop[structIterator].strParamsList[0] << endl;
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "XTYRED";
                        nvtxRangePush(str.c_str());
                    #endif

                    if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "temp2BUF")) // xty 1 reduction
                    {
                        XTY_GPU_RED_MM(d_temp2XTYBUF, temp2, blocksize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBRBUF")) // xty 2 reduction
                    {
                        XTY_GPU_RED_MM(d_gramRBRXTYBUF, gramRBR, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PBPBUF")) // xty 3 reduction
                    {
                        XTY_GPU_RED_MM(d_gramPBPXTYBUF, gramPBP, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "XARBUF")) // xty 4 reduction
                    {
                        XTY_GPU_RED_MM(d_gramXARXTYBUF, gramXAR, blocksize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RARBUF")) //xty 5 reduction
                    {
                        XTY_GPU_RED_MM(d_gramRARXTYBUF, gramRAR, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "XAPBUF")) //xty 6 reduction
                    {
                        XTY_GPU_RED_MM(d_gramXAPXTYBUF, gramXAP, blocksize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RAPBUF")) //xty 7 reduction
                    {
                        XTY_GPU_RED_MM(d_gramRAPXTYBUF, gramRAP, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PAPBUF")) //xty 8 reduction
                    {
                        XTY_GPU_RED_MM(d_gramPAPXTYBUF, gramPAP, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "XBPBUF")) //xty 9 reduction
                    {
                        XTY_GPU_RED_MM(d_gramXBPXTYBUF, gramXBP, blocksize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBPBUF")) //xty 10 reduction
                    {
                        XTY_GPU_RED_MM(d_gramRBPXTYBUF, gramRBP, currentBlockSize, currentBlockSize, block_width, d_memory, iterationNumber);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][2][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 5) // XY 
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                    task_id = taskInfo_secondLoop[structIterator].taskID; 
                    
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "XY t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif
                    
                    if(task_id == 1)
                    {
                        XY_GPU_MM(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                    }
                    else if(task_id == 2)
                    {
                        XY_GPU_MM(blockVectorX, temp2, temp3_R, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                    }
                    else if(task_id == 3)
                    {
                        XY_GPU_MM(d_activeBlockVectorR, gramRBR, temp3_R, M, currentBlockSize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, false, false);
                    }
                    else if (task_id == 4)
                    {
                        XY_GPU_MM(activeBlockVectorP, gramPBP, temp3_R, M, currentBlockSize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                    }
                    else if(task_id == 5)
                    {
                        XY_GPU_MM(activeBlockVectorAP, gramPBP, temp3_R, M, currentBlockSize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                    }
                    else if(task_id == 6)
                    {
                        XY_GPU_MM(d_activeBlockVectorR, coordX2, blockVectorP, M, currentBlockSize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, true, false, false);
                    }
                    else if(task_id == 7)
                    {
                        XY_GPU_MM(activeBlockVectorP, coordX3, blockVectorP, M, currentBlockSize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 1.0, false, false, false); //newP
                    }
                    else if(task_id == 8)
                    {
                        XY_GPU_MM(activeBlockVectorAR, coordX2, blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                    }
                    else if(task_id == 9)
                    {
                        XY_GPU_MM(activeBlockVectorAP, coordX3, blockVectorAP, M, currentBlockSize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 1.0, false, false, false); // newAP
                    }
                    else if(task_id == 10)
                    {
                        // XY_GPU_MM(blockVectorX, coordX1, d_activeBlockVectorR, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, true); //newX
                        XY_GPU_MM(blockVectorX, coordX1, temp3_R, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false); //newX
                    }
                    else if(task_id == 11)
                    {
                        // XY_GPU_MM(blockVectorAX, coordX1, d_activeBlockVectorR, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, true);
                        XY_GPU_MM(blockVectorAX, coordX1, temp3_R, M, blocksize, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, 1.0, 0.0, false, false, false);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][3][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 6) // ADD
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "ADD t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    if(task_id == 1)
                    {
                        // mat_addition_GPU_MM(blockVectorP, newP, blockVectorP, M, blocksize, block_width, block_id, d_memory, task_id, iterationNumber);
                    }
                    else if(task_id == 2)
                    {
                        // mat_addition_GPU_MM(blockVectorAP, newAP, blockVectorAP, M, blocksize, block_width, block_id, d_memory, task_id, iterationNumber);
                    }
                    if(task_id == 3)
                    {
                        // mat_addition_GPU_MM(d_activeBlockVectorR, blockVectorP, blockVectorX, numrows, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, true, false, false);
                        mat_addition_GPU_MM(temp3_R, blockVectorP, blockVectorX, numrows, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, false, false, false);
                    }
                    else if(task_id == 4)
                    {
                        // mat_addition_GPU_MM(d_activeBlockVectorR, blockVectorAP, blockVectorAX, M, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, true, false, false);
                        mat_addition_GPU_MM(temp3_R, blockVectorAP, blockVectorAX, M, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, false, false, false);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][4][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 7) // DLACPY 
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "CPY t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    if(task_id == 1)
                    {
                        custom_dlacpy_GPU_MM(temp3_R, d_activeBlockVectorR, M, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false, true);
                    }
                    else if(task_id == 2)
                    {
                        custom_dlacpy_GPU_MM(temp3_R, activeBlockVectorP, M, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false, false);
                    }
                    else if(task_id == 3)
                    {
                        custom_dlacpy_GPU_MM(temp3_R, activeBlockVectorAP, M, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false, false);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][5][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 8 && 0) //UPDATE
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    
                    tstart = omp_get_wtime();
                    
                    #if defined(NSYS)
                        str = "UP t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    if(task_id == 1)
                    {
                        updateBlockVector_GPU_MM(d_activeBlockVectorR, d_activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, true);
                    }
                    else if(task_id == 2)
                    {
                        updateBlockVector_GPU_MM(activeBlockVectorP, d_activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false);
                    }
                    else if(task_id == 3)
                    {
                        updateBlockVector_GPU_MM(activeBlockVectorAP, d_activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][10][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 9) //SUB
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    // cout << "SUB: task_id - " << task_id << " block_id - " << block_id  << " iterationNumber - " << iterationNumber << endl;
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "SUB t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    if(task_id == 1)
                    {
                        mat_sub_GPU_MM(blockVectorAX, blockVectorR, blockVectorR, numrows, blocksize, block_width, block_id, d_memory, task_id, iterationNumber, false, false, false);
                    }
                    else if(task_id == 2)
                    {
                        mat_sub_GPU_MM(d_activeBlockVectorR, temp3_R, d_activeBlockVectorR, M, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, true, false, true);      
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][6][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 10) //MULT
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                    tstart = omp_get_wtime();
                    
                    #if defined(NSYS)
                        str = "MULT t: " + to_string(1) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    mat_mult_GPU_MM(blockVectorR, blockVectorR, d_activeBlockVectorR, M, blocksize, block_width, block_id, d_memory, iterationNumber);

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][7][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 11) // COL 
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0];
                    buf_id = taskInfo_secondLoop[structIterator].numParamsList[1];
                    tstart = omp_get_wtime();
                    
                    #if defined(NSYS)
                        str = "COL t: " + to_string(1) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    sum_sqrt_GPU_COL_MM(d_activeBlockVectorR, residualNorms, M, blocksize, block_width, block_id, buf_id, d_RNBUF, d_memory, iterationNumber);
                    
                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][8][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 12) // RNRED
                {
                    tstart = omp_get_wtime();
                    sum_sqrt_task_RNRED(d_RNBUF, d_residualNorms, blocksize);
                    taskTiming[iterationNumber - 1][8][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 13) // SQRT
                {
                    tstart = omp_get_wtime();
                    sum_sqrt_task_SQRT(d_residualNorms, blocksize);
                    taskTiming[iterationNumber - 1][8][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 14) // GET 
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    
                    tstart = omp_get_wtime();

                    #if defined(NSYS)
                        str = "GET t: " + to_string(task_id) + " b: " + to_string(block_id) + " i: " + to_string(iterationNumber);
                        nvtxRangePush(str.c_str());
                    #endif

                    if(task_id == 1)
                    { 
                        getActiveBlockVector_GPU_MM(d_activeBlockVectorR, d_activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, true);
                    }
                    else if(task_id == 2)
                    {
                        getActiveBlockVector_GPU_MM(activeBlockVectorP, d_activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false);
                    }
                    else if(task_id == 3)
                    {
                        getActiveBlockVector_GPU_MM(activeBlockVectorAP, d_activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width, block_id, d_memory, task_id, iterationNumber, false);
                    }

                    #if defined(NSYS)
                        nvtxRangePop();
                    #endif

                    taskTiming[iterationNumber - 1][9][1] += omp_get_wtime() - tstart;
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 15) //TRANS
                {
                    #pragma omp task firstprivate(d_memory)\
                    depend(in: gramRAR[0 : currentBlockSize * currentBlockSize], currentBlockSize)\
                    depend(out: transGramRAR[0 : currentBlockSize * currentBlockSize])
                    {
                        //This task can be run on GPU?
                        if(isOnDevice(make_pair(gramRAR, 0)))
                        {
                            cudaError_t cuberror1 = cudaMemcpy(gramRAR, d_memory+(unsigned long int) mp[make_pair(gramRAR, 0)][0] * num_per_blk, 
                                                        currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                            cudaDeviceSynchronize();
                            if(cuberror1 != 0){ printf("cudaMemcpy failed gramRAR: %d\n", cuberror1); errorFlag = true;}
                        }

                        transpose(gramRAR, transGramRAR, currentBlockSize, currentBlockSize);
                    } //end task
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 16) //SPEUPDATE
                {
                    if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RAR"))
                    {
                        #pragma omp task firstprivate(d_memory) depend(in: currentBlockSize)\
                        depend(in: identity_PAP[0 : currentBlockSize * currentBlockSize])\
                        depend(inout: transGramRAR[0 : currentBlockSize * currentBlockSize])\
                        depend(out : gramRAR[0 : currentBlockSize * currentBlockSize])
                        {
                            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 
                                            0.5, transGramRAR, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramRAR, currentBlockSize);
                            
                            //copying gramRAR from host to device
                            if(isOnDevice(make_pair(gramRAR, 0)))
                            {
                                cudaError_t cuberror2 = cudaMemcpy(d_memory+(unsigned long int) mp[make_pair(gramRAR, 0)][0] * num_per_blk, gramRAR, 
                                                            currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                                cudaDeviceSynchronize();
                                if(cuberror2 != 0){ printf("cudaMemcpy failed gramRAR: %d\n", cuberror2); errorFlag = true;}
                            }
                                //else ==> complete later
                        } //end task
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PAP"))
                    {
                        #pragma omp task firstprivate(d_memory) depend(in : currentBlockSize)\
                        depend(inout: identity_PAP[0 : currentBlockSize * currentBlockSize])\
                        depend(inout: gramPAP[0 : currentBlockSize * currentBlockSize])
                        { 
                            pair<double *, int> prPAP = make_pair(gramPAP, 0); 
                            if(isOnDevice(prPAP))
                            {
                                cudaError_t cuberror1 = cudaMemcpy(gramPAP, d_memory+(unsigned long int) mp[prPAP][0] * num_per_blk,  
                                                            currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                cudaDeviceSynchronize();
                                if(cuberror1 != 0){ printf("cudaMemcpy failed gramPAP: %d\n", cuberror1); errorFlag = true;}
                            }

                            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, currentBlockSize, currentBlockSize, currentBlockSize, 
                                    0.5, gramPAP, currentBlockSize, identity_PAP, currentBlockSize, 0.5, gramPAP, currentBlockSize);
                            
                            //copying gramRAR from host to device
                            if(isOnDevice(prPAP))
                            {
                                cudaError_t cuberror2 = cudaMemcpy(d_memory+(unsigned long int) mp[prPAP][0] * num_per_blk, gramPAP, 
                                                            currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                                cudaDeviceSynchronize();
                                if(cuberror2 != 0){ printf("cudaMemcpy failed gramPAP: %d\n", cuberror2); errorFlag = true;}
                            }
                        } //end task
                    }
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 17) //CHOL
                {
                    if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBR"))
                    {
                        #pragma omp task private(i, j) firstprivate(h, t, d_memory)\
                        depend(in: currentBlockSize, trans_gramRBR[0 : currentBlockSize * currentBlockSize])\
                        depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize]) depend(out: info)
                        {
                            if(isOnDevice(make_pair(gramRBR, 0)))
                            {
                                cudaError_t cuberror1 = cudaMemcpy(gramRBR, d_memory+(unsigned long int) mp[make_pair(gramRBR, 0)][0] * num_per_blk, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                cudaDeviceSynchronize();
                                if(cuberror1 != 0){ printf("cudaMemcpy failed gramRBR: %d\n", cuberror1); errorFlag = true;}
                            }

                            // OP: [gramRBR,cholFlag] = chol(gramRBR); 
                            transpose(gramRBR, trans_gramRBR, currentBlockSize, currentBlockSize);    
                            dpotrf_(&uplo, &currentBlockSize, trans_gramRBR, &currentBlockSize, &info);
                            if(info != 0)
                                cout << "dportf_ error 2!!" << endl;
                            transpose(trans_gramRBR, gramRBR, currentBlockSize, currentBlockSize);
                            for(i = 0 ; i < currentBlockSize ; i++)
                            {
                                for(j = 0 ; j < i ; j++)
                                {
                                    gramRBR[i * currentBlockSize + j] = 0.0;
                                }
                            }
                        }//end task    
                    } //end CHOL,RBR
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PBP"))
                    {
                        #pragma omp task private(i, j) firstprivate(h, t, d_memory)\
                        depend(inout: gramPBP[0 : currentBlockSize * currentBlockSize])\
                        depend(inout: trans_gramPBP[0 : currentBlockSize * currentBlockSize])\
                        depend(out: info)
                        {
                            if(isOnDevice(make_pair(gramPBP, 0)))
                            {
                                cudaError_t cuberror1 = cudaMemcpy(gramPBP, d_memory+(unsigned long int) mp[make_pair(gramPBP, 0)][0] * num_per_blk, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                cudaDeviceSynchronize();
                                if(cuberror1 != 0){ printf("cudaMemcpy failed gramPBP: %d\n", cuberror1); errorFlag = true;}
                            }

                            transpose(gramPBP, trans_gramPBP, currentBlockSize, currentBlockSize);
                            dpotrf_( &uplo, &currentBlockSize, trans_gramPBP, &currentBlockSize, &info );
                            if(info != 0)
                            {
                                cout << "BLOPEX:lobpcg:DirectionNotFullRank...The direction matrix is not full rank." << endl; errorFlag = true;
                            }
                            transpose(trans_gramPBP, gramPBP, currentBlockSize, currentBlockSize);
                            
                            // making the lower part of gramPBP zero
                            for(i = 0 ; i < currentBlockSize ; i++)
                            {
                                for(j = 0 ; j < i ; j++)
                                {
                                    gramPBP[i * currentBlockSize + j] = 0.0;
                                }
                            }
                        } //end task
                    }//end CHOL,PBP
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 18) //INV
                {
                    if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "RBR"))
                    {
                        // cout << "INV" << endl;

                        #pragma omp task firstprivate(h, t, d_memory) depend(in: info, currentBlockSize)\
                        depend(inout: gramRBR[0 : currentBlockSize * currentBlockSize])
                        { 
                            // OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask)/gramRBR; 
                            if(info == 0)
                            {   
                                inverse(gramRBR, currentBlockSize, currentBlockSize);

                                //copying gramRBR from host to device
                                if(isOnDevice(make_pair(gramRBR, 0)))
                                {
                                    cudaError_t cuberror2 = cudaMemcpy(d_memory+(unsigned long int) mp[make_pair(gramRBR, 0)][0] * num_per_blk, gramRBR, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                                    cudaDeviceSynchronize();
                                    if(cuberror2 != 0){ printf("cudaMemcpy failed gramRBR: %d\n", cuberror2); errorFlag = true;}
                                }
                                //else ==> complete later
                            }
                            else
                                printf("INV,RBR info: %d\n", info);
                        } //end task
                    } //end INV,RBR
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "PBP"))
                    {   
                        #pragma omp task firstprivate(h, t, d_memory) depend(in : info, currentBlockSize, gramPBP)\
                        depend(inout: gramPBP[0 : currentBlockSize * currentBlockSize])
                        {
                            if(info == 0)
                            {
                                inverse(gramPBP, currentBlockSize, currentBlockSize);
                                //copying gramRBR from host to device
                                if(isOnDevice(make_pair(gramPBP, 0)))
                                {
                                    cudaError_t cuberror2 = cudaMemcpy(d_memory+(unsigned long int) mp[make_pair(gramPBP, 0)][0] * num_per_blk, gramPBP, currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                                    cudaDeviceSynchronize();
                                    if(cuberror2 != 0){ printf("cudaMemcpy failed gramRBR: %d\n", cuberror2); errorFlag = true;}
                                }
                                //else ==> complete later
                            }
                            else
                                printf("INV,PBP info: %d\n", info);
                        } //end task
                    }//end INV,PBP
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 19) // SETZERO
                {
                    block_id = taskInfo_secondLoop[structIterator].numParamsList[0]; 
                    task_id = taskInfo_secondLoop[structIterator].taskID;
                    
                    if(task_id == 1)
                    {
                        #if defined(DEBUG)
                            cout << "SETZERO : task_id - " << task_id << " block_id - " << block_id << " iterationNumber " << iterationNumber << endl;
                        #endif
                        
                        tstart = omp_get_wtime();

                        #if defined(NSYS)
                            str = "SETZERO t: " + to_string(task_id) + " b: " + to_string(block_id) ;
                            nvtxRangePush(str.c_str());
                        #endif

                        i = block_id * block_width; // starting point of the block
                        blksz = block_width;
                        if(i + blksz > M)
                            blksz = M - i;

                        pair<double *, int> prAR = make_pair(activeBlockVectorAR, block_id);

                        if(!isOnDevice(prAR))
                        {
                            status = reserveOnDevice(activeBlockVectorAR, M, currentBlockSize, block_id, block_id * block_width * currentBlockSize, blksz * currentBlockSize, 1.0, iterationNumber);
                            if(status !=0)
                                cout << "dst reservation is failed on SETZERO :"  << " block_id : " << block_id << endl;
                        }
                        else
                        {
                            mp[prAR][5] = 1.0;
                        }
                        unsigned long int actAR_offset = (unsigned long int) mp[prAR][0] * num_per_blk;
                        taskTiming[iterationNumber - 1][0][0] += omp_get_wtime() - tstart;
                        // Hot Spot - 28% time was spent here?
                        #pragma omp target is_device_ptr(d_memory)\
                        firstprivate(blksz, i, block_width, actAR_offset)\
                        depend(in :  M, blocksize, currentBlockSize)\
                        depend(out : activeBlockVectorAR[block_id * block_width * currentBlockSize : blksz * currentBlockSize])
                        #pragma omp teams distribute parallel for //collapse(2) --> adding collpase() slows down
                        for(j = 0 ; j < blksz ; j++)
                        {
                            for(k = 0 ; k < currentBlockSize ; k++)
                            {
                                d_memory[actAR_offset + j * currentBlockSize + k] = 0.0;
                            }
                        }

                        #if defined(NSYS)
                            nvtxRangePop();
                        #endif

                        taskTiming[iterationNumber - 1][0][1] += omp_get_wtime() - tstart;    
                    }
                }
                else if(taskInfo_secondLoop[structIterator].opCode == 20) // CONV
                {
                    #pragma omp task firstprivate(blocksize, iterationNumber)\
                    depend(inout: activeMask[0 : blocksize], currentBlockSize, convergeFlag)\
                    depend(in: residualNorms[0 : blocksize])\
                    depend(out: activeRSize, explicitGramFlag, activePSize, restart)
                    {   
                        // currentBlockSize = sum(activeMask);
                        // printf("* CONV 2 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                            
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
                        // printf("CONV 2 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d\n\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart);
                
                        int flag = 1;
                        for(i = 0 ; i < blocksize ; i++)
                        {
                            if(residualNorms[i] < 4.0538e-10)
                            {
                                flag = 0;
                            }
                        }
                        if(flag == 0 )
                            explicitGramFlag = 1;
                        else
                            explicitGramFlag = 0;
                    } //end task
                } //end CONV
                else if(taskInfo_secondLoop[structIterator].opCode == 21)
                {
                    if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "CONSTRUCTGA1"))
                    {
                        gramASize = blocksize + currentBlockSize + currentBlockSize;
                        coordX = (double *) malloc(gramASize * blocksize * sizeof(double));
                        //declare it only once up there, fix it later
                        gramA = (double *) malloc(gramASize * gramASize * sizeof(double));
                        
                        #pragma omp task firstprivate(gramASize, d_memory)\
                        depend(in: lambda[0 : blocksize * blocksize], gramXAR[0 : blocksize * currentBlockSize], gramXAP[0 : blocksize * currentBlockSize])\
                        depend(inout: activeMask[0 : blocksize]) depend(in: activeRSize, explicitGramFlag, activePSize, restart)\
                        depend(out: gramA[0 : gramASize * gramASize])
                        {

                            pair<double *, int> prXAR = make_pair(gramXAR, 0);
                            pair<double *, int> prXAP = make_pair(gramXAP, 0);
                            if(isOnDevice(prXAR))
                            {
                                cudaError_t error_rbr = cudaMemcpy(gramXAR, d_memory+(unsigned long int) mp[prXAR][0] * num_per_blk, 
                                                                    blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                if(error_rbr != 0){ printf("cudaMemcpy failed gramXAR: %d\n", error_rbr); errorFlag = true;}
                            }

                            if(isOnDevice(prXAP))
                            {
                                cudaError_t error_xap = cudaMemcpy(gramXAP, d_memory+(unsigned long int) mp[prXAP][0] * num_per_blk, 
                                                                        blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                if(error_xap != 0){ printf("cudaMemcpy failed gramXAP: %d\n", error_xap); errorFlag = true;}
                            }

                            cudaDeviceSynchronize();

                            mat_copy(lambda, blocksize, blocksize, gramA, 0, 0, gramASize);
                            mat_copy(gramXAR, blocksize, currentBlockSize, gramA, 0, blocksize, gramASize);
                            mat_copy(gramXAP, blocksize, currentBlockSize, gramA, 0, blocksize + currentBlockSize, gramASize);
                            // printf("GA 1 gramASize: %d\n", gramASize);
                        } //end task 
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "CONSTRUCTGA2"))
                    {
                        gramASize = blocksize + currentBlockSize + currentBlockSize;
                        
                        #pragma omp task firstprivate(gramASize, d_memory)\
                        depend(in : gramPAP[0 : currentBlockSize * currentBlockSize])\
                        depend(in : gramXAR[0 : blocksize * currentBlockSize], gramRAR[0 : currentBlockSize * currentBlockSize])\
                        depend(in : gramRAP[0 : currentBlockSize * currentBlockSize], gramXAP[0 : blocksize * currentBlockSize])\
                        depend(inout: activeMask[0 : blocksize]) depend(in: activeRSize, explicitGramFlag, activePSize, restart)\
                        depend(inout : gramA[0 : gramASize * gramASize])
                        {

                            pair<double *, int> prRAP = make_pair(gramRAP, 0);
                            if(isOnDevice(prRAP))
                            {
                                cudaError_t error_rap = cudaMemcpy(gramRAP, d_memory+(unsigned long int) mp[prRAP][0] * num_per_blk, 
                                                                    currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                cudaDeviceSynchronize();
                                if(error_rap != 0){ printf("cudaMemcpy failed gramRAP: %d\n", error_rap); errorFlag = true;}
                            }

                            transpose(gramXAR, transGramXAR, currentBlockSize, blocksize);
                                
                            mat_copy(transGramXAR, currentBlockSize, blocksize, gramA, blocksize, 0, gramASize);
                            mat_copy(gramRAR, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize, gramASize);
                            mat_copy(gramRAP, currentBlockSize, currentBlockSize, gramA, blocksize, blocksize+currentBlockSize, gramASize);

                            transpose(gramXAP, transGramXAP, currentBlockSize, blocksize);
                            transpose(gramRAP, transGramRAP, currentBlockSize, currentBlockSize);

                            mat_copy(transGramXAP, currentBlockSize, blocksize, gramA, blocksize+currentBlockSize, 0, gramASize);
                            mat_copy(transGramRAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize, gramASize);
                            mat_copy(gramPAP, currentBlockSize, currentBlockSize, gramA, blocksize+currentBlockSize, blocksize+currentBlockSize, gramASize);
                            // printf("GA 2 gramASize: %d\n", gramASize);
                        } //end task
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "CONSTRUCTGB"))
                    {
                        //fix it later, define up there
                        gramASize = blocksize + currentBlockSize + currentBlockSize;
                        gramB = (double *) malloc(gramASize * gramASize * sizeof(double));

                        //code: 19
                        #pragma omp task firstprivate(gramASize)\
                        depend(in: identity_PAP[0 : currentBlockSize * currentBlockSize], activePSize, activeRSize)\
                        depend(in : gramXBP[0 : blocksize * currentBlockSize], gramRBP[0 : currentBlockSize * currentBlockSize])\
                        depend(inout: activeMask[0 : blocksize]) depend(in: activeRSize, explicitGramFlag, activePSize, restart)\
                        depend(out: gramB[0 : gramASize * gramASize]) 
                        {

                            pair<double *, int> prXBP = make_pair(gramXBP, 0);
                            pair<double *, int> prRBP = make_pair(gramRBP, 0);
                            if(isOnDevice(prXBP))
                            {
                                cudaError_t error_xbp = cudaMemcpy(gramXBP, d_memory+(unsigned long int) mp[prXBP][0] * num_per_blk, 
                                                                    blocksize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                if(error_xbp != 0){ printf("cudaMemcpy failed gramXBP: %d\n", error_xbp); errorFlag = true;}
                            }

                            if(isOnDevice(prRBP))
                            {
                                cudaError_t error_xbp = cudaMemcpy(gramRBP, d_memory+(unsigned long int) mp[prRBP][0] * num_per_blk, 
                                                                        currentBlockSize * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
                                if(error_xbp != 0){ printf("cudaMemcpy failed gramRBP: %d\n", error_xbp); errorFlag = true;}
                            }

                            cudaDeviceSynchronize();

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
                            // printf("GB gramASize: %d\n", gramASize);
                            // cout << "CONSTRUCTGB - " << " activePSize: " << activePSize <<  " activeRSize; " << activeRSize << endl;
                        } //end task
                    }
                    else if(!strcmp(taskInfo_secondLoop[structIterator].strParamsList[0], "EIGEN"))
                    {
                        #pragma omp task default(shared) private(i, j, k)\
                        firstprivate(gramASize, d_memory)\
                        depend(inout: gramA[0 : gramASize * gramASize])\
                        depend(inout: activeMask[0 : blocksize]) depend(in: activeRSize, explicitGramFlag, activePSize, restart)\
                        depend(in: gramB[0 : gramASize * gramASize], currentBlockSize, activeRSize, activePSize)\
                        depend(out: coordX1[0 : blocksize * blocksize], coordX2[0 : currentBlockSize * blocksize], coordX3[0 : currentBlockSize * blocksize], lambda[0 : blocksize * blocksize])
                        {
                            // printf("EIG 1 => convergeFlag: %d currentBlockSize: %d activePSize: %d activeRSize: %d restart: %d  gramASize: %d\n", convergeFlag, currentBlockSize, activePSize, activeRSize, restart, gramASize);
                            // printf("EIG 2 gramASize: %d\n", gramASize);
                            cudaError_t lambda_error;

                            eigen_value = (double *) malloc(gramASize * sizeof(double));
                        
                            info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, itype, jobz, uplo, gramASize, gramA, gramASize, gramB, gramASize, eigen_value);
                            if(info != 0)
                                printf("LAPACKE_dsygv error: The algorithm failed to compute eigenvalues.\n" );
 
                            diag(eigen_value, lambda, blocksize);

                            pair<double *, int> prLambda = make_pair(lambda, 0);
                            if(isOnDevice(prLambda))
                            {
                                lambda_error = cudaMemcpy(d_memory+(unsigned long int) mp[prLambda][0] * num_per_blk, lambda, blocksize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                                // cudaDeviceSynchronize();
                                if(lambda_error != 0){ printf("cudaMemcpy failed gramRBR: %d\n", lambda_error); errorFlag = true;}
                            }

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

                            pair<double *, int> x1 = make_pair(coordX1, 0);
                            pair<double *, int> x2 = make_pair(coordX2, 0);
                            pair<double *, int> x3 = make_pair(coordX3, 0);

                            if(isOnDevice(x1))
                            {
                                lambda_error = cudaMemcpy(d_memory+(unsigned long int) mp[x1][0] * num_per_blk, coordX1, 
                                                            blocksize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                                if(lambda_error != 0){ printf("cudaMemcpy failed coordX1: %d\n", lambda_error); errorFlag = true;}
                            }
                            if(isOnDevice(x2))
                            {
                                lambda_error = cudaMemcpy(d_memory+(unsigned long int) mp[x2][0] * num_per_blk, coordX2, 
                                                            currentBlockSize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                                if(lambda_error != 0){ printf("cudaMemcpy failed coordX2: %d\n", lambda_error); errorFlag = true;}
                            }
                            if(isOnDevice(x3))
                            {
                                lambda_error = cudaMemcpy(d_memory+(unsigned long int) mp[x3][0] * num_per_blk, coordX3, 
                                                            currentBlockSize * blocksize * sizeof(double), cudaMemcpyHostToDevice);
                                if(lambda_error != 0){ printf("cudaMemcpy failed coordX3: %d\n", lambda_error); errorFlag = true;}
                            }
                            cudaDeviceSynchronize();

                        } //end task
                    }
                } //end of taskID 21 
                else if(taskInfo_secondLoop[structIterator].opCode == 27) // MEMCPY
                {
                    // for(i = 0 ; i < nrowblk ; i++) //for each block
                    // {
                    //     blksz = block_width;
                    //     if(i * block_width + blksz > M)
                    //         blksz = M - i * block_width;
                        
                    //     #pragma omp task firstprivate(M, t, h, nrowblk, blksz, i, block_width)\
                    //     depend(in: activeBlockVectorR[i * block_width * currentBlockSize : blksz * currentBlockSize])\
                    //     depend(out: d_activeBlockVectorR[i * block_width * currentBlockSize : blksz * currentBlockSize])
                    //     {
                    //         cudaError_t cuberror3;
                    //         pair<double *, int> prActR = make_pair(activeBlockVectorR, i);
                            
                    //         if(isOnDevice(prActR)) //device to device copy
                    //         {
                    //             cuberror3 = cudaMemcpy(d_activeBlockVectorR + (i * block_width * currentBlockSize), 
                    //                                     d_memory + (unsigned long int) mp[prActR][0] * num_per_blk, 
                    //                                     blksz * currentBlockSize * sizeof(double), cudaMemcpyDeviceToDevice);
                    //         }
                    //         else //host to device copy
                    //         {
                    //             cuberror3 = cudaMemcpy(d_activeBlockVectorR + (i * block_width * currentBlockSize), 
                    //                                 activeBlockVectorR + (i * block_width * currentBlockSize), 
                    //                                 blksz * currentBlockSize * sizeof(double), cudaMemcpyHostToDevice);
                    //         }
                    //         if(cuberror3 != 0){ printf("cudaMemcpy failed in MEMCPY: %d\n", cuberror3); errorFlag = true;}
                    //     }
                    // }

                    #pragma omp taskwait
                    cudaDeviceSynchronize();
                }
            }//end taskinfo loop

            #pragma omp taskwait
            cudaDeviceSynchronize();
            
            #if defined(NSYS)
                nvtxRangePop();
            #endif

            loop_finish_time = get_seconds();
            loopTime[iterationNumber - 1] =  loop_finish_time - loop_start_time;
            // saving lambada of each iteration
            for(i = 0 ; i < blocksize ; i++)
            {
                saveLamda[i][iterationNumber - 1] = lambda[i * blocksize + i];
            }
            printMapInfo(); printMemoryUsageStat(iterationNumber); cout << "loopTime[" << iterationNumber - 1 << "]: " << loopTime[iterationNumber - 1] << endl;
            iterationNumber++;
            
        } //end while ==> Iteration
    } // end master
    } // end parallel 



    // printMapInfo(); printMemoryUsageStat();

    // cout << "loopTime[1]: " << loopTime[1] << endl;
    // *** M * B
    // cout << "currentBlockSize: " << currentBlockSize << endl;
    // status = copyToHost(blockVectorAX, M, blocksize, 60, block_width, block_width * blocksize);
    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;
    // print_mat(blockVectorAX + (60 * block_width * blocksize + i * 171), 10, blocksize);

    // *** M * CB
    // cout << endl << endl;
    // pair<double *, int> tempPR = make_pair(blockVectorX, 152);
    // if(isOnDevice(tempPR)){ cout << "tempPR is on device: "  << (unsigned long int)mp[tempPR][0] << " num_element: " << (unsigned long int)mp[tempPR][1] << endl; //}
    // status = copyToHost(blockVectorX, M, currentBlockSize, 152, block_width, (unsigned long int)mp[tempPR][1]);
    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;
    // }
    // print_mat(blockVectorX + (152 * block_width * currentBlockSize + 171), 10, currentBlockSize);
    // cout << endl << endl;
    

//    // *** M * CB
//     // cout << endl << endl;

    // for(i = 0 ; i < nrowblk ; i++)
    // {
    //     cout << "block_id : " << i << endl;
    //     blksz = block_width;
    //     if(i * block_width + blksz > M)
    //         blksz = M - i * block_width;
        
    //     // cout << "block_id : " << i <<  " blksz: " << blksz << " currentBlockSize: " << currentBlockSize << endl;

    //     if(isOnDevice(make_pair(blockVectorAX, i)))
    //     { 
    //         status = copyToHost(blockVectorAX, M, blocksize, i, block_width, blksz * currentBlockSize);
    //         if(status != 0)
    //             cout << "Error in copyToHost" << endl;
    //     }
    //     // else
    //     //     cout << "Not in device" << endl;

    //     print_mat(blockVectorAX + (i * block_width * currentBlockSize + 171), 10, currentBlockSize);
    //     cout << endl << endl;
    // }

    // status = copyToHost(activeBlockVectorAR, M, currentBlockSize, i, block_width, blksz * currentBlockSize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(activeBlockVectorAR + (i * block_width * currentBlockSize + i * 171), 10, currentBlockSize);

    // cout << endl << endl;
    


    // status = omp_target_memcpy(activeBlockVectorR + ( 5 * block_width * currentBlockSize), d_activeBlockVectorR + ( 5 * block_width * currentBlockSize), block_width * currentBlockSize * sizeof(double), 0, 0, h, t);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(activeBlockVectorR + (5 * block_width * currentBlockSize), 10, currentBlockSize);

    // cout << endl << endl;

    // *** B * CB

    // status = copyToHost(lambda, blocksize, currentBlockSize, 0, blocksize, blocksize * currentBlockSize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(lambda, blocksize, currentBlockSize);
    // cout << endl;


    // *** CB * CB

    // status = copyToHost(gramRBR, currentBlockSize, currentBlockSize, 0, currentBlockSize, currentBlockSize * currentBlockSize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(gramRBR, currentBlockSize, currentBlockSize);
    // cout << endl;


    // ***
    // status = copyToHost(blockVectorX, M, blocksize, 5, block_width, block_width * blocksize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(blockVectorX + (5 * block_width * blocksize), 10, blocksize);


    // status = omp_target_memcpy(residualNorms, d_residualNorms, blocksize * sizeof(double), 0, 0, h, t);
    // if( status != 0 ){ printf("omp_target_memcpy failed residualNorms ==> %d\n", cuberror); return 0; }

    // print_mat(residualNorms, 1, blocksize);
    
    // status = omp_target_memcpy(activeMask, d_activeMask, blocksize * sizeof(int), 0, 0, h, t);
    // if( status != 0 ){ printf("omp_target_memcpy failed activeMask ==> %d\n", cuberror); return 0; }
    // for(i = 0 ; i < blocksize ; i++)
    //     cout << activeMask[i] << " ";
    // cout << endl;

    // print_mat(blockVectorR + (5 * block_width * blocksize), 10, blocksize);
   
    // cout << endl << endl;

    // print_mat(lambda, blocksize, blocksize);
   
    // cout << endl << endl;
    
    // status = copyToHost(lambda, blocksize, blocksize, 0, blocksize, blocksize * blocksize);

    // if(status != 0)
    //     cout << "Error in copyToHost" << endl;

    // print_mat(lambda, blocksize, blocksize);
    
    // int totalBlock, nMemBlock, maxBlock = -1, minBlock = 100000000;
    // for(i = 0 ; i < nrowblk ; i++)
    // {
    //     nMemBlock = requiredDeviceBlocks(nnz_per_tile[i] * sizeof(double), memGranularity);
    //     cout << "SpMM[" << i << "] :" << nMemBlock << endl;
    //     if(maxBlock < nMemBlock)
    //         maxBlock = nMemBlock;
    //     if(minBlock > nMemBlock)
    //         minBlock = nMemBlock;
    //     totalBlock += nMemBlock;
    // }
    // cout << "totalBlock: " << totalBlock <<  " avgBlock: " << totalBlock * 1.0 / nrowblk << " minBlock: " << minBlock << " maxBlock: " << maxBlock << endl; 


    //printing eigen values
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




    // printf("Finished second and rest of the loops.....\n");

    printf("\nLoopTime: \n");
    printf("\n");
    double totalSum = 0;
    for(j = 0 ; j < maxIterations ; j++)
    {
        totalSum += loopTime[j];
        printf("%.4lf,", loopTime[j]);
    }
    printf("%.4lf", totalSum/(maxIterations));
    printf("\n\n");

    summarize_timing_variables();
    return 0;
}


/*
Module list required

module load cdt/19.03
module swap PrgEnv-{intel,cray}
module swap craype-{haswell,x86-skylake}
module unload cray-libsci
module load cudatoolkit craype-accel-nvidia70
module load cuda
*/
