//#include "utils.h"
#include "kernels.h"

int hpx_main(hpx::program_options::variables_map& vm)
{
    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;
    using hpx::dataflow;
    using hpx::util::unwrapping;

    /* parse commend line arguments */
    int block_width = vm["block_width"].as<int>();
    std::string str_file = vm["matrix_file"].as<std::string>();
    int blocksize = vm["blocksize"].as<int>();
    int maxIterations = vm["iteration"].as<int>();
    nthreads = vm["nthreads"].as<int>();
    int currentBlockSize = blocksize;
    int gramSize = blocksize + currentBlockSize + currentBlockSize;

    int len = (int)str_file.length();
    char *file_name = new char[len + 1];
    strcpy(file_name, str_file.c_str());

    int *colptrs, *irem;
    double *xrem;
    block *A;

    int M;
    int i, j, iter;
    int flag;
    double threshold, totalSum;
    double *loopTime;

    /* read the matrix in csc format */
    read_custom(file_name, colptrs, irem, xrem);
    M = numrows;

    /* convert it to csb */
    csc2blkcoord(A, colptrs, irem, xrem, block_width);

    /* deleting CSC storage memory */
    delete [] file_name;

    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    /* define lobpcg variables */
    double *gramA, *transGramA, *gramB, *transGramB, *eigenvalue, *coordX;
    double *blockVectorX, *blockVectorAX, *blockVectorR, *blockVectorP, *blockVectorAP, *newX, *newAX;
    double *activeBlockVectorR, *activeBlockVectorAR, *activeBlockVectorP, *activeBlockVectorAP, *temp3, *newP, *newAP;
    double *gramXAX, *transGramXAX, *gramXBX, *transGramXBX, *lambda, *identity_BB;
    double *gramXAR, *gramXAP, *gramXBP, *zeros_BC, *temp2;
    double *transGramXAR, *transGramXAP, *transGramXBP, *zeros_CB;
    double *gramRBR, *transGramRBR, *gramRAR, *transGramRAR, *gramPBP, *transGramPBP, *gramRAP, *transGramRAP, *gramRBP, *transGramRBP, *gramPAP, *identity_PAP;
    double *saveLambda;
    int *activeMask;
    double *tempLambda, *residualNorms;

    double *gramXAXBUF, *gramXBXBUF;
    double *gramXARBUF, *gramXAPBUF, *gramXBPBUF, *temp2BUF;
    double *gramRBRBUF, *gramRARBUF, *gramPBPBUF, *gramRAPBUF, *gramRBPBUF, *gramPAPBUF;
    double *residualNormsBUF;

    /* allocating memory for lobpcg variables */
    blockVectorX = (double *) malloc(M * blocksize * sizeof(double));
    blockVectorAX = (double *) malloc(M * blocksize * sizeof(double));
    blockVectorR = (double *) malloc(M * blocksize * sizeof(double));
    blockVectorP = (double *) malloc(M * blocksize * sizeof(double));
    blockVectorAP = (double *) malloc(M * blocksize * sizeof(double));
    newX = (double *) malloc(M * blocksize * sizeof(double));
    newAX = (double *) malloc(M * blocksize * sizeof(double));
    
    activeBlockVectorR = (double *) malloc(M * currentBlockSize * sizeof(double));
    activeBlockVectorAR = (double *) malloc(M * currentBlockSize * sizeof(double));
    activeBlockVectorP = (double *) malloc(M * currentBlockSize * sizeof(double));
    activeBlockVectorAP = (double *) malloc(M * currentBlockSize * sizeof(double));
    temp3 = (double *) malloc(M * currentBlockSize * sizeof(double));
    newP = (double *) malloc(M * currentBlockSize * sizeof(double));
    newAP = (double *) malloc(M * currentBlockSize * sizeof(double));

    gramXAX = (double *) malloc(blocksize * blocksize * sizeof(double));
    transGramXAX = (double *) malloc(blocksize * blocksize * sizeof(double));
    gramXBX = (double *) malloc(blocksize * blocksize * sizeof(double));
    transGramXBX = (double *) malloc(blocksize * blocksize * sizeof(double));
    lambda = (double *) malloc(blocksize * blocksize * sizeof(double));
    identity_BB = (double *) malloc(blocksize * blocksize * sizeof(double));
    
    gramXAR = (double *) malloc(blocksize * currentBlockSize * sizeof(double));
    gramXAP = (double *) malloc(blocksize * currentBlockSize * sizeof(double));
    gramXBP = (double *) malloc(blocksize * currentBlockSize * sizeof(double));
    zeros_BC = (double *) malloc(blocksize * currentBlockSize * sizeof(double));
    temp2 = (double *) malloc(blocksize * currentBlockSize * sizeof(double));

    transGramXAR = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    transGramXAP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    transGramXBP = (double *) malloc(currentBlockSize * blocksize * sizeof(double));
    zeros_CB = (double *) malloc(currentBlockSize * blocksize * sizeof(double));

    gramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    transGramRBR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    gramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    transGramRAR = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    gramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    transGramPBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    gramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    transGramRAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    gramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    transGramRBP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    gramPAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));
    identity_PAP = (double *) malloc(currentBlockSize * currentBlockSize * sizeof(double));

    saveLambda = (double *) malloc(blocksize * maxIterations * sizeof(double));
    
    activeMask = (int *) malloc(blocksize * sizeof(int));
    tempLambda = (double *) malloc(blocksize * sizeof(double));
    residualNorms = (double *) malloc(blocksize * sizeof(double));

    gramXAXBUF = (double *) malloc(nrowblks * blocksize * blocksize * sizeof(double));
    gramXBXBUF = (double *) malloc(nrowblks * blocksize * blocksize * sizeof(double));
    
    gramXARBUF = (double *) malloc(nrowblks * blocksize * currentBlockSize * sizeof(double));
    gramXAPBUF = (double *) malloc(nrowblks * blocksize * currentBlockSize * sizeof(double));
    gramXBPBUF = (double *) malloc(nrowblks * blocksize * currentBlockSize * sizeof(double));
    temp2BUF = (double *) malloc(nrowblks * blocksize * currentBlockSize * sizeof(double));

    gramRBRBUF = (double *) malloc(nrowblks * currentBlockSize * currentBlockSize * sizeof(double));
    gramRARBUF = (double *) malloc(nrowblks * currentBlockSize * currentBlockSize * sizeof(double));
    gramPBPBUF = (double *) malloc(nrowblks * currentBlockSize * currentBlockSize * sizeof(double));
    gramRAPBUF = (double *) malloc(nrowblks * currentBlockSize * currentBlockSize * sizeof(double));
    gramRBPBUF = (double *) malloc(nrowblks * currentBlockSize * currentBlockSize * sizeof(double));
    gramPAPBUF = (double *) malloc(nrowblks * currentBlockSize * currentBlockSize * sizeof(double));

    residualNormsBUF = (double *) malloc(nrowblks * blocksize * sizeof(double));

    loopTime = (double *) malloc(maxIterations * sizeof(double));

    gramA = (double *) malloc(gramSize * gramSize * sizeof(double));
    gramB = (double *) malloc(gramSize * gramSize * sizeof(double));
    transGramA = (double *) malloc(gramSize * gramSize * sizeof(double));
    transGramB = (double *) malloc(gramSize * gramSize * sizeof(double));
    eigenvalue = (double *) malloc(gramSize * sizeof(double));
    coordX = (double *) malloc(gramSize * blocksize * sizeof(double));


    /* defining future variables */
    hpx::shared_future<void> future_tempLambda;
    hpx::shared_future<void> future_saveLambda;
    hpx::shared_future<void> future_identity_BB;
    hpx::shared_future<void> future_identity_PAP;
    hpx::shared_future<void> future_zeros_BC;
    hpx::shared_future<void> future_zeros_CB;
    hpx::shared_future<void> future_gramXBX;
    std::vector<hpx::shared_future<void>> future_gramXBXBUF(nrowblks);
    hpx::shared_future<void> future_gramXAX;
    std::vector<hpx::shared_future<void>> future_gramXAXBUF(nrowblks);
    hpx::shared_future<void> future_transGramXBX;
    std::vector<hpx::shared_future<void>> future_blockVectorX(nrowblks);
    std::vector<hpx::shared_future<void>> future_newX(nrowblks);
    std::vector<hpx::shared_future<void>> future_blockVectorAX(nrowblks);
    hpx::shared_future<void> future_transGramXAX;
    hpx::shared_future<void> future_lambda;
    std::vector<hpx::shared_future<void>> future_newAX(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramXARBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramXAPBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramXBPBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_temp2BUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramRBRBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramRARBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramPBPBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramRAPBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramRBPBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_gramPAPBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_residualNormsBUF(nrowblks);
    std::vector<hpx::shared_future<void>> future_blockVectorR(nrowblks);
    hpx::shared_future<void> future_residualNorms;
    hpx::shared_future<int> future_currentBlockSize;
    hpx::shared_future<void> future_temp2;
    std::vector<hpx::shared_future<void>> future_activeBlockVectorR(nrowblks);
    std::vector<hpx::shared_future<void>> future_temp3(nrowblks);
    std::vector<hpx::shared_future<void>> future_newP(nrowblks);
    std::vector<hpx::shared_future<void>> future_newAP(nrowblks);
    hpx::shared_future<void> future_gramRBR;
    hpx::shared_future<void> future_transGramRBR;
    std::vector<hpx::shared_future<void>> future_activeBlockVectorAR(nrowblks);
    hpx::shared_future<void> future_gramXAR;
    hpx::shared_future<void> future_gramRAR;
    hpx::shared_future<void> future_transGramRAR;
    hpx::shared_future<void> future_transGramXAR;
    std::vector<hpx::shared_future<void>> future_gramA(9);
    std::vector<hpx::shared_future<void>> future_gramB(9);
    hpx::shared_future<void> future_transGramA;
    hpx::shared_future<void> future_transGramB;
    hpx::shared_future<void> future_eigenvalue;
    hpx::shared_future<void> future_gramAF;
    hpx::shared_future<void> future_gramBF;
    hpx::shared_future<void> future_coordX;
    std::vector<hpx::shared_future<void>> future_blockVectorP(nrowblks);
    std::vector<hpx::shared_future<void>> future_blockVectorAP(nrowblks);
    std::vector<hpx::shared_future<void>> future_activeBlockVectorP(nrowblks);
    std::vector<hpx::shared_future<void>> future_activeBlockVectorAP(nrowblks);
    hpx::shared_future<void> future_gramPBP;
    hpx::shared_future<void> future_transGramPBP;
    hpx::shared_future<void> future_gramXAP;
    hpx::shared_future<void> future_transGramXAP;
    hpx::shared_future<void> future_gramRAP;
    hpx::shared_future<void> future_transGramRAP;
    hpx::shared_future<void> future_gramPAP;
    hpx::shared_future<void> future_gramXBP;
    hpx::shared_future<void> future_transGramXBP;
    hpx::shared_future<void> future_gramRBP;
    hpx::shared_future<void> future_transGramRBP;

    /* initialize vectors in parallel for first touch policy optimization */
    /*
    auto OpFill = unwrapping(&Fill);

    std::vector<hpx::shared_future<void>> reset_future(nthreads);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, blockVectorX, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, blockVectorAX, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, blockVectorR, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, blockVectorP, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, blockVectorAP, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, newX, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, newAX, M * blocksize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, activeBlockVectorR, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, activeBlockVectorAR, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, activeBlockVectorP, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, activeBlockVectorAP, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, temp3, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, newP, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);

    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpFill, newAP, M * currentBlockSize, i);
    }
    hpx::wait_all(reset_future);
    */
    
    using executor = hpx::execution::experimental::fork_join_executor;
    std::vector<int> v(nthreads);
    std::iota(std::begin(v), std::end(v), 0);
    executor exec;

    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, blockVectorX, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, blockVectorAX, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, blockVectorR, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, blockVectorP, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, blockVectorAP, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, newX, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, newAX, M * blocksize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, activeBlockVectorR, M * currentBlockSize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, activeBlockVectorAR, M * currentBlockSize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, activeBlockVectorP, M * currentBlockSize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, activeBlockVectorAP, M * currentBlockSize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, temp3, M * currentBlockSize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, newP, M * currentBlockSize);
    hpx::parallel::execution::bulk_sync_execute(exec, &Fill, v, newAP, M * currentBlockSize);

    /* defining function unwrappers */
    auto OpReset = unwrapping(&Reset);
    auto OpIdentity = unwrapping(&Identity);
    auto OpXTY_part = unwrapping(&XTY_part);
    auto OpXTY_red = unwrapping(&XTY_red);
    auto OpTranspose = unwrapping(&Transpose);
    auto OpCholesky = unwrapping(&Cholesky);
    auto OpResetBelowDiagonal = unwrapping(&ResetBelowDiagonal);
    auto OpCopy = unwrapping(&Copy);
    auto OpInverse = unwrapping(&Inverse);
    auto OpXY_part = unwrapping(&XY_part);
    auto OpCopy_part = unwrapping(&Copy_part);
    auto OpReset_part = unwrapping(&Reset_part);
    auto OpSpMM_part = unwrapping(&SpMM_part);
    auto OpXY = unwrapping(&XY);
    auto OpEigenComp = unwrapping(&EigenComp);
    auto OpDiag = unwrapping(&Diag);
    auto OpSubtract_part = unwrapping(&Subtract_part);
    auto OpMultiply_part = unwrapping(&Multiply_part);
    auto OpReducedNorm_part = unwrapping(&ReducedNorm_part);
    auto OpReducedNorm_red = unwrapping(&ReducedNorm_red);
    auto OpUpdateActiveMask = unwrapping(&UpdateActiveMask);
    auto OpGatherActiveVectors_part = unwrapping(&GatherActiveVectors_part);
    auto OpScatterActiveVectors_part = unwrapping(&ScatterActiveVectors_part);
    auto OpCopyBlock = unwrapping(&CopyBlock);
    auto OpCopyCoordX = unwrapping(&CopyCoordX);
    auto OpAdd_part = unwrapping(&Add_part);
    auto OpXTY = unwrapping(&XTY);

    /* non-loop */
    srand(0);
    for(i = 0; i != M * blocksize; ++i)
    {
        blockVectorX[i] = (double)rand()/(double)RAND_MAX;    
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_blockVectorX[i] = hpx::make_ready_future();
        future_blockVectorR[i] = hpx::make_ready_future();
    }
    for(i = 0; i != 9; ++i)
    {
        future_gramA[i] = hpx::make_ready_future();
        future_gramB[i] = hpx::make_ready_future();
    }
    future_gramXAR = hpx::make_ready_future();
    future_gramXAP = hpx::make_ready_future();
    future_gramXBP = hpx::make_ready_future();
    future_temp2 = hpx::make_ready_future();
    future_gramRBR = hpx::make_ready_future();
    future_gramRAR = hpx::make_ready_future();
    future_gramPBP = hpx::make_ready_future();
    future_gramRAP = hpx::make_ready_future();
    future_gramRBP = hpx::make_ready_future();
    future_gramPAP = hpx::make_ready_future();

    // std::vector<hpx::parallel::execution::default_executor > execs;
    std::vector<hpx::execution::parallel_executor> execs;
    // hpx::execution::parallel_executor exec_prefer;
    execs.reserve(nthreads);
    for(i = 0; i != nthreads; ++i)
    {
        /*
        hpx::parallel::execution::default_executor exec(
                hpx::threads::thread_schedule_hint(
                    hpx::threads::thread_schedule_hint_mode::thread, i));
        */
        
        hpx::execution::parallel_executor exec(
                hpx::threads::thread_schedule_hint(
                    hpx::threads::thread_schedule_hint_mode::thread, i));
        
        execs.push_back(exec);
        
        /*
        auto exec_prefer_instance = hpx::experimental::prefer(hpx::execution::experimental::make_with_hint, exec_prefer, i);
        execs.push_back(exec_prefer_instance);
        */
    }

    for(i = 0; i != blocksize; ++i)
    {
        activeMask[i] = 1;
    }
    threshold = 1e-10;
    totalSum = 0.0;

    future_tempLambda = dataflow(hpx::launch::async, OpReset, tempLambda, blocksize, 1);
    future_saveLambda = dataflow(hpx::launch::async, OpReset, saveLambda, blocksize, maxIterations);

    future_identity_BB = dataflow(hpx::launch::async, OpIdentity, identity_BB, blocksize);
    future_identity_PAP = dataflow(hpx::launch::async, OpIdentity, identity_PAP, currentBlockSize);
    future_zeros_BC = dataflow(hpx::launch::async, OpReset, zeros_BC, blocksize, currentBlockSize);
    future_zeros_CB = dataflow(hpx::launch::async, OpReset, zeros_CB, currentBlockSize, blocksize);

    for(i = 0; i != nrowblks; ++i)
    {
        future_gramXBXBUF[i] = dataflow(execs[threadIndex[i]], OpReset, gramXBXBUF + i*blocksize*blocksize, blocksize, blocksize);
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_gramXAXBUF[i] = dataflow(execs[threadIndex[i]], OpReset, gramXAXBUF + i*blocksize*blocksize, blocksize, blocksize);
    }

    for(i = 0; i != nrowblks; ++i)
    {
        future_gramXBXBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_gramXBXBUF[i], future_blockVectorX[i], blockVectorX, blockVectorX, gramXBXBUF, blocksize, blocksize, M, 1.0, 0.0, block_width, i);
    }
    future_gramXBX = dataflow(hpx::launch::async, OpXTY_red, future_gramXBXBUF, gramXBXBUF, gramXBX, blocksize, blocksize);
    
    future_transGramXBX = dataflow(hpx::launch::async, OpTranspose, future_gramXBX, gramXBX, transGramXBX, blocksize, blocksize);
    future_transGramXBX = dataflow(hpx::launch::async, OpCholesky, future_transGramXBX, transGramXBX, blocksize);
    future_gramXBX = dataflow(hpx::launch::async, OpTranspose, future_transGramXBX, transGramXBX, gramXBX, blocksize, blocksize);
    future_gramXBX = dataflow(hpx::launch::async, OpResetBelowDiagonal, future_gramXBX, gramXBX, blocksize);
    future_transGramXBX = dataflow(hpx::launch::async, OpCopy, future_gramXBX, gramXBX, transGramXBX, blocksize, blocksize);
    future_transGramXBX = dataflow(hpx::launch::async, OpInverse, future_transGramXBX, transGramXBX, blocksize);

    for(i = 0; i != nrowblks; ++i)
    {
        future_newX[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorX[i], future_transGramXBX, blockVectorX, transGramXBX, newX, M, blocksize, blocksize, 1.0, 0.0, block_width, i);
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_blockVectorX[i] = dataflow(execs[threadIndex[i]], OpCopy_part, future_newX[i], newX, blockVectorX, M, blocksize, block_width, i);
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_blockVectorAX[i] = dataflow(execs[threadIndex[i]], OpReset_part, blockVectorAX, M, blocksize, block_width, i);
    }
    for(j = 0; j != ncolblks; ++j)
    {
        for(i = 0; i != nrowblks; ++i)
        {
            if(A[i*ncolblks + j].nnz > 0)
            {
                future_blockVectorAX[i] = dataflow(execs[threadIndex[i]], OpSpMM_part, future_blockVectorAX[i], future_blockVectorX[j], A, blockVectorX, blockVectorAX, blocksize, i, j);
            }
        }
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_gramXAXBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_blockVectorX[i], future_blockVectorAX[i], blockVectorX, blockVectorAX, gramXAXBUF, blocksize, blocksize, M, 1.0, 0.0, block_width, i);
    }
    future_gramXAX = dataflow(hpx::launch::async, OpXTY_red, future_gramXAXBUF, gramXAXBUF, gramXAX, blocksize, blocksize);
    
    future_transGramXAX = dataflow(hpx::launch::async, OpTranspose, future_gramXAX, gramXAX, transGramXAX, blocksize, blocksize);
    future_gramXAX = dataflow(hpx::launch::async, OpXY, future_transGramXAX, future_identity_BB, transGramXAX, identity_BB, gramXAX, blocksize, blocksize, blocksize, 0.5, 0.5);

    future_transGramXAX = dataflow(hpx::launch::async, OpTranspose, future_gramXAX, gramXAX, transGramXAX, blocksize, blocksize);
    future_transGramXAX = dataflow(hpx::launch::async, OpEigenComp, future_transGramXAX, future_tempLambda, transGramXAX, identity_BB, tempLambda, blocksize);
    future_gramXAX = dataflow(hpx::launch::async, OpTranspose, future_transGramXAX, transGramXAX, gramXAX, blocksize, blocksize);
    future_lambda = dataflow(hpx::launch::async, OpDiag, future_transGramXAX, tempLambda, lambda, blocksize);

    for(i = 0; i != nrowblks; ++i)
    {
        future_newX[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorX[i], future_gramXAX, blockVectorX, gramXAX, newX, M, blocksize, blocksize, 1.0, 0.0, block_width, i);
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_blockVectorX[i] = dataflow(execs[threadIndex[i]], OpCopy_part, future_newX[i], newX, blockVectorX, M, blocksize, block_width, i);
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_newAX[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorAX[i], future_gramXAX, blockVectorAX, gramXAX, newAX, M, blocksize, blocksize, 1.0, 0.0, block_width, i);
    }
    for(i = 0; i != nrowblks; ++i)
    {
        future_blockVectorAX[i] = dataflow(execs[threadIndex[i]], OpCopy_part, future_newAX[i], newAX, blockVectorAX, M, blocksize, block_width, i);
    }
    
    
    /* first loop */
    for(iter = 0; iter != maxIterations; ++iter)
    {
        hpx::chrono::high_resolution_timer t;
        for(i = 0; i != nrowblks; ++i)
        {
            future_gramXARBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramXAR, gramXARBUF + i*blocksize*currentBlockSize, blocksize, currentBlockSize);
            future_gramXAPBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramXAP, gramXAPBUF + i*blocksize*currentBlockSize, blocksize, currentBlockSize);
            future_gramXBPBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramXBP, gramXBPBUF + i*blocksize*currentBlockSize, blocksize, currentBlockSize);
            future_temp2BUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_temp2, temp2BUF + i*blocksize*currentBlockSize, blocksize, currentBlockSize);
            
            future_gramRBRBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramRBR, gramRBRBUF + i*currentBlockSize*currentBlockSize, currentBlockSize, currentBlockSize);
            future_gramRARBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramRAR, gramRARBUF + i*currentBlockSize*currentBlockSize, currentBlockSize, currentBlockSize);
            future_gramPBPBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramPBP, gramPBPBUF + i*currentBlockSize*currentBlockSize, currentBlockSize, currentBlockSize);
            future_gramRAPBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramRAP, gramRAPBUF + i*currentBlockSize*currentBlockSize, currentBlockSize, currentBlockSize);
            future_gramRBPBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramRBP, gramRBPBUF + i*currentBlockSize*currentBlockSize, currentBlockSize, currentBlockSize);
            future_gramPAPBUF[i] = dataflow(execs[threadIndex[i]], OpReset, future_gramPAP, gramPAPBUF + i*currentBlockSize*currentBlockSize, currentBlockSize, currentBlockSize);
            
            future_residualNormsBUF[i] = dataflow(execs[threadIndex[i]], OpReset, residualNormsBUF + i*blocksize, 1, blocksize);
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorR[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorR[i], future_blockVectorX[i], future_lambda, blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, 1.0, 0.0, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorR[i] = dataflow(execs[threadIndex[i]], OpSubtract_part, future_blockVectorAX[i], future_blockVectorR[i], blockVectorAX, blockVectorR, blockVectorR, M, blocksize, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_newX[i] = dataflow(execs[threadIndex[i]], OpMultiply_part, future_blockVectorR[i], blockVectorR, blockVectorR, newX, M, blocksize, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_residualNormsBUF[i] = dataflow(execs[threadIndex[i]], OpReducedNorm_part, future_newX[i], newX, residualNormsBUF, M, blocksize, block_width, i);
        }
        future_residualNorms = dataflow(hpx::launch::async, OpReducedNorm_red, future_residualNormsBUF, residualNormsBUF, residualNorms, blocksize);
        future_currentBlockSize = dataflow(hpx::launch::async, OpUpdateActiveMask, future_residualNorms, residualNorms, activeMask, threshold, blocksize);
    
        currentBlockSize = future_currentBlockSize.get();
        flag = (currentBlockSize == blocksize);
        gramSize = (iter == 0) ? (blocksize + currentBlockSize) : (blocksize + currentBlockSize + currentBlockSize);

        if(currentBlockSize == 0)
        {
            printf("Converged at iteration #%d\n", iter + 1);
            break;
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_activeBlockVectorR[i] = dataflow(execs[threadIndex[i]], OpGatherActiveVectors_part, future_blockVectorR[i], blockVectorR, activeBlockVectorR, activeMask, M, blocksize, currentBlockSize, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_temp2BUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_blockVectorX[i], future_activeBlockVectorR[i], blockVectorX, activeBlockVectorR, temp2BUF, blocksize, currentBlockSize, M, 1.0, 0.0, block_width, i);
        }
        future_temp2 = dataflow(hpx::launch::async, OpXTY_red, future_temp2BUF, temp2BUF, temp2, blocksize, currentBlockSize);
        for(i = 0; i != nrowblks; ++i)
        {
            future_temp3[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorX[i], future_temp2, blockVectorX, temp2, temp3, M, currentBlockSize, blocksize, 1.0, 0.0, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_activeBlockVectorR[i] = dataflow(execs[threadIndex[i]], OpSubtract_part, future_activeBlockVectorR[i], future_temp3[i], activeBlockVectorR, temp3, activeBlockVectorR, M, currentBlockSize, block_width, i);
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_gramRBRBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_activeBlockVectorR[i], activeBlockVectorR, activeBlockVectorR, gramRBRBUF, currentBlockSize, currentBlockSize, M, 1.0, 0.0, block_width, i);
        }
        future_gramRBR = dataflow(hpx::launch::async, OpXTY_red, future_gramRBRBUF, gramRBRBUF, gramRBR, currentBlockSize, currentBlockSize);
        future_transGramRBR = dataflow(hpx::launch::async, OpTranspose, future_gramRBR, gramRBR, transGramRBR, currentBlockSize, currentBlockSize);
        future_transGramRBR = dataflow(hpx::launch::async, OpCholesky, future_transGramRBR, transGramRBR, currentBlockSize);
        future_gramRBR = dataflow(hpx::launch::async, OpTranspose, future_transGramRBR, transGramRBR, gramRBR, currentBlockSize, currentBlockSize);
        future_gramRBR = dataflow(hpx::launch::async, OpResetBelowDiagonal, future_gramRBR, gramRBR, currentBlockSize);
        future_gramRBR = dataflow(hpx::launch::async, OpInverse, future_gramRBR, gramRBR, currentBlockSize);
        for(i = 0; i != nrowblks; ++i)
        {
            future_temp3[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorR[i], future_gramRBR, activeBlockVectorR, gramRBR, temp3, M, currentBlockSize, currentBlockSize, 1.0, 0.0, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_activeBlockVectorR[i] = dataflow(execs[threadIndex[i]], OpCopy_part, future_temp3[i], temp3, activeBlockVectorR, M, currentBlockSize, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorR[i] = dataflow(execs[threadIndex[i]], OpScatterActiveVectors_part, future_activeBlockVectorR[i], activeBlockVectorR, blockVectorR, activeMask, M, blocksize, currentBlockSize, block_width, i);
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_activeBlockVectorAR[i] = dataflow(execs[threadIndex[i]], OpReset_part, activeBlockVectorAR, M, currentBlockSize, block_width, i);
        }
        for(j = 0; j != ncolblks; ++j)
        {
            for(i = 0; i != nrowblks; ++i)
            {
                if(A[i*ncolblks + j].nnz > 0)
                {
                    future_activeBlockVectorAR[i] = dataflow(execs[threadIndex[i]], OpSpMM_part, future_activeBlockVectorAR[i], future_activeBlockVectorR[j], A, activeBlockVectorR, activeBlockVectorAR, blocksize, i, j);
                }
            }
        }

        if(iter != 0)
        {
            for(i = 0; i != nrowblks; ++i)
            {
                future_activeBlockVectorP[i] = dataflow(execs[threadIndex[i]], OpGatherActiveVectors_part, future_blockVectorP[i], blockVectorP, activeBlockVectorP, activeMask, M, blocksize, currentBlockSize, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_gramPBPBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_activeBlockVectorP[i], activeBlockVectorP, activeBlockVectorP, gramPBPBUF, currentBlockSize, currentBlockSize, M, 1.0, 0.0, block_width, i);
            }
            future_gramPBP = dataflow(hpx::launch::async, OpXTY_red, future_gramPBPBUF, gramPBPBUF, gramPBP, currentBlockSize, currentBlockSize);
            future_transGramPBP = dataflow(hpx::launch::async, OpTranspose, future_gramPBP, gramPBP, transGramPBP, currentBlockSize, currentBlockSize);
            future_transGramPBP = dataflow(hpx::launch::async, OpCholesky, future_transGramPBP, transGramPBP, currentBlockSize);
            future_gramPBP = dataflow(hpx::launch::async, OpTranspose, future_transGramPBP, transGramPBP, gramPBP, currentBlockSize, currentBlockSize);
            future_gramPBP = dataflow(hpx::launch::async, OpResetBelowDiagonal, future_gramPBP, gramPBP, currentBlockSize);
            future_gramPBP = dataflow(hpx::launch::async, OpInverse, future_gramPBP, gramPBP, currentBlockSize);
            
            for(i = 0; i != nrowblks; ++i)
            {
                future_newP[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorP[i], future_gramPBP, activeBlockVectorP, gramPBP, newP, M, currentBlockSize, currentBlockSize, 1.0, 0.0, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_activeBlockVectorP[i] = dataflow(execs[threadIndex[i]], OpCopy_part, future_newP[i], newP, activeBlockVectorP, M, currentBlockSize, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_blockVectorP[i] = dataflow(execs[threadIndex[i]], OpScatterActiveVectors_part, future_activeBlockVectorP[i], activeBlockVectorP, blockVectorP, activeMask, M, blocksize, currentBlockSize, block_width, i);
            }
            
            for(i = 0; i != nrowblks; ++i)
            {
                future_activeBlockVectorAP[i] = dataflow(execs[threadIndex[i]], OpGatherActiveVectors_part, future_blockVectorAP[i], blockVectorAP, activeBlockVectorAP, activeMask, M, blocksize, currentBlockSize, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_newAP[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorAP[i], future_gramPBP, activeBlockVectorAP, gramPBP, newAP, M, currentBlockSize, currentBlockSize, 1.0, 0.0, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_activeBlockVectorAP[i] = dataflow(execs[threadIndex[i]], OpCopy_part, future_newAP[i], newAP, activeBlockVectorAP, M, currentBlockSize, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_blockVectorAP[i] = dataflow(execs[threadIndex[i]], OpScatterActiveVectors_part, future_activeBlockVectorAP[i], activeBlockVectorAP, blockVectorAP, activeMask, M, blocksize, currentBlockSize, block_width, i);
            }
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_gramXARBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_blockVectorAX[i], future_activeBlockVectorR[i], blockVectorAX, activeBlockVectorR, gramXARBUF, blocksize, currentBlockSize, M, 1.0, 0.0, block_width, i);
        }
        future_gramXAR = dataflow(hpx::launch::async, OpXTY_red, future_gramXARBUF, gramXARBUF, gramXAR, blocksize, currentBlockSize);
        for(i = 0; i != nrowblks; ++i)
        {
            future_gramRARBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_activeBlockVectorAR[i], future_activeBlockVectorR[i], activeBlockVectorAR, activeBlockVectorR, gramRARBUF, currentBlockSize, currentBlockSize, M, 1.0, 0.0, block_width, i);
        }
        future_gramRAR = dataflow(hpx::launch::async, OpXTY_red, future_gramRARBUF, gramRARBUF, gramRAR, currentBlockSize, currentBlockSize);
        future_transGramRAR = dataflow(hpx::launch::async, OpTranspose, future_gramRAR, gramRAR, transGramRAR, currentBlockSize, currentBlockSize);
        future_gramRAR = dataflow(hpx::launch::async, OpXY, future_transGramRAR, future_identity_PAP, transGramRAR, identity_PAP, gramRAR, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, 0.5);
    
        if(iter == 0)
        {
            if(flag)
            {
                future_transGramXAR = dataflow(hpx::launch::async, OpTranspose, future_gramXAR, gramXAR, transGramXAR, blocksize, currentBlockSize);

                future_gramA[0] = dataflow(hpx::launch::async, OpCopyBlock, future_lambda, lambda, gramA, blocksize, blocksize, gramSize, 0, 0);
                future_gramA[1] = dataflow(hpx::launch::async, OpCopyBlock, future_gramXAR, gramXAR, gramA, blocksize, currentBlockSize, gramSize, 0, blocksize);
                future_gramA[2] = dataflow(hpx::launch::async, OpCopyBlock, future_transGramXAR, transGramXAR, gramA, currentBlockSize, blocksize, gramSize, blocksize, 0);
                future_gramA[3] = dataflow(hpx::launch::async, OpCopyBlock, future_gramRAR, gramRAR, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize);

                future_gramB[0] = dataflow(hpx::launch::async, OpIdentity, gramB, gramSize);
            }
        }
        else
        {
            for(i = 0; i != nrowblks; ++i)
            {
                future_gramXAPBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_blockVectorAX[i], future_activeBlockVectorP[i], blockVectorAX, activeBlockVectorP, gramXAPBUF, blocksize, currentBlockSize, M, 1.0, 0.0, block_width, i);
            }
            future_gramXAP = dataflow(hpx::launch::async, OpXTY_red, future_gramXAPBUF, gramXAPBUF, gramXAP, blocksize, currentBlockSize);
            for(i = 0; i != nrowblks; ++i)
            {
                future_gramRAPBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_activeBlockVectorAR[i], future_activeBlockVectorP[i], activeBlockVectorAR, activeBlockVectorP, gramRAPBUF, currentBlockSize, currentBlockSize, M, 1.0, 0.0, block_width, i);
            }
            future_gramRAP = dataflow(hpx::launch::async, OpXTY_red, future_gramRAPBUF, gramRAPBUF, gramRAP, currentBlockSize, currentBlockSize);
            for(i = 0; i != nrowblks; ++i)
            {
                future_gramPAPBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_activeBlockVectorAP[i], future_activeBlockVectorP[i], activeBlockVectorAP, activeBlockVectorP, gramPAPBUF, currentBlockSize, currentBlockSize, M, 1.0, 0.0, block_width, i);
            }
            future_gramPAP = dataflow(hpx::launch::async, OpXTY_red, future_gramPAPBUF, gramPAPBUF, gramPAP, currentBlockSize, currentBlockSize);
            future_gramPAP = dataflow(hpx::launch::async, OpXTY, future_gramPAP, future_identity_PAP, gramPAP, identity_PAP, gramPAP, currentBlockSize, currentBlockSize, currentBlockSize, 0.5, 0.5);
            
            for(i = 0; i != nrowblks; ++i)
            {
                future_gramXBPBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_blockVectorX[i], future_activeBlockVectorP[i], blockVectorX, activeBlockVectorP, gramXBPBUF, blocksize, currentBlockSize, M, 1.0, 0.0, block_width, i);
            }
            future_gramXBP = dataflow(hpx::launch::async, OpXTY_red, future_gramXBPBUF, gramXBPBUF, gramXBP, blocksize, currentBlockSize);
            for(i = 0; i != nrowblks; ++i)
            {
                future_gramRBPBUF[i] = dataflow(execs[threadIndex[i]], OpXTY_part, future_activeBlockVectorR[i], future_activeBlockVectorP[i], activeBlockVectorR, activeBlockVectorP, gramRBPBUF, currentBlockSize, currentBlockSize, M, 1.0, 0.0, block_width, i);
            }
            future_gramRBP = dataflow(hpx::launch::async, OpXTY_red, future_gramRBPBUF, gramRBPBUF, gramRBP, currentBlockSize, currentBlockSize);

            if(flag)
            {
                future_transGramXAR = dataflow(hpx::launch::async, OpTranspose, future_gramXAR, gramXAR, transGramXAR, blocksize, currentBlockSize);
                future_transGramXAP = dataflow(hpx::launch::async, OpTranspose, future_gramXAP, gramXAP, transGramXAP, blocksize, currentBlockSize);
                future_transGramRAP = dataflow(hpx::launch::async, OpTranspose, future_gramRAP, gramRAP, transGramRAP, currentBlockSize, currentBlockSize);
                
                future_transGramXBP = dataflow(hpx::launch::async, OpTranspose, future_gramXBP, gramXBP, transGramXBP, blocksize, currentBlockSize);
                future_transGramRBP = dataflow(hpx::launch::async, OpTranspose, future_gramRBP, gramRBP, transGramRBP, currentBlockSize, currentBlockSize);

                future_gramA[0] = dataflow(hpx::launch::async, OpCopyBlock, future_lambda, lambda, gramA, blocksize, blocksize, gramSize, 0, 0);
                future_gramA[1] = dataflow(hpx::launch::async, OpCopyBlock, future_gramXAR, gramXAR, gramA, blocksize, currentBlockSize, gramSize, 0, blocksize);
                future_gramA[2] = dataflow(hpx::launch::async, OpCopyBlock, future_gramXAP, gramXAP, gramA, blocksize, currentBlockSize, gramSize, 0, blocksize+currentBlockSize);
                future_gramA[3] = dataflow(hpx::launch::async, OpCopyBlock, future_transGramXAR, transGramXAR, gramA, currentBlockSize, blocksize, gramSize, blocksize, 0);
                future_gramA[4] = dataflow(hpx::launch::async, OpCopyBlock, future_gramRAR, gramRAR, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize);
                future_gramA[5] = dataflow(hpx::launch::async, OpCopyBlock, future_gramRAP, gramRAP, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize+currentBlockSize);
                future_gramA[6] = dataflow(hpx::launch::async, OpCopyBlock, future_transGramXAP, transGramXAP, gramA, currentBlockSize, blocksize, gramSize, blocksize+currentBlockSize, 0);
                future_gramA[7] = dataflow(hpx::launch::async, OpCopyBlock, future_transGramRAP, transGramRAP, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize);
                future_gramA[8] = dataflow(hpx::launch::async, OpCopyBlock, future_gramPAP, gramPAP, gramA, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize+currentBlockSize);
                
                future_gramB[0] = dataflow(hpx::launch::async, OpCopyBlock, future_identity_BB, identity_BB, gramB, blocksize, blocksize, gramSize, 0, 0);
                future_gramB[1] = dataflow(hpx::launch::async, OpCopyBlock, future_zeros_BC, zeros_BC, gramB, blocksize, currentBlockSize, gramSize, 0, blocksize);
                future_gramB[2] = dataflow(hpx::launch::async, OpCopyBlock, future_gramXBP, gramXBP, gramB, blocksize, currentBlockSize, gramSize, 0, blocksize+currentBlockSize);
                future_gramB[3] = dataflow(hpx::launch::async, OpCopyBlock, future_zeros_CB, zeros_CB, gramB, currentBlockSize, blocksize, gramSize, blocksize, 0);
                future_gramB[4] = dataflow(hpx::launch::async, OpCopyBlock, future_identity_PAP, identity_PAP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize);
                future_gramB[5] = dataflow(hpx::launch::async, OpCopyBlock, future_gramRBP, gramRBP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize, blocksize+currentBlockSize);
                future_gramB[6] = dataflow(hpx::launch::async, OpCopyBlock, future_transGramXBP, transGramXBP, gramB, currentBlockSize, blocksize, gramSize, blocksize+currentBlockSize, 0);
                future_gramB[7] = dataflow(hpx::launch::async, OpCopyBlock, future_transGramRBP, transGramRBP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize);
                future_gramB[8] = dataflow(hpx::launch::async, OpCopyBlock, future_identity_PAP, identity_PAP, gramB, currentBlockSize, currentBlockSize, gramSize, blocksize+currentBlockSize, blocksize+currentBlockSize);
            }
        }
        future_eigenvalue = dataflow(hpx::launch::async, OpReset, eigenvalue, gramSize, 1);
        future_transGramA = dataflow(hpx::launch::async, OpTranspose, future_gramA, gramA, transGramA, gramSize, gramSize);
        future_transGramB = dataflow(hpx::launch::async, OpTranspose, future_gramB, gramB, transGramB, gramSize, gramSize);
        future_eigenvalue = dataflow(hpx::launch::async, OpEigenComp, future_transGramA, future_transGramB, future_eigenvalue, transGramA, transGramB, eigenvalue, gramSize);
        future_gramAF = dataflow(hpx::launch::async, OpTranspose, future_eigenvalue, transGramA, gramA, gramSize, gramSize);
        future_gramBF = dataflow(hpx::launch::async, OpTranspose, future_gramBF, future_eigenvalue, transGramB, gramB, gramSize, gramSize);

        future_lambda = dataflow(hpx::launch::async, OpDiag, future_eigenvalue, eigenvalue, lambda, blocksize);

        future_coordX = dataflow(hpx::launch::async, OpCopyCoordX, future_gramAF, gramA, coordX, gramSize, blocksize);

        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorP[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorR[i], future_coordX, activeBlockVectorR, coordX + blocksize*blocksize, blockVectorP, M, blocksize, currentBlockSize, 1.0, 0.0, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorAP[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorAR[i], future_coordX, activeBlockVectorAR, coordX + blocksize*blocksize, blockVectorAP, M, blocksize, currentBlockSize, 1.0, 0.0, block_width, i);
        }

        if(iter != 0)
        {
            for(i = 0; i != nrowblks; ++i)
            {
                future_blockVectorP[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorP[i], future_coordX, future_blockVectorP[i], activeBlockVectorP, coordX + (blocksize+currentBlockSize) * blocksize, blockVectorP, M, blocksize, currentBlockSize, 1.0, 1.0, block_width, i);
            }
            for(i = 0; i != nrowblks; ++i)
            {
                future_blockVectorAP[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_activeBlockVectorAP[i], future_coordX, future_blockVectorAP[i], activeBlockVectorAP, coordX + (blocksize+currentBlockSize) * blocksize, blockVectorAP, M, blocksize, currentBlockSize, 1.0, 1.0, block_width, i);
            }
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_newX[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorX[i], future_coordX, blockVectorX, coordX, newX, M, blocksize, blocksize, 1.0, 0.0, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorX[i] = dataflow(execs[threadIndex[i]], OpAdd_part, future_blockVectorP[i], future_newX[i], blockVectorP, newX, blockVectorX, M, blocksize, block_width, i);
        }

        for(i = 0; i != nrowblks; ++i)
        {
            future_newAX[i] = dataflow(execs[threadIndex[i]], OpXY_part, future_blockVectorAX[i], future_coordX, blockVectorAX, coordX, newAX, M, blocksize, blocksize, 1.0, 0.0, block_width, i);
        }
        for(i = 0; i != nrowblks; ++i)
        {
            future_blockVectorAX[i] = dataflow(execs[threadIndex[i]], OpAdd_part, future_blockVectorAP[i], future_newAX[i], blockVectorAP, newAX, blockVectorAX, M, blocksize, block_width, i);
        }

        future_saveLambda = dataflow(hpx::launch::async, OpCopyBlock, future_saveLambda, future_eigenvalue, eigenvalue, saveLambda, blocksize, 1, maxIterations, 0, iter);


        loopTime[iter] = t.elapsed();
    }
    hpx::wait_all(future_blockVectorX);
    hpx::wait_all(future_blockVectorAX);
    hpx::wait_all(future_blockVectorR);
    hpx::wait_all(future_saveLambda);
    hpx::wait_all(future_lambda);
    hpx::wait_all(future_gramBF);

    for(i = 0; i < maxIterations; ++i)
    {
        printf("%.4lf,", loopTime[i]);
        totalSum += loopTime[i];
    }
    printf("%.4lf\n", totalSum/maxIterations);

    for(i = 0; i < blocksize; ++i)
    {
        for(j = 0; j < maxIterations; ++j)
        {
            printf("%.4lf", saveLambda[i*maxIterations + j]);
            if(j != maxIterations - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
    
    /* deallocation */
    for(i = 0; i != nrowblks; ++i)
    {
        for(j = 0; j != ncolblks; ++j)
        {
            if(A[i * ncolblks + j].nnz > 0)
            {
                delete [] A[i * ncolblks + j].rloc;
                delete [] A[i * ncolblks + j].cloc;
                delete [] A[i * ncolblks + j].val;
            }
        }
    }
    delete [] A;
    
    free(gramA);
    free(gramB);
    free(transGramA);
    free(transGramB);
    free(eigenvalue);
    free(coordX);

    free(gramXBX);
    free(transGramXBX);
    free(gramXAX);
    free(transGramXAX);
    free(tempLambda);

    free(blockVectorX);
    free(blockVectorAX);
    free(blockVectorR);
    free(blockVectorP);
    free(blockVectorAP);
    free(newX);
    free(newAX);
    
    free(activeBlockVectorR);
    free(activeBlockVectorAR);
    free(activeBlockVectorP);
    free(temp3);

    free(lambda);
    free(identity_BB);
    
    free(gramXAR);
    free(gramXAP);
    free(gramXBP);
    free(zeros_BC);
    free(temp2);

    free(transGramXAR);
    free(transGramXAP);
    free(transGramXBP);
    free(zeros_CB);

    free(gramRBR);
    free(transGramRBR);
    free(gramRAR);
    free(transGramRAR);
    free(gramPBP);
    free(transGramPBP);
    free(gramRAP);
    free(transGramRAP);
    free(gramRBP);
    free(transGramRBP);
    free(gramPAP);
    free(identity_PAP);

    free(saveLambda);
    
    free(activeMask);
    free(residualNorms);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("block_width", value<int>()->default_value(1024), "CSB block size")
        ("matrix_file", value<std::string>()->default_value("~/DeepSparse/deepsparse/Matrices/NERSC/inline_1.cus"), "Custom matrix file")
        ("blocksize", value<int>()->default_value(10), "RHS")
        ("iteration", value<int>()->default_value(10), "Solver iterations")
        ("nthreads", value<int>()->default_value(14), "# of threads")
        ;

    // Initialize and run HP
    // return hpx::init(desc_commandline, argc, argv);
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    return hpx::init(argc, argv, init_args);
}
