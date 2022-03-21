#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_numeric.hpp>

#include <boost/range/irange.hpp>

#include <hpx/runtime_local/get_worker_thread_num.hpp>

#include <hpx/execution.hpp>
#include <hpx/property.hpp>
#include <hpx/thread.hpp>

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>

#include <mkl.h>

#define TID (hpx::get_worker_thread_num())
#define MAXTHREADS 64

/* global variables */
int order, np, nx, block_width;
int numrows, numcols, nnonzero;
int nrowblks, ncolblks;
int eig_wanted, nthreads;
int **top;
int *blockIndex, *threadIndex;

hpx::chrono::high_resolution_timer start_time;

struct block
{
    int nnz;
    int roffset, coffset;
    // unsigned short int *rloc, *cloc;
    int *rloc, *cloc;
    double *val;
};

void read_custom(char* filename, int *&colptrs, 
        int *&irem, double *&xrem)
{
    FILE *fp;

    fp = fopen(filename, "rb");

    if (fp == NULL)
    {
        std::cout << "invalid matrix file name :(" << std::endl;
        return;
    }

    fread(&numrows, sizeof(int), 1, fp);
    std::cout << "row: " << numrows << std::endl;

    fread(&numcols, sizeof(int), 1, fp);
    std::cout << "colum: "<< numcols << std::endl;

    if (numrows != numcols)
    {
        std::cout << "matrix should be square" << std::endl;
        return;
    }

    fread(&nnonzero, sizeof(float), 1, fp);
    std::cout << "non zero: " << nnonzero << std::endl;

    colptrs = new int[numcols + 1];
    irem = new int[nnonzero];
    xrem = new double[nnonzero];
    float *txrem = new float[nnonzero];
    std::cout << "Memory allocation finished" << std::endl;

    fread(colptrs, sizeof(int), numcols+1, fp);
    std::cout << "finished reading colptrs" << std::endl;

    fread(irem, sizeof(int), nnonzero, fp);
    std::cout << "finished reading irem" << std::endl;

    fread(txrem, sizeof(float), nnonzero, fp);
    std::cout << "finished reading xrem" << std::endl;
    for(int i = 0 ; i < nnonzero ; i++)
    {
        xrem[i] = txrem[i];
    }

    for(int i = 0 ; i != numcols+1 ; i++)
    {   
        colptrs[i]--;
    }

    for(int i = 0 ; i != nnonzero ; i++)
    {
        irem[i]--;
    }

    delete []txrem;
}

void init_blocks(int tid, block *matrixBlock, int *colptrs, int *irem, double *xrem)
{
    //int start = (nrowblks/nthreads)*tid;
    //int end = (tid == (nthreads-1)) ? (nrowblks) : (start + (nrowblks/nthreads));
    int blkr, blkc;
    //for(blkr = start; blkr != end; blkr++)
    for(blkr = blockIndex[tid]; blkr != blockIndex[tid+1]; blkr++)
    {
        for(blkc = 0 ; blkc != ncolblks ; blkc++)
        {
            top[blkr][blkc] = 0;
            matrixBlock[blkr * ncolblks + blkc].nnz = 0;
        }
    }
}

void init_entries(int tid, block *matrixBlock, int *colptrs, int *irem, double *xrem)
{
    //int start = (ncolblks/nthreads)*tid;
    //int end = (tid == (nthreads-1)) ? (ncolblks) : (start + (ncolblks/nthreads));
    int i, c, r, k, k1, k2, blkr, blkc;

    //for(i = start; i < end; i++)
    for(i = blockIndex[tid]; i != blockIndex[tid+1]; i++)
    {
        for(c = i * block_width; c < std::min( (i+1)*block_width, numcols) ; c++)
        {
            k1 = colptrs[c];
            k2 = colptrs[c + 1]; 
            blkc = c / block_width;

            for(k = k1; k != k2 ; k++)
            {
                r = irem[k];
                blkr = r / block_width;
                matrixBlock[blkr * ncolblks + blkc].rloc[top[blkr][blkc]] = r - matrixBlock[blkr * ncolblks + blkc].roffset;
                matrixBlock[blkr * ncolblks + blkc].cloc[top[blkr][blkc]] = c - matrixBlock[blkr * ncolblks + blkc].coffset;
                matrixBlock[blkr * ncolblks + blkc].val[top[blkr][blkc]] = xrem[k];

                top[blkr][blkc]++;
            }
        }
    }
}

void csc2blkcoord(block *&matrixBlock, int *colptrs, int *irem, double *xrem)
{
    int i, j, r, c, k, k1, k2, blkr, blkc, tmp;

    nrowblks = (numrows + block_width - 1) / block_width;
    ncolblks = (numcols + block_width - 1) / block_width;
    std::cout << "nrowblks = " << nrowblks << " numrows = " << numrows << " block_width = " << block_width << std::endl;
    std::cout << "ncolblks = " << ncolblks << std::endl;

    matrixBlock = new block[nrowblks * ncolblks];

    top = new int*[nrowblks];

    for(i = 0; i != nrowblks; i++)
    {
        top[i] = new int[ncolblks];
    }

    /*
    using hpx::dataflow;
    using hpx::util::unwrapping;

    auto OpInitBlocks = unwrapping(&init_blocks);
    auto OpInitEntries = unwrapping(&init_entries);

    std::vector<hpx::shared_future<void>> ib_future(nthreads);
    std::vector<hpx::shared_future<void>> ie_future(nthreads);
    

    for(i = 0; i < nthreads; i++)
    {
        //init_blocks(i, matrixBlock, colptrs, irem, xrem);
        ib_future[i] = dataflow(hpx::launch::async, OpInitBlocks, i, matrixBlock, colptrs, irem, xrem);
    }

    hpx::wait_all(ib_future);    
    */

    using executor = hpx::execution::experimental::fork_join_executor;
    std::vector<int> v(nthreads);
    std::iota(std::begin(v), std::end(v), 0);
    executor exec;

    hpx::parallel::execution::bulk_sync_execute(exec, &init_blocks, v, matrixBlock, colptrs, irem, xrem);


    std::cout << "finished inital memory allocation.." << std::endl;


    /* calculatig nnz per block */
    for(c = 0; c != numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1];
        blkc = c / block_width;

        for(k = k1; k != k2 ; k++)
        {
            r = irem[k];
            blkr = r / block_width;
            if (blkr >= nrowblks || blkc >= ncolblks)
            {
                std::cout << "(" << blkr << ", " << blkc << ") doesn't exist" << std::endl;
            }
            else
            {
                matrixBlock[blkr * ncolblks + blkc].nnz++;  
            }    
        }
    }

    std::cout << "finished counting nnz in each block" << std::endl;

    for(blkc = 0; blkc != ncolblks; blkc++)
    {
        for(blkr = 0; blkr != nrowblks; blkr++)
        {
            matrixBlock[blkr * ncolblks + blkc].roffset = blkr * block_width;
            matrixBlock[blkr * ncolblks + blkc].coffset = blkc * block_width;

            if (matrixBlock[blkr * ncolblks + blkc].nnz > 0)
            {
                // matrixBlock[blkr * ncolblks + blkc].rloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                // matrixBlock[blkr * ncolblks + blkc].cloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].rloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].val = new double[matrixBlock[blkr * ncolblks + blkc].nnz];
            }
            else
            {
                matrixBlock[blkr * ncolblks + blkc].rloc = NULL;
                matrixBlock[blkr * ncolblks + blkc].cloc = NULL;
            }
        }
    }

    std::cout << "allocated memory foreach block" << std::endl;

    /*
    for(i = 0; i < nthreads; i++)
    {
        //init_entries(i, matrixBlock, colptrs, irem, xrem);
        ie_future[i] = dataflow(hpx::launch::async, OpInitEntries, i, matrixBlock, colptrs, irem, xrem);
    }

    hpx::wait_all(ie_future);
    */

    hpx::parallel::execution::bulk_sync_execute(exec, &init_entries, v, matrixBlock, colptrs, irem, xrem);

    printf("conversion completed\n\n");

    for(i = 0 ; i != nrowblks; i++)
    {
        delete [] top[i];
    }
    delete [] top;
}

/* ---------------------- KERNELS BEGIN ----------------------- */

/*void reset(double *V, int size, int tid)
{
    int start = (size/nthreads) * tid;
    int end = (tid == nthreads - 1) ? size : ( start + (size/nthreads));

    for(int i = start; i < end; ++i)
        V[i] = 0.0;
}
*/

void reset(int tid, double *V, int size)
{
    int start = (size/nthreads) * tid;
    int end = (tid == nthreads - 1) ? size : ( start + (size/nthreads));

    for(int i = start; i < end; ++i)
        V[i] = 0.0;
}

void reset_buffer(double *V, int size)
{
    for(int i = 0; i != size; ++i)
        V[i] = 0.0;
}

void fill(double *V, int part, double value)
{
    int start = part * nx;
    int end = std::min(start + nx, order);

    for(int i = start; i != end; ++i)
        V[i] = value;
}

void fill_nonloop(double *V, int part, double value)
{
    int start = part * nx;
    int end = std::min(start + nx, order);

    for(int i = start; i != end; ++i)
        V[i] = value;
}

void copy_vector(double *dest, double *src, int part, int offset)
{
    int start = part * nx;
    int end = std::min(start + nx, order);
    for(int i = start; i != end; ++i)
    {
        dest[i * eig_wanted + offset] = src[i];
    }
}

void copy_vector_nonloop(double *dest, double *src, int part, int offset)
{
    int start = part * nx;
    int end = std::min(start + nx, order);
    for(int i = start; i != end; ++i)
    {
        dest[i * eig_wanted + offset] = src[i];
    }
}

void matvec(block *A, double *X, double *Y, int r, int c)
{
    block &blk = A[r * np + c];
    for(int i = 0; i != blk.nnz; ++i)
    {
        Y[ blk.rloc[i] + blk.roffset ] += blk.val[i] * X[ blk.cloc[i] + blk.coffset ];
    }
}

void dot_product(double *X, double *Y, double *buf, int part)
{
    int start = part * nx;
    int end = std::min(start + nx, order);
    int buf_id = part;

    buf[buf_id] = cblas_ddot(end - start, X + start, 1, Y + start, 1);
}

void custom_dgemv_transpose(double *M, double *X, double *Y, int part, int avail_eigs)
{
    int start = part * nx;
    int end = std::min(start + nx, order);
    int buf_id = part;

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, avail_eigs, 1, end - start,
            1.0, M + (start * eig_wanted), eig_wanted, X + (start * 1), 1, 0.0, Y + (buf_id), np);
}

void custom_dgemv(double *M, double *X, double *Y, int part, int avail_eigs)
{
    int start = part * nx;
    int end = std::min(start + nx, order);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, end - start, 1, avail_eigs,
            1.0, M + (start * eig_wanted), eig_wanted, X, 1, 0.0, Y + (start * 1), 1);
}

void subtract_vector(double *X, double *Y, int part)
{
    int start = part * nx;
    int end = std::min(start + nx, order);

    for(int i = start; i != end; ++i)
    {
        X[i] -= Y[i];
    }
}

void find_norm(double *V, double *buf, int part)
{
    int start = part * nx;
    int end = std::min(start + nx, order);
    int buf_id = part;

    buf[buf_id] = cblas_ddot(end - start, V + start, 1, V + start, 1);
}

void normalize(double *X, double *Y, int part, double *beta, int itr)
{
    double norm = beta[itr];
    int start = part * nx;
    int end = std::min(start + nx, order);

    for(int i = start; i != end; ++i)
    {
        X[i] = Y[i] / norm;
    }
}

void single_reduce(double *V, double *buff, int itr, bool is_sqrt)
{
    V[itr] = 0.0;

    for(int i = 0; i != np; ++i)
    {
        V[itr] += buff[i];
    }

    if (is_sqrt)
        V[itr] = sqrt(V[itr]);
}

void vector_reduce(double *V, double *buff)
{
    int i, j;

    for(i = 0; i != eig_wanted; ++i)
    {
        V[i] = 0.0;

        for(j = 0; j != np; ++j)
        {
            V[i] += buff[i*np + j];
        }
    }
}


/* ---------------------- KERNELS END ------------------------- */

void init_mapping()
{
    int i, j;
    int *blockCount;
    int numerator, denominator;
    double threshold;

    blockIndex = new int[nthreads + 1];
    threadIndex = new int[np];
    blockCount = new int[nthreads];

    for(i = 0; i < nthreads; ++i)
    {
        blockCount[i] = np / nthreads;
    }

    numerator = np % nthreads;
    denominator = nthreads;
    threshold = ((double)numerator)/denominator;

    for(i = 0; i < nthreads; ++i)
    {
        if(((double)numerator)/denominator >= threshold && numerator)
        {
            ++blockCount[i];
            --numerator;
            --denominator;
        }
        else
        {
            --denominator;
        }
    }

    blockIndex[0] = 0;
    for(i = 1; i <= nthreads; ++i)
    {
        blockIndex[i] = blockIndex[i-1] + blockCount[i-1];
    }

    for(i = 0; i < nthreads; ++i)
    {
        for(j = blockIndex[i]; j < blockIndex[i+1]; ++j)
        {
            threadIndex[j] = i;
        }
    }

    delete [] blockCount;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    /* parse commend line arguments */
    block_width = vm["block_width"].as<int>();
    std::string str_file = vm["matrix_file"].as<std::string>();
    eig_wanted = vm["eig_wanted"].as<int>();
    nthreads = vm["nthreads"].as<int>();

    int len = str_file.length();
    char *file_name = new char[len + 1];
    strcpy(file_name, str_file.c_str());

    /* define lanczos variables */
    double *q, *qq, *z;
    double *Q, *QpZ, *QQpZ;
    double *alpha, *beta;

    double *alphaBUFF, *normBUFF, *QpZBUFF;

    int *colptrs, *irem;
    double *xrem;

    block *A;

    int i, j, iter;
    double total_time;

    /* read the matrix in csc format */
    read_custom(file_name, colptrs, irem, xrem);

    order = numrows;

    np = (numrows + block_width - 1) / block_width;
    nx = block_width;

    init_mapping();
    
    /* convert it to csb */
    csc2blkcoord(A, colptrs, irem, xrem);

    /* deleting CSC storage memory */
    delete [] file_name;

    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    /* allocating memory forlanczos variables */
    q = new double[numcols];
    qq = new double[numcols];
    z = new double[numcols];
    Q = new double[numcols * eig_wanted];
    QpZ = new double[eig_wanted];
    QQpZ = new double[numcols];
    alpha = new double[eig_wanted];
    beta = new double[eig_wanted];

    alphaBUFF = new double[np];
    normBUFF = new double[np];
    QpZBUFF = new double[eig_wanted * np];

    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;
    using hpx::dataflow;
    using hpx::util::unwrapping;

    /* initialize vectors in parallel for first touch policy optimization */
    /*
    auto OpReset = unwrapping(&reset);
    
    std::vector<hpx::shared_future<void>> reset_future(nthreads);
    
    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpReset, q, numcols, i);
    }
    hpx::wait_all(reset_future);
        
    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpReset, qq, numcols, i);
    }
    hpx::wait_all(reset_future);
        
    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpReset, z, numcols, i);
    }
    hpx::wait_all(reset_future);
        
    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpReset, Q, numcols * eig_wanted, i);
    }
    hpx::wait_all(reset_future);
        
    for(i = 0; i < nthreads; ++i)
    {
        reset_future[i] = dataflow(hpx::launch::async, OpReset, QQpZ, numcols, i);
    }
    hpx::wait_all(reset_future);
    */
    
    using executor = hpx::execution::experimental::fork_join_executor;
    std::vector<int> v(nthreads);
    std::iota(std::begin(v), std::end(v), 0);
    executor exec;

    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, q, numcols);
    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, qq, numcols);
    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, z, numcols);
    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, Q, numcols * eig_wanted);
    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, QQpZ, numcols);

        
    /* to unwrap futures passed to functions */
    auto OpFill = unwrapping(&fill);
    auto OpFillNonLoop = unwrapping(&fill_nonloop);
    auto OpCopyVec = unwrapping(&copy_vector);
    auto OpCopyVecNonLoop = unwrapping(&copy_vector_nonloop);
    auto OpMatVec = unwrapping(&matvec);
    auto OpDotV = unwrapping(&dot_product);
    auto OpDGEMV_T = unwrapping(&custom_dgemv_transpose);
    auto OpDGEMV = unwrapping(&custom_dgemv);
    auto OpSubV = unwrapping(&subtract_vector);
    auto OpFindNorm = unwrapping(&find_norm);
    auto OpNormalize = unwrapping(&normalize);
    auto OpSingleRed = unwrapping(&single_reduce);
    auto OpVecRed = unwrapping(&vector_reduce);

    std::vector<hpx::shared_future<void>> qq_future(np);
    std::vector<hpx::shared_future<void>> z_future(np);
    std::vector<hpx::shared_future<void>> Q_future(np);
    std::vector<hpx::shared_future<void>> QQpZ_future(np);
    std::vector<hpx::shared_future<void>> alpha_future(np);
    std::vector<hpx::shared_future<void>> norm_future(np);
    std::vector<hpx::shared_future<void>> QpZ_future(np);
    hpx::shared_future<void> alphaRed_future;
    hpx::shared_future<void> normRed_future;
    hpx::shared_future<void> QpZRed_future;

    printf("nthreads = %d\n", nthreads);
    
    //std::vector<hpx::parallel::execution::default_executor > execs;
    std::vector<hpx::execution::parallel_executor> execs;
    execs.reserve(nthreads);
    //hpx::execution::parallel_executor exec_prefer;
    for(i = 0; i != nthreads; ++i)
    {
        //hpx::parallel::execution::default_executor exec(
        hpx::execution::parallel_executor exec(
            hpx::threads::thread_schedule_hint(
                  hpx::threads::thread_schedule_hint_mode::thread, i));
        //auto exec_prefer_instance = hpx::experimental::prefer(hpx::execution::experimental::make_with_hint, exec_prefer, i);
        //execs.push_back(exec_prefer_instance);
        execs.push_back(exec);
    }

    // q = ones(numcols)
    for(i = 0; i != np; ++i)
    {
        fill_nonloop(q, i, 1.0);
    }

    // qq = q / ||q||
    for(i = 0; i != np; ++i)
    {
        //qq_future[i] = dataflow(hpx::launch::async, OpFillNonLoop, qq, i, 1.0/sqrt(numcols));
        qq_future[i] = dataflow(execs[threadIndex[i]], OpFillNonLoop, qq, i, 1.0/sqrt(numcols));
    }

    // Q[column 0] = qq
    for(i = 0; i != np; ++i)
    {
        //Q_future[i] = dataflow(hpx::launch::async, OpCopyVecNonLoop, qq_future[i], Q, qq, i, 0);
        Q_future[i] = dataflow(execs[threadIndex[i]], OpCopyVecNonLoop, qq_future[i], Q, qq, i, 0);
    }

    start_time.restart();

    for(iter = 0; iter != eig_wanted; ++iter)
    // for(iter = 0; iter != 0; ++iter)
    {
        // z = zeros(numcols)
        for(i = 0; i != np; ++i)
        {
            //z_future[i] = dataflow(hpx::launch::async, OpFill, qq_future[i], z, i, 0.0);
            z_future[i] = dataflow(execs[threadIndex[i]], OpFill, qq_future[i], z, i, 0.0);
        }

        // z = A * qq
        for(i = 0; i != np; ++i)
        {
            for(int j = 0; j != np; ++j)
            {
                if(A[i * np + j].nnz > 0)
                {
                    //z_future[i] = dataflow(hpx::launch::async, OpMatVec, z_future[i], qq_future[j], A, qq, z, i, j);
                    
                    // locality-aware scheduling with a specified hint

                    // we use fork-join executor (similar omp parallel for)
                    // to initialize data for first touch optimization
                    // 1D partitioned initialization of vectors/vector blocks 
                    // and the matrix (1D row part)

                    // np = number of partitions (nrowblks), assuming np >> nthreads
                    // Y[i] += A[i,j]*X[j], let alpha = np/nthreads
                    // thread #(i/alpha) is close to Y[i] & A[i,j] 
                    // whereas thread #(j/alpha) is close to X[j]

                    // hint for the thread id would be #(i/alpha)
                    // greedy apprach in general - the thread that owns 
                    // most of the data for the corresponding task
                    // gets assigned to that task

                    z_future[i] = dataflow(execs[threadIndex[i]], OpMatVec, z_future[i], qq_future[j], A, qq, z, i, j);

                }
            }
        }

        // alpha[iter] = qq' * z
        for(i = 0; i != np; ++i)
        {
            //alpha_future[i] = dataflow(hpx::launch::async, OpDotV, qq_future[i], z_future[i], qq, z, alphaBUFF, i);
            alpha_future[i] = dataflow(execs[threadIndex[i]], OpDotV, qq_future[i], z_future[i], qq, z, alphaBUFF, i);
        }
        alphaRed_future = dataflow(hpx::launch::async, OpSingleRed, alpha_future, alpha, alphaBUFF, iter, 0);

        // QpZ = Q' * z
        for(i = 0; i != np; ++i)
        {
            //QpZ_future[i] = dataflow(hpx::launch::async, OpDGEMV_T, Q_future[i], z_future[i], Q, z, QpZBUFF, i, iter + 1);
            QpZ_future[i] = dataflow(execs[threadIndex[i]], OpDGEMV_T, Q_future[i], z_future[i], Q, z, QpZBUFF, i, iter + 1);
        }
        QpZRed_future = dataflow(hpx::launch::async, OpVecRed, QpZ_future, QpZ, QpZBUFF);

        // QQpZ = zeros(numcols);
        for(i = 0; i != np; ++i)
        {
            //QQpZ_future[i] = dataflow(hpx::launch::async, OpFill, z_future[i], QQpZ, i, 0.0);
            QQpZ_future[i] = dataflow(execs[threadIndex[i]], OpFill, z_future[i], QQpZ, i, 0.0);
        }

        // QQpZ = Q * QpZ
        for(i = 0; i != np; ++i)
        {
            //QQpZ_future[i] = dataflow(hpx::launch::async, OpDGEMV, Q_future[i], QpZRed_future, QQpZ_future[i], Q, QpZ, QQpZ, i, iter + 1);
            QQpZ_future[i] = dataflow(execs[threadIndex[i]], OpDGEMV, Q_future[i], QpZRed_future, QQpZ_future[i], Q, QpZ, QQpZ, i, iter + 1);
        }

        // z = z - QQpZ
        for(i = 0; i != np; ++i)
        {
            //z_future[i] = dataflow(hpx::launch::async, OpSubV, alphaRed_future, z_future[i], QQpZ_future[i], z, QQpZ, i);
            z_future[i] = dataflow(execs[threadIndex[i]], OpSubV, alphaRed_future, z_future[i], QQpZ_future[i], z, QQpZ, i);
        }

        // beta[iter] = ||z||
        for(i = 0; i != np; ++i)
        {
            //norm_future[i] = dataflow(hpx::launch::async, OpFindNorm, z_future[i], z, normBUFF, i);
            norm_future[i] = dataflow(execs[threadIndex[i]], OpFindNorm, z_future[i], z, normBUFF, i);
        }
        normRed_future = dataflow(hpx::launch::async, OpSingleRed, norm_future, beta, normBUFF, iter, 1);
        
        // qq = z / beta[iter]
        for(i = 0; i != np; ++i)
        {
            //qq_future[i] = dataflow(hpx::launch::async, OpNormalize, alphaRed_future, z_future[i], normRed_future, qq, z, i, beta, iter);
            qq_future[i] = dataflow(execs[threadIndex[i]], OpNormalize, alphaRed_future, z_future[i], normRed_future, qq, z, i, beta, iter);
        }

        // Q[column iter + 1] = qq
        if(iter != eig_wanted - 1)
        {
            for(i = 0; i != np; ++i)
            {
                //Q_future[i] = dataflow(hpx::launch::async, OpCopyVec, qq_future[i], Q, qq, i, iter + 1);
                Q_future[i] = dataflow(execs[threadIndex[i]], OpCopyVec, qq_future[i], Q, qq, i, iter + 1);
            }
        }

    }

    hpx::wait_all(alpha_future);
    hpx::wait_all(alphaRed_future);
    hpx::wait_all(norm_future);
    hpx::wait_all(normRed_future);
    hpx::wait_all(QpZ_future);
    hpx::wait_all(QpZRed_future);
    hpx::wait_all(z_future);
    hpx::wait_all(qq_future);
    hpx::wait_all(Q_future);
    hpx::wait_all(QQpZ_future);

    total_time = start_time.elapsed();
    for(i = 0; i != eig_wanted; ++i)
        printf("%.4lf,", total_time/eig_wanted);
    printf("%.4lf\n", total_time/eig_wanted);

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", alpha[i]);
        if (i != eig_wanted - 1)
        {
            printf(",");
        }
    }
    printf("\n");

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", beta[i]);
        if (i != eig_wanted - 1)
        {
            printf(",");
        }
    }
    printf("\n");

    LAPACKE_dsterf(eig_wanted,alpha,beta);

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf",alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");

    for(i = 0; i != nrowblks; ++i)
    {
        for(j = 0; j != ncolblks; ++j)
        {
            if (A[i * ncolblks + j].nnz > 0)
            {
                delete [] A[i * ncolblks + j].rloc;
                delete [] A[i * ncolblks + j].cloc;
                delete [] A[i * ncolblks + j].val;
            }
        }
    }
    delete [] A;

    delete [] q;
    delete [] qq;
    delete [] z;
    delete [] Q;
    delete [] QpZ;
    delete [] QQpZ;
    delete [] alpha;
    delete [] beta;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("block_width", value<int>()->default_value(1024), "CSB block size")
        ("matrix_file", value<std::string>()->default_value("~/DeepSparse/deepsparse/Matrices/NERSC/inline_1.cus"), "Custom matrix file")
        ("eig_wanted", value<int>()->default_value(10), "RHS")
        ("nthreads", value<int>()->default_value(14), "# of threads")
        ;

    // Initialize and run HPX
    // return hpx::init(desc_commandline, argc, argv);
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    return hpx::init(argc, argv, init_args);
}
