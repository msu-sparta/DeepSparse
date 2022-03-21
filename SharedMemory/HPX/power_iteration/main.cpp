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

/* global variables */
int order, np, nx, block_width;
int numrows, numcols, nnonzero;
int nrowblks, ncolblks;
int nthreads;
int **top;
int *blockIndex, *threadIndex;

struct block
{
    int nnz;
    int roffset, coffset;
    unsigned short int *rloc, *cloc;
    //int *rloc, *cloc;
    double *val;
};

void read_custom(char* filename, int *&colptrs, 
        int *&irem, double *&xrem)
{
    FILE *fp;

    fp = fopen(filename, "rb");

    if (fp == NULL)
    {
        std::cout << "invalid matrix file name" << std::endl;
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
    for (int i = 0 ; i < nnonzero ; i++)
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
                matrixBlock[blkr * ncolblks + blkc].rloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                //matrixBlock[blkr * ncolblks + blkc].rloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                //matrixBlock[blkr * ncolblks + blkc].cloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
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

    hpx::parallel::execution::bulk_sync_execute(exec, &init_entries, v, matrixBlock, colptrs, irem, xrem);

    printf("conversion completed\n\n");

    for(i = 0 ; i != nrowblks; i++)
    {
        delete [] top[i];
    }
    delete [] top;
}

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

/* ---------------------- KERNELS BEGIN ----------------------- */
void reset_vector(double *V, int part)
{
    int start = part*nx;
    int end = std::min(start + nx, order);

    for (int i = start; i != end; ++i)
        V[i] = 0.0;
}

void reset(int tid, double *V, int size)
{
    int start = (size/nthreads) * tid;
    int end = (tid == nthreads - 1) ? size : ( start + (size/nthreads));
    
    for(int i = start; i < end; ++i)
        V[i] = 0.0;
}

void matvec(block *A, double *X, double *Y,
        int r, int c)
{
    block &blk = A[r * np + c];

    for (int i = 0; i != blk.nnz; i++)
    {
        Y[ blk.rloc[i] + blk.roffset ] += blk.val[i] * X[ blk.cloc[i] + blk.coffset ];
    }

}

void find_norm(double *V, double *buf, int part)
{
    double rv = 0.0;
    int start = part*nx;
    int end = std::min(start + nx, order);
    for (int i = start; i != end; ++i)
    {
        rv += V[i]*V[i];
    }

    buf[part] = rv;
}

double reduce(double *buf, double *norms, int iter)
{
    double rv = 0;

    for (int i = 0; i != np; ++i)
        rv += buf[i];

    return norms[iter] = sqrt(rv);
}

void normalize(double *V, int part, double norm)
{
    int start = part*nx;
    int end = std::min(start + nx, order);
    
    for (int i = start; i != end; ++i)
    {
        V[i] /= norm;
    }
}

void copy_vector(double *src, double *dest, int part)
{
    int start = part*nx;
    int end = std::min(start + nx, order);
   
    for (int i = start; i != end; ++i)
    {
        dest[i] = src[i];
    }
}

/* ---------------------- KERNELS END ------------------------- */

int hpx_main(hpx::program_options::variables_map& vm)
{
    /* parse commend line arguments */
    block_width = vm["block_width"].as<int>();
    std::string str_file = vm["matrix_file"].as<std::string>();
    int iterations = vm["iterations"].as<int>();
    nthreads = vm["nthreads"].as<int>();

    int len = str_file.length();
    char *file_name = new char[len + 1];
    strcpy(file_name, str_file.c_str());

    /* read the matrix in csc format */
    int *colptrs, *irem;
    double *xrem;
    read_custom(file_name, colptrs, irem, xrem);
    order = numrows;

    np = (numrows + block_width - 1) / block_width;
    nx = block_width;

    init_mapping();

    /* convert it to csb */
    block *A;
    csc2blkcoord(A, colptrs, irem, xrem);

    /* deleting CSC storage memory */
    delete [] file_name;
    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    /* namespaces to be used*/
    using hpx::parallel::for_each;
    using hpx::parallel::execution::par;
    using hpx::dataflow;
    using hpx::util::unwrapping;

    /* initialize vectors and matrix */
    double *X, *Y, *redBuf;
    X = new double[order];
    Y = new double[order];
    redBuf = new double[order];

    using executor = hpx::execution::experimental::fork_join_executor;
    std::vector<int> v(nthreads);
    std::iota(std::begin(v), std::end(v), 0);
    executor exec;

    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, X, order);
    hpx::parallel::execution::bulk_sync_execute(exec, &reset, v, Y, order);

    for (int i = 0; i != order; ++i)
    {
        X[i] = 0.5;
    }

    /* define norm and timing lists to be printed out */
    double *norms = new double[iterations]; 
    std::vector<double> timings(iterations);
   
    /* define and setup futures that are required
     * for the dataflow */
    std::vector<hpx::shared_future<void>> norm_squared(np);
    hpx::shared_future<double> norm_red;
    std::vector<hpx::shared_future<void>> X_futures(np);
    std::vector<hpx::shared_future<void>> Y_futures(np);

    for (int i = 0; i != np; ++i)
    {
        X_futures[i] = hpx::make_ready_future();
        Y_futures[i] = hpx::make_ready_future();
    }

    /* to unwrap futures passed to functions */
    auto OpResVec = unwrapping(&reset_vector);
    auto OpMatVec = unwrapping(&matvec);
    auto OpCopyVec = unwrapping(&copy_vector);
    auto OpFindNorm = unwrapping(&find_norm);
    auto OpNormalize = unwrapping(&normalize);
    auto OpReduce = unwrapping(&reduce);

    printf("nthreads = %d\n", nthreads);

    std::vector<hpx::execution::parallel_executor> execs;
    execs.reserve(nthreads);

    for(int i = 0; i != nthreads; ++i)
    {
        hpx::execution::parallel_executor exec(
             hpx::threads::thread_schedule_hint(
                 hpx::threads::thread_schedule_hint_mode::thread, i));
        execs.push_back(exec);
    }

    hpx::chrono::high_resolution_timer t;
    for (int iter = 0; iter != iterations; ++iter)
    {

        // Y = 0
        for (int i = 0; i != np; ++i)
        {
            //Y_futures[i] = dataflow(hpx::launch::async, OpResVec, X_futures[i], Y, i);
            Y_futures[i] = dataflow(execs[threadIndex[i]], OpResVec, X_futures[i], Y, i);
        }
        
        // Y = A*X
        for (int i = 0; i != np; ++i)
        {
            for (int j = 0; j != np; ++j)
            {
                if(A[i*np+j].nnz > 0)
                {
                    //Y_futures[i] = dataflow(hpx::launch::async, OpMatVec, X_futures[j], Y_futures[i], A, X, Y, i, j);
                    Y_futures[i] = dataflow(execs[threadIndex[i]], OpMatVec, X_futures[j], Y_futures[i], A, X, Y, i, j);
                }
            }
        }

        // norm = ||Y||
        for (int i = 0; i != np; ++i)
        {
            //norm_squared[i] = dataflow(hpx::launch::async, OpFindNorm, Y_futures[i], Y, i);
            norm_squared[i] = dataflow(execs[threadIndex[i]], OpFindNorm, Y_futures[i], Y, redBuf, i);
        }
        norm_red = dataflow(hpx::launch::async, OpReduce, norm_squared, redBuf, norms, iter);

        // Y = Y/norm
        for (int i = 0; i != np; ++i)
        {
            //Y_futures[i] = dataflow(hpx::launch::async, OpNormalize, Y, i, norms[iter]);
            Y_futures[i] = dataflow(execs[threadIndex[i]], OpNormalize, Y, i, norm_red);
        }
        
        // X = Y
        for (int i = 0; i != np; ++i)
        {
            //X_futures[i] = dataflow(hpx::launch::async, OpCopyVec, Y_futures[i], Y, X, i);
            X_futures[i] = dataflow(execs[threadIndex[i]], OpCopyVec, Y_futures[i], Y, X, i);
        }
    }
    
    hpx::wait_all(X_futures);
    double totalTime = t.elapsed();
        
    double totalSum = 0;
    for (int iter = 0 ; iter != iterations ; ++iter)
    {
        printf("%.4lf,", totalTime/iterations);
    }
    printf("%.4lf\n", totalTime/iterations);

    for (int iter = 0 ; iter != iterations ; ++iter)
    {
        printf("%.4lf", norms[iter]);
        if(iter != iterations - 1)
            printf(",");
    }
    printf("\n");

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("block_width", value<int>()->default_value(1024), "CSB block size")
        ("matrix_file", value<std::string>()->default_value(""), "Custom matrix file")
        ("iterations", value<int>()->default_value(10), "# iterations")
        ("nthreads", value<int>()->default_value(14), "# of threads")
    ;
    
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    return hpx::init(argc, argv, init_args);
}
