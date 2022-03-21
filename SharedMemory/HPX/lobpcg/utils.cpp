#include "utils.h"

int numrows, numcols, nnonzero;
int nrowblks, ncolblks;
int nthreads;
int **top;
int *blockIndex, *threadIndex;

void init_mapping()
{
    int i, j;
    int *blockCount;
    int numerator, denominator;
    double threshold;

    blockIndex = new int[nthreads + 1];
    threadIndex = new int[nrowblks];
    blockCount = new int[nthreads];

    for(i = 0; i < nthreads; ++i)
    {
        blockCount[i] = nrowblks / nthreads;
    }

    numerator = nrowblks % nthreads;
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

void init_blocks(int tid, block *matrixBlock, int *colptrs, int *irem, double *xrem, int block_width)
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

void init_entries(int tid, block *matrixBlock, int *colptrs, int *irem, double *xrem, int block_width)
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

            for (k = k1; k != k2 ; k++)
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

void csc2blkcoord(block *&matrixBlock, int *colptrs, int *irem, double *xrem, int block_width)
{
    int i, r, c, k, k1, k2, blkr, blkc;

    nrowblks = (numrows + block_width - 1) / block_width;
    ncolblks = (numcols + block_width - 1) / block_width;
    
    init_mapping();

    std::cout << "nrowblks = " << nrowblks << " numrows = " << numrows << " block_width = " << block_width << std::endl;
    std::cout << "ncolblks = " << ncolblks << std::endl;

    matrixBlock = new block[nrowblks * ncolblks];

    top = new int*[nrowblks];

    for (i = 0; i != nrowblks; i++)
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
    ib_future[i] = dataflow(hpx::launch::async, OpInitBlocks, i, matrixBlock, colptrs, irem, xrem, block_width);
    }

    hpx::wait_all(ib_future);    
    */

    using executor = hpx::execution::experimental::fork_join_executor;
    std::vector<int> v(nthreads);
    std::iota(std::begin(v), std::end(v), 0);
    executor exec;
    hpx::parallel::execution::bulk_sync_execute(exec, &init_blocks, v, matrixBlock, colptrs, irem, xrem, block_width);

    std::cout << "finished inital memory allocation.." << std::endl;


    /* calculatig nnz per block */
    for (c = 0; c != numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1];
        blkc = c / block_width;

        for (k = k1; k != k2 ; k++)
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

    for (blkc = 0; blkc != ncolblks; blkc++)
    {
        for (blkr = 0; blkr != nrowblks; blkr++)
        {
            matrixBlock[blkr * ncolblks + blkc].roffset = blkr * block_width;
            matrixBlock[blkr * ncolblks + blkc].coffset = blkc * block_width;

            if (matrixBlock[blkr * ncolblks + blkc].nnz > 0)
            {
                matrixBlock[blkr * ncolblks + blkc].rloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                // matrixBlock[blkr * ncolblks + blkc].rloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                // matrixBlock[blkr * ncolblks + blkc].cloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].val = new double[matrixBlock[blkr * ncolblks + blkc].nnz];
            }
            else
            {
                matrixBlock[blkr * ncolblks + blkc].rloc = NULL;
                matrixBlock[blkr * ncolblks + blkc].cloc = NULL;
            }
        }
    }

    std::cout << "allocated memory for each block" << std::endl;

    /*
       for(i = 0; i < nthreads; i++)
       {
    //init_entries(i, matrixBlock, colptrs, irem, xrem);
    ie_future[i] = dataflow(hpx::launch::async, OpInitEntries, i, matrixBlock, colptrs, irem, xrem, block_width);
    }

    hpx::wait_all(ie_future);
    */

    hpx::parallel::execution::bulk_sync_execute(exec, &init_entries, v, matrixBlock, colptrs, irem, xrem, block_width);

    printf("conversion completed\n\n");

    for(i = 0 ; i != nrowblks; i++)
    {
        delete [] top[i];
    }
    delete [] top;
}

