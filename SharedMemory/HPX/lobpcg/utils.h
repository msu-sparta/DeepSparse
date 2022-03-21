#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

#include <mkl.h>

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <boost/range/irange.hpp>

#include <hpx/modules/executors.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/runtime_local/get_worker_thread_num.hpp>

#include <hpx/execution.hpp>
#include <hpx/property.hpp>
#include <hpx/thread.hpp>

extern int numrows, numcols, nnonzero;
extern int nrowblks, ncolblks;
extern int nthreads;
extern int *threadIndex, *blockIndex;

struct block
{
    int nnz;
    int roffset, coffset;
    unsigned short int *rloc, *cloc;
    // int *rloc, *cloc;
    double *val;
};

void read_custom(char* filename, int *&colptrs, int *&irem, double *&xrem);
void csc2blkcoord(block *&matrixBlock, int *colptrs, int *irem, double *xrem, int block_width);
