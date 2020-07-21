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

    double norm;
    double *loopTime, *save_norm;
    double tstart, tend;

    filename = argv[1];
    block_width = atoi(argv[2]);
    maxIterations = atoi(argv[3]);

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

    for(iterationNumber = 0; iterationNumber < maxIterations; iterationNumber++)
    {
        tstart = omp_get_wtime();

        /* Y_temp = zeros(numcols) */
        cblas_dscal(numcols, 0.0, Y_temp, 1);

        /* Y_temp = A * Y */
        spmv_blkcoord_loop(numrows, numcols, nthrds, Y, Y_temp, matrixBlock);

        /* Y = Y_temp */
        cblas_dcopy(numcols, Y_temp, 1, Y, 1);

        /* norm = ||Y|| */
        norm = cblas_dnrm2(numcols, Y, 1);

        /* Y /= norm */
        cblas_dscal(numcols, 1.0/norm, Y, 1);

        tend = omp_get_wtime();
        loopTime[iterationNumber] = tend - tstart;
        save_norm[iterationNumber] = norm;
    }

    double totalSum = 0;
    for(i = 0 ; i < maxIterations ; i++)
    {
        totalSum += loopTime[i];
        printf("%.4lf,", loopTime[i]);
    }
    printf("%.4lf\n", totalSum/maxIterations);

    for(i = 0 ; i < maxIterations ; i++)
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
