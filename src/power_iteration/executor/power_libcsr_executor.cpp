#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    (void)argc;

    int i, block_width;
    int iterationNumber, maxIterations;
    char *filename;
    double *xrem;

    int *ia, *ja;
    double *acsr;

    double norm;
    double *loopTime, *save_norm;
    double tstart, tend;

    filename = argv[1];
    block_width = atoi(argv[2]);
    maxIterations = atoi(argv[3]);

    wblk = block_width;

    read_custom(filename, xrem);

#pragma omp parallel
#pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    /* mkl_dcsrcsc (csc to csr) params*/
    int job_dcsrcsc[] = {1, 0, 0, 0, 0, 1};
    int dcsrcsc_info = -1;

    ia = (int *) malloc((numrows + 1) * sizeof(int));
    ja = (int *) malloc(nnonzero * sizeof(int));
    acsr = (double *) malloc(nnonzero * sizeof(double));

    mkl_dcsrcsc(job_dcsrcsc, &numrows, acsr, ja, ia, xrem, irem, colptrs, &dcsrcsc_info);

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
    char transA;
    double alpha, beta;
    char matdescra[6];

    transA = 'N';
    alpha = 1.0;
    beta = 0.0;
    matdescra[0] = 'G';
    matdescra[1] = 'L';
    matdescra[2] = 'U';
    matdescra[3] = 'C';

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

        /* Y_temp = A * Y */
        mkl_dcsrmv(&transA , &numrows , &numcols , &alpha , matdescra , acsr , ja , ia , ia+1 , Y , &beta , Y_temp);

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

    free(ia);
    free(ja);
    free(acsr);

    free(Y);
    free(Y_temp);
    
    free(loopTime);
    free(save_norm);

    return 0;
}
