#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    int i, block_width;
    int iterationNumber, eig_wanted;
    char *filename;
    
    double *xrem;
    block *matrixBlock;

    double *q, *qq;
    double norm_q;
    double *Q;
    double *z;
    double *alpha, *beta, *QpZ, *QQpZ;

    double tstart, tend;
    double total_time;

    filename = argv[1];
    block_width = atoi(argv[2]);
    eig_wanted = atoi(argv[3]);

    wblk = block_width;

    read_custom(filename, xrem);
    csc2blkcoord(matrixBlock,xrem);

    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }
    
    /* deleting csc storage memory*/
    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    q = (double *) malloc(numcols * sizeof(double));
    qq = (double *) malloc(numcols * sizeof(double));
    alpha = (double *) malloc(eig_wanted * sizeof(double));
    beta = (double *) malloc(eig_wanted * sizeof(double));
    QpZ = (double *) malloc(eig_wanted * sizeof(double));
    QQpZ = (double *) malloc(numcols * sizeof(double));
    z = (double *) malloc(numcols * sizeof(double));
    Q = (double*) malloc(numcols * (eig_wanted+1) * sizeof(double));

    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        q[i] = 1.0;
    }

    norm_q = sqrt(numcols);
    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        qq[i] = q[i]/norm_q;
    }

    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        Q[i*(eig_wanted)] = qq[i];
    }

    total_time = 0.0;

    for(iterationNumber = 0; iterationNumber < eig_wanted; ++iterationNumber)
    {
        // TODO: turn it to an mkl call
        // z = zeros(numcols)
        tstart = omp_get_wtime();
        #pragma omp parallel for default(shared)
        for(i = 0; i < numcols; ++i)
        {
            z[i] = 0.0;
        }

        // z = A * qq
        spmv_blkcoord_loop(nrows, ncols, nthrds, qq, z, matrixBlock);

        //alpha[iterationNumber] = qq' * z
        cblas_dgemv(CblasRowMajor, CblasTrans, numcols, 1, 1.0, qq,
                1, z, 1, 0, alpha+iterationNumber, 1);

        // QpZ = Q' * z
        cblas_dgemv(CblasRowMajor, CblasTrans, numcols, iterationNumber+1,
                1.0, Q, eig_wanted, z, 1, 0, QpZ, 1);

        // QQpZ = Q * QpZ
        cblas_dgemv(CblasRowMajor, CblasNoTrans, numcols, iterationNumber+1,
                1.0, Q, eig_wanted, QpZ, 1, 0, QQpZ, 1);

        //TODO: turn it to an mkl call
        // z = z - QQpZ
        #pragma omp parallel for default(shared)
        for(i = 0; i < numcols; ++i)
        {
            z[i] = z[i] - QQpZ[i];
        }

        // beta[iterationNumber] = ||z||
        cblas_dgemv(CblasRowMajor, CblasTrans, numcols, 1, 1.0, z, 1, z,
                1, 0, beta+iterationNumber, 1);
        beta[iterationNumber] = sqrt(beta[iterationNumber]);

        // qq = z / beta[iterationNumber]
        #pragma omp parallel for default(shared)
        for(i = 0; i < numcols; ++i)
        {
            qq[i] = z[i]/beta[iterationNumber];
        }

        // Q[column iterationNumber + 1] = qq 
        #pragma omp parallel for default(shared)
        for(i = 0; i < numcols; ++i)
        {
            Q[(i*eig_wanted)+iterationNumber+1] = qq[i];
        }

        tend = omp_get_wtime();
        printf("%.4lf,",tend-tstart);
        total_time += (tend - tstart);
    }

    printf("%.4lf\n",total_time/eig_wanted);

    for(i = 0 ; i < eig_wanted; ++i)
    {
        printf("%.4lf",alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");
    for(i = 0 ; i < eig_wanted; ++i)
    {
        printf("%.4lf",beta[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");
    LAPACKE_dsterf(eig_wanted,alpha,beta);

    for(i = 0 ; i < eig_wanted; ++i)
    {
        printf("%.4lf",alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");

    // deallocation
    for(i = 0; i < nrowblks; i++)
    {
        for(int j = 0; j < ncolblks; j++)
        {
            if(matrixBlock[i * ncolblks + j].nnz > 0)
            {
                delete [] matrixBlock[i * ncolblks + j].rloc;
                delete [] matrixBlock[i * ncolblks + j].cloc;
                delete [] matrixBlock[i * ncolblks + j].val;
            }
        }
    }
    delete [] matrixBlock;
    free(q);
    free(qq);
    free(alpha);
    free(beta);
    free(z);
    free(Q);
    free(QpZ);
    free(QQpZ);

    return 0;
}
