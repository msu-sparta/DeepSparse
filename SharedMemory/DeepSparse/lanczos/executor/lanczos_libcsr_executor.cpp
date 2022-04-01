#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    (void)argc;

    int i, block_width;
    int iterationNumber, eig_wanted;
    char *filename;
    double *xrem;

    int *ia, *ja;
    double *acsr;

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

    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }

    /* mkl_dcsrcsc (csc to csr) params*/
    int job_dcsrcsc[] = {1, 0, 0, 0, 0, 1};
    int dcsrcsc_info = -1;

    /* mkl_dcsrmv params */
    char transa;
    double alphaS, betaS;
    char matdescra[6];

    ia = (int *) malloc((numrows + 1) * sizeof(int));
    ja = (int *) malloc(nnonzero * sizeof(int));
    acsr = (double *) malloc(nnonzero * sizeof(double));

    mkl_dcsrcsc(job_dcsrcsc, &numrows, acsr, ja, ia, xrem, irem, colptrs, &dcsrcsc_info);

    /* deleting csc storage memory*/
    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    transa = 'n';
    alphaS = 1.0;
    betaS = 0.0;
    matdescra[0] = 'G';
    matdescra[1] = 'L';
    matdescra[2] = 'U';
    matdescra[3] = 'C';

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

    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        QQpZ[i] = 0.0;
    }

    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        z[i] = 0.0;
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
        mkl_dcsrmv(&transa, &numrows, &numcols, &alphaS, matdescra,
                acsr, ja, ia, ia+1, qq, &betaS, z);

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

    free(q);
    free(qq);
    free(alpha);
    free(beta);
    free(z);
    free(Q);
    free(QpZ);
    free(QQpZ);

    free(acsr);
    free(ja);
    free(ia);

    return 0;
}
