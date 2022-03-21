#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <cmath>
#include "mpi.h"
#include <mkl.h>
#include <omp.h>
#define MASTER_THREAD 0
using namespace std;

// Command-line variables
int order = 1024;

void read_custom(char* filename, int *&colptrs, 
        int *&irem, double *&xrem)
{
    FILE *fp;
    int numrows, numcols, nnonzero;

    fp = fopen(filename, "rb");

    if (fp == NULL)
    {
        cout << "invalid matrix file name" << endl;
        return;
    }

    fread(&numrows, sizeof(int), 1, fp);
    cout << "nrows: " << numrows << endl;

    fread(&numcols, sizeof(int), 1, fp);
    cout << "ncols: "<< numcols << endl;

    if (numrows != numcols)
    {
        cout << "matrix should be square" << endl;
        return;
    }

    fread(&nnonzero, sizeof(float), 1, fp);
    cout << "nnz: " << nnonzero << endl;

    colptrs = new int[numcols + 1];
    irem = new int[nnonzero];
    xrem = new double[nnonzero];
    float *txrem = new float[nnonzero];
    cout << "Memory allocation finished" << endl;

    fread(colptrs, sizeof(int), numcols+1, fp);
    cout << "finished reading colptrs" << endl;

    fread(irem, sizeof(int), nnonzero, fp);
    cout << "finished reading irem" << endl;

    fread(txrem, sizeof(float), nnonzero, fp);
    cout << "finished reading xrem" << endl;

    #pragma omp parallel for default(shared)
    for(int i = 0 ; i < nnonzero; i++)
    {
        xrem[i] = txrem[i];
    }

    #pragma omp parallel for default(shared)
    for(int i = 0 ; i <= numcols; i++)
    {   
        colptrs[i]--;
    }

    #pragma omp parallel for default(shared)
    for(int i = 0 ; i < nnonzero ; i++)
    {
        irem[i]--;
    }
    
    delete [] txrem;

    fclose(fp);
}

void reset_kernel(double *V, int size)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
        V[i] = 0.0;
}

void matvec(int *ia, int *ja, double *acsr, double *X, double *Y, int size)
{
    int i, j;
    #pragma omp parallel for default(shared) private(i, j)
    for(i = 0; i < size; ++i)
    {
        for(j = ia[i]; j != ia[i+1]; ++j)
        {
            Y[i] = Y[i] + acsr[j] * X[ja[j]];
        }
    }
}

double find_norm(double *V, int size)
{
    double rv = 0.0;
    
    #pragma omp parallel for default(shared) reduction(+:rv)
    for (int i = 0; i < size; ++i)
    {
        rv += V[i]*V[i];
    }

    return rv;
}

double find_dotp(double *V1, double *V2, int size)
{
    double rv = 0.0;
    
    #pragma omp parallel for default(shared) reduction(+:rv)
    for (int i = 0; i < size; ++i)
    {
        rv += V1[i]*V2[i];
    }

    return rv;
}

void normalize(double *src, double *dst, int size, double norm)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
    {
        dst[i] = src[i]/norm;
    }
}

void copy_kernel(double *src, double *dst, int size, int width, int offset)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
    {
        dst[i*width + offset] = src[i];
    }
}

void subtract_kernel(double *src, double *dst, int size)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
    {
        dst[i] -= src[i];
    }
}

void xty_kernel(double *x, double *y, double *xty, int size, int width)
{
    int i, j;

    for (i = 0; i < width; ++i)
    {
        xty[i] = 0.0;
    }

    #pragma omp parallel for default(shared) private(i,j) reduction(+:xty[:width])
    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < width; j++)
        {
            xty[j] += x[i*width+j] * y[i];
        }
    }
}

void xy_kernel(double *x, double *y, double *xy, int size, int width)
{
    int i, j;

    #pragma omp parallel for default(shared) private(i,j)
    for (i = 0; i < size; ++i)
    {
        for (j = 0; j < width; j++)
        {
            xy[i] += x[i*width+j] * y[j];
        }
    }
}

void print(double *arr, int n, int m)
{
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
        {
            printf("%.2lf ", arr[i*m + j]);
        }
        printf("\n");
    }
}

void Lanczos(char *filename, int eig_wanted)
{
    int my_rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nrows, ncols, nnz;
    
    FILE *fp;
    fp = fopen(filename, "rb");
    fread(&order, sizeof(int), 1, fp);
    fread(&ncols, sizeof(int), 1, fp);
    fread(&nnz, sizeof(int), 1, fp);
    fclose(fp);

    nrows = (order/nprocs) + (my_rank < order%nprocs);

    double *acsr, *acsr_global;
    int *ia, *ia_global, *ja, *ja_global;
    int *vec_recvcounts, *vec_displs;
    int *ia_sendcounts, *ja_sendcounts, *ja_displs;

    double timer, *timings;
    double *alpha, *beta;
    double alpha_local, beta_local;

    ia = new int[order];
    ia_sendcounts = new int[nprocs];    
    
    vec_recvcounts = new int[nprocs];
    vec_displs = new int[nprocs];

    if(my_rank == MASTER_THREAD)
    {
        cout << "# of processes: " << nprocs << endl;

        ja_sendcounts = new int[nprocs];    
        ja_displs = new int[nprocs];    
    }

    timings = new double[eig_wanted];
    alpha = new double[eig_wanted];
    beta = new double[eig_wanted];


    for(int i = 0; i < nprocs; ++i)
    {
        vec_displs[i] = (i == 0) ? 0 : vec_displs[i-1] + vec_recvcounts[i-1];
        vec_recvcounts[i] = (order/nprocs) + (i < order%nprocs);
        ia_sendcounts[i] = vec_recvcounts[i] + 1;
    }

    if(my_rank == MASTER_THREAD)
    {
        double *xrem;
        int *colptrs, *irem;
        
        read_custom(filename, colptrs, irem, xrem);

        int job_dcsrcsc[] = {1, 0, 0, 0, 0, 1};
        int dcsrcsc_info = -1;

        ia_global = new int[order + 1];
        ja_global = new int[nnz];
        acsr_global = new double[nnz];

        mkl_dcsrcsc(job_dcsrcsc, &order, acsr_global, ja_global, ia_global, xrem, irem, colptrs, &dcsrcsc_info);

        delete [] colptrs;
        delete [] irem;
        delete [] xrem;

        for(int i = 0; i != nprocs; ++i)
        {
            ja_displs[i] = (i == 0) ? 0 : ja_displs[i-1] + ja_sendcounts[i-1];
            ja_sendcounts[i] = (i+1 != nprocs) ? ia_global[vec_displs[i+1]] - ia_global[vec_displs[i]] : ia_global[order] - ia_global[vec_displs[i]];
        }

    }

    MPI_Scatterv(ia_global, ia_sendcounts, vec_displs, MPI_INT, ia, ia_sendcounts[my_rank], MPI_INT, MASTER_THREAD, MPI_COMM_WORLD);
    
    int roffset = ia[0];
    #pragma omp parallel for default(shared)
    for(int i = 0; i <= nrows; ++i)
    {
        ia[i] -= roffset;
    }

    ja = new int[ia[nrows]];
    acsr = new double[ia[nrows]];
    
    MPI_Scatterv(ja_global, ja_sendcounts, ja_displs, MPI_INT, ja, ia[nrows], MPI_INT, MASTER_THREAD, MPI_COMM_WORLD);
    MPI_Scatterv(acsr_global, ja_sendcounts, ja_displs, MPI_DOUBLE, acsr, ia[nrows], MPI_DOUBLE, MASTER_THREAD, MPI_COMM_WORLD);

    double *QpZ, *qq, *z, *QQpZ, *Q;
    double *QpZ_local, *qq_global;

    QpZ = new double[eig_wanted];
    QpZ_local = new double[eig_wanted];
    qq = new double[nrows];
    z = new double[nrows];
    QQpZ = new double[nrows];
    Q = new double[nrows*eig_wanted];
    qq_global = new double[order];

    #pragma omp parallel for default(shared)
    for(int i = 0; i < eig_wanted; ++i)
    {
        QpZ[i] = 0.0;
    }
    #pragma omp parallel for default(shared)
    for(int i = 0; i < eig_wanted; ++i)
    {
        QpZ_local[i] = 0.0;
    }
    #pragma omp parallel for default(shared)
    for(int i = 0; i < nrows; ++i)
    {
        qq[i] = 1.0/sqrt(order);
    }
    #pragma omp parallel for default(shared)
    for(int i = 0; i < nrows; ++i)
    {
        z[i] = 0.0;
    }
    #pragma omp parallel for default(shared)
    for(int i = 0; i < nrows; ++i)
    {
        QQpZ[i] = 0.0;
    }
    #pragma omp parallel for default(shared)
    for(int i = 0; i < nrows*eig_wanted; ++i)
    {
        Q[i] = 0.0;
    }

    // Q[column 0] = qq
    copy_kernel(qq, Q, nrows, eig_wanted, 0);

    for (int iter = 0; iter != eig_wanted; ++iter)
    {
        if (my_rank == MASTER_THREAD)
        {
            timer = MPI_Wtime();
        }

        // z = 0
        reset_kernel(z, nrows);

        // gather qq from other processes for the following 1D-SpMV
        MPI_Allgatherv(qq, nrows, MPI_DOUBLE, qq_global, vec_recvcounts, vec_displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // z = A*qq
        matvec(ia, ja, acsr, qq_global, z, nrows);

        // alpha[iter] = qq'z
        alpha_local = find_dotp(qq, z, nrows);
        MPI_Allreduce(&alpha_local, alpha+iter, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // QpZ = Q'z - TODO: limit it to first iter columns
        xty_kernel(Q, z, QpZ_local, nrows, eig_wanted);
        MPI_Allreduce(QpZ_local, QpZ, eig_wanted, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // QQpZ = 0
        reset_kernel(QQpZ, nrows);

        // QQpZ = Q*QpZ - TODO: limit it to first iter columns
        xy_kernel(Q, QpZ, QQpZ, nrows, eig_wanted);

        // z = z - QQpZ
        subtract_kernel(QQpZ, z, nrows);

        // beta[iter] = ||z||
        beta_local = find_norm(z, nrows);
        MPI_Allreduce(&beta_local, beta+iter, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        beta[iter] = sqrt(beta[iter]);

        // qq = z/beta[iter]
        normalize(z, qq, nrows, beta[iter]);

        // Q[column iter+1] = qq
        if (iter != eig_wanted - 1)
        {
            copy_kernel(qq, Q, nrows, eig_wanted, iter+1);
        }

        if (my_rank == MASTER_THREAD)
        {
            timings[iter] = MPI_Wtime() - timer;
        }
    }
 
    if (my_rank == MASTER_THREAD)
    {
        double totalSum = 0;
        for (int iter = 0 ; iter != eig_wanted; ++iter)
        {
            totalSum += timings[iter];
            printf("%.4lf,", timings[iter]);
        }
        printf("%.4lf\n", totalSum/eig_wanted);

        for (int iter = 0; iter != eig_wanted; ++iter)
        {
            printf("%.4lf", alpha[iter]);
            if (iter != eig_wanted - 1)
                printf(",");
        }           
        printf("\n");
                    
        for (int iter = 0; iter != eig_wanted; ++iter)
        {               
            printf("%.4lf", beta[iter]);
            if (iter != eig_wanted - 1)
                printf(",");
        }       
        printf("\n");

        LAPACKE_dsterf(eig_wanted, alpha, beta);
        
        for (int iter = 0; iter != eig_wanted; ++iter)
        {
            printf("%.4lf", alpha[iter]);
            if (iter != eig_wanted - 1)
                printf(",");
        }
        printf("\n");
    }

    delete [] ia;
    delete [] ia_sendcounts;
    delete [] vec_recvcounts;
    delete [] vec_displs;
    delete [] ja;
    delete [] acsr;

    delete [] timings;
    delete [] alpha;
    delete [] beta;

    delete [] QpZ;
    delete [] qq;
    delete [] z;
    delete [] QQpZ;
    delete [] Q;
    delete [] qq_global;

    if (my_rank == MASTER_THREAD)
    {
        delete [] ja_sendcounts;
        delete [] ja_displs;
        delete [] ia_global;
        delete [] ja_global;
        delete [] acsr_global;
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, & argv);

    char *filename = argv[1];
    int eig_wanted = atoi(argv[2]);

    Lanczos(filename, eig_wanted);

    MPI_Finalize();
    
    return 0;
}
