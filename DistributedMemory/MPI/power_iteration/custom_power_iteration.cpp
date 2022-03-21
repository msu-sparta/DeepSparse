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
    
    delete []txrem;

    fclose(fp);
}

void reset_vector(double *V, int size)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
        V[i] = 0.0;
}

void matvec(int *ia, int *ja, double *acsr, double *X, double *Y,
        int nrows)
{
    int i, j;
    #pragma omp parallel for default(shared) private(i, j)
    for(i = 0; i < nrows; ++i)
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
    for (int i = 0; i <= size; ++i)
    {
        rv += V[i]*V[i];
    }

    return rv;
}

void normalize(double *V, int size, double norm)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
    {
        V[i] /= norm;
    }
}

void copy_vector(double *src, double *dest, int size)
{
    #pragma omp parallel for default(shared)
    for (int i = 0; i < size; ++i)
    {
        dest[i] = src[i];
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

void PowerMethod(char *filename, int iterations)
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

    double *X, *X_global, *Y;
    double *acsr, *acsr_global;
    int *ia, *ia_global, *ja, *ja_global;
    int *vec_recvcounts, *vec_displs;
    int *ia_sendcounts, *ja_sendcounts, *ja_displs;

    double timer, *timings, *norms;
    double norm_squared, global_norm;

    X = new double[nrows];
    X_global = new double[order];
    Y = new double[nrows];
    ia = new int[order];
    ia_sendcounts = new int[nprocs];    
    
    vec_recvcounts = new int[nprocs];
    vec_displs = new int[nprocs];

    if(my_rank == MASTER_THREAD)
    {
        cout << "# of processes: " << nprocs << endl;

        ja_sendcounts = new int[nprocs];    
        ja_displs = new int[nprocs];    
        timings = new double[iterations];
        norms = new double[iterations];
    }

    #pragma omp parallel for default(shared)
    for(int i = 0; i < nrows; ++i)
    {
        X[i] = 1.0;
    }

    // first touch optimizations
    #pragma omp parallel for default(shared)
    for(int i = 0; i < order; ++i)
    {
        X_global[i] = 0.0;
    }

    #pragma omp parallel for default(shared)
    for(int i = 0; i < nrows; ++i)
    {
        Y[i] = 0.0;
    }

    #pragma omp parallel for default(shared)
    for(int i = 0; i <= nrows; ++i)
    {
        ia[i] = 0.0;
    }

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

    ja = new int[ia[nrows]];
    acsr = new double[ia[nrows]];

    #pragma omp parallel for default(shared)
    for(int i = 0; i < ia[nrows]; ++i)
    {
        ja[i] = 0.0;
    }

    #pragma omp parallel for default(shared)
    for(int i = 0; i < ia[nrows]; ++i)
    {
        acsr[i] = 0.0;
    }
    
    MPI_Scatterv(ja_global, ja_sendcounts, ja_displs, MPI_INT, ja, ia[nrows], MPI_INT, MASTER_THREAD, MPI_COMM_WORLD);
    MPI_Scatterv(acsr_global, ja_sendcounts, ja_displs, MPI_DOUBLE, acsr, ia[nrows], MPI_DOUBLE, MASTER_THREAD, MPI_COMM_WORLD);

    for (int iter = 0; iter != iterations; ++iter)
    {
        if(my_rank == MASTER_THREAD)
        {
            timer = MPI_Wtime();
        }

        // Y = 0
        reset_vector(Y, nrows);

        MPI_Allgatherv(X, nrows, MPI_DOUBLE, X_global, vec_recvcounts, vec_displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Y = A*X
        matvec(ia, ja, acsr, X_global, Y, nrows);

        // norm = ||Y||
        norm_squared = find_norm(Y, nrows);
        
        MPI_Allreduce(&norm_squared, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        global_norm = sqrt(global_norm);

        if(my_rank == MASTER_THREAD)
        {
            norms[iter] = global_norm;
        }

        // Y = Y/norm
        normalize(Y, nrows, global_norm);
        
        // X = Y
        copy_vector(Y, X, nrows);

        if(my_rank == MASTER_THREAD)
        {
            timings[iter] = MPI_Wtime() - timer;
        }
    }
 
    if(my_rank == MASTER_THREAD)
    {
        double totalSum = 0;
        for (int iter = 0 ; iter != iterations ; ++iter)
        {
            totalSum += timings[iter];
            printf("%.4lf,", timings[iter]);
        }
        printf("%.4lf\n", totalSum/iterations);
        printf("Total time: %.4lf\n", totalSum);
        
        for (int iter = 0 ; iter != iterations ; ++iter)
        {
            printf("%.4lf", norms[iter]);
            if(iter != iterations - 1)
                printf(",");
        }
        printf("\n");
    }   
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, & argv);

    char *filename = argv[1];
    int iterations = atoi(argv[2]);

    PowerMethod(filename, iterations);

    MPI_Finalize();
    
    return 0;
}
