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

void PowerMethod(char *filename, int iterations)
{
    int my_rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

#pragma omp parallel
    {
        if(my_rank == 0)
        {
            printf("Hello World... from thread = %d\n",
                    omp_get_thread_num());
        }
    }

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
    ia = new int[nrows+1];
    ia_sendcounts = new int[nprocs];    
    
    vec_recvcounts = new int[nprocs];
    vec_displs = new int[nprocs];

    if(my_rank == MASTER_THREAD)
    {
        cout << "# of processes: " << nprocs << endl;
        cout << "# of threads per process: " << omp_get_num_threads() << endl;;
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

    for (int iter = 0; iter != iterations; ++iter)
    {
        if(my_rank == MASTER_THREAD)
        {
            timer = MPI_Wtime();
        }

        MPI_Allgatherv(X, nrows, MPI_DOUBLE, X_global, vec_recvcounts, vec_displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Y = A*X
        mkl_dcsrmv(&transA , &nrows, &ncols, &alpha, matdescra, acsr, ja, ia, ia+1, X_global, &beta, Y);

        // X = Y
        cblas_dcopy(nrows, Y, 1, X, 1);

        // norm = ||X||
        norm_squared = cblas_ddot(nrows, X, 1, X, 1);

        MPI_Allreduce(&norm_squared, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        global_norm = sqrt(global_norm);

        // X = X/norm
        cblas_dscal(nrows, 1.0/global_norm, X, 1);

        if(my_rank == MASTER_THREAD)
        {
            norms[iter] = global_norm;
        }

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
