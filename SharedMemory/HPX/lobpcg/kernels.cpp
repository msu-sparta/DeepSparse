//#include "utils.h"
#include "kernels.h"

void Fill(int tid, double *V, int size)
{
    int start = (size/nthreads) * tid;
    int end = (tid == nthreads - 1) ? size : ( start + (size/nthreads));
    
    for(int i = start; i < end; ++i)
        V[i] = 0;
}

void XY(double *a, double *b, double *c, int m, int n, int k, double alpha, double beta)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
}

void XY_part(double *a, double *b, double *c, int m, int n, int k, double alpha, double beta, int block_width, int part)
{
    int rows = std::min(m, (part+1)*block_width) - part*block_width;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, n, k, alpha, a + part*block_width*k, k, b, n, beta, c + part*block_width*n, n);
}

void XTY(double *a, double *b, double *c, int m, int n, int k, double alpha, double beta)
{
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, a, m, b, n, beta, c, n);
}

void XTY_part(double *a, double *b, double *c, int m, int n, int k, double alpha, double beta, int block_width, int part)
{
    int cols = std::min(k, (part+1)*block_width) - part*block_width;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, cols, alpha, a + part*block_width*m, m, b + part*block_width*n, n, beta, c + part*m*n, n);
}

void XTY_red(double *src, double *dst, int m, int n)
{
    int i, j, k;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[i*n + j] = 0.0;
            for(k = 0; k != nrowblks; ++k)
            {
                dst[i*n + j] += src[(k*m + i)*n + j];
            }
        }
    }

}

void Transpose(double *src, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[j * m + i] = src[i * n + j];
        }
    }
}

void Cholesky(double *a, int m)
{
    int info;
    const char uplo = 'U';

    dpotrf_(&uplo, &m, a, &m, &info);
    if(info != 0)
    {
        printf("Error: Cholesky, info: %d\n", info);
    }
}

void Copy(double *src, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[i * n + j] = src[i * n + j];
        }
    }
}

void Copy_part(double *src, double *dst, int m, int n, int block_width, int part)
{
    int i, j, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[i * n + j] = src[i * n + j];
        }
    }
}

void Inverse(double *a, int m)
{
    int info, lda, lwork;
    int *ipiv;
    double work_query;
    double *a_t, *work;

    a_t = (double *) malloc(m * m * sizeof(double));
    ipiv = (int *) malloc((m+1) * sizeof(int));

    lda = m;
    lwork = -1;

    Transpose(a, a_t, m, m);

    dgetrf_(&m, &m, a_t, &lda, ipiv, &info);
    if(info < 0)
    {
        printf("Error: LU, info: %d\n", info);
    }

    dgetri_(&m, a_t, &lda, ipiv, &work_query, &lwork, &info);
    if(info < 0)
    {
        printf("Error: (LU)^-1 #1, info: %d\n", info);
    }

    lwork = (int)work_query;
    work = (double *) malloc(lwork * sizeof(double));

    dgetri_(&m, a_t, &lda, ipiv, work, &lwork, &info);
    if(info < 0)
    {
        printf("Error: (LU)^-1 #2, info: %d\n", info);
    }

    Transpose(a_t, a, m, m);

    free(a_t);
    free(ipiv);
    /* TODO */
    free(work);
}

void Print(double *a, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            printf("%.6lf ", a[i * n + j]);
        }
        printf("\n");
    }
}

void ResetBelowDiagonal(double *a, int m)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != i; ++j)
        {
            a[i * m + j] = 0.0;
        }
    }
}

void Reset(double *a, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            a[i * n + j] = 0.0;
        }
    }
}

void Reset_part(double *a, int m, int n, int block_width, int part)
{
    int i, j, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            a[i * n + j] = 0.0;
        }
    }
}

void SpMM(block *A, double *X, double *Y, int RHS)
{
    int r, c, i, j;

    for(r = 0; r != nrowblks; ++r)
    {
        for(c = 0; c != ncolblks; ++c)
        {
            block &blk = A[r * ncolblks + c];
            for(i = 0; i != blk.nnz; ++i)
            {
                for(j = 0; j != RHS; ++j)
                {
                    Y[ (blk.rloc[i] + blk.roffset) * RHS + j] += blk.val[i] * X[ (blk.cloc[i] + blk.coffset) * RHS + j];
                }
            }
        }
    }
}

void SpMM_part(block *A, double *X, double *Y, int RHS, int r, int c)
{
    int i, j;

    block &blk = A[r * ncolblks + c];
    for(i = 0; i != blk.nnz; ++i)
    {
        for(j = 0; j != RHS; ++j)
        {
            Y[ (blk.rloc[i] + blk.roffset) * RHS + j] += blk.val[i] * X[ (blk.cloc[i] + blk.coffset) * RHS + j];
        }
    }
}

void Identity(double *a, int m)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != m; ++j)
        {
            a[i * m + j] = ( (i == j) ? 1.0 : 0.0 );
        }
    }
}

void EigenComp(double *a, double *b, double *w, int blocksize)
{
    const int itype = 1;
    const char jobz = 'V';
    const char uplo = 'U';
    int info, lwork;
    double work_query;
    double *work;

    lwork = -1;
    dsygv_(&itype, &jobz, &uplo, &blocksize, a, &blocksize, b, &blocksize, w, &work_query, &lwork, &info);
    if(info != 0)
    {
        printf("Error: EIG #1, info: %d\n", info);
    }

    lwork = (int)work_query;
    work = (double *) malloc(lwork * sizeof(double));
    dsygv_(&itype, &jobz, &uplo, &blocksize, a, &blocksize, b, &blocksize, w, work, &lwork, &info);
    if(info != 0)
    {
        printf("Error: EIG #1, info: %d\n", info);
    }
    /* TODO */
    free(work);
}

void Diag(double *src, double *dst, int m)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != m; ++j)
        {
            dst[i * m + j] = ( (i == j) ? src[i] : 0.0 );
        }
    }
}

void Subtract(double *src1, double *src2, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j !=n; ++j)
        {
            dst[i*n + j] = src1[i*n + j] - src2[i*n + j];
        }
    }
}

void Subtract_part(double *src1, double *src2, double *dst, int m, int n, int block_width, int part)
{
    int i, j, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        for(j = 0; j !=n; ++j)
        {
            dst[i*n + j] = src1[i*n + j] - src2[i*n + j];
        }
    }
}

void Multiply(double *src1, double *src2, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[i*n + j] = src1[i*n + j] * src2[i*n + j];
        }
    }
}

void Multiply_part(double *src1, double *src2, double *dst, int m, int n, int block_width, int part)
{
    int i, j, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);

    for(i = start; i != end; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[i*n + j] = src1[i*n + j] * src2[i*n + j];
        }
    }
}

void ReducedNorm(double *src, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[j] += src[i*n + j];
        }
    }

    for(i = 0; i != n; i++)
    {
        dst[i] = sqrt(dst[i]);
    }
}

void ReducedNorm_part(double *src, double *dst, int m, int n, int block_width, int part)
{
    int i, j, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[part*n + j] += src[i*n + j];
        }
    }
}

void ReducedNorm_red(double *src, double *dst, int m)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        dst[i] = 0.0;
        for(j = 0; j != nrowblks; ++j)
        {
            dst[i] += src[j*m + i];
        }
        dst[i] = sqrt(dst[i]);
    }
}

int UpdateActiveMask(double *src, int *dst, double threshold, int m)
{
    int i, activeN;

    activeN = 0;
    for(i = 0; i != m; ++ i)
    {
        if(src[i] > threshold && dst[i] == 1)
        {
            dst[i] = 1;
            activeN++;
        }
        else
        {
            dst[i] = 0;
        }
    }

    return activeN;
}

void GatherActiveVectors(double *src, double *dst, int *mask, int m, int n, int activeN)
{
    int i, j, k;

    for(i = 0; i != m; ++i)
    {
        k = 0;
        for(j = 0; j != n; ++j)
        {
            if(mask[j] == 1)
            {
                dst[i*activeN + k] = src[i*n + j];
                k++;
            }
        }
    }
}

void GatherActiveVectors_part(double *src, double *dst, int *mask, int m, int n, int activeN, int block_width, int part)
{
    int i, j, k, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        k = 0;
        for(j = 0; j != n; ++j)
        {
            if(mask[j] == 1)
            {
                dst[i*activeN + k] = src[i*n + j];
                k++;
            }
        }
    }
}

void ScatterActiveVectors(double *src, double *dst, int *mask, int m, int n, int activeN)
{
    int i, j, k;

    for(i = 0; i != m; ++i)
    {
        k = 0;
        for(j = 0; j != n; ++j)
        {
            if(mask[j] == 1)
            {
                dst[i*n + j] = src[i*activeN + k];
                k++;
            }
        }
    }
}

void ScatterActiveVectors_part(double *src, double *dst, int *mask, int m, int n, int activeN, int block_width, int part)
{
    int i, j, k, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        k = 0;
        for(j = 0; j != n; ++j)
        {
            if(mask[j] == 1)
            {
                dst[i*n + j] = src[i*activeN + k];
                k++;
            }
        }
    }
}


void CopyBlock(double *src, double *dst, int m, int n, int ld, int rOffset, int cOffset)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j != n; ++j)
        {
            dst[(i+rOffset) * ld + (j+cOffset)] = src[i*n + j];
        }
    }
}

void CopyCoordX(double *src, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; i++)
    {
        for(j = 0; j != std::min(m,n); ++j)
        {
            dst[i*n + j] = src[i*m + j];
        }
    }
}

void Add(double *src1, double *src2, double *dst, int m, int n)
{
    int i, j;

    for(i = 0; i != m; ++i)
    {
        for(j = 0; j !=n; ++j)
        {
            dst[i*n + j] = src1[i*n + j] + src2[i*n + j];
        }
    }
}

void Add_part(double *src1, double *src2, double *dst, int m, int n, int block_width, int part)
{
    int i, j, start, end;

    start = part*block_width;
    end = std::min(start+block_width, m);
    for(i = start; i != end; ++i)
    {
        for(j = 0; j !=n; ++j)
        {
            dst[i*n + j] = src1[i*n + j] + src2[i*n + j];
        }
    }
}
