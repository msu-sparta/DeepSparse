#include "exec_util.h"

long position = 0 ;
int *colptrs, *irem;
int nrows, ncols, nnz, numrows, numcols, nnonzero, nthrds ;
int wblk, nrowblks, ncolblks, nthreads;
int *nnzPerRow;
//block<double> *matrixBlock;
block *matrixBlock;

//template<typename T>
void read_custom(char* filename, double *&xrem)
{
    FILE *fp;

    double tstart, tend;

    tstart = omp_get_wtime();

    fp = fopen(filename, "rb");

        if(fp == NULL)
    {
            cout << "invalid matrix file name" << endl;
        return;
    }

    fread(&numrows, sizeof(int), 1, fp);
    cout<<"row: "<<numrows<<endl;

    fread(&numcols, sizeof(int), 1, fp);
    cout<<"colum: "<<numcols<<endl;

    fread(&nnonzero, sizeof(int), 1, fp);
    cout<<"non zero: "<<nnonzero<<endl;

    colptrs = new int[numcols + 1];
    irem = new int[nnonzero];
    //xrem = new T[nnonzero];
    xrem = new double[nnonzero];

    int *tcolptrs = new int[numcols + 1];
    int *tirem = new int[nnonzero];
    float *txrem = new float[nnonzero];

    cout << "Memory allocation finished" << endl;

    fread(tcolptrs, sizeof(int), numcols+1, fp);
        cout << "finished reading tcolptrs" << endl;

    fread(tirem, sizeof(int), nnonzero, fp);
    cout << "finished reading tirem" << endl;

    fread(txrem, sizeof(float), nnonzero, fp);
    cout << "finished reading txrem" << endl;


    #pragma omp parallel for
    for(int i = 0 ; i < nnonzero; i++)
    {
        xrem[i] = txrem[i];
    }

    #pragma omp parallel for
    for(int i = 0 ; i < numcols+1; i++)
    {	
    	colptrs[i] = tcolptrs[i] - 1;
    }

    #pragma omp parallel for
    for(int i = 0 ; i < nnonzero; i++)
    {
    	irem[i] = tirem[i] - 1;
    }

    delete []tcolptrs;
    delete []tirem;
    delete []txrem;

    tend = omp_get_wtime();

    cout << "Matrix is read in " << tend - tstart << " seconds." << endl;
}

void csc2blkcoord(block *&matrixBlock, double *xrem)
{
    int i, j, r, c, k, blkr, blkc;
    int **top;
    nrowblks = (numrows + wblk - 1) / wblk;
    ncolblks = (numcols + wblk - 1) / wblk;
    cout << "wblk = " << wblk << endl;
    cout << "nrowblks = " << nrowblks << endl;
    cout << "ncolblks = " << ncolblks << endl;

    matrixBlock = new block[nrowblks * ncolblks];
    top = new int*[nrowblks];
    nnzPerRow = (int *) malloc(nrowblks * sizeof(int));

    for(i = 0 ; i < nrowblks ; i++)
    {
        top[i] = new int[ncolblks];
        nnzPerRow[i] = 0;
    }

    #pragma omp parallel for default(shared) private(blkr, blkc)
    for(blkr = 0 ; blkr < nrowblks ; blkr++)
    {
        for(blkc = 0 ; blkc < ncolblks ; blkc++)
        {
            top[blkr][blkc] = 0;
            matrixBlock[blkr * ncolblks + blkc].nnz = 0;
        }
    }

    cout<<"calculatig nnz per block"<<endl;

    //calculatig nnz per block
    #pragma omp parallel for default(shared) private(i, c, k, blkc, r, blkr)
    for(i = 0; i < ncolblks; i++)
    {
        for(c = i * wblk ; c < min( (i+1)*wblk, numcols) ; c++)
        {
            blkc = c / wblk;

            for(k = colptrs[c]; k < colptrs[c+1]; k++)
            {
                r = irem[k];
                blkr = r / wblk;
                if( blkr >= nrowblks || blkc >= ncolblks)
                {
                    cout << "(" << blkr << ", " << blkc << ") doesn't exist" << endl;
                }
                else
                {
                    matrixBlock[blkr * ncolblks + blkc].nnz++;  
                }    
            }
        }
    }

    cout<<"finished counting nnz in each block"<<endl;

    for(blkc = 0 ; blkc < ncolblks; blkc++)
    {
        for(blkr = 0 ; blkr < nrowblks ; blkr++)
        {
            matrixBlock[blkr * ncolblks + blkc].roffset = blkr * wblk;
            matrixBlock[blkr * ncolblks + blkc].coffset = blkc * wblk;

            if(matrixBlock[blkr * ncolblks + blkc].nnz > 0)
            {
                nnzPerRow[blkr] += matrixBlock[blkr * ncolblks + blkc].nnz;
#ifdef SHORT_INT
                matrixBlock[blkr * ncolblks + blkc].rloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
#else
                matrixBlock[blkr * ncolblks + blkc].rloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
#endif
                //matrixBlock[blkr * ncolblks + blkc].val = new T[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].val = new double[matrixBlock[blkr * ncolblks + blkc].nnz];
            }
            else
            {
                matrixBlock[blkr * ncolblks + blkc].rloc = NULL;
                matrixBlock[blkr * ncolblks + blkc].cloc = NULL;
            }
        }
    }

    cout<<"allocating memory for each block"<<endl;

    printf("numrow = %d numcols = %d\n", numrows,numcols);

    #pragma omp parallel for default(shared) private(i, c, k, blkc, r, blkr)
    for(i = 0; i < ncolblks; i++)
    {
        for(c = i * wblk ; c < min( (i+1)*wblk, numcols) ; c++)
        {
            blkc = c / wblk;
            for(k = colptrs[c]; k < colptrs[c+1]; k++)
            {
                r = irem[k];
                blkr = r / wblk;

                matrixBlock[blkr * ncolblks + blkc].rloc[top[blkr][blkc]] = r - matrixBlock[blkr * ncolblks + blkc].roffset;
                matrixBlock[blkr * ncolblks + blkc].cloc[top[blkr][blkc]] = c - matrixBlock[blkr * ncolblks + blkc].coffset;
                matrixBlock[blkr * ncolblks + blkc].val[top[blkr][blkc]] = xrem[k];

                top[blkr][blkc] = top[blkr][blkc] + 1;
            }
        }
    }

    for(i = 0 ; i < nrowblks ; i++)
    {
        delete [] top[i];
    }
    delete [] top;

    cout<<"matrix conversion completed"<<endl;
}

void transpose(double *src, double *dst, const int N, const int M)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

void inverse(double *arr, int m, int n)
{   
    /**************
    input: arr[m*n] in row major format.
    **************/

    int lda_t = m;
    int lda = n;
    int info;
    int lwork = -1;
    double* work = NULL;
    double work_query;
    int *ipiv = new int[n+1]();

    double *arr_t = new double[m*n]();
    transpose(arr, arr_t, n, m);
    dgetrf_( &n, &m, arr_t, &lda_t, ipiv, &info );
    if(info < 0)
    {
       cout << "dgetrf_: Transpose error!!" << endl;
       exit(1);
    }
   //transpose(arr_t, arr, m, n);
   //LAPACKE_dgetri(LAPACK_ROW_MAJOR, n,arr,n,ipiv);

   /* Query optimal working array(s) size */
   dgetri_( &m, arr_t, &lda_t, ipiv, &work_query, &lwork, &info );
   if(info<0)
   {
       cout<<"dgetri_ 1: Transpose error!!"<<endl;
       //exit(1);
   }
   lwork = (int)work_query;
   //cout<<"lwork: "<<lwork<<endl;
   work = new double[lwork]();
   dgetri_( &m, arr_t, &lda_t, ipiv, work, &lwork, &info );
   if(info<0)
   {
       cout<<"dgetri_ 2: Transpose error!!"<<endl;
       //exit(1);
   }
   transpose(arr_t, arr, m, n);
   delete []arr_t;
   delete []ipiv;
}



void print_mat(double *arr, const int row, const int col) 
{
    //cout.setf(ios::fixed);
    //cout.setf(ios::showpoint);
    //cout.precision(12);
    for(int i = 0 ; i < row ; i++)
    {
        for(int j = 0 ; j < col ; j++)
        {
            //cout << arr[i * col + j] << " ";
            printf("%.18lf ", arr[i*col + j]);
        }
        //cout << endl;
        printf("\n");
    }
}

void make_identity_mat(double *arr, const int row, const int col)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            if(i == j)
                arr[i * row + j] = 1.00;
            else
                arr[i * row + j] = 0.00;
        }
    }
}

void diag(double *src, double *dst, const int size)
{
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0; i < size ; i++)
    {
        for(j = 0 ; j < size ; j++)
        {
            if(i == j)
            {
                dst[i * size + j] = src[i];
            }
            else
                dst[i * size + j] =0.0;
        }
    }
}



void sum_sqrt(double *src, double *dst, const int row, const int col)
{
    int i, j;
    
    #pragma omp parallel for default(shared) private(j)
    for(i = 0 ; i < col ; i++) //i->col
    {
        for(j = 0 ; j < row ; j++) //j->row
        {
            dst[i] += src[j * col + i];
        }
    }

    #pragma omp parallel for default(shared)
    for(i = 0; i < col ; i++) //i->col
    {
        dst[i] = sqrt(dst[i]);
    }
}

void update_activeMask(int *activeMask, double *residualNorms, double residualTolerance, int blocksize)
{
    int i;
    #pragma omp parallel for
    for(i=0; i<blocksize; i++)
    {
        if((residualNorms[i]>residualTolerance) && activeMask[i]==1)
            activeMask[i]=1;
        else
            activeMask[i]=0;
    }
}

void getActiveBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize)
{
    //activeBlockVectorR -> M*currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize

    int i, j, k=0;
    #pragma omp parallel for firstprivate(k) private(j) default(shared)
    for(i=0; i<M; i++)
    {
        k=0;
        for(j=0; j<blocksize; j++)
        {
             if(activeMask[j] == 1)
             {
                activeBlockVectorR[i*currentBlockSize+k] = blockVectorR[i*blocksize+j];
                k++;
             }
        }
    }
}
void updateBlockVector(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize)
{
    //activeBlockVectorR -> M*currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize
    int i, j, k = 0;
    #pragma omp parallel for firstprivate(k) private(j) default(shared)
    for(i=0; i<M; i++)
    {
        k=0;
        for(j=0; j<blocksize; j++)
        {
             if(activeMask[j] == 1)
             {
                blockVectorR[i*blocksize+j]= activeBlockVectorR[i*currentBlockSize+k];
                k++;
             }
        }
    }
}
void mat_copy(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst)
{
    int i,j;
    #pragma omp parallel for private(j) default(shared)
    for(i=0; i<row; i++)
    {
        for(j=0; j<col; j++)
        {
            dst[(start_row+i)*ld_dst+(start_col+j)]=src[i*col+j];
        }

    }
}


void print_eigenvalues( MKL_INT n, double* wr, double* wi ) 
{
    MKL_INT j;
    for( j = 0; j < n; j++ ) 
    {
        if( wi[j] == (double)0.0 ) 
        {
            printf( " %.15f", wr[j] );
        } 
        else 
        {
            printf( " (%.15f,%6.2f)", wr[j], wi[j] );
        }
    }
    printf( "\n" );
}

void custom_dlacpy(double *src, double *dst, int m, int n)
{
    //src[m*n] and dst[m*n]
    int i, j;
    #pragma omp parallel for private(j) default(shared)
    for(i = 0 ; i < m ; i++) //each row
    {
        for(j = 0 ; j < n ; j++) //each column
        {
            dst[i * n + j] = src[i * n + j];
        }
    }
}


bool checkEquals( double* a, double* b, size_t outterSize, size_t innerSize)
{
    for(size_t i = 0 ; i < outterSize * innerSize ; ++i)
    {
        if(abs(a[i]-b[i])>0.2)
        {
            cout << i << " " << a[i] << " " << b[i] << endl;
            return false;
        }
    }
    return true;
}

int buildTaskInfoStruct(struct TaskInfo *&taskInfo, char *partFile)
{
    char taskName[20000], ary[200], structToStr[20000];
    char buffer [33];
    char **splitParams;
    int partNo, tokenCount, priority, opCode, numParamsCount, strParamsCount;
    int vertexCount = 0, taskCounter = 0, index;
    string numStr;
    int i, j;

    //opening partition file
    ifstream partitionFile(partFile);
    
    if(partitionFile.fail())
    {
        cout << "File doesn't exits" << endl;
    }

    partitionFile >> vertexCount;

    //printf("vertexCount: %d\n", vertexCount);
    //cout << "vertexCount: " << vertexCount << endl;

    taskInfo = (struct TaskInfo *) malloc(vertexCount * sizeof(struct TaskInfo));

    while(partitionFile >> taskName)
    {
        tokenCount = split(taskName, ',', &splitParams);
        
        taskInfo[taskCounter].opCode = atoi(splitParams[0]);  
        taskInfo[taskCounter].numParamsCount = atoi(splitParams[1]);  //1
            
        index = 1; //numParamsCount
            
        if(taskInfo[taskCounter].numParamsCount > 0)
        {
            taskInfo[taskCounter].numParamsList = (int *) malloc(taskInfo[taskCounter].numParamsCount * sizeof(int));
            for(i = 0 ; i < taskInfo[taskCounter].numParamsCount ; i++)
            {
                taskInfo[taskCounter].numParamsList[i] = atoi(splitParams[index + 1 + i]);
            }   
        }
            
        taskInfo[taskCounter].strParamsCount = atoi(splitParams[index + taskInfo[taskCounter].numParamsCount + 1]); //2
            
        if(taskInfo[taskCounter].strParamsCount > 0)
        {
            taskInfo[taskCounter].strParamsList = (char **) malloc(taskInfo[taskCounter].strParamsCount * sizeof(char *)); //3
            for(i = 0 ; i < taskInfo[taskCounter].strParamsCount ; i++)
            {
                taskInfo[taskCounter].strParamsList[i] = (char *) malloc((strlen(splitParams[index + taskInfo[taskCounter].numParamsCount + 2 + i]) + 1) * sizeof(char));
                strcpy(taskInfo[taskCounter].strParamsList[i], splitParams[index + taskInfo[taskCounter].numParamsCount + 2 + i]);
            }   
        }
            
        taskInfo[taskCounter].taskID = atoi(splitParams[tokenCount - 3]);
        taskInfo[taskCounter].partitionNo = atoi(splitParams[tokenCount - 2]);
        taskInfo[taskCounter].priority = atoi(splitParams[tokenCount - 1]);
        
        // if(opCode == 15 || opCode == 16 || opCode == 17 || opCode == 18 || opCode == 20)
        // {
        //     //if(taskInfo[taskCounter].numParamsCount == 0 && taskInfo[taskCounter].strParamsCount == 1)
        //     printf("%s ---> %d,%d,%d,%s,%d,%d,%d\n", taskName, taskInfo[taskCounter].opCode, taskInfo[taskCounter].numParamsCount, taskInfo[taskCounter].strParamsCount, taskInfo[taskCounter].strParamsList[0], taskInfo[taskCounter].taskID, taskInfo[taskCounter].partitionNo, taskInfo[taskCounter].priority);    
        // }
        // structToString(taskInfo[taskCounter], structToStr);

        // //printf("%s --> %s\n", taskName, structToStr);

        // if(strcmp(taskName, structToStr) != 0)
        //     printf("%s --> %s\n", taskName, structToStr);
        //printf("taskCounter - %d: %d,%d\n", taskCounter, taskInfo[taskCounter].opCode, taskInfo[taskCounter].numParamsCount);
        taskCounter++;

    }//end while

    partitionFile.close();

    //cout << "Finish allocating taskInfo" << endl;

    return vertexCount;
}

void structToString(struct TaskInfo taskInfo, char *structToStr)
{
    int i;
    char buffer[50];
    
    myitoa(taskInfo.opCode, buffer);
    strcpy(structToStr, buffer);
    
    strcat(structToStr, ",");
    
    myitoa(taskInfo.numParamsCount, buffer);
    strcat(structToStr, buffer);
    strcat(structToStr, ",");

    if(taskInfo.numParamsCount > 0)
    {
        for(i = 0 ; i < taskInfo.numParamsCount ; i++)
        {
            myitoa(taskInfo.numParamsList[i], buffer);
            strcat(structToStr, buffer);
            strcat(structToStr, ",");
        }
    }

    myitoa(taskInfo.strParamsCount, buffer);
    strcat(structToStr, buffer);
    strcat(structToStr, ",");

    if(taskInfo.strParamsCount > 0)
    {
        for(i = 0 ; i < taskInfo.strParamsCount ; i++)
        {
            strcat(structToStr, taskInfo.strParamsList[i]);
            strcat(structToStr, ",");
        }
    }
    
    
    myitoa(taskInfo.taskID, buffer);
    strcat(structToStr, buffer);
    
    strcat(structToStr, ",");
    
    myitoa(taskInfo.partitionNo, buffer);
    strcat(structToStr, buffer);

    strcat(structToStr, ",");

    myitoa(taskInfo.priority, buffer);
    strcat(structToStr, buffer);

    //printf("%s\n", structToStr);
}

int readPartBoundary(int *&partBoundary, char *partBoundaryFile)
{
    int partCount = 0, vertexNo, counter = 0;
    int i, j;

    //opening partition file
    ifstream partitionBoundaryFile(partBoundaryFile);

    partitionBoundaryFile >> partCount;

    //printf("vertexCount: %d\n", vertexCount);
    //cout << "vertexCount: " << vertexCount << endl;

    partBoundary = (int *) malloc(partCount * sizeof(int));

    while(partitionBoundaryFile >> partBoundary[counter])
    {
        counter++;
    }//end while

    partitionBoundaryFile.close();

    //cout << partBoundaryFile << " : " << counter << endl;
    partCount--;
    return partCount;
}

void str_rev(char *str)
{
    char *p1, *p2;
    if (! str || ! *str)
        return;
    int len = strlen(str);
    for (p1 = str, p2 = str + len - 1; p2 > p1; ++p1, --p2)
    {
        *p1 ^= *p2;
        *p2 ^= *p1;
        *p1 ^= *p2;
    }
}

// Implementation of itoa() 
void myitoa(int num, char* str) 
{ 
    int i = 0, base = 10; 
    bool isNegative = false; 

    /* Handle 0 explicitely, otherwise empty string is printed for 0 */
    if (num == 0) 
    { 
        str[i++] = '0'; 
        str[i] = '\0'; 
    } 
    else
    {

        // In standard itoa(), negative numbers are handled only with 
        // base 10. Otherwise numbers are considered unsigned. 
        if (num < 0) 
        { 
            isNegative = true; 
            num = -num; 
        } 

        // Process individual digits 
        while (num != 0) 
        { 
            int rem = num % base; 
            str[i++] = (rem > 9)? (rem-10) + 'a' : rem + '0'; 
            num = num/base; 
        } 

        // If number is negative, append '-' 
        if (isNegative) 
            str[i++] = '-'; 

        str[i] = '\0'; // Append string terminator 
        // Reverse the string 
        str_rev(str); 
    }
    //return str; 
} 

void print_summary(double **timingStat, int iterationNumber)
{
    int i, j, workerCount;
    int numOperation = 24 ; ////// added while merging
    double *avgTaskTiming = (double *) malloc(numOperation * sizeof(double));
    double totalTime = 0;
    double **taskTiming = NULL;
    for(i = 0 ; i < numOperation; i++)
    {
        avgTaskTiming[i] = 0.0;
        workerCount = 0;
        for(j = 0 ; j < nthrds ; j++)
        {
            avgTaskTiming[i] += taskTiming[i][j];
            if(taskTiming[i][j] > 0)
                workerCount++;
        }
        if(workerCount > 0 && (i == 12)) //only two inv task
        {
            avgTaskTiming[i] = avgTaskTiming[i] / 2;
        }
        else if(workerCount > 0 && (i != 20) && (i != 19) && (i != 18) && (i != 17) && (i != 23))
        {
            avgTaskTiming[i] = avgTaskTiming[i] / nthrds;
        }
        totalTime += avgTaskTiming[i];

        //printf("task: %d workerCount: %d avg Timing: %.10lf\n", i, workerCount, avgTaskTiming[i]);
        
    }
    //printf("%10s %.6lf sec.\n", "SETZERO", avgTaskTiming[0]);
    timingStat[0][iterationNumber - 1] = avgTaskTiming[0];

    //printf("%10s %.6lf sec.\n", "XY", avgTaskTiming[1]);
    timingStat[1][iterationNumber - 1] = avgTaskTiming[1];
    
    //printf("%10s %.6lf sec.\n", "XTY", (avgTaskTiming[10] + avgTaskTiming[11]));
    timingStat[2][iterationNumber - 1] = avgTaskTiming[10] + avgTaskTiming[11];
    
    //printf("%10s %.6lf sec.\n", "ADD", avgTaskTiming[21]);
    timingStat[3][iterationNumber - 1] = avgTaskTiming[21];
    
    //printf("%10s %.6lf sec.\n", "SUB", avgTaskTiming[2]);
    timingStat[4][iterationNumber - 1] = avgTaskTiming[2];
    
    //printf("%10s %.6lf sec.\n", "MULT", avgTaskTiming[3]);
    timingStat[5][iterationNumber - 1] = avgTaskTiming[3];
    
    //printf("%10s %.6lf sec.\n", "SPMM", avgTaskTiming[15] + avgTaskTiming[16]);
    timingStat[6][iterationNumber - 1] = avgTaskTiming[15] + avgTaskTiming[16];

    //printf("%10s %.6lf sec.\n", "GET", avgTaskTiming[9]);
    timingStat[7][iterationNumber - 1] = avgTaskTiming[9];
    
    //printf("%10s %.6lf sec.\n", "UPDATE", avgTaskTiming[13]);
    timingStat[8][iterationNumber - 1] = avgTaskTiming[13];
    
    //printf("%10s %.6lf sec.\n", "EIGEN", avgTaskTiming[20]);
    timingStat[9][iterationNumber - 1] = avgTaskTiming[20];
    
    //printf("%10s %.6lf sec.\n", "DLACPY", avgTaskTiming[14]);
    timingStat[10][iterationNumber - 1] = avgTaskTiming[14];

    //printf("Total Time: %.6lf\n", totalTime);
    timingStat[11][iterationNumber - 1] = totalTime;
}

int split (const char *str, char c, char ***arr)
{
    int count = 1;
    int token_len = 1;
    int i = 0;
    const char *p;
    char *t;

    p = str;
    while (*p != '\0')
    {
        if (*p == c)
            count++;
        p++;
    }

    *arr = (char**) malloc(sizeof(char*) * count);
    if (*arr == NULL)
        return -1;

    p = str;
    while (*p != '\0')
    {
        if (*p == c)
        {
            (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
            if ((*arr)[i] == NULL)
                return -1;

            token_len = 0;
            i++;
        }
        p++;
        token_len++;
    }
    (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
    if ((*arr)[i] == NULL)
        return -1;

    i = 0;
    p = str;
    t = ((*arr)[i]);
    while (*p != '\0')
    {
        if (*p != c && *p != '\0')
        {
            *t = *p;
            t++;
        }
        else
        {
            *t = '\0';
            i++;
            t = ((*arr)[i]);
        }
        p++;
    }

    return count;
}
