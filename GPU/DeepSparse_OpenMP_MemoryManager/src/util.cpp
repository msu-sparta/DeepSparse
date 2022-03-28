#ifndef UTIL_H
#define UTIL_H

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <fstream>
#include <vector>
using namespace std;

#if defined(USE_MKL)
#include <mkl.h>
#endif
#if defined(USE_LAPACK)
#include <lapacke.h>
#include <cblas.h>
#endif

#ifdef USE_CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

#include <omp.h>
#include "../inc/util.h"
#include "../inc/memory_manager_v5.h"

long position = 0, maxIterations;
int *colptrs, *irem;

// Sparse matrix in CSR format storage
int  *ia , *ja;
double *d_ja;
double *acsr;

int numcols, numrows, nnonzero, nthrds = 1;
int nrows, ncols, nnz;
int wblk, nrowblks, ncolblks, blocksize, block_width;

template<typename T>
struct block
{
    int nnz;
    int roffset, coffset;
    unsigned short int *rloc, *cloc;
    T *val;
};

struct TaskInfo
{
  int opCode;
  int numParamsCount;
  int *numParamsList;
  int strParamsCount;
  char **strParamsList;
  int taskID;
  int partitionNo;
  int priority;
};

struct TaskInfo *taskInfo_nonLoop, *taskInfo_firstLoop, *taskInfo_secondLoop;

int numTasks = 15;
vector<string> function_name{"SETZERO", "SPMM", "XTY", "XY",  "ADD", "DLACPY", "SUB", "MULT", "COL", "GET", "UPDATE", "EIGEN", "INVERSE", "TRANSPOSE", "CHOL"};
double ***taskTiming;

void initialize_timing_variables()
{
    int i, j, k;
    taskTiming = (double ***) malloc(sizeof(double **) * maxIterations);
    for(i = 0 ; i < maxIterations ; i++)
        taskTiming[i] = (double **) malloc(sizeof(double*) * numTasks);
    
    for(i = 0 ; i < maxIterations ; i++)
        for(j = 0 ; j < numTasks ; j++)
            taskTiming[i][j] = (double *) malloc(sizeof(double) * 2); // 2 ==> data transfer time & compute time
    
    #pragma omp parallel for default(shared) private(j, k)
    for(i = 0 ; i < maxIterations ; i++)
    {
        for(j = 0 ; j < numTasks ; j++)
        {
            for(k = 0 ; k < 2 ; k++)
            {
                taskTiming[i][j][k] = 0.0;
            }
        }
    }
}

void summarize_timing_variables()
{
    int i, j, k;
    double kernel_time, memcpy_time;
    double total = 0 , kernel_total = 0, mem_total = 0;

    for(j = 0 ; j < numTasks ; j++)
    {
        kernel_time = 0, memcpy_time = 0;

        for(i = 0 ; i < maxIterations ; i++)
        {
            if(taskTiming[i][j][0] > 0 || taskTiming[i][j][1] > 0)
            {
                 
                taskTiming[i][j][1] = taskTiming[i][j][1] - taskTiming[i][j][0];
                kernel_time += taskTiming[i][j][1];
                memcpy_time += taskTiming[i][j][0];
            }
                
            // for(k = 0 ; k < 2 ; k++)
            // {
                
            // }
        }
        if(kernel_time > 0 || memcpy_time > 0)
        {
            memcpy_time = memcpy_time/maxIterations;
            kernel_time = kernel_time/maxIterations;

            cout << function_name[j] << "," << kernel_time << "," << memcpy_time << "," << kernel_time + memcpy_time << endl;

            total += (kernel_time + memcpy_time);
            kernel_total += kernel_time;
            mem_total += memcpy_time;
        }
    }
    cout << "Total," << kernel_total << "," << mem_total << "," << total << endl;

}

int omp_thread_count() 
{
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

double get_seconds() 
{
	struct timeval now;
	gettimeofday(&now, NULL);
	const double seconds = (double) now.tv_sec;
	const double usec    = (double) now.tv_usec;
	return seconds + (usec * 1.0e-6);
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

    //how about using strtok
    i = 0;
    char *tempStr = (char *) malloc ((strlen(str)+1) * sizeof(char));
    strcpy(tempStr, str);

    char *token = strtok(tempStr, ",");
 
    // Keep printing tokens while one of the
    // delimiters present in str[].
    while (token != NULL)
    {
        (*arr)[i] = (char*) malloc( sizeof(char) * (strlen(token)+1) );
        strcpy((*arr)[i],token);
        token = strtok(NULL, ",");
        i++;
    }
    return count;
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
} 

void read_csr(char *filename)
{
    int i, j;
    FILE *file = fopen(filename, "rb");
    if (filename != NULL)
    {
        fread(&numrows, sizeof(int), 1, file);
        cout << "row: " << numrows << endl;
        fread(&numcols, sizeof(int), 1, file);
        cout << "colum: " << numcols << endl;

        fread(&nnonzero, sizeof(float), 1, file);
        cout << "non zero: " << nnonzero << endl;

        ia = (int *) malloc((numrows + 1) * sizeof(int)); //colsptr
        ja = (int *) malloc(nnonzero * sizeof(int)); //irem
        acsr = (double *) malloc(nnonzero * sizeof(double)); //xrem

        fread(ia, sizeof(int), numrows + 1, file);
        cout << "finished reading ia"<<endl;

        fread(ja, sizeof(int), nnonzero, file);
        cout << "finished reading ja"<<endl;
        
        fread(acsr, sizeof(double), nnonzero, file);
        cout << "finished reading acsr"<<endl;
        fclose(file);

        // for mm_v3 the following code blocks should be commented out. It is contained inside the main program in mm_v3
        // Setting up device memory
        // double fixed_memory = (numrows + nnonzero + 1) * sizeof(int) + (numrows * blocksize) * sizeof(double); //+ max_nnz * sizeof(double); 
        // capacity = 14.5 * 1e+9 - fixed_memory;
        // cout << "capacity: " << capacity << " elem: " << capacity / sizeof(double) << " fixed_memory: " << fixed_memory << endl;
        // unsigned long int total_element = capacity / sizeof(double);
        // available_mem = capacity = total_element * sizeof(double);

        // cout << "capacity: " << capacity << " elem: " << capacity / sizeof(double) << endl;

        // totalMemBlock = (unsigned long int) floor(capacity/(sizeof(double) * num_per_blk));
        // cout << "totalMemBlock: " << totalMemBlock << endl;
        // // pair<double *, int> nullPr = make_pair(nullptr, -1);
        // memView.resize(totalMemBlock); // ==> for mm_v2
        // // for mm_v4, we need to properly fill the memView vector
        // unsigned long int ii, total_block = totalMemBlock;
        // // pair<double *, int> tempPR;
        // for(ii = 0 ; ii < totalMemBlock ; ii++)
        // {
        //     memView[ii] = make_pair(nullptr, total_block);
        //     // memView[ii] = tempPR;
        //     total_block--;
        // }
        // setMatrixInfo(nullptr, -1, -1, "nullptr");
    }
    else
    {
        cout << "FIle opening error" << endl;
    }
}


void read_csr_DJA(char *filename)
{
    int i, j;
    FILE *file = fopen(filename, "rb");
    if (filename != NULL)
    {
        fread(&numrows, sizeof(int), 1, file);
        cout << "row: " << numrows << endl;
        fread(&numcols, sizeof(int), 1, file);
        cout << "colum: " << numcols << endl;

        fread(&nnonzero, sizeof(float), 1, file);
        cout << "non zero: " << nnonzero << endl;

        ia = (int *) malloc((numrows + 1) * sizeof(int)); //colsptr
        d_ja = (double *) malloc(nnonzero * sizeof(double)); //irem
        acsr = (double *) malloc(nnonzero * sizeof(double)); //xrem

        fread(ia, sizeof(int), numrows + 1, file);
        cout << "finished reading ia"<<endl;

        fread(d_ja, sizeof(double), nnonzero, file);
        cout << "finished reading d_ja" << endl;
        
        fread(acsr, sizeof(double), nnonzero, file);
        cout << "finished reading acsr" << endl;
        fclose(file);
    }
    else
    {
        cout << "FIle opening error" << endl;
    }
}


void read_csr_DJA_2(char* filename)
{
	int i, j;
  	ifstream file (filename, ios::in|ios::binary);
  	if (file.is_open())
  	{

        /*fread(&numrows, sizeof(int), 1, file);
        cout << "row: " << numrows << endl;
        fread(&numcols, sizeof(int), 1, file);
        cout << "colum: " << numcols << endl;

        fread(&nnonzero, sizeof(float), 1, file);
        cout << "non zero: " << nnonzero << endl;

        ia = (int *) malloc((numrows + 1) * sizeof(int)); //colsptr
        d_ja = (double *) malloc(nnonzero * sizeof(double)); //irem
        acsr = (double *) malloc(nnonzero * sizeof(double)); //xrem

        fread(ia, sizeof(int), numrows + 1, file);
        cout << "finished reading ia"<<endl;

        fread(d_ja, sizeof(double), nnonzero, file);
        cout << "finished reading d_ja" << endl;
        
        fread(acsr, sizeof(double), nnonzero, file);
        cout << "finished reading acsr" << endl;
        fclose(file);*/

      	/* int a = 0, c = 0;
      	long int b = 0;

      	double d = 0;
      	file.read ((char*)&numrows,sizeof(numrows));
      	cout<<"row: "<<numrows<<endl;
      	file.read(reinterpret_cast<char*>(&numcols), sizeof(numcols));
      	cout<<"colum: "<<numcols<<endl;

      	file.read(reinterpret_cast<char*>(&nnonzero), sizeof(nnonzero));
      	cout<<"non zero: "<<nnonzero<<endl;

      	// colptrs = new int[numcols + 1];
      	// irem = new int[nnonzero];
      	d_ja = new double[nnonzero];

      	cout << "Memory allocaiton finished" << endl;

      	position = 0;
      	while(!file.eof() && position <= numcols)
      	{
        	file.read(reinterpret_cast<char*>(&a), sizeof(a)); //irem(j)
        	// colptrs[position++] = a;
          position++;
    	}
    	cout << "finished reading colptrs" << endl;
    	position = 0;
    	while(!file.eof() && position < nnonzero)
    	{
        	file.read(reinterpret_cast<char*>(&d), sizeof(d));
        	// irem[position++] = a;
          d_ja[position++] = d;
    	}
    cout << "Finished reading d_ja" << endl;
    	position = 0;
    	while(!file.eof() && position < nnonzero)
    	{
        	file.read(reinterpret_cast<char*>(&d), sizeof(d)); //irem(j)
        	// xrem[position++] = d;
          position++;
      }
      cout << "Done reading in DJA_2" << endl; */
	}
  else {
    cout << "File not found" << endl;
  }
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
    taskInfo = (struct TaskInfo *) malloc(vertexCount * sizeof(struct TaskInfo));

    while(partitionFile >> taskName)
    {
        tokenCount = split(taskName, ',', &splitParams);
        
        taskInfo[taskCounter].opCode = atoi(splitParams[0]);  
        taskInfo[taskCounter].numParamsCount = atoi(splitParams[1]);  //1
            
        index = 1; // numParamsCount
            
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
        taskCounter++;
    }//end while

    partitionFile.close();
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
}

int readPartBoundary(int *&partBoundary, char *partBoundaryFile)
{
    int partCount = 0, vertexNo, counter = 0;
    int i, j;

    //opening partition file
    ifstream partitionBoundaryFile(partBoundaryFile);
    partitionBoundaryFile >> partCount;
    partBoundary = (int *) malloc(partCount * sizeof(int));

    while(partitionBoundaryFile >> partBoundary[counter])
    {
        counter++;
    }//end while

    partitionBoundaryFile.close();
    partCount--;
    return partCount;
}

template<typename T>
void read_custom(char* filename, T *&xrem)
{
    int i,j;
    ifstream file (filename, ios::in|ios::binary);
    if (file.is_open())
    {
        int a = 0, c = 0;
        long int b = 0;
        float d = 0.0;
        file.read((char*)&numrows, sizeof(numrows));
        cout << "row: " << numrows << endl;
        file.read(reinterpret_cast<char*>(&numcols), sizeof(numcols));
        cout << "colum: " << numcols << endl;

        file.read(reinterpret_cast<char*>(&nnonzero), sizeof(nnonzero));
        cout << "non zero: " << nnonzero << endl;

        colptrs = new int[numcols + 1];
        irem = new int[nnonzero];
        xrem = new T[nnonzero];
        cout << "Memory allocaiton finished" << endl;
        position = 0;

        while(!file.eof() && position<=numcols)
        {
            file.read(reinterpret_cast<char*>(&a), sizeof(a)); //irem(j)
            colptrs[position++] = a - 1;
        }
        cout << "finished reading colptrs" << endl;
        position = 0;
        while(!file.eof() && position < nnonzero)
        {
            file.read(reinterpret_cast<char*>(&a), sizeof(a)); //irem(j)
            irem[position++] = a-1;
        }

        position=0;
        while(!file.eof() && position<nnonzero)
        {
            file.read(reinterpret_cast<char*>(&d), sizeof(d)); //irem(j)
            xrem[position++] = d;
        }  
    }
}

template struct block<double>;
template struct block<float>;
template void read_custom<double>(char* filename, double *&xrem);

template<typename T>
bool checkEquals( T* a, T* b, size_t outterSize, size_t innerSize)
{
    for(size_t i = 0 ; i < outterSize * innerSize ; ++i)
    {
        if(abs(a[i] - b[i]) > 0.2)
        {
            cout << i << " " << a[i] << " " << b[i] << endl;
            return false;
        }
    }
    return true;
}

template<typename T>
void csc2blkcoord(block<T> *&matrixBlock, T *xrem)
{
    int i, j, r, c, k, k1, k2, blkr, blkc, tmp;
    int **top;
    nrowblks = ceil(numrows / (float)(wblk));
    ncolblks = ceil(numcols / (float)(wblk));
    cout << " nrowblks = " << nrowblks << endl;
    cout << " ncolblks = " << ncolblks << endl;

    matrixBlock = new block<T>[nrowblks * ncolblks];
    top = new int*[nrowblks];

    for(i = 0 ; i < nrowblks ; i++)
    {
        top[i] = new int[ncolblks];
    }

    for(blkr = 0 ; blkr < nrowblks ; blkr++)
    {
        for(blkc = 0 ; blkc < ncolblks ; blkc++)
        {
            top[blkr][blkc] = 0;
            matrixBlock[blkr * ncolblks + blkc].nnz = 0;
        }
    }
    cout << "Finish memory allocation for block.." << endl;

    //calculatig nnz per block
    for(c = 0 ; c < numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1] - 1;
        blkc = ceil((c + 1) / (float)wblk);
        
        for(k = k1 - 1 ; k < k2 ; k++)
        {
            r = irem[k];
            blkr = ceil(r / (float)wblk);
            matrixBlock[(blkr - 1) * ncolblks + (blkc - 1)].nnz++;
        }
    }
    for(blkc = 0 ; blkc < ncolblks ; blkc++)
    {
        for(blkr = 0 ; blkr < nrowblks ; blkr++)
        {
            matrixBlock[blkr * ncolblks + blkc].roffset = blkr * wblk;
            matrixBlock[blkr * ncolblks + blkc].coffset = blkc * wblk;
      
            if(matrixBlock[blkr * ncolblks + blkc].nnz > 0)
            {
                matrixBlock[blkr * ncolblks + blkc].rloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new unsigned short int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].val = new T[matrixBlock[blkr * ncolblks + blkc].nnz];
            }
            else
            {
                matrixBlock[blkr * ncolblks +blkc].rloc = NULL;
                matrixBlock[blkr * ncolblks + blkc].cloc = NULL;
            }
        }
    }
  
    for(c = 0 ; c < numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1] - 1;
        blkc = ceil((c + 1)/(float)wblk);

        for(k=k1-1;k<k2;k++)
        {
            r = irem[k];
            blkr = ceil(r/(float)wblk);

            matrixBlock[(blkr - 1) * ncolblks + blkc - 1].rloc[top[blkr - 1][blkc - 1]] = r - matrixBlock[(blkr - 1) * ncolblks + blkc - 1].roffset;
            matrixBlock[(blkr - 1) * ncolblks + blkc - 1].cloc[top[blkr - 1][blkc - 1]] = (c + 1) -  matrixBlock[(blkr - 1) * ncolblks + blkc - 1].coffset;
            matrixBlock[(blkr - 1) * ncolblks + blkc - 1].val[top[blkr - 1][blkc - 1]] = xrem[k];

            top[blkr - 1][blkc - 1] = top[blkr - 1][blkc - 1] + 1;
        }
    }
}

template void csc2blkcoord<double>(block<double> *&matrixBlock, double *xrem); 

void print_mat(double *arr, const int row, const int col) 
{
    cout.setf(ios::fixed);
    cout.setf(ios::showpoint);
    for(int i = 0 ; i < row ; i++)
    {
        cout << i << " : ";
        for(int j = 0 ; j < col ; j++)
        {
          cout << arr[i * col + j] << " ";
        }
        cout << endl;
    }
}

void print_eigenvalues(int n, double* wr, double* wi ) 
{
    int j;
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

#endif
