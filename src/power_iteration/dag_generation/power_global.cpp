#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <unordered_map>
#include <climits>
using namespace std;

#include <omp.h>
#include "../../common/util.h"


void mainloop(int blocksize , int block_width);

int main(int argc, char *argv[])
{

    int blocksize, block_width;

    int i;

    double tstart , tend;


    stringstream bs(argv[1]);
    bs >> blocksize;
    stringstream bw(argv[2]);
    bw >> block_width;

    /* csb format variables */

    double *xrem;

    char *filename = argv[3];

    wblk = block_width; 
    
    //read_custom<double>(filename, xrem);
    read_custom(filename, xrem);

    printf("Finsihed reading CUS file\n");
    //exit(1);

    //csc2blkcoord<double>(matrixBlock, xrem);
    csc2blkcoord(matrixBlock, xrem);
    
    printf("Finsihed Converting CUS TO CSB\n");


    #pragma omp parallel
    #pragma omp master
    {
        nthreads = omp_get_num_threads();
    }

    /* initializing nrowblksString */
    
    tstart = omp_get_wtime();
    int intTOstringCount = (nrowblks > nthreads) ? nrowblks : nthreads;
    nrowblksString = (char**) malloc(intTOstringCount * sizeof(char *)); // Allocate row pointers
    
    for(i = 0 ; i < intTOstringCount ; i++)
    {
        nrowblksString[i] = (char *) malloc(7 * sizeof(char));
        myitoa(i, nrowblksString[i]);
        //printf("%d : %s\n", i, nrowblksString[i]);
    }

    tend = omp_get_wtime();
    printf("nrowblksString time: %lf sec. \n", tend - tstart);
    

    /* graphn Gen timing breakdown*/
    graphGenTime = (double *) malloc(total_func * sizeof(double));
    
    for(i = 0 ; i < total_func; i++)
    {
        graphGenTime[i] = 0.0;
    }


    int guessEdgeCount = 8000000;
    int guessNodeCount = 8000000;
    edgeU = (int *) malloc(guessEdgeCount * sizeof(int));
    edgeV = (int *) malloc(guessEdgeCount * sizeof(int));
    edgeW = (double *) malloc(guessEdgeCount * sizeof(double));
    vertexWeight = (double *) malloc(guessEdgeCount * sizeof(double));
    
    vertexName.reserve(guessEdgeCount);
    vertexName.max_load_factor(0.25);

    //allocating memory for global graph
    globalGraph = (char **) malloc(guessNodeCount * sizeof(char *));
    for (i = 0 ; i < guessNodeCount ; i++)
    {
        globalGraph[i] = (char*) malloc(100 * sizeof(char));
    }

    mainloop(blocksize, block_width);
}




void mainloop(int blocksize , int block_width){
    
    int i, j, k; 
    int pseudo_tid = 0, nbuf = 16;
    char ary[150], i_string[8];
    int nthreads = 1;

    ////initialize edgeCount and nodeCount to zero

    edgeCount = 0 ; 
    nodeCount = 0 ; 
    globalNodeCount = 0 ; 

   int *Y_vertexNo = (int *) malloc(ncolblks * sizeof(int));
    int *SUBMAX_vertexNo = (int *) malloc(nrowblks * sizeof(int));

    int **SPMV_vertexNo = (int **) malloc(nrowblks * sizeof(int *));
    
    for(i = 0 ; i < ncolblks ; i++)
    {
        SPMV_vertexNo[i] = (int *) malloc(ncolblks * sizeof(int));
    }    
    for(i = 0 ; i < nrowblks ; i++)
    {
        Y_vertexNo[i] = -1;
        SUBMAX_vertexNo[i] = -1;
        for(j = 0 ; j < ncolblks ; j++)
            SPMV_vertexNo[i][j] = -1;
    }
    //printf("Hello world!\n");

    for(i = 0 ; i < ncolblks ; i++)
    {
         //Y
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "_Y,");
        strcat(ary, nrowblksString[i]);
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;
        Y_vertexNo[i] = nodeCount - 1;
        
        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
        //printf("node: %s id: %d\n", ary, Y_vertexNo[i]);
    }

    
    
    int SETZERO_id = -1, Y_id = -1;
    for(i = 0 ; i < nrowblks ; i++)
    {   
        //what if there is no nnz in an entire row?? --> think about it later

        //Y
        // memset(&ary[0], 0, sizeof(ary));
        // strcat(ary, "_Y,");
        // strcat(ary, nrowblksString[i]);
        // vertexName[strdup(ary)] = nodeCount;
        // vertexWeight[nodeCount] = block_width * sizeof(double);
        // nodeCount++;
        // Y_id = nodeCount - 1;
        //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);

        //SETZERO
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SETZERO,"); //block_id
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",1"); //task_id
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;
        SETZERO_id = nodeCount - 1;
        //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
            
        for(j = 0 ; j < ncolblks ; j++)
        {
            //SPMV
            if(matrixBlock[i * ncolblks + j].nnz > 0)
            {
                //A 
                // memset(&ary[0], 0, sizeof(ary));
                // strcat(ary, "_A,");
                // strcat(ary, nrowblksString[i]);
                // strcat(ary, ",");
                // strcat(ary, nrowblksString[j]);
                // vertexName[strdup(ary)] = nodeCount;
                // vertexWeight[nodeCount] = matrixBlock[i * ncolblks + j].nnz * sizeof(double); 
                // nodeCount++;
                //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);
                
                pseudo_tid = ((pseudo_tid % nthreads) > (nthreads - 1) ? 0 : (pseudo_tid % nthreads) ); //pseudo_tid changes itself from 0 to 15 in a cyclic fashion

                memset(&ary[0], 0, sizeof(ary));
                strcat(ary, "SPMV,");
                strcat(ary, nrowblksString[i]); //row_id
                strcat(ary, ",");
                strcat(ary, nrowblksString[j]); //col_id
                strcat(ary, ",");
                strcat(ary, nrowblksString[pseudo_tid]); //buf_id
                vertexName[strdup(ary)] = nodeCount;
                vertexWeight[nodeCount] = block_width * sizeof(double); 
                nodeCount++;
                SPMV_vertexNo[i][j] = nodeCount - 1;
                //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);
                pseudo_tid++;

                //Global Graph
                strcpy(globalGraph[globalNodeCount], ary);
                globalNodeCount++;

                // edgeU[edgeCount] = nodeCount - 2; //from A
                // edgeV[edgeCount] = nodeCount - 1;
                // edgeW[edgeCount] = matrixBlock[i * ncolblks + j].nnz * sizeof(double);
                // //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                // edgeCount++;

                edgeU[edgeCount] = Y_vertexNo[j]; //from Y
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double) + matrixBlock[i * ncolblks + j].nnz * sizeof(double);
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;

                edgeU[edgeCount] = SETZERO_id; //from SETZERO
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double);
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;
            }
        }
    }

    //SUBMAX
    

    //MAX node
    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "MAX");
    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = sizeof(double);
    nodeCount++;
    int MAX_id = nodeCount - 1;
    //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);

    pseudo_tid = 0;

    for(i = 0 ; i < nrowblks ; i++)
    {   
        //what if there is no nnz in an entire row?? --> think about it later
        pseudo_tid = ((pseudo_tid % nthreads) > (nthreads - 1) ? 0 : (pseudo_tid % nthreads) ); //pseudo_tid changes itself from 0 to 15 in a cyclic fashion
        
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SUBMAX,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[pseudo_tid]); //buf_id
        strcat(ary, ",1"); //task_id
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        SUBMAX_vertexNo[i] = nodeCount;
        nodeCount++;
        pseudo_tid++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        for(j = 0 ; j < ncolblks ; j++)
        {
            if(SPMV_vertexNo[i][j] != -1)
            {
                //SPMV --> SUBMAX
                edgeU[edgeCount] = SPMV_vertexNo[i][j]; 
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double);
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;
            }
        }
        
        //SUBMAX --> MAX
        edgeU[edgeCount] = nodeCount - 1; //from SUBMAX
        edgeV[edgeCount] = MAX_id; //to MAX
        edgeW[edgeCount] = sizeof(double);
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;
    }

    //Global Graph
    strcpy(globalGraph[globalNodeCount], "MAX");
    globalNodeCount++;

    for(i = 0 ; i < nrowblks ; i++)
    {  
        //NORM
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "NORM,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",1"); //task_id
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        //SUBMAX --> NORM
        edgeU[edgeCount] = SUBMAX_vertexNo[i]; //from MAX
        edgeV[edgeCount] = nodeCount - 1; //to NORM
        edgeW[edgeCount] = sizeof(double);
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        //MAX --> NORM
        edgeU[edgeCount] = MAX_id; //from MAX
        edgeV[edgeCount] = nodeCount - 1; //to NORM
        edgeW[edgeCount] = sizeof(double);
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        //DLACPY
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DLACPY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",1"); //task_id
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        //NORM --> DLACPY
        edgeU[edgeCount] = nodeCount - 2; //from NORM
        edgeV[edgeCount] = nodeCount - 1; //to DLACPY
        edgeW[edgeCount] = sizeof(double);
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;
    }

    printf("Total Node: %d\nTotal edges: %d\n", nodeCount, edgeCount);

    buildTaskInfoStruct_main(globalNodeCount, globalGraph , "power_mainloop", blocksize , "msdoor");



}