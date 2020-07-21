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
    int nthreads = 16;


    //memory chunk DS
    char main_task[100];
    char tmp_input1[100];
    char tmp_input2[100];
    char setzero_task[100];
    char y_task[100];
    char QQ_task[100];
    char spmv_task[100];
    char norm_task[100];
    char dlacpy_task[100];
    char submax_task[100];


    char sub_task[200];
    char dotV_task[200];
    char daxpy_task[200];
    memory_chunk temp_chunk;

    int **pseudo_tid_map;

    pseudo_tid_map = (int**)calloc(nrowblks+1,sizeof(int*));

    for(i = 0 ; i <= nrowblks ; i++)
    {
        pseudo_tid_map[i] = (int*) calloc(nrowblks + 1 , sizeof(int));
    }


    //printf("inside lanczos mainloop nrowblks = %d colblks = %d\n",nrowblks,ncolblks);


    ////initialize edgeCount and nodeCount to zero

    edgeCount = 0 ; 
    nodeCount = 0 ; 
    globalNodeCount = 0 ; 




    int *QQ_vertexNo = (int *) malloc(ncolblks * sizeof(int));

    int **SPMV_vertexNo = (int **) malloc(nrowblks * sizeof(int *));

    for(i = 0 ; i < ncolblks ; i++)
    {
        SPMV_vertexNo[i] = (int *) malloc(ncolblks * sizeof(int));
    }    
    for(i = 0 ; i < nrowblks ; i++)
    {
        QQ_vertexNo[i] = -1;
        for(j = 0 ; j < ncolblks ; j++)
            SPMV_vertexNo[i][j] = -1;
    }
    //printf("Hello world!\n");


    for(i = 0 ; i < ncolblks ; i++)
    {
        //Y
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "_QQ,");
        strcat(ary, nrowblksString[i]);
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;
        QQ_vertexNo[i] = nodeCount - 1;

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
        // strcat(ary, "_QQ,");
        // strcat(ary, nrowblksString[i]);
        // vertexName[strdup(ary)] = nodeCount;
        // vertexWeight[nodeCount] = block_width * sizeof(double);
        // nodeCount++;
        // Y_id = nodeCount - 1;
        // //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);
        // strcpy(y_task,ary);

        //SETZERO
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SETZERO,"); //block_id
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",1"); //task_id
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;
        SETZERO_id = nodeCount - 1;

        strcpy(setzero_task,ary);
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
                /*memset(&ary[0], 0, sizeof(ary));
                  strcat(ary, "_A,");
                  strcat(ary, nrowblksString[i]);
                  strcat(ary, ",");
                  strcat(ary, nrowblksString[j]);
                  vertexName[strdup(ary)] = nodeCount;
                  vertexWeight[nodeCount] = matrixBlock[i * ncolblks + j].nnz * sizeof(double); 
                  nodeCount++;*/
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
                pseudo_tid_map[i][j] = pseudo_tid-1;

                strcpy(spmv_task,ary);

                //printf("spmv task %s\n",spmv_task);

                //Global Graph
                strcpy(globalGraph[globalNodeCount], ary);
                globalNodeCount++;

                //printf("globalGraph[%d] = %s globalNodeCount %d\n",globalNodeCount-1,globalGraph[globalNodeCount-1] ,globalNodeCount);


                // edgeU[edgeCount] = nodeCount - 2; //from A
                // edgeV[edgeCount] = nodeCount - 1;
                // edgeW[edgeCount] = matrixBlock[i * ncolblks + j].nnz * sizeof(double);
                // printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                // edgeCount++;

                edgeU[edgeCount] = QQ_vertexNo[j]; //from Y[j] --> corresponding col
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double) + matrixBlock[i * ncolblks + j].nnz * sizeof(double); //adding matrix volume in this edge, similar to lobpcg
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;

                memset(&QQ_task[0], 0, sizeof(QQ_task));
                strcat(QQ_task, "_QQ,");
                strcat(QQ_task, nrowblksString[j]);

                // edge Y to SPMV
                strcpy(tmp_input1,QQ_task);
                strcpy(temp_chunk.memory_name,tmp_input1);
                temp_chunk.value = edgeW[edgeCount-1];

                ////inp_map[strdup(spmv_task)][strdup(QQ_task)] = temp_chunk;
                //out_map[strdup(QQ_task)][strdup(spmv_task)] = temp_chunk;

                //                printf("input_map[%s][%s] = %s %lf\n", spmv_task,QQ_task,tmp_input1,edgeW[edgeCount-1]);  

                edgeU[edgeCount] = SETZERO_id; //from SETZERO
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double);
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;

                // edge SETZERO to SPMV input memory name Y_temp
                strcpy(tmp_input1,"Z");
                strcat(tmp_input1,",");
                strcat(tmp_input1,nrowblksString[i]);

                strcpy(temp_chunk.memory_name,tmp_input1);
                temp_chunk.value = edgeW[edgeCount-1];

                ////inp_map[strdup(spmv_task)][strdup(setzero_task)] = temp_chunk;
                //out_map[strdup(setzero_task)][strdup(spmv_task)] = temp_chunk;

                //                 printf("input_map[%s][%s] = %s %lf\n", spmv_task,setzero_task,tmp_input1,edgeW[edgeCount-1]);      
            }
        }
    }

    //SUBMAX



    //printf("spmv nodes done\n");




    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RED,alpha,0");
    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = sizeof(double);
    nodeCount++;
    int redAlpha_id = nodeCount - 1 ; 

    //Global Graph
    //strcpy(globalGraph[globalNodeCount], ary);
    //globalNodeCount++;


    pseudo_tid = 0;



    for (i = 0 ; i < nrowblks ; i++){

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XTY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[pseudo_tid]); //buf_id
        strcat(ary,",1");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        nodeCount++;
        pseudo_tid++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;


        // QQ ---->  XTY
        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "_QQ,");
        strcat(QQ_task, nrowblksString[i]);



        //NORM --> DLACPY
        edgeU[edgeCount] = QQ_vertexNo[i]; 
        edgeV[edgeCount] = nodeCount - 1; 
        edgeW[edgeCount] = block_width * sizeof(double);
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"_QQ");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        ////inp_map[strdup(ary)][strdup(QQ_task)] = temp_chunk;
        //out_map[strdup(QQ_task)][strdup(ary)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",ary, QQ_task, tmp_input1,edgeW[edgeCount-1]);

        for(j = 0 ; j < ncolblks ; j++)
        {
            if(SPMV_vertexNo[i][j] != -1)
            {
                //SPMV --> XTY
                edgeU[edgeCount] = SPMV_vertexNo[i][j]; 
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double);
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;

                // #### Hier #####
                strcpy(spmv_task, "SPMV,");
                strcat(spmv_task, nrowblksString[i]);
                strcat(spmv_task, ",");
                strcat(spmv_task, nrowblksString[j]);
                strcat(spmv_task, ",");
                strcat(spmv_task, nrowblksString[pseudo_tid_map[i][j]]);

                //printf("\n\nspmm_task  %s\n\n",spmv_task);

                strcpy(tmp_input1,"Z");
                strcat(tmp_input1,",");
                strcat(tmp_input1,nrowblksString[i]);

                strcpy(temp_chunk.memory_name,tmp_input1);
                temp_chunk.value = edgeW[edgeCount-1];

                ////inp_map[strdup(ary)][strdup(spmv_task)] = temp_chunk;
                //out_map[strdup(spmv_task)][strdup(ary)] = temp_chunk;


                //                   printf("input_map[%s][%s] = %s %lf\n",ary,spmv_task, tmp_input1,edgeW[edgeCount-1]);                

            }   

        }



        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "RED,alpha,0");
        //strcat(QQ_task, nrowblksString[i]);



        //NORM --> DLACPY
        edgeU[edgeCount] = nodeCount - 1; 
        edgeV[edgeCount] = redAlpha_id; 
        edgeW[edgeCount] = nthreads * sizeof(double);
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"AlphaBUFF");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(QQ_task)][strdup(ary)] = temp_chunk;
        //out_map[strdup(ary)][strdup(QQ_task)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",QQ_task, ary, tmp_input1,edgeW[edgeCount-1]);



    }

    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RED,alpha,0");


    //Global Graph
    strcpy(globalGraph[globalNodeCount], ary);
    globalNodeCount++;


    //RED,QPZ node
    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RED,QpZ,0");
    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = sizeof(double);
    nodeCount++;
    int redqPz_id = nodeCount - 1;
    //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);

    //Global Graph
    //strcpy(globalGraph[globalNodeCount], ary);
    //globalNodeCount++;

    pseudo_tid = 0;

    for (i = 0 ; i < nrowblks ; i++){

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DGEMV,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[pseudo_tid]); //buf_id
        strcat(ary,",1");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        nodeCount++;
        pseudo_tid++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;


        // Q ---->  dgemv, , 1
        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "_Q,");
        strcat(QQ_task, nrowblksString[i]);




        edgeU[edgeCount] = QQ_vertexNo[i]; 
        edgeV[edgeCount] = nodeCount - 1; //to DLACPY
        edgeW[edgeCount] = block_width * sizeof(double) * 51.0;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"_Q");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(ary)][strdup(QQ_task)] = temp_chunk;
        //out_map[strdup(QQ_task)][strdup(ary)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",ary, QQ_task, tmp_input1,edgeW[edgeCount-1]);

        for(j = 0 ; j < ncolblks ; j++)
        {
            if(SPMV_vertexNo[i][j] != -1)
            {
                //SPMV --> XTY
                edgeU[edgeCount] = SPMV_vertexNo[i][j]; 
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * sizeof(double);
                //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                edgeCount++;

                // #### Hier #####
                strcpy(spmv_task, "SPMV,");
                strcat(spmv_task, nrowblksString[i]);
                strcat(spmv_task, ",");
                strcat(spmv_task, nrowblksString[j]);
                strcat(spmv_task, ",");
                strcat(spmv_task, nrowblksString[pseudo_tid_map[i][j]]);

                //printf("\n\nspmm_task  %s\n\n",spmv_task);

                strcpy(tmp_input1,"Z");
                strcat(tmp_input1,",");
                strcat(tmp_input1,nrowblksString[i]);

                strcpy(temp_chunk.memory_name,tmp_input1);
                temp_chunk.value = edgeW[edgeCount-1];

                //inp_map[strdup(ary)][strdup(spmv_task)] = temp_chunk;
                //out_map[strdup(spmv_task)][strdup(ary)] = temp_chunk;


                //             printf("input_map[%s][%s] = %s %lf\n",ary,spmv_task, tmp_input1,edgeW[edgeCount-1]);                

            }   

        }

        // DGEMV , 1 ----> RED,QpZ

        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "RED,QpZ,0");


        edgeU[edgeCount] = nodeCount - 1; 
        edgeV[edgeCount] = redqPz_id; 
        edgeW[edgeCount] = nthreads *  sizeof(double) * 51.0;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"QpZBUF");
        strcat(tmp_input1,",0");
        //strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(QQ_task)][strdup(ary)] = temp_chunk;
        //out_map[strdup(ary)][strdup(QQ_task)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",QQ_task, ary, tmp_input1,edgeW[edgeCount-1]);         


    }



    //RED,QPZ node
    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RED,QpZ,0");

    //printf("nodeName: %s nodeId: %d\n", ary, nodeCount);

    //Global Graph
    strcpy(globalGraph[globalNodeCount], ary);
    globalNodeCount++;

    memset(&norm_task[0], 0, sizeof(norm_task));
    strcat(norm_task, "NORM");
    vertexName[strdup(norm_task)] = nodeCount;
    vertexWeight[nodeCount] = sizeof(double);
    nodeCount++;
    int norm_task_id = nodeCount - 1 ; 

    //Global Graph


    pseudo_tid = 0 ; 

    for (i = 0 ; i < nrowblks ; i++){

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DGEMV,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[pseudo_tid]); //buf_id
        strcat(ary,",2");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;


        // Q ---->  dgemv, , 2
        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "_Q,");
        strcat(QQ_task, nrowblksString[i]);




        edgeU[edgeCount] = QQ_vertexNo[i]; 
        edgeV[edgeCount] = nodeCount - 1; //to DLACPY
        edgeW[edgeCount] = block_width * sizeof(double) * 51.0;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"_Q");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(ary)][strdup(QQ_task)] = temp_chunk;
        //out_map[strdup(QQ_task)][strdup(ary)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",ary, QQ_task, tmp_input1,edgeW[edgeCount-1]);

        //  RED,QpZ -----> dgemv, 2

        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "RED,QpZ,0");


        edgeU[edgeCount] = redqPz_id; 
        edgeV[edgeCount] = nodeCount - 1; 
        edgeW[edgeCount] = sizeof(double) * 51.0;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"QpZ");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(QQ_task)][strdup(ary)] = temp_chunk;
        //out_map[strdup(ary)][strdup(QQ_task)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",ary, QQ_task, tmp_input1,edgeW[edgeCount-1]);




        memset(&sub_task[0], 0, sizeof(sub_task));
        strcat(sub_task, "SUB,");
        strcat(sub_task, nrowblksString[i]);
        strcat(sub_task,",1");
        vertexName[strdup(sub_task)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        nodeCount++;


        //Global Graph
        strcpy(globalGraph[globalNodeCount], sub_task);
        globalNodeCount++;

        edgeU[edgeCount] = nodeCount - 2; 
        edgeV[edgeCount] = nodeCount - 1; 
        edgeW[edgeCount] = block_width * sizeof(double) ;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;
        //////ADD dotV node


        memset(&dotV_task[0], 0, sizeof(sub_task));
        strcat(dotV_task, "DOTV,");
        strcat(dotV_task, nrowblksString[i]);
        strcat(dotV_task, ",");
        strcat(dotV_task, nrowblksString[pseudo_tid]); //buf_id
        strcat(dotV_task,",1");
        vertexName[strdup(dotV_task)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        nodeCount++;

        pseudo_tid++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], dotV_task);
        globalNodeCount++;

        edgeU[edgeCount] = nodeCount - 2; 
        edgeV[edgeCount] = nodeCount - 1; 
        edgeW[edgeCount] = block_width * sizeof(double) ;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;




        strcpy(tmp_input1,"QQpZ");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(sub_task)][strdup(ary)] = temp_chunk;
        //out_map[strdup(ary)][strdup(sub_task)] = temp_chunk;

        //        printf("input_map[%s][%s] = %s %lf\n",sub_task, ary, tmp_input1,edgeW[edgeCount-1]);         




        edgeU[edgeCount] = nodeCount - 1; 
        edgeV[edgeCount] = norm_task_id; 
        edgeW[edgeCount] = block_width * sizeof(double) ;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"Z");
        strcat(tmp_input1,",");
        strcat(tmp_input1,nrowblksString[i]);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        //inp_map[strdup(norm_task)][strdup(dotV_task)] = temp_chunk;
        //out_map[strdup(dotV_task)][strdup(norm_task)] = temp_chunk;

        //      printf("input_map[%s][%s] = %s %lf\n",norm_task, sub_task, tmp_input1,edgeW[edgeCount-1]);         









    }

    strcpy(globalGraph[globalNodeCount], norm_task);
    globalNodeCount++;




    for (i = 0 ; i < nrowblks ; i++){

        memset(&daxpy_task[0], 0, sizeof(daxpy_task));
        strcat(daxpy_task, "DAXPY,");
        strcat(daxpy_task, nrowblksString[i]);
        strcat(daxpy_task,",1");
        vertexName[strdup(daxpy_task)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        nodeCount++;
        pseudo_tid++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], daxpy_task);
        globalNodeCount++;


        edgeU[edgeCount] = norm_task_id; 
        edgeV[edgeCount] = nodeCount - 1 ; 
        edgeW[edgeCount] = sizeof(double) ;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        strcpy(tmp_input1,"beta");


        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];


        //inp_map[strdup(daxpy_task)][strdup(norm_task)] = temp_chunk;
        //out_map[strdup(norm_task)][strdup(daxpy_task)] = temp_chunk;

        //    printf("input_map[%s][%s] = %s %lf\n",daxpy_task, norm_task, tmp_input1,edgeW[edgeCount-1]);
    }

    for (i = 0 ; i < nrowblks ; i++){

        //DLACPY
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DLACPY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary,",1");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * sizeof(double);
        nodeCount++;
        strcpy(dlacpy_task,ary);

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        edgeU[edgeCount] = nodeCount - 2 ; 
        edgeV[edgeCount] = nodeCount - 1 ; 
        edgeW[edgeCount] = block_width * sizeof(double) ;
        //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
        edgeCount++;

        memset(&QQ_task[0], 0, sizeof(QQ_task));
        strcat(QQ_task, "_QQ,");
        strcat(QQ_task, nrowblksString[i]);


        strcpy(tmp_input1,QQ_task);
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];


        //inp_map[strdup(dlacpy_task)][strdup(daxpy_task)] = temp_chunk;
        //out_map[strdup(daxpy_task)][strdup(dlacpy_task)] = temp_chunk;

        //   printf("input_map[%s][%s] = %s %lf\n",dlacpy_task, daxpy_task, tmp_input1,edgeW[edgeCount-1]);
    }

    printf("Total Node: %d\nTotal edges: %d\n", nodeCount, edgeCount);

    //buildTaskInfoStruct_main(globalNodeCount, globalGraph , "lanczos_mainloop", blocksize , "msdoor");
    buildTaskInfoStruct_main(globalNodeCount, globalGraph , "", blocksize , "");
}
