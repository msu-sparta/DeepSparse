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

//#include "new_edgeweight_try/inc/rMLGP.h"



/* gen_graph_v29.cpp using map (Each csb blcok is task): 
* 1. working on v28
* 2. Only generating DAG for the first non-loop part
*/



std::ofstream node_file("node.txt");
std::ofstream edge_file("edge.txt");


//------ global parametes (block) ---------

long position = 0 ;
int *colptrs, *irem;
int nrows, ncols, nnz, numrows, numcols, nnonzero, nthrds = 32;
int wblk, nrowblks, ncolblks, nthreads;
int *nnzPerRow;
block<double> *matrixBlock;





//typedef std::unordered_map<const char * , int, my_hash, eqstr> VertexType;

VertexType vertexName;

int *edgeU, *edgeV;
double *edgeW;
double *vertexWeight;
char **nrowblksString;
int nodeCount = 0 , edgeCount = 0;

stringstream convertToString;
/* gen timing breakdown */
int total_func = 14;
double *graphGenTime;

// ##### Hier Partitioner DS ##### //
int **pseudo_tid_map;
int small_block;

TaskDetail td;
InOutVariable iov;
task_allinout_memory all_mem_in_task;
input_map inp_map;
output_map out_map;

char global_filename[1000];





/* global graph for OpenMP scheduler */
char **globalGraph;
int globalNodeCount = 0;


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

void print_vector(vector<string> nodeName)
{
    int i;
    for(i = 0 ; i < nodeName.size() ; i++)
    {
        cout << nodeName[i] << endl;
    }
}

void print_map(VertexType mymap) //( unordered_map<string, int> mymap)
{
    std::cout << "MAP contains:" << endl;
    int count = 0, len;
    int task = 0;
    size_t maxLength = 0;
    string maxString = "";
    
    for (VertexType::iterator it = mymap.begin(); it != mymap.end(); ++it)
    {
        cout << it->first << " => " << it->second << " => " << vertexWeight[it->second]<< endl;
        len = strlen(it->first);
        if(strlen(it->first) > maxLength)
        {
            maxLength = len;
            maxString = it->first;
        }
        count++;
    }

    std::cout << "count: " << count << std::endl;
    cout << "task: " << task << endl;
    cout << "maxLength: " << maxLength << " maxString: " << maxString << endl; 

    /* bucket STAT */

    unsigned total_bucket = mymap.bucket_count();
    unsigned maxBucketSize = 0, sz;

    for (unsigned i = 0 ; i < total_bucket ; ++i) 
    {
        sz = mymap.bucket_size(i);
        //cout << "bucket " << i << " : " << sz << endl; 
        if(sz > maxBucketSize)
            maxBucketSize = sz;
    }

    cout << "Total Bucket: " << total_bucket << " maxBucketSize: " << maxBucketSize << endl;
}

int main(int argc, char *argv[])
{

	int M, N, index = 0;
    int blocksize, block_width, currentBlockSize;
    int iterationNumber = 2, maxIterations = 2;
    int i, j;

    double tstart, tend, total_time, t1, t2;

    stringstream bs(argv[3]);
    bs >> blocksize;
    stringstream bw(argv[4]);
    bw >> block_width;

    /* csb format variables */

    double *xrem;

    char *filename = argv[5];

    wblk = block_width; 
    
    read_custom<double>(filename, xrem);

    printf("Finsihed reading CUS file\n");
    //exit(1);

    csc2blkcoord<double>(matrixBlock, xrem);
    
    printf("Finsihed Converting CUS TO CSB\n");
    
    M = numrows;
    N = numcols;

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


    printf("Rows: %d, Cols: %d\n", M, N);
    printf("Block Size: %d Block Width: %d nthreads: %d nrowblks: %d ncolblks: %d\n", blocksize, block_width, nthreads, nrowblks, ncolblks);

    currentBlockSize = blocksize;

    tstart = omp_get_wtime();

    /////// DAG before loop 

    /* special node ids */
    int _lambda_id, activeMask_id, residualNorms_id, CONSTRUCTGA1_id, EIGEN_id, CONSTRUCTGA2_id;
    int CONVERGE_id, CONSTRUCTGB_id;
    int xty_id, xy_id, dlacpy_id;
    char ary[150], i_string[8];

    t1 = omp_get_wtime();

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "_X,");
        strcat(ary, nrowblksString[i]);
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
    }

    //_XTY(blockVectorX, blockVectorX, gramXBX, M, blocksize, blocksize, block_width);
    xty_id = 20;
    _XTY(1, "_X", "null", "null", "null", "null", -1,
        1, "_X", "null", "null", "null", "null", -1,
        "_X", "_X", "XBX", M, blocksize, blocksize, block_width, xty_id);

    //CHOL,XBX
    vertexName[strdup("CHOL,XBX")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    //Global Graph
    strcpy(globalGraph[globalNodeCount], strdup("CHOL,XBX"));
    globalNodeCount++;

    edgeU[edgeCount] = vertexName["RED,XBXBUF,0"];
    edgeV[edgeCount] = nodeCount - 1; 
    edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
    edgeCount++;
    
    vertexName[strdup("DLACPY,0,20")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    //Global Graph
    strcpy(globalGraph[globalNodeCount], strdup("DLACPY,0,20"));
    globalNodeCount++;

    edgeU[edgeCount] = nodeCount - 2;
    edgeV[edgeCount] = nodeCount - 1; 
    edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
    edgeCount++;

    vertexName[strdup("INV,XBX")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    //Global Graph
    strcpy(globalGraph[globalNodeCount], strdup("INV,XBX"));
    globalNodeCount++;

    edgeU[edgeCount] = nodeCount - 2;
    edgeV[edgeCount] = nodeCount - 1; 
    edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
    edgeCount++;

    

    xy_id = 20;
    _XY(1, "_X", "null", "null", "null", "null", -1,
        1, "INV,XBX", "null", "null", "null", "null", -1,
        "_X", "INV(XBX)", "X", M, blocksize, blocksize, block_width, xy_id);

    dlacpy_id = 21;

    int *X_vertexNo = (int *) malloc(nrowblks * sizeof(int));

    custom_dlacpy_v1(2, "null", "X", "XY", "X", "INV(XBX)", xy_id,
                "newX", "X", M, blocksize, block_width, dlacpy_id, -1 , X_vertexNo); // -1 is the taskwait_node_no dummy

    int **SPMM_non_loop = (int **) malloc(nrowblks * sizeof(int *));
    for(i = 0 ; i < ncolblks ; i++)
        SPMM_non_loop[i] = (int *) malloc(ncolblks * sizeof(int));
        
    for(i = 0 ; i < nrowblks ; i++)
        for(j = 0 ; j < ncolblks ; j++)
            SPMM_non_loop[i][j] = -1;
    
    spmm_blkcoord_csbTask(2, "null", "AX", "SETZERO", "AX",
                         "A", "X", "AX", M, N, blocksize, block_width, currentBlockSize, -1, X_vertexNo, SPMM_non_loop); // -1 is the taskwait_node_no dummy


    //new -> takes task id for 2nd edge but not for SPMM
    xty_id = 21;
    _XTY_v3(2, "null", "AX", "SPMMRED", "SPMMBUF", 
            2, "null", "newX", "DLACPY", "X", 21,
            "X", "AX", "XAX", M, blocksize, blocksize, block_width, xty_id, SPMM_non_loop);

    vertexName[strdup("SPEUPDATE,XAX")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    //Global Graph
    strcpy(globalGraph[globalNodeCount], strdup("SPEUPDATE,XAX"));
    globalNodeCount++;

    edgeU[edgeCount] = vertexName["RED,XAXBUF,0"];
    edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"]; -> last insert
    edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
    edgeCount++;

    vertexName[strdup("EIGEN")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize  * sizeof(double);
    EIGEN_id = nodeCount;
    nodeCount++;

    //Global Graph
    strcpy(globalGraph[globalNodeCount], strdup("EIGEN"));
    globalNodeCount++;

    edgeU[edgeCount] = vertexName["SPEUPDATE,XAX"];
    edgeV[edgeCount] = nodeCount - 1;
    edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
    edgeCount++;


    //xy 21
    xy_id = 21;
    _XY_v1(2, "null", "newX", "DLACPY", "X", 21,
            1, "EIGEN", "null", "null", "null", "null",
            "X", "EIGEN", "X", M, blocksize, blocksize, block_width, xy_id);

    dlacpy_id = 22;
    custom_dlacpy(2, "null", "X", "XY", "EIGEN", "X", xy_id,
                 "newX", "X", M, currentBlockSize, block_width, dlacpy_id);

    xy_id = 22;
    _XY_v2(2, "null", "X", "SPMMRED", "SPMMBUF", 
            "EIGEN",
            "AX", "EIGEN", "AX", M, blocksize, blocksize, block_width, xy_id, SPMM_non_loop);

    dlacpy_id = 23;
    custom_dlacpy(2, "null", "AX", "XY", "EIGEN", "AX", xy_id,
                 "newAX", "AX", M, currentBlockSize, block_width, dlacpy_id);

    tend = omp_get_wtime();
    total_time = tend - tstart;

    cout << "Done!!!!" << endl;
    cout << "Total time: " << total_time << " sec." << endl; 
    cout << "Timing break down: " << endl;

    char function_name[][20] = {"XTY", "XY", "SPMM", "ADD", "SUB", "MULT", "GET", "UPDATE", "SUM_SQRT", "DLACPY", "findIndex", "for loops", "main push_back", "nested loop"};
    for(i = 0 ; i < total_func; i++)
    {
        cout << right;
        cout << setw(15) << function_name[i] << " : " << setw(10) << graphGenTime[i] << setw(4) << " sec." << endl;
    }
    cout << endl;

    cout << "Node Count: " << nodeCount << endl;
    cout << "Edge Count: " << edgeCount << endl;
    cout << "map size: " << vertexName.size() << endl; 

    cout << "globalNodeCount: " << globalNodeCount << endl;

    std::ofstream globalgraph_file("global_graph_nonloop.txt");
    for(i = 0 ; i < globalNodeCount; i++)
    {
        globalgraph_file << globalGraph[i] << " " << 0 << endl;
        //printf("%d => %s\n", (i + 1), globalGraph[i]);
    }



    ///counting the number of outgoing edges ////////////////
    int *out_edges = (int*)calloc(globalNodeCount , sizeof(int));
    long int tot_out_edge = 0;
    for(i = 0 ; i < edgeCount ; i++){
	out_edges[edgeU[i]]++;
 //   	printf("(%d)%s --> (%d)%s\n",edgeU[i],globalGraph[edgeU[i]],edgeV[i],globalGraph[edgeV[i]]);

    }

    FILE *global_out_edge_file = fopen("global_out_edge_file.txt","w");
    for(i = 0 ; i < globalNodeCount ; i++){
    	fprintf(global_out_edge_file,"(%d)%s --> %d\n",i+1,globalGraph[i],out_edges[i] );
	tot_out_edge += out_edges[i];
    }

	fclose(global_out_edge_file);

    printf("\ntotal_outgoing edge = %ld\n",tot_out_edge);


    printf("Finish writing global graph\n");
    
    exit(1);
    
    //writing graph in dot format in file

    /*std::ofstream graph_file("MatA100-graph-v29-nonloop.dot");
    graph_file << "digraph G {" << endl;
    for(i = 0 ; i < nodeCount ; i++)
    {
        string name= "";
        
        for (auto it = vertexName.begin(); it != vertexName.end(); ++it )
            if (it->second == i)
                name = it->first;

        if(name != "")
        {
            std::size_t found = name.find("SPMM");
            
            if (found != std::string::npos)
                graph_file << i + 1 << " [label=\"" << name << "\", color=deepskyblue, style = filled];" << endl;  
            else  
                graph_file << i + 1 << " [label=\"" << name << "\", color=salmon2, style = filled];" << endl;
        }
        //else
        //{
        //    graph_file << i + 1 << " [label=" << name << ", color=salmon2, style = filled];" << endl;
        //}
    }
    for(i = 0 ; i < edgeCount ; i++)
    {
        graph_file << edgeU[i] + 1 << " ->"<< edgeV[i] + 1 << " ;" << endl;
    }

    graph_file << "}" << endl; 

    printf("Finish writig dot file\n");*/

    //exit(1);


    char** vertex_name_string;
    vertex_name_string = (char **) malloc((nodeCount + 1) * sizeof(char *));
    for (i = 0 ; i <= nodeCount ; ++i)
    {
        vertex_name_string[i] = (char*) malloc(100 * sizeof(char));
    }




    //checking split funtion
    char **splitParams;
    int paramCount;
    for (std::pair<const char*, int> element : vertexName)
    {
        // std::cout << element.first << " :: " << element.second << std::endl;
        //  printf("%d = %s \n",element.second,element.first);
        //  element.first.copy(vertex_name_string[element.second],element.first.length());
        //  vertex_name_string[element.second][element.first.length()] = '\0';
        //  printf("i = %d %s\n",element.second,vertex_name_string[element.second]);
        strcpy(vertex_name_string[element.second], element.first);
        
        //checking taskName
        //paramCount = split(vertex_name_string[element.second], ',', &splitParams); 
        //for(i = 0 ; i < paramCount ; i++)
        //    printf("%s  ", splitParams[i]);
            //printf("%s      (%d) ==> %s     (%d)\n", vertex_name_string[element.second], strlen(vertex_name_string[element.second]), element.first, strlen(element.first));
        //printf("\n");
    }
/*
    MLGP_option opt;
    
    processArgs_rMLGP(argc, argv, &opt);

    printf("Calling run_rMLGP from main\n");
    
    run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, &vertex_name_string[0], vertexWeight, numrows, numcols, nrowblks, ncolblks, block_width);

    printf("run_rMLGP Finshed\n");
    //run_rMLGP(opt.file_name, opt);
    
    free_opt(&opt);
*/
	return 0;
}

void mat_addition(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int block_width, int add_id)
{
    /* funciton code: 3 */

    double tstart, tend;
    tstart = omp_get_wtime();

    int i, edge2_id;
    char i_string[8], task_id1_char[4], task_id2_char[4], add_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    myitoa(add_id, add_id_char);

    if(edge2Format == 1) // edge coming from a single matrix 
    {
        edge2_id = vertexName[edge2_var];
    }

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "ADD,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, add_id_char);
       
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] =  block_width * col * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
        

        if(edge1Format == 1) //edge coming from a single matrix 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;

        }

        if(edge2Format == 1) // edge coming from a single matrix 
        {
            edgeU[edgeCount] = edge2_id;
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else // coming from another operation of format : func_(inp1, inp2, blk) 
        {   
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
    }

    tend = omp_get_wtime();
    graphGenTime[3] += (tend - tstart);
}

void _XY(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int M, int N, int P, int block_width, int xy_id)
{
    //cout << "in _XY" << endl;
    /**********************************************
    Input: X[M*N], Y[N*P]
    Output: result[M*P]
    nthrds : global variable, total # of threads
    ***********************************************/
    
    /* funciton code: 1 */

    double tstart, tend;
    tstart = omp_get_wtime();

    int i, edge2_id;
    char i_string[8], task_id1_char[4], task_id2_char[4], xy_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    myitoa(xy_id, xy_id_char);

    if(edge2Format == 1) 
    {
        edge2_id = vertexName[edge2_var];
    }

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, xy_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * P * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
     
        if(edge1Format == 1)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++;
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++;
        }

        if(edge2Format == 1) //whole edge2_var
        {   
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++; 
        }
        else
        {   
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
               
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++; 
        }
    }

    tend = omp_get_wtime();
    graphGenTime[1] += (tend - tstart);
}

void _XY_v1(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], int task_id,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], 
            char input1[], char input2[], char output[], int M, int N, int P, int block_width, int xy_id)
{
    //cout << "in _XY_v1" << endl;

    /**********************************************
    Input: X[M*N], Y[N*P]
    Output: result[M*P]
    nthrds : global variable, total # of threads
    ***********************************************/
   

    /* funciton code: 1 */
    
    double tstart, tend;
    tstart = omp_get_wtime();

    int i, edge2_id;
    char i_string[8], task_id1_char[4], xy_id_char[4];
    char ary[150];
        
    myitoa(task_id, task_id1_char);
    myitoa(xy_id, xy_id_char);

    if(edge2Format == 1) 
    {
        edge2_id = vertexName[edge2_var];
    }

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, xy_id_char);
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * P * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 1)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++;
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
         
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++;       
        }

        if(edge2Format == 1) //whole edge2_var
        {
            edgeU[edgeCount] = edge2_id;
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = N * P * sizeof(double);
            edgeCount++;  
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = N * P * sizeof(double);
            edgeCount++;      
        }
    }

    tend = omp_get_wtime();
    graphGenTime[1] += (tend - tstart);
}

void _XY_v2(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            char edge2_var[],
            char input1[], char input2[], char output[], int M, int N, int P, int block_width, int xy_id, int **SPMM_vertexNo)
{
    /**********************************************
    Input: X[M*N], Y[N*P]
    Output: result[M*P]
    nthrds : global variable, total # of threads
    ***********************************************/

    /* funciton code: 1 */
    
    double tstart, tend;
    tstart = omp_get_wtime();

    int i, l;
    char i_string[8], xy_id_char[4];
    char ary[150];
        
    myitoa(xy_id, xy_id_char);

    int edge2_id = vertexName[edge2_var]; 

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XY,");
        strcat(ary, nrowblksString[i]); //row_id
        strcat(ary, ",");
        strcat(ary, xy_id_char); //xy_id

        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * P * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 1)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
         
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++;

             

        }
        else
        {
            /*memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++;*/ 

            for(l = 0; l < ncolblks; l++) //all SPMM CSB block task of row i to xy task
            {
                if(SPMM_vertexNo[i][l] != -1)
                {
                    edgeU[edgeCount] = SPMM_vertexNo[i][l];
                    edgeV[edgeCount] = nodeCount - 1;
                    //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                    edgeW[edgeCount] = block_width * N * sizeof(double);
                    edgeCount++;
                }
            }      
        }

        //whole edge2_var, EIGEN
        edgeU[edgeCount] = edge2_id;
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = N * P * sizeof(double);
        edgeCount++;          
    }

    tend = omp_get_wtime();
    graphGenTime[1] += (tend - tstart);
}


void _XTY(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id)
{
    /*********************************************************
    _XTY_v1: adding partial sums block by block, not row by row
    Input: X[row*col], Y[row*p]
    Output: result[col*P]
    nthrds : global variable, total # of threads
    buf : how to free/deallocate corresponding memory location
    blocksize: each chunk
    **********************************************************/

    /* funciton code: 0 */
    
    double tstart, tend;
    tstart = omp_get_wtime(); 

    int i, j, k, l;
    int nbuf = 16;
    int pseudo_tid, max_pesudo_tid = -1;

    char i_string[8], j_string[8], l_string[8], k_string[4], task_id1_char[4], task_id2_char[4], xty_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    myitoa(xty_id, xty_id_char);

    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);
        vertexName[strdup(ary)] = nodeCount;

        if((j + block_width) > col)
            vertexWeight[nodeCount] = (col - j) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double); 
        nodeCount++;
    }

    for(i = 0, k = 0 ; i < nrowblks ; i++, k++)
    {
        pseudo_tid = ((i % nbuf) > (nthreads - 1) ? 0 : (i % nbuf) );
        k = pseudo_tid;
        max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );
        
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XTY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[k]);
        strcat(ary, ",");
        strcat(ary, xty_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        if( (i * block_width + block_width) > row)
            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 2) //coming from another task
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;        
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }

        if(edge2Format == 2)
        {   
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;      
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "RED,");
            strcat(ary, output);
            strcat(ary, "BUF,");
            strcat(ary, nrowblksString[l]);
            
            edgeU[edgeCount] = nodeCount - 1;
            edgeV[edgeCount] = vertexName[ary];
            edgeW[edgeCount] = vertexWeight[nodeCount]; //=> changed //sizeof(double); 
            edgeCount++;
        }
    }

    //For global graph only 
    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
    }

    tend = omp_get_wtime();
    graphGenTime[0] += (tend - tstart);

    
}

void _XTY_v1(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id)
{
    /*********************************************************
    _XTY_v1: adding partial sums block by block, not row by row
    Input: X[row*col], Y[col*p]
    Output: result[col*P]
    nthrds : global variable, total # of threads
    buf : how to free/deallocate corresponding memory location
    blocksize: each chunk
    **********************************************************/
    //cout << "_XTY_v1 : row: " << row << " col : " << col << " p: " << p << " id: " << xty_id << endl; 
    double tstart, tend;
    tstart = omp_get_wtime();
    
    int i, j, k, l;
    int nbuf = 16;
    int pseudo_tid, max_pesudo_tid = -1;

    char i_string[8], j_string[8], l_string[8], k_string[4], task_id1_char[4], task_id2_char[4], xty_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    myitoa(xty_id, xty_id_char);

    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {   
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);

        vertexName[strdup(ary)] = nodeCount;
        if((j + block_width) > col)
            vertexWeight[nodeCount] = (col - j) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;
    }

    for(i = 0, k = 0 ; i < nrowblks ; i++, k++)
    {
        pseudo_tid = ((i % nbuf) > (nthreads - 1) ? 0 : (i % nbuf) );
        k = pseudo_tid;
        max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );
       
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XTY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[k]);
        strcat(ary, ",");
        strcat(ary, xty_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        if( (i * block_width + block_width) > row)
        {

            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
            //cout << "Hi " << i * block_width + block_width << " " << row << " " << vertexWeight[nodeCount] << endl;
        }
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;


        if(edge1Format == 2) //coming from another task
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }

        if(edge2Format == 2)
        {   
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else
        {   
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }

        for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "RED,");
            strcat(ary, output);
            strcat(ary, "BUF,");
            strcat(ary, nrowblksString[l]);
            
            edgeU[edgeCount] = nodeCount - 1;
            edgeV[edgeCount] = vertexName[ary];
            edgeW[edgeCount] = vertexWeight[nodeCount]; //sizeof(double);
            edgeCount++;
        }
    }

    //For global graph only 
    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
    }

    tend = omp_get_wtime();
    graphGenTime[0] += (tend - tstart);
}

void _XTY_v2(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id)
{
    /* funciton code: 0 */
    
    double tstart, tend;
    tstart = omp_get_wtime();
    
    int i, j, k, l;
    int nbuf = 16;
    int pseudo_tid, max_pesudo_tid = -1;

    char i_string[8], j_string[8], l_string[8], k_string[4], task_id1_char[4], task_id2_char[4], xty_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    myitoa(xty_id, xty_id_char);

    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);
        
        vertexName[strdup(ary)] = nodeCount;
        if((j + block_width) > col)
            vertexWeight[nodeCount] = (col - j) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;
    }

    for(i = 0, k = 0 ; i < nrowblks ; i++, k++)
    {
        pseudo_tid = ((i % nbuf) > (nthreads - 1) ? 0 : (i % nbuf) );
        k = pseudo_tid;
        max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XTY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[k]);
        strcat(ary, ",");
        strcat(ary, xty_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        if( (i * block_width + block_width) > row)
            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 2) //coming from another task
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;      
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }

        if(edge2Format == 2)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
         
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "RED,");
            strcat(ary, output);
            strcat(ary, "BUF,");
            strcat(ary, nrowblksString[l]);
         
            edgeU[edgeCount] = nodeCount - 1;
            edgeV[edgeCount] = vertexName[ary];
            edgeW[edgeCount] = vertexWeight[nodeCount]; //sizeof(double);
            edgeCount++;
        }
    }

    //For global graph only 
    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
    }

    tend = omp_get_wtime();
    graphGenTime[0] += (tend - tstart);   
}



void _XTY_v3(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id, int **SPMM_vertexNo)
{
    double tstart, tend;
    tstart = omp_get_wtime();
    
    int i, j, k, l;
    int nbuf = 16;
    int pseudo_tid, max_pesudo_tid = -1;

    char i_string[8], j_string[8], l_string[8], k_string[4], task_id2_char[4], xty_id_char[4];
    char ary[150];
        
    myitoa(task_id_2, task_id2_char);
    myitoa(xty_id, xty_id_char);

    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);
       
        vertexName[strdup(ary)] = nodeCount;
        if((j + block_width) > col)
            vertexWeight[nodeCount] = (col - j) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;
    }

    for(i = 0, k = 0 ; i < nrowblks ; i++, k++)
    {
        pseudo_tid = ((i % nbuf) > (nthreads - 1) ? 0 : (i % nbuf) );
        k = pseudo_tid;
        max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );
        
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XTY,");
        strcat(ary, nrowblksString[i]); //row_id
        strcat(ary, ",");
        strcat(ary, nrowblksString[k]); //buf_id
        strcat(ary, ","); 
        strcat(ary, xty_id_char); //xty_id
        
        vertexName[strdup(ary)] = nodeCount;
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if( (i * block_width + block_width) > row)
            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        

        if(edge1Format == 2) //coming from another task, SPMM
        {   
            /*memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]); 

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;*/

            //edge from all CSB block task in row i, to this task

            for(l = 0; l < ncolblks; l++)
            {
                if(SPMM_vertexNo[i][l] != -1)
                {
                    edgeU[edgeCount] = SPMM_vertexNo[i][l];
                    edgeV[edgeCount] = nodeCount - 1;
                    edgeW[edgeCount] = block_width * col * sizeof(double);
                    //printf("%d %d\n", edgeU[edgeCount], edgeV[edgeCount]);
                    edgeCount++;

                }
            }      
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }

        if(edge2Format == 2)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "RED,");
            strcat(ary, output);
            strcat(ary, "BUF,");
            strcat(ary, nrowblksString[l]);

            edgeU[edgeCount] = nodeCount - 1;
            edgeV[edgeCount] = vertexName[ary];
            edgeW[edgeCount] = vertexWeight[nodeCount]; //sizeof(double);
            edgeCount++;
        }
    }

    //For global graph only 
    for(j = 0, l = 0 ; j < col ; j = j + block_width, l++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,");
        strcat(ary, output);
        strcat(ary, "BUF,");
        strcat(ary, nrowblksString[l]);

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
    }

    tend = omp_get_wtime();
    graphGenTime[0] += (tend - tstart);
}

void strrev(char *input, int length) 
{
    char tmp;
    //int length = strlen(input);
    int last_pos = length - 1;
    for(int i = 0; i < length/2; i++)
    {
        tmp = input[i];
        input[i] = input[last_pos - i];
        input[last_pos - i] = tmp;
    }

    //printf("%s\n", input);
}

void string_rev(char *p)
{
   char *q = p;

   if(q == NULL)
       return;

   while(*q) ++q;
   for(--q; p < q; ++p, --q)
       *p = *p ^ *q,
       *q = *p ^ *q,
       *p = *p ^ *q;

}  

void str_reverse_in_place(char *str, int len)
{
    char *p1 = str;
    char *p2 = str + len - 1;

    while (p1 < p2) {
        char tmp = *p1;
        *p1++ = *p2;
        *p2-- = tmp;
    }

    //return str;
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
    //return str;
}

void myitoa(int x, char* dest) 
{
    //int i = (int) log10((double) x);
    int i = 0;
    if(x == 0)
    {
        dest[0] = '0';
        dest[1] = '\0';
    }
    else
    {
        while(x > 0) 
        {
            dest[i] = (x % 10) + '0';
            x = x / 10;
            //i = i - 1;
            i++;
        }
        dest[i] = '\0';
    }
    //strrev(dest, i+1);
    //string_rev(dest);
    str_rev(dest);
}

void spmm_blkcoord_csbTask(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int currentBlockSize, int taskwait_node_no, int *actR_vertexNo, int **SPMM_vertexNo)
{
    int i, j, k;
    int nbuf = 1; // how many partial SPMM results?
    //int nnz = 20; //nnz in each csb block, for time-being it is set to 20
    int pseudo_tid = 0, max_pesudo_tid = 0;
    char ary[150];

    double tstart, tend, t1, t2, fetch_time = 0, insert_time = 0, sprintf_time = 0, conversion_time = 0 ;
    int total_insert = 0 , total_fetch = 0;
    char i_string[8], j_string[8], k_string[4];
    
    int buf_setzero_node_no = -1;
    int buf_reduction_node_no = -1;
    int offset, modulus;

    tstart = omp_get_wtime();

    for(i = 0 ; i < nrowblks ; i++)
    {
        //t1 = omp_get_wtime();   
     
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SETZERO,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",1");
     
        //t2 = omp_get_wtime();
        //sprintf_time += (t2 - t1);
        //t1 = omp_get_wtime();

        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        buf_setzero_node_no = nodeCount; //saving it to use later in nested looop
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        //t2 = omp_get_wtime();
        //insert_time += (t2 - t1);
        //total_insert++;
        

        //t1 = omp_get_wtime();
        
        //memset(&ary[0], 0, sizeof(ary));
        //strcat(ary, "SPMMRED,");
        //strcat(ary, nrowblksString[i]);
        
        //t2 = omp_get_wtime();
        //sprintf_time += (t2 - t1);

        //t1 = omp_get_wtime();

        //vertexName[strdup(ary)] = nodeCount;
        //vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        //buf_reduction_node_no = nodeCount;
        //nodeCount++; 

        //t2 = omp_get_wtime();
        //insert_time += (t2 - t1);
        //total_insert++;
       
        

        for(j = 0 ; j < ncolblks ; j++)
        {
            if(matrixBlock[i * ncolblks + j].nnz > 0)
            {
                pseudo_tid = ( (((i * ncolblks) + j) % nbuf) > (nthreads - 1) ? 0 : ( ((i * ncolblks) + j) % nbuf ) );
                k = pseudo_tid;
                max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );
                
                //t1 = omp_get_wtime();
                
                memset(&ary[0], 0, sizeof(ary));
                strcat(ary, "SPMM,");
                strcat(ary, nrowblksString[i]);
                strcat(ary, ",");
                strcat(ary, nrowblksString[j]);
                strcat(ary, ",");
                strcat(ary, nrowblksString[k]);
                //strcat(ary, ")");

                //t2 = omp_get_wtime();
                //sprintf_time += (t2 - t1);

                //t1 = omp_get_wtime();

                vertexName[strdup(ary)] = nodeCount;
                vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
                SPMM_vertexNo[i][j] = nodeCount; //saving SPMM node number for later use
                nodeCount++;

                //Global Graph
                strcpy(globalGraph[globalNodeCount], ary);
                globalNodeCount++;

                //t2 = omp_get_wtime();
                //insert_time += (t2 - t1);
                //total_insert++;
                

                //this will come from actR  
                edgeU[edgeCount] = actR_vertexNo[j]; //SPMM(r,c) requires actR(c) 
                edgeV[edgeCount] = nodeCount - 1;
                edgeW[edgeCount] = block_width * currentBlockSize * sizeof(double) + matrixBlock[i * ncolblks + j].nnz * sizeof(double); //instead of adding _A(i,j) can we add the weight of _A(i,j) to actR(j) weight? 
                edgeCount++;

                //SETZERO AR to SPMM
                edgeU[edgeCount] = buf_setzero_node_no; 
                edgeV[edgeCount] = nodeCount - 1; 
                edgeW[edgeCount] = block_width * p * sizeof(double); //single block of a particular partial buffer
                edgeCount++;

                //SPMM to SPMMRED
                //edgeU[edgeCount] = nodeCount - 1; 
                //edgeV[edgeCount] = buf_reduction_node_no;
                //edgeW[edgeCount] = block_width * p * sizeof(double); //what should be the proper weight of the edges going out from SPMM task? should be on nnz of that block???
                //edgeCount++;
            }
        }
        //SETZERO(actAR) to SPMMRED
        //edgeU[edgeCount] = SETZERO_SPMM_OUTPUT_vertexNo[i]; 
        //edgeV[edgeCount] = buf_reduction_node_no;
        //edgeW[edgeCount] = block_width * p * sizeof(double);
        //edgeCount++;
    }

    tend = omp_get_wtime();
    graphGenTime[2] += (tend - tstart);

    //cout << "SPMM fetch time: " << fetch_time << " sec. " << "total_fetch: " << total_fetch << endl;
    //cout << "SPMM insert time: " << insert_time << " sec. " << "total_insert: " << total_insert << endl;
    ///cout << "SPMM sprintf time: " << sprintf_time << " sec." << endl;
    //cout << "Conversion time: " << conversion_time << " sec." << endl;
}

void custom_dlacpy(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id,
            char input1[], char output[], int row, int col, int block_width, int dlacpy_id)
{
    /* funciton code: 9 */
    
    double tstart, tend;
    tstart = omp_get_wtime();
    
    int i;

    char i_string[8], task_id1_char[4], dlacpy_id_char[4];
    char ary[150];
        
    myitoa(task_id, task_id1_char);
    myitoa(dlacpy_id, dlacpy_id_char);

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DLACPY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, dlacpy_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;

         //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;


        if(edge1Format == 1) //edge coming from a single matrix 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;

        }
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
    }

    tend = omp_get_wtime();
    graphGenTime[9] += (tend - tstart);
}

//called only once before SPMM
void custom_dlacpy_v1(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id,
            char input1[], char output[], int row, int col, int block_width, int dlacpy_id, int taskwait_node_no, int *actR_vertexNo)
{
    /* funciton code: 9 */
    
    double tstart, tend;
    tstart = omp_get_wtime();

    int i;
    
    char i_string[8], task_id1_char[4], dlacpy_id_char[4];
    char ary[150];
        
    myitoa(task_id, task_id1_char);
    myitoa(dlacpy_id, dlacpy_id_char);

    for(i = 0 ; i < nrowblks ; i++)
    {       
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DLACPY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, dlacpy_id_char);
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        actR_vertexNo[i] = nodeCount; //saving it for SPMM tasks
        nodeCount++;

         //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 1) //edge coming from a single matrix 
        {       
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);   
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;

        }
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
    }

    tend = omp_get_wtime();
    graphGenTime[9] += (tend - tstart);
}






void getActiveBlockVector(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            char edge2[], int edge2_id,
            char input1[], char input2[], char output[], int row, int col, int currentBlockSize, int block_width, int get_id ) //edge2 is activeMask node
{
    /* funciton code: 6 */
    
    double tstart, tend;
    tstart = omp_get_wtime();

    int i;// edge2_id;
    char i_string[8], task_id1_char[4], get_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(get_id, get_id_char);

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "GET,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, get_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        nodeCount++;
         //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 1) //edge coming from a single matrix
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
         
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;       
        }
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;       
        }

        edgeU[edgeCount] = edge2_id;
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = block_width * col * sizeof(double);
        edgeCount++;   
    }

    tend = omp_get_wtime();
    graphGenTime[6] += (tend - tstart);
}

void updateBlockVector(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], int task_id,
            char edge2[], int edge2_id,
            char input1[], char input2[], char output[], int row, int col, int currentBlockSize, int block_width, int update_id) //edge2 is activeMask node
{ 
    /* funciton code: 7 */
    
    double tstart, tend;
    tstart = omp_get_wtime();
    
    int i;
    char i_string[8], task_id1_char[4], update_id_char[4];
    char ary[150];
        
    myitoa(task_id, task_id1_char);
    myitoa(update_id, update_id_char);
    
    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "UPDATE,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, update_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        nodeCount++;

         //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        if(edge1Format == 1) //edge coming from a single matrix 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;       
        }
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }

        edgeU[edgeCount] = edge2_id; 
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = block_width * col * sizeof(double);
        edgeCount++; 
    }

    tend = omp_get_wtime();
    graphGenTime[7] += (tend - tstart);
}


void mat_sub(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
			int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
			char input1[], char input2[], char output[], int row, int col, int block_width, int sub_id)
{
    /**********************************************
    Input: X[M*N], Y[N*P]
    Output: result[M*P]
    nthrds : global variable, total # of threads
    ***********************************************/

    /* funciton code: 4 */
    
    double tstart, tend;
    tstart = omp_get_wtime();

    int i;
    char i_string[8], task_id1_char[4], task_id2_char[4], sub_id_char[4];
    char ary[150];
        
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    myitoa(sub_id, sub_id_char);

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SUB,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, sub_id_char);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;

         //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

    	if(edge1Format == 1) //edge coming from a single matrix 
    	{
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
              
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;		
    	}
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            //strcat(ary, ")");

            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;	
        }

        if(edge2Format == 1) // edge coming from a single matrix 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
              
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        else // coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;
        }
        
    }

    tend = omp_get_wtime();
    graphGenTime[4] += (tend - tstart);
}



void mat_mult(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
			int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
			char input1[], char input2[], char output[], int row, int col, int block_width)
{
    /* funciton code: 5 */
    
    string dummyString = "";
    double tstart, tend;
    int i, edge2_id;
    char i_string[8], task_id1_char[4], task_id2_char[4];
    char ary[150];
       
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);
    
    tstart = omp_get_wtime();
    if(edge2Format == 1) // edge coming from a single matrix 
    {
        edge2_id = vertexName[edge2_var]; 
    }
    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "MULT,");
        strcat(ary, nrowblksString[i]);
        
        vertexName[strdup(ary)] = nodeCount; 
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

    	if(edge1Format == 1) //edge coming from a single matrix 
    	{
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_var);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;		
    	}
        else //coming from another operation of format : func_(inp1, inp2, blk) 
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge1_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id1_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;		
        }

        if(edge2Format == 1) // edge coming from a single matrix 
        {
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;		
        }
        else // coming from another operation of format : func_(inp1, inp2, blk)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;		
        }
    }

    tend = omp_get_wtime();
    graphGenTime[5] += (tend - tstart);
}

void sum_sqrt(char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[],
            char edge2[], int edge2_id,
            char input1[], char output[], int row, int col, int block_width)
{
    /* funciton code: 3 */
    double tstart, tend;
    tstart = omp_get_wtime();

    int i, k;
    int nbuf = 16;
    int pseudo_tid, max_pesudo_tid = -1;

    char *tempRNRED = (char *) malloc(50 * sizeof(char));
    char *tempSQRT = (char *) malloc(50 * sizeof(char));

    char i_string[8], k_string[4], task_id2_char[4];
    char ary[150];

    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RNRED,");
    strcat(ary, "RNBUF");
    
    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = col * sizeof(double);
    int sumsqrt_buf_id = nodeCount;
    nodeCount++;

    //Global Graph
    strcpy(tempRNRED, ary);

    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "SQRT,");
    strcat(ary, output);

    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = col * sizeof(double);
    int sqrt_id = nodeCount;
    nodeCount++;

    //Global Graph
    strcpy(tempSQRT, ary);

    for(i = 0, k = 0 ; i < nrowblks ; i++, k++)
    {
        pseudo_tid = ((i % nbuf) > (nthreads - 1) ? 0 : (i % nbuf) );
        k = pseudo_tid;
        max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );
        
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "COL,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[k]);

        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = col * sizeof(double);
        nodeCount++;

        //Global Graph
        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, edge1_func);
        strcat(ary, ",");
        strcat(ary, nrowblksString[i]);
        
        edgeU[edgeCount] = vertexName[ary];
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = block_width * col * sizeof(double);
        edgeCount++;
        
        //edge 2

        edgeU[edgeCount] = nodeCount - 1;
        edgeV[edgeCount] = sumsqrt_buf_id; 
        edgeW[edgeCount] = col * sizeof(double);
        edgeCount++;
    }

    edgeU[edgeCount] = edge2_id; 
    edgeV[edgeCount] = sumsqrt_buf_id; 
    edgeW[edgeCount] = col * sizeof(double);
    edgeCount++;

    edgeU[edgeCount] = sumsqrt_buf_id; 
    edgeV[edgeCount] = sqrt_id;
    edgeW[edgeCount] = col * sizeof(double);
    edgeCount++;

    //Global Graph
    strcpy(globalGraph[globalNodeCount], tempRNRED);
    globalNodeCount++;
    strcpy(globalGraph[globalNodeCount], tempSQRT);
    globalNodeCount++;

    free(tempRNRED);
    free(tempSQRT);

    tend = omp_get_wtime();
    graphGenTime[8] += (tend - tstart);
}

template<typename T>
void read_custom(char* filename, T *&xrem)
{
 /*   int i, j;
    ifstream file (filename, ios::in|ios::binary);
    if (file.is_open())
    {
        //file.seekg(0, ios::end); // set the pointer to the end
        //long size = file.tellg() ; // get the length of the file
        //cout << "Size of file: " << size << endl;
        //file.seekg(0, ios::beg); // set the pointer to the beginning

        unsigned char MyBytes[4];  //set values to this also.
        

        int a = 0;
        float d = 0;
        unsigned int b = 0;

        file.read ((char*)&numrows, sizeof(numrows));
        cout << "numrows: " << numrows << endl;
        
        file.read(reinterpret_cast<char*>(&numcols), sizeof(numcols));
        cout << "numcols: " << numcols << endl;

        file.read(reinterpret_cast<char*>(&nnonzero), sizeof(nnonzero));
        cout << "nnonzero: " << nnonzero << endl;

        //cout << "sizeof(unsigned int): " << sizeof(unsigned int) << " sizeof(float): " << sizeof(d) << endl;

        colptrs = new int[numcols+1];
        irem = new int[nnonzero];
        xrem = new T[nnonzero];
        cout << "Memory allocaiton finished" << endl;
        
        position = 0;
        while(!file.eof() && position <= numcols)
        {
            //int Int32 = 0;
            //file.read((char *)MyBytes, 4);
            //Int32 = (Int32 << 8) + MyBytes[3];
            //Int32 = (Int32 << 8) + MyBytes[2];
            //Int32 = (Int32 << 8) + MyBytes[1];
            //Int32 = (Int32 << 8) + MyBytes[0];
            //colptrs[position++] = Int32;

            a = 0;
            //file.read(reinterpret_cast<char*>(&b), sizeof(b)); 
            file.read ((char*)&a, sizeof(a));
            colptrs[position] = a;
            position++;
        }
        cout << "finished reading colptrs" << endl;

        position = 0;
        while(!file.eof() && position < nnonzero)
        {
            a = 0;
            //file.read(reinterpret_cast<char*>(&a), sizeof(a)); 
            file.read ((char*)&a, sizeof(a));
            irem[position++] = a;
        }
        cout << "finished reading irem" << endl;

        position = 0 ;
        while(!file.eof() && position < nnonzero)
        {
            d = 0;
            file.read(reinterpret_cast<char*>(&d), sizeof(d));
            //file.read ((char*)&d, sizeof(d)); 
            xrem[position++] = d;
         
        }
        cout << "finished reading xrem" << endl;
    }
    else
    {
        cout << "file open error" << endl;
    }

    /*for(i = numcols - 10 ; i <=numcols ; i++)
        cout << colptrs[i] << " ";
    cout << endl;

    for(i = nnonzero - 10 ; i < nnonzero ; i++)
        cout << irem[i] << " ";
    cout << endl;

    for(i = nnonzero - 10 ; i < nnonzero ; i++)
        cout << xrem[i] << " ";
    cout << endl;*/

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

    fread(&nnonzero, sizeof(float), 1, fp);
    cout<<"non zero: "<<nnonzero<<endl;

    colptrs = new int[numcols + 1];
        irem = new int[nnonzero];
        xrem = new T[nnonzero];
        float *txrem = new float[nnonzero];
    cout << "Memory allocation finished" << endl;

    fread(colptrs, sizeof(int), numcols+1, fp);
        cout << "finished reading colptrs" << endl;

    fread(irem, sizeof(int), nnonzero, fp);
    cout << "finished reading irem" << endl;

    fread(txrem, sizeof(float), nnonzero, fp);
        cout << "finished reading xrem" << endl;
   for(int i = 0 ; i < nnonzero ; i++){
    
        xrem[i] = txrem[i];
    }
    
    delete []txrem;
    tend = omp_get_wtime();
    cout << "Matrix is read in " << tend - tstart << " seconds." << endl;


}

template<typename T>
void csc2blkcoord(block<T> *&matrixBlock, T *xrem)
{
    int i, j, r, c, k, k1, k2, blkr, blkc, tmp;
    int **top;
    nrowblks = ceil(numrows / (float)(wblk));
    ncolblks = ceil(numcols / (float)(wblk));
    cout << "wblk = " << wblk << endl;
    cout << "nrowblks = " << nrowblks << endl;
    cout << "ncolblks = " << ncolblks << endl;

    matrixBlock = new block<T>[nrowblks * ncolblks];
    top = new int*[nrowblks];
    //top = (int **) malloc(nrowblks * sizeof(int *));
    nnzPerRow = (int *) malloc(nrowblks * sizeof(int));

    for(i = 0 ; i < nrowblks ; i++)
    {
        top[i] = new int[ncolblks];
        //top[i] = (int *) malloc(ncolblks * sizeof(int));
        nnzPerRow[i] = 0;
    }

    for(blkr = 0 ; blkr < nrowblks ; blkr++)
    {
        for(blkc = 0 ; blkc < ncolblks ; blkc++)
        {
            top[blkr][blkc] = 0;
            matrixBlock[blkr * ncolblks + blkc].nnz = 0;
        }
    }
    cout << "here" << endl;
    cout<<"Finish memory allocation for block.."<<endl;

    //cout<<"K1: "<<colptrs[0]<<" K2: "<<colptrs[1]<<endl;

    //cout<<"calculatig nnz per block"<<endl;

    //calculatig nnz per block
    for(c = 0 ; c < numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1] - 1;
        blkc = ceil((c + 1) / (float)wblk);
        //cout<<"K1: "<<k1<<" K2: "<<k2<<" blkc: "<<blkc<<endl;

        for(k = k1 - 1 ; k < k2 ; k++)
        {
            r = irem[k];
            blkr = ceil(r/(float)wblk);
            if((blkr - 1) >= nrowblks || (blkc - 1) >= ncolblks)
            {
                cout << "(" << blkr - 1 << ", " << blkc - 1 << ") doesn't exist" << endl;
            }
            else
            {
                matrixBlock[(blkr - 1) * ncolblks + (blkc - 1)].nnz++;  
            }    
        }
    }

    cout<<"finished counting nnz in each block"<<endl;

    for(blkc = 0 ; blkc < ncolblks; blkc++)
    {
        for(blkr = 0 ; blkr < nrowblks ; blkr++)
        {
            //cout<<"br: "<<blkr<<" bc: "<<blkc<<" roffset: "<<blkr*wblk<<" coffset: "<<blkc*wblk<<endl;
            matrixBlock[blkr * ncolblks + blkc].roffset = blkr * wblk;
            matrixBlock[blkr * ncolblks + blkc].coffset = blkc * wblk;
            //cout<<"here 1"<<endl;

            if(matrixBlock[blkr * ncolblks + blkc].nnz > 0)
            {
                nnzPerRow[blkr] += matrixBlock[blkr * ncolblks + blkc].nnz;
                matrixBlock[blkr * ncolblks + blkc].rloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].cloc = new int[matrixBlock[blkr * ncolblks + blkc].nnz];
                matrixBlock[blkr * ncolblks + blkc].val = new T[matrixBlock[blkr * ncolblks + blkc].nnz];
            }
            else
            {
                matrixBlock[blkr * ncolblks + blkc].rloc = NULL;
                matrixBlock[blkr * ncolblks + blkc].cloc = NULL;
            }
        }
    }

    cout<<"allocating memory for each block"<<endl;

    //for(blkr=0;blkr<nrowblks;blkr++)
    //{
        //printf("nnzPerRow[%d] : %d\n", blkr, nnzPerRow[blkr]);
    //}
    //cout<<"end for"<<endl;

    for(c = 0 ; c < numcols ; c++)
    {
        k1 = colptrs[c];
        k2 = colptrs[c + 1] - 1;
        blkc = ceil((c + 1) / (float)wblk);

        for(k = k1 - 1 ; k < k2 ; k++)
        {
            r = irem[k];
            blkr = ceil(r / (float)wblk);

            matrixBlock[(blkr - 1) * ncolblks+blkc - 1].rloc[top[blkr-1][blkc-1]] = r - matrixBlock[(blkr - 1) * ncolblks + blkc - 1].roffset;
            matrixBlock[(blkr - 1) * ncolblks+blkc - 1].cloc[top[blkr-1][blkc-1]] = (c + 1) -  matrixBlock[(blkr - 1) * ncolblks + blkc - 1].coffset;
            matrixBlock[(blkr - 1) * ncolblks+blkc - 1].val[top[blkr-1][blkc-1]] = xrem[k];

            top[blkr-1][blkc-1]=top[blkr-1][blkc-1]+1;
        }
    }

    for(i = 0 ; i < nrowblks ; i++)
    {
        delete [] top[i];
    }
    delete [] top;
}


void myprint()
{
    printf("hello world\n");
}

void get_new_csb_block(int newWblk, int*** nnzBlock, int* nrowblocks, int* ncolblocks)
{
    wblk = newWblk;
    cout<<"new wblk = "<<wblk<<endl;

    int i,j;

    int nrowblks = ceil(numrows / (float)(wblk));
    int ncolblks = ceil(numcols / (float)(wblk));

    //char *filename = "550k.cus";
    //char* filename = "Z5.N5.Nm7.Mj1.p45/matrix.cus";
    block<double> *newMatrixBlock;
    double* newXrem;
    read_custom<double>(global_filename, newXrem);
    csc2blkcoord<double>(newMatrixBlock, newXrem);

    int total_spmm_blocks = 0;

    (*nnzBlock) = (int**)calloc((nrowblks+10),sizeof(int*));
    for(i = 0 ; i < nrowblks ; i++){
        (*nnzBlock)[i] = (int*)calloc((ncolblks+10),sizeof(int));
    }

    for(i = 0 ; i < nrowblks ; ++i){
        for(j = 0 ; j < ncolblks ; j++){
            if(newMatrixBlock[i*ncolblks+j].nnz > 0 ){
                //printf("%d %d\n",i,j);
                (*nnzBlock)[i][j] = 1;
                total_spmm_blocks++;
            }
        }
    }
    *nrowblocks = nrowblks;
    *ncolblocks = ncolblks;
    //printf("%d\n",(*nnzBlock)[489][0]);
    printf("total spmm blocks %d\n",total_spmm_blocks);
}


void get_input_output(const char* node, char* input1, char* input2, char* output)
{
    strcpy(input1,td[strdup(node)].input1);
    //printf("%s %s\n", td[strdup(node)].input1,input1);
    strcpy(input2,td[strdup(node)].input2);
    //printf("%s %s\n", td[strdup(node)].input2,input2);
    strcpy(output,td[strdup(node)].output);
    //printf("%s %s\n", td[strdup(node)].output,output);
}


void clear_InOutVariable()
{
    iov.clear();
}

int search_in_InOutVariable(const char* var)
{
    if(iov.count(var) != 0)
        return 1;
    else iov[strdup(var)] = 0;
    return 0;
}

void get_output_of_a_task(const char* task_name, char* output)
{

    strcpy(output,td[strdup(task_name)].output);
}


void fill_allinout_memory_map(const char* task_name, const char* parent_task, double memory_amount)
{

    char* parent_output = (char*)malloc(100*sizeof(char));
    get_output_of_a_task(parent_task,parent_output);
    all_mem_in_task[strdup(task_name)][strdup(parent_output)] = memory_amount;
    //printf("all_mem_in_task[%s][%s] = %lf\n",task_name,parent_output,memory_amount);
}

int get_internal_map_size(const char* task_name)
{
    return all_mem_in_task[strdup(task_name)].size();
}

double get_task_one_input_memory(const char* task_name, const char* input_mem)
{
    return all_mem_in_task[strdup(task_name)][strdup(input_mem)];
}

void get_all_the_keys_input(const char* task_name, char **keys){

    //printf("%s\n", task_name);

    /*std::unordered_map<const char * , double, my_hash, eqstr> task_map = all_mem_in_task[strdup(task_name)];

    for (std::unordered_map<const char * , double, my_hash, eqstr>::iterator it = task_map.begin(); it != task_map.end(); ++it)
    {
       // printf("map %s %lf\n",it->first,it->second);

    }*/
    int i = 0;
    internal_map task_map = inp_map[strdup(task_name)];

    for (internal_map::iterator it = task_map.begin(); it != task_map.end(); ++it)
    {
       // printf("map %s %lf\n",it->first,it->second);
        strcpy(keys[i],it->first);
        i++;

    }


}

void get_all_the_keys_output(const char* task_name, char **keys){

    //printf("%s\n", task_name);

    /*std::unordered_map<const char * , double, my_hash, eqstr> task_map = all_mem_in_task[strdup(task_name)];

    for (std::unordered_map<const char * , double, my_hash, eqstr>::iterator it = task_map.begin(); it != task_map.end(); ++it)
    {
       // printf("map %s %lf\n",it->first,it->second);

    }*/
    int i = 0;
    internal_map task_map = out_map[strdup(task_name)];

    for (internal_map::iterator it = task_map.begin(); it != task_map.end(); ++it)
    {
       // printf("map %s %lf\n",it->first,it->second);
        strcpy(keys[i],it->first);
        i++;

    }


}

int get_inp_mem_size(const char* task){



    //printf("%s\n", task);

    internal_map task_map = inp_map[strdup(task)];

    // for (internal_map::iterator it = task_map.begin(); it != task_map.end() ; ++it)
    // {
    //     printf("int map key %s \n",it->first);
    // }

    return task_map.size();
}

int get_out_mem_size(const char* task){



    //printf("%s\n", task);
    internal_map task_map = out_map[strdup(task)];

    // for (internal_map::iterator it = task_map.begin(); it != task_map.end() ; ++it)
    // {
    //     printf("int map key %s \n",it->first);
    // }

    return task_map.size();
}

void get_incoming_memory_name(const char* task_name, const char* parent_task_name, char* memory_name){

    memory_chunk temp_chunk;

    temp_chunk = inp_map[strdup(task_name)][strdup(parent_task_name)];
    strcpy(memory_name,temp_chunk.memory_name);

}
double get_incoming_memory_value(const char* task_name, const char* parent_task_name)
{

    memory_chunk temp_chunk;

    temp_chunk = inp_map[strdup(task_name)][strdup(parent_task_name)];

    return temp_chunk.value;
}

void get_outgoing_memory_name(const char* task_name, const char* child_task_name, char* memory_name){

    memory_chunk temp_chunk;

    temp_chunk = out_map[strdup(task_name)][strdup(child_task_name)];
    strcpy(memory_name,temp_chunk.memory_name);

}
double get_outgoing_memory_value(const char* task_name, const char* child_task_name){

    memory_chunk temp_chunk;

    temp_chunk = out_map[strdup(task_name)][strdup(child_task_name)];

    return temp_chunk.value;


}



