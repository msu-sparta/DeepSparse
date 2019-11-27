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

#include "../../common/partitioner/part_inc/rMLGP.h"



void  nonloop(int blocksize, int block_width, int argc, char *argv[]);

void firstloop(int blocksize , int block_width, int argc, char *argv[]);

void secondloop(int blocksize , int block_width, int argc, char *argv[]);


int main(int argc, char *argv[]){

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


    int guessEdgeCount = 20000000;
    int guessNodeCount = 80000000;
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

    nonloop(blocksize, block_width, argc, argv);
    firstloop(blocksize, block_width, argc, argv);
    secondloop(blocksize, block_width, argc, argv);




}



void  nonloop(int blocksize, int block_width, int argc, char *argv[])
{

	int M, N, index = 0;
    int currentBlockSize;
    int iterationNumber = 2, maxIterations = 2;
    int i, j;

    double tstart, tend, total_time, t1, t2;


    
    M = numrows;
    N = numcols;

    

  	////initialize edgeCount and nodeCount to zero

    edgeCount = 0 ; 
    nodeCount = 0 ; 
    globalNodeCount = 0 ;


    char main_task[100];
    char tmp_input1[100];
    char tmp_input2[100];
    memory_chunk temp_chunk;



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

    edgeU[edgeCount] = vertexName["RED,XBXBUF,0"];
    edgeV[edgeCount] = nodeCount - 1; 
    edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
    edgeCount++;


/////hier

 // ##### Hier #####
    strcpy(tmp_input1,"gramXBX,0");
    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map["CHOL,XBX"]["RED,XBXBUF,0"] = temp_chunk;
    out_map["RED,XBXBUF,0"]["CHOL,XBX"] = temp_chunk;

    //printf("input_map[CHOL,XBX][RED,XBXBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


    
    vertexName[strdup("DLACPY,0,20")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    edgeU[edgeCount] = nodeCount - 2;
    edgeV[edgeCount] = nodeCount - 1; 
    edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
    edgeCount++;

// ##### Hier #####
    strcpy(tmp_input1,"gramXBX,0");
    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map["DLACPY,0,20"]["CHOL,XBX"] = temp_chunk;
    out_map["CHOL,XBX"]["DLACPY,0,20"] = temp_chunk;

    //printf("input_map[DLACPY,0,20][CHOL,XBX] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);






    vertexName[strdup("INV,XBX")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    edgeU[edgeCount] = nodeCount - 2;
    edgeV[edgeCount] = nodeCount - 1; 
    edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
    edgeCount++;

// ##### Hier #####
    strcpy(tmp_input1,"tempGramXBX,0");
    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map["INV,XBX"]["DLACPY,0,20"] = temp_chunk;
    out_map["DLACPY,0,20"]["INV,XBX"] = temp_chunk;

    //printf("input_map[INV,XBX][DLACPY,0,20] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




    

    xy_id = 20;
    _XY(1, "_X", "null", "null", "null", "null", -1,
        1, "INV,XBX", "null", "null", "null", "null", -1,
        "_X", "tempGramXBX", "newX", M, blocksize, blocksize, block_width, xy_id);

    dlacpy_id = 21;

    int *X_vertexNo = (int *) malloc(nrowblks * sizeof(int));

    custom_dlacpy_v1(2, "null", "X", "XY", "X", "INV(XBX)", xy_id,
                "newX", "blockVectorX", M, blocksize, block_width, dlacpy_id, -1 , X_vertexNo); // -1 is the taskwait_node_no dummy

    int **SPMM_non_loop = (int **) malloc(nrowblks * sizeof(int *));
    for(i = 0 ; i < ncolblks ; i++)
        SPMM_non_loop[i] = (int *) malloc(ncolblks * sizeof(int));
        
    for(i = 0 ; i < nrowblks ; i++)
        for(j = 0 ; j < ncolblks ; j++)
            SPMM_non_loop[i][j] = -1;
    
    spmm_blkcoord_csbTask(2, "null", "AX", "SETZERO", "AX",
                         "A", "blockVectorX", "blockVectorAX", M, N, blocksize, block_width, currentBlockSize, -1, X_vertexNo, SPMM_non_loop); // -1 is the taskwait_node_no dummy


    //new -> takes task id for 2nd edge but not for SPMM
    xty_id = 21;
    _XTY_v3(2, "null", "AX", "SPMMRED", "SPMMBUF", 
            2, "null", "newX", "DLACPY", "X", 21,
            "blockVectorX", "blockVectorAX", "XAX", M, blocksize, blocksize, block_width, xty_id, SPMM_non_loop);

    vertexName[strdup("SPEUPDATE,XAX")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
    nodeCount++;

    edgeU[edgeCount] = vertexName["RED,XAXBUF,0"];
    edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"]; -> last insert
    edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
    edgeCount++;

    strcpy(tmp_input1,"gramXAX,0");
    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map["SPEUPDATE,XAX"]["RED,XBXBUF,0"] = temp_chunk;
    out_map["RED,XAXBUF,0"]["SPEUPDATE,XBX"] = temp_chunk;

    //printf("input_map[SPEUPDATE,XBX][RED,XAXBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




    vertexName[strdup("EIGEN")] = nodeCount;
    vertexWeight[nodeCount] = blocksize * blocksize  * sizeof(double);
    EIGEN_id = nodeCount;
    nodeCount++;

    edgeU[edgeCount] = vertexName["SPEUPDATE,XAX"];
    edgeV[edgeCount] = nodeCount - 1;
    edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
    edgeCount++;


    strcpy(tmp_input1,"gramXAX,0");
    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map["EIGEN"]["SPEUPDATE,XAX"] = temp_chunk;
    out_map["SPEUPDATE,XAX"]["EIGEN"] = temp_chunk;

    //printf("input_map[EIGEN][SPEUPDATE,XAX] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



    //xy 21
    xy_id = 21;
    _XY_v1(2, "null", "newX", "DLACPY", "X", 21,
            1, "EIGEN", "null", "null", "null", "null",
            "_X", "EIGEN", "_X", M, blocksize, blocksize, block_width, xy_id);

    dlacpy_id = 22;
    custom_dlacpy(2, "null", "X", "XY", "EIGEN", "X", xy_id,
                 "newX", "_X", M, currentBlockSize, block_width, dlacpy_id);

    xy_id = 22;
    _XY_v2(2, "null", "X", "SPMMRED", "SPMMBUF", 
            "EIGEN",
            "_AX", "EIGEN", "_AX", M, blocksize, blocksize, block_width, xy_id, SPMM_non_loop);

    dlacpy_id = 23;
    custom_dlacpy(2, "null", "AX", "XY", "EIGEN", "AX", xy_id,
                 "newAX", "_AX", M, currentBlockSize, block_width, dlacpy_id);

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

    MLGP_option opt;
    
    processArgs_rMLGP(argc, argv, &opt);
    opt.co_stop_size = 30;
    opt.co_stop_level = 1000;
     // opt.conpar = 0;
    // opt.inipart = 11;
    opt.use_binary_input = 0;
    printf("Calling run_rMLGP from main\n");
    
    //run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, &vertex_name_string[0], vertexWeight, numrows, numcols, nrowblks, ncolblks, block_width);

     run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, (const char**) &vertex_name_string[0], vertexWeight,0,"msdoor");


    printf("run_rMLGP Finshed\n");
    //run_rMLGP(opt.file_name, opt);

    free(pseudo_tid_map);
    
    free_opt(&opt);

    //return 0;

}



///////////firstloop/////////////////////////////////////


    /*
        usage: ./lobpcg_gen_graph_v24.x  <nblk> <block_width> 
        $$ change the custom format sparse matrix file path on line 202 
    */

    //cout << "USHRT_MAX: " << USHRT_MAX << " INT_MAX: " << INT_MAX << " LONG_MAX: " << LONG_MAX << endl;
    //exit(1);
void firstloop(int blocksize , int block_width, int argc, char *argv[])
{
    int M, N, index = 0;
    int currentBlockSize;
    int iterationNumber = 2, maxIterations = 2;
    int i, j;

    double tstart, tend, total_time, t1, t2;

    
    M = numrows;
    N = numcols;



    ////initialize edgeCount and nodeCount to zero

    edgeCount = 0 ; 
    nodeCount = 0 ; 
    globalNodeCount = 0 ; 


    char main_task[100];
    char tmp_input1[100];
    char tmp_input2[100];
    memory_chunk temp_chunk;



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

    

    for(iterationNumber = 1 ; iterationNumber <= 1 ; iterationNumber++)
    {
        // OP: blockVectorR = blockVectorAX - blockVectorX*spdiags(lambda,0,blockSize,blockSize); 
        // part-1: blockVectorR = blockVectorX*spdiags(lambda,0,blockSize,blockSize) 

        t1 = omp_get_wtime();

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "_lambda");
        vertexName[strdup(ary)] =  nodeCount;
        vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);
        _lambda_id = nodeCount;
        nodeCount++;
        
        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        t1 = omp_get_wtime();

        for(i = 0 ; i < nrowblks ; i++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "_X,");
            strcat(ary, nrowblksString[i]);
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
            nodeCount++;
            
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "_AX,");
            strcat(ary, nrowblksString[i]);
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
            nodeCount++;
        }

        t2 = omp_get_wtime();
        graphGenTime[11] += (t2 - t1);

        //_XY(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width);
        //xy 1
        xy_id = 1;
        _XY(1, "_X", "null", "null", "null", "null", -1,
            1, "_lambda", "null", "null", "null", "null", -1,
            "_X", "_lambda", "blockVectorR", M, blocksize, blocksize, block_width, xy_id);


        //sub 1
        // part-2: blockVectorR = blockVectorAX - blockVectorR
        int sub_id = 1;
        mat_sub(1, "_AX", "null", "null", "null", "null", -1,
                2, "null", "R", "XY", "_X", "_lambda", xy_id,
                "_AX", "blockVectorR", "blockVectorR", M, blocksize, block_width, sub_id);

        //OP: residualNorms = full(sqrt(sum(conj(blockVectorR).*blockVectorR)'));
        //complex task-1
        //node_file << "residualNorms_RESET(residualNorms);" << endl;

        t1 = omp_get_wtime();

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RESET,RN");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = blocksize * sizeof(double);
        residualNorms_id = nodeCount;
        nodeCount++;

        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);
        //mult 1
        mat_mult(2, "null", "R", "SUB", "AX", "R", sub_id,
                 2, "null", "R", "SUB", "AX", "R", sub_id,
                "blockVectorR", "blockVectorR", "newX", M, blocksize, block_width);
        //sumsqrt 1
        //sum_sqrt_task newX, residualNorms, M, blocksize, block_width
        sum_sqrt("newX", "MULT", "R", "R", "RESET,RN", residualNorms_id,
            "newX", "RN", M, blocksize, block_width);

        // OP: activeMask = full(residualNorms > residualTolerance) & activeMask; 
        //update_activeMask_task(activeMask, residualNorms, residualTolerance, blocksize);
        
        t1 = omp_get_wtime();
        
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RESET,actMask");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = blocksize * sizeof(double);
        activeMask_id = nodeCount;
        nodeCount++;

        edgeU[edgeCount] = vertexName["SQRT,RN"];
        edgeV[edgeCount] = activeMask_id; //vertexName["activeMask_UPDATE(activeMask)"];
        edgeW[edgeCount] = blocksize * sizeof(double);
        edgeCount++;


        // #### Hier ####
        strcpy(tmp_input1, "residualNorms");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["RESET,actMask"]["SQRT,RN"] = temp_chunk;
        out_map["SQRT,RN"]["RESET,actMask"] = temp_chunk;

        //printf("input_map[RESET,actMask][SQRT,RN] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




                
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "CONV,actMask");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        CONVERGE_id = nodeCount;
        nodeCount++; 

        edgeU[edgeCount] = activeMask_id; 
        edgeV[edgeCount] = CONVERGE_id; 
        edgeW[edgeCount] = blocksize * sizeof(double);
        edgeCount++;


        // #### Hier ####
        strcpy(tmp_input1,"actMask");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["CONV,actMask"]["RESET,actMask"] = temp_chunk;
        out_map["RESET,actMask"]["CONV,actMask"] = temp_chunk;

        //printf("input_map[CONV,actMask][RESET,actMask] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        //if convergeFlag == 0 block starts here

        // OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask) - ...
        //                                  blockVectorX*(blockVectorX'*blockVectorR(:,activeMask)); 

        // partial c implementation 
        // blockVectorR(:,activeMask) -> activeBlockVectorR 
        //getActiveBlockVector_task(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width);
        
        //get 1
        int get_id = 1;

        getActiveBlockVector(2, "null", "R", "SUB", "AX", "R", sub_id,
            "RESET,actMask", activeMask_id,
            "blockVectorR", "actMask", "actR", M, blocksize, currentBlockSize, block_width, get_id);

        t1 = omp_get_wtime();
        string dummyString = "";
        for(i = 0 ; i < nrowblks ; i++)
        {
            edgeU[edgeCount] = CONVERGE_id;
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "GET,");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",1");
        
            edgeV[edgeCount] = vertexName[ary];
            edgeW[edgeCount] = sizeof(double);
            edgeCount++;

            strcpy(tmp_input1,"actMask");


            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[ary]["CONV,actMask"] = temp_chunk;
            out_map["CONV,actMask"][ary] = temp_chunk;
            

            //printf("input_map[%s][CONV,actMask] = %s %lf\n",ary,tmp_input1,edgeW[edgeCount-1]);


             
        }

        t2 = omp_get_wtime();
        graphGenTime[11] += (t2 - t1);

        //_XTY_v1(blockVectorX, activeBlockVectorR, temp2, M, blocksize, currentBlockSize, block_width);
        //xty 1
        xty_id = 1;
        _XTY(1, "_X", "null", "null", "null", "null", -1,
                2, "null", "actR", "GET", "R", "actMask", get_id,
                "_X", "activeBlockVectorR", "temp2", M, blocksize, currentBlockSize, block_width, xty_id);

        /* OP: temp3 = blockVectorX * temp2 */
        //_XY(blockVectorX, temp2, temp3_R, M, blocksize, currentBlockSize, block_width); //temp2(0)_REDUCTION(temp2BUF);
        //xy 2
        xy_id = 2;
        _XY(1, "_X", "null", "null", "null", "null", -1,
            2, "RED,temp2BUF,0", "null", "null", "null", "null", -1,
            "_X", "temp2", "temp3", M, blocksize, currentBlockSize, block_width, xy_id);

        //mat_sub_task(activeBlockVectorR, temp3_R, activeBlockVectorR, M, currentBlockSize, block_width);
        //sub 2
        sub_id = 2;
        mat_sub(2, "null", "actR", "GET", "R", "actMask", get_id,
                2, "null", "temp3R",  "XY", "_X", "temp2", xy_id,
                "activeBlockVectorR", "temp3", "activeBlockVectorR", M, currentBlockSize, block_width, sub_id);

        //_XTY_v1(activeBlockVectorR, activeBlockVectorR, gramRBR, M, currentBlockSize, currentBlockSize, block_width);
        
        //change temp3R 
        //xty 2
        xty_id = 2;
        _XTY(2, "null", "actR", "SUB", "actR", "temp3R", sub_id,
            2, "null", "actR", "SUB", "actR", "temp3R", sub_id,
            "activeBlockVectorR", "activeBlockVectorR", "RBR", M, currentBlockSize, currentBlockSize, block_width, xty_id);

        //complex task 1
        t1 = omp_get_wtime();

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "CHOL,RBR");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,RBRBUF,0");

        edgeU[edgeCount] = vertexName[ary];
        edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRBR(0)_CHOL(gramRBR)"]; -> inserted just above!
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // ##### Hier #####
        strcpy(tmp_input1,"RBR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["CHOL,RBR"][strdup(ary)] = temp_chunk;
        out_map[strdup(ary)]["CHOL,RBR"] = temp_chunk;

        //printf("input_map[CHOL,RBR][%s] = %s %lf\n", ary,tmp_input1,edgeW[edgeCount-1]);


        //complex task 2
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "INV,RBR");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        edgeU[edgeCount] = nodeCount - 2; //vertexName["gramRBR(0)_CHOL(gramRBR)"]; -> 2nd last insert
        edgeV[edgeCount] = nodeCount - 1; //vertexName[nodeName, "gramRBR(0)_INV(gramRBR)"]; -> inserted just above
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // ##### Hier #####
        strcpy(tmp_input1,"RBR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["INV,RBR"]["CHOL,RBR"] = temp_chunk;
        out_map["CHOL,RBR"]["INV,RBR"] = temp_chunk;


        //printf("input_map[INV,RBR][CHOL,RBR] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);
        


        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        // OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask)/gramRBR;

        //_XY(activeBlockVectorR, gramRBR, temp3_R, M, currentBlockSize, currentBlockSize, block_width);
        //xy 3
        xy_id = 3;
        _XY(2, "null", "actR", "SUB", "actR", "temp3R", sub_id,
           2, "INV,RBR", "null", "null", "null", "null", -1,
           "activeBlockVectorR", "RBR", "temp3_R", M, currentBlockSize, currentBlockSize, block_width, xy_id);

        //pareparing to generate SPMM tasks
        t1 = omp_get_wtime();
        
        int taskwait_node_no = nodeCount; //-> not used anywhere
        
        t2 = omp_get_wtime(); 
        graphGenTime[12] += (t2 - t1);


        //last actR
        //custom_dlacpy_task(temp3_R, activeBlockVectorR, M, currentBlockSize, block_width);
        //dlacy 1
        dlacpy_id = 1;
        
        int *actR_vertexNo = (int *) malloc(nrowblks * sizeof(int));

        custom_dlacpy_v1(2, "null", "temp3R", "XY", "actR", "RBR", xy_id, 
                    "temp3_R", "activeBlockVectorR", M, currentBlockSize, block_width, dlacpy_id, taskwait_node_no, actR_vertexNo);


        /*t1 = omp_get_wtime();
        //set activeBlockVectorAR for SPMM (activeBlockVectorAR is the actual ooutput of the SPMM task)

        int *SETZERO_SPMM_OUTPUT_vertexNo = (int *) malloc(nrowblks * sizeof(int)); 

        for(i = 0 ; i < nrowblks ; i ++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "SETZERO,");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",1");
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
            SETZERO_SPMM_OUTPUT_vertexNo[i] = nodeCount;
            nodeCount++;
        }

        t2 = omp_get_wtime();
        graphGenTime[11] += (t2 - t1);*/


        int spmm_id = 1;
        //spmm_blkcoord_v1<double>(numrows, numcols, currentBlockSize, nthrds, activeBlockVectorR, activeBlockVectorAR, matrixBlock, block_width);
        
        //spmm_blkcoord_v1(2, "null", "activeBlockVectorAR", "SETZERO", "activeBlockVectorAR",
        //                "A", "activeBlockVectorR", "activeBlockVectorAR", M, N, currentBlockSize, block_width, spmm_id);

        int **SPMM_vertexNo = (int **) malloc(nrowblks * sizeof(int *));
        for(i = 0 ; i < ncolblks ; i++)
            SPMM_vertexNo[i] = (int *) malloc(ncolblks * sizeof(int));
        
        for(i = 0 ; i < nrowblks ; i++)
            for(j = 0 ; j < ncolblks ; j++)
                SPMM_vertexNo[i][j] = -1;
    
        spmm_blkcoord_csbTask(2, "null", "actAR", "SETZERO", "actAR",
                         "A", "activeBlockVectorR", "activeBlockVectorAR", M, N, currentBlockSize, block_width, currentBlockSize, taskwait_node_no, actR_vertexNo, SPMM_vertexNo);

        /*for(i = 0 ; i < nrowblks ; i++)
        {
            for(j = 0 ; j < ncolblks ; j++)
            {
                printf("%d ", SPMM_vertexNo[i][j]);
            }
            printf("\n");
        }*/

        //can we do this update after SPMM tasks???
        //updateBlockVector_task(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width); 
        //update 1
        int update_id = 1;
        updateBlockVector(2, "null", "actR", "DLACPY", "temp3R", dlacpy_id,
            "CONV,actMask", activeMask_id,
            "activeBlockVectorR", "actMask", "activeBlockVectorR", M, blocksize, currentBlockSize, block_width, update_id);


        if(iterationNumber > 1)
        {
            // OP : gramPBP=blockVectorP(:,activeMask)'*blockVectorP(:,activeMask); 
            //merged with _blockVectorX, _blockVectorAX up there
            
            //getActiveBlockVector_task(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width);
            //get 2
            get_id = 2;
            getActiveBlockVector(1, "_P", "null", "null", "null", "null", -1, 
                                "RESET,actMask", activeMask_id, 
                                "_P", "actMask", "actP", M, blocksize, currentBlockSize, block_width, get_id);

            //_XTY_v1(activeBlockVectorP, activeBlockVectorP, gramPBP, M, currentBlockSize, currentBlockSize, block_width);
            //xty 3
            xty_id = 3;
            _XTY(2, "null", "actP", "GET", "_P", "actMask", get_id,
                2, "null", "actP", "GET", "_P", "actMask", get_id,
                "actP", "actP", "PBP", M, currentBlockSize, currentBlockSize, block_width, xty_id);

            //complex task 3 gramPBP(0)_CHOL(gramPBP)
            t1 = omp_get_wtime();

            vertexName[strdup("CHOL,PBP")] = nodeCount;
            vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            nodeCount++;

            edgeU[edgeCount] = vertexName["RED,PBPBUF,0"];
            edgeV[edgeCount] = nodeCount - 1; //vertexName["gramPBP(0)_CHOL(gramPBP)"]; -> inserted up there! last insert
            edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            edgeCount++;
            
            //complex task 4 gramPBP(0)_INV(gramPBP)
            vertexName[strdup("INV,PBP")] = nodeCount;
            vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            nodeCount++;

            edgeU[edgeCount] = nodeCount - 2; //vertexName["gramPBP(0)_CHOL(gramPBP)"]; -> second last insert
            edgeV[edgeCount] = nodeCount - 1; //vertexName["gramPBP(0)_INV(gramPBP)"]; -> last insert
            edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            edgeCount++;

            t2 = omp_get_wtime();
            graphGenTime[12] += (t2 - t1);

            //_XY(activeBlockVectorP, gramPBP, temp3_P, M, currentBlockSize, currentBlockSize, block_width);
            //xy 4
            xy_id = 4;
            _XY(2, "null", "actP", "GET", "_P", "actMask", get_id,
               1, "INV,PBP", "null", "null", "null", "null", -1,
               "actP", "PBP", "temp3P", M, currentBlockSize, currentBlockSize, block_width, xy_id);
            
            //last actP
            //custom_dlacpy_task(temp3_P, activeBlockVectorP, M, currentBlockSize, block_width);
            //dlacpy 2
            dlacpy_id = 2;
            custom_dlacpy(2, "null", "temp3P", "XY", "actP", "PBP", xy_id,
                        "temp3P", "actP", M, currentBlockSize, block_width, dlacpy_id);
            
            //updateBlockVector_task(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width);
            //update 2
            update_id = 2;
            updateBlockVector(2, "null", "actP", "DLACPY", "temp3P", dlacpy_id,
                "RESET,actMask", activeMask_id,
                "actP", "actMask", "P", M, blocksize, currentBlockSize, block_width, update_id);

            //getActiveBlockVector_task(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width);
            //merged with _blockVectorX, _blockVectorAX up there

            //get 3
            get_id = 3;
            getActiveBlockVector(1, "_AP", "null", "null", "null", "null", -1,
                                "RESET,actMask", activeMask_id,
                                "_AP", "actMask", "actAP", M, blocksize, currentBlockSize, block_width, get_id);

            //_XY(activeBlockVectorAP, gramPBP, temp3_AP, M, currentBlockSize, currentBlockSize, block_width);
            //xy 5
            xy_id = 5;
            _XY(2, "null", "actAP", "GET", "_AP", "actMask", get_id,
               1, "INV,PBP", "null", "null", "null", "null", -1,
               "actAP", "PBP", "temp3AP", M, currentBlockSize, currentBlockSize, block_width, xy_id);
            //last actAP
            //custom_dlacpy_task(temp3_AP, activeBlockVectorAP, M, currentBlockSize, block_width);
            //dlacpy 3
            dlacpy_id = 3;
            custom_dlacpy(2, "null", "temp3AP", "XY", "actAP", "PBP", xy_id,
                        "temp3AP", "actAP", M, currentBlockSize, block_width, dlacpy_id);

            //blockVectorAP
            //updateBlockVector_task(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width);
            //update 3
            update_id = 3;
            updateBlockVector(2, "null", "actAP", "DLACPY", "temp3AP", dlacpy_id,
                "RESET,actMask", activeMask_id,
                "actAP", "actMask", "AP", M, blocksize, currentBlockSize, block_width, update_id);
        } //end if(tir > 1)

        // OP: gramXAR=full(blockVectorAX'*blockVectorR(:,activeMask));
        //     gramRAR=full(blockVectorAR(:,activeMask)'*blockVectorR(:,activeMask));
        //     gramRAR=(gramRAR'+gramRAR)*0.5;

        //_XTY_v1(blockVectorAX, activeBlockVectorR, gramXAR, M, blocksize, currentBlockSize, block_width);

        //different type of task format (2n one) 
        //xty 4
        xty_id = 4;
        
        //cout << "before calling xty 4: row:" << M << " col: " << blocksize << " currentBlockSize: " << currentBlockSize << " id: " << xty_id << endl;

        _XTY_v1(1, "_AX", "null", "null", "null", "null", -1,
                2, "null", "actR", "DLACPY", "temp3R", 1,
                "_AX", "activeBlockVectorR", "XAR", M, blocksize, currentBlockSize, block_width, xty_id);

        //_XTY_v1(activeBlockVectorAR, activeBlockVectorR, gramRAR, M, currentBlockSize, currentBlockSize, block_width); 
        //xty 5
        xty_id = 5;
        
        //from v10
        //_XTY_v1(2, "null", "activeBlockVectorAR", "SPMM", "A", "activeBlockVectorR", 1,
        //    2, "null", "activeBlockVectorR", "DLACPY", "temp3R", 1,
        //    "activeBlockVectorAR", "activeBlockVectorR", "gramRAR", M, currentBlockSize, currentBlockSize, block_width, xty_id);
        
        //from v12
        //_XTY_v2(2, "null", "activeBlockVectorAR", "SPMMREDUCTION", "tempactiveBlockVectorARSPMMBUF", 
        //2, "null", "activeBlockVectorR", "DLACPY", "temp3R",
        //"activeBlockVectorAR", "activeBlockVectorR", "gramRAR", M, currentBlockSize, currentBlockSize, block_width);

        //new -> takes task id for 2nd edge but not for SPMM
        _XTY_v3(2, "null", "actAR", "SPMMRED", "SPMMBUF", 
                2, "null", "actR", "DLACPY", "temp3R", 1,
                "activeBlockVectorAR", "activeBlockVectorR", "RAR", M, currentBlockSize, currentBlockSize, block_width, xty_id, SPMM_vertexNo);

        t1 = omp_get_wtime();

        vertexName[strdup("TRANS,RAR")] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        edgeU[edgeCount] = vertexName["RED,RARBUF,0"];
        edgeV[edgeCount] = nodeCount - 1; //vertexName["transGramRAR(0)_TRANS(gramRAR)"]; -> last insert
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // #### Hier ####
        strcpy(tmp_input1,"RAR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["TRANS,RAR"]["RED,RARBUF,0"] = temp_chunk;
        out_map["RED,RARBUF,0"]["TRANS,RAR"] = temp_chunk;

        //printf("input_map[TRANS,RAR][RED,RARBUF] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);

        
        vertexName[strdup("SPEUPDATE,RAR")] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        edgeU[edgeCount] = vertexName["RED,RARBUF,0"];
        edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"]; -> last insert
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;


        // #### Hier ####
        strcpy(tmp_input1,"RAR,0");


        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["SPEUPDATE,RAR"]["RED,RARBUF,0"] = temp_chunk;
        out_map["RED,RARBUF,0"]["SPEUPDATE,RAR"] = temp_chunk;

        //printf("input_map[SPEUPDATE,RAR][RED,RARBUF] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


        edgeU[edgeCount] = nodeCount - 2; //vertexName["transGramRAR(0)_TRANS(gramRAR)"]; -> 2nd last insert
        edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"]; -> last insert
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        strcpy(tmp_input1,"transRAR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["SPEUPDATE,RAR"]["TRANS,RAR"] = temp_chunk;
        out_map["TRANS,RAR"]["SPEUPDATE,RAR"] = temp_chunk;

        //printf("input_map[SPEUPDATE,RAR][TRANS,RAR] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        //inner for loop start here

        if(iterationNumber == 1) //equivalent to if(restart == 0) 
        {
            

            //CONSTRUCTGA1  //gramXAR, lambda, gramXAP
            vertexName[strdup("CONSTRUCTGA")] = nodeCount;
            vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
            CONSTRUCTGA1_id = nodeCount;
            nodeCount++;
            
            //gramXAP -> gramXAP(0)_REDUCTION(gramXAPBUF);
            
            edgeU[edgeCount] = _lambda_id; 
            edgeV[edgeCount] = CONSTRUCTGA1_id; 
            edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
            edgeCount++;


            // ##### Hier #####
            strcpy(tmp_input1,"_lambda");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA"]["_lambda"] = temp_chunk;
            out_map["_lambda"]["CONSTRUCTGA"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA][_lambda] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);

            
            //gramXAR -> gramXAR(0)_REDUCTION(gramXARBUF);

            edgeU[edgeCount] = vertexName["RED,XARBUF,0"];
            edgeV[edgeCount] = CONSTRUCTGA1_id; //vertexName["CONSTRUCTGA1"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"XAR,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA"]["RED,XARBUF,0"] = temp_chunk;
            out_map["RED,XARBUF,0"]["CONSTRUCTGA"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA][RED,XARBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


            

            edgeU[edgeCount] = vertexName["SPEUPDATE,RAR"]; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"];
            edgeV[edgeCount] = CONSTRUCTGA1_id; //vertexName["CONSTRUCTGA2"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

             // ##### Hier #####
            strcpy(tmp_input1,"RAR,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA"]["SPEUPDATE,RAR"] = temp_chunk;
            out_map["SPEUPDATE,RAR"]["CONSTRUCTGA"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA][SPEUPDATE,RAR] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



             vertexName[strdup("CONSTRUCTGB")] = nodeCount;
            vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
            CONSTRUCTGB_id = nodeCount;
            nodeCount++;
            
            edgeU[edgeCount] = CONVERGE_id; 
            edgeV[edgeCount] = CONSTRUCTGB_id; 
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

             // ##### Hier #####
            strcpy(tmp_input1,"actMask");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGB"]["CONV,actMask"] = temp_chunk;
            out_map["CONV,actMask"]["CONSTRUCTGB"] = temp_chunk;

            //printf("input_map[CONSTRUCTGB][CONV,actMask] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



            t2 = omp_get_wtime();
            graphGenTime[12] += (t2 - t1);           
        } //end inner loop

        t1  = omp_get_wtime();

        vertexName[strdup("EIGEN")] = nodeCount;
        vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
        EIGEN_id = nodeCount;
        nodeCount++;

        edgeU[edgeCount] = CONSTRUCTGA1_id; //vertexName["CONSTRUCTGA2"];
        edgeV[edgeCount] = EIGEN_id; //vertexName["EIGEN"];
        edgeW[edgeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                    (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
        edgeCount++;

         // ##### Hier #####
        strcpy(tmp_input1,"gramA");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["EIGEN"]["CONSTRUCTGA"] = temp_chunk;
        out_map["CONSTRUCTGA"]["EIGEN"] = temp_chunk;

        //printf("input_map[EIGEN][CONSTRUCTGA] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



        edgeU[edgeCount] = CONSTRUCTGB_id; //vertexName["CONSTRUCTGB"];
        edgeV[edgeCount] = EIGEN_id; //vertexName["EIGEN"];
        edgeW[edgeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                    (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
        edgeCount++;

         // ##### Hier #####
        strcpy(tmp_input1,"gramB");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["EIGEN"]["CONSTRUCTGB"] = temp_chunk;
        out_map["CONSTRUCTGB"]["EIGEN"] = temp_chunk;

        //printf("input_map[EIGEN][CONSTRUCTGB] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        int add_id = -1;
        
        //last part after EIGEN
        if(iterationNumber == 1)
        {
            // blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:)
            // partil result- part1:- blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:)

            //_XY(activeBlockVectorR, coordX + (blocksize * blocksize), blockVectorP, M, currentBlockSize, blocksize, block_width);
            //xy 6
            xy_id = 6;
            _XY_v1(2, "null", "actR", "DLACPY", "temp3R", 1,
                    1, "EIGEN", "null", "null", "null", "null",
                    "activeBlockVectorR", "EIGEN", "_P", M, currentBlockSize, blocksize, block_width, xy_id);

            
            //  OP: blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) 

            //_XY(activeBlockVectorAR, coordX+(blocksize*blocksize), blockVectorAP, M, currentBlockSize, blocksize, block_width);
            //xy 8
            xy_id = 8;
            //_XY(2, "null", "activeBlockVectorAR", "SPMM", "A", "activeBlockVectorR", spmm_id,
            //    1, "EIGEN", "null", "null", "null", "null", -1,
            //    "activeBlockVectorAR", "EIGEN", "blockVectorAP", M, currentBlockSize, blocksize, block_width, xy_id);

            _XY_v2(2, "null", "actAR", "SPMMRED", "SPMMBUF", 
                    "EIGEN",
                    "activeBlockVectorAR", "EIGEN", "_AP", M, currentBlockSize, blocksize, block_width, xy_id, SPMM_vertexNo);
        }

        // OP: blockVectorX = blockVectorX*coordX(1:blockSize,:) + blockVectorP; 
        //_XY(blockVectorX, coordX, newX, M, blocksize, blocksize, block_width);
        //xy 10
        xy_id = 10;
        _XY(1, "_X", "null", "null", "null", "null", -1,
            1, "EIGEN", "null", "null", "null", "null", -1,
            "_X", "EIGEN", "newX", M, blocksize, blocksize, block_width, xy_id);

        //mat_addition_task(newX, blockVectorP, blockVectorX, M, blocksize, block_width);
        //add 3
        add_id = 3; //P from xy_id 6
        mat_addition(2, "null", "newX", "XY", "X", "EIGEN", xy_id,
                    2, "null", "P", "XY", "P", "newP", 6,
                    "newX", "_P", "_X", M, blocksize, block_width, add_id); //blockVectorP(0)_ADD(blockVectorP_newP_0);
        
        /* OP: blockVectorAX=blockVectorAX*coordX(1:blockSize,:) + blockVectorAP; */
        
        //_XY(blockVectorAX, coordX, newAX, M, blocksize, blocksize, block_width); 
        //xy 11
        xy_id = 11;
        _XY(1, "_AX", "null", "null", "null", "null", -1,
            1, "EIGEN", "null", "null", "null", "null", -1,
            "_AX", "EIGEN", "newAX", M, blocksize, blocksize, block_width, xy_id);

        //mat_addition_task(newAX, blockVectorAP, blockVectorAX, M, blocksize, block_width);
        //add 4
        add_id = 4; //AP from xy_id 10
        mat_addition(2, "null", "newAX", "XY", "AX", "EIGEN", xy_id,
                    2, "null", "AP", "XY", "AP", "newAP", 8,
                    "newAX", "AP", "AX", M, blocksize, block_width, add_id);

    } // end for

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

    
    
    //writing graph in dot format in file

    /*std::ofstream graph_file("MatA100-firstloop-graph-v30.dot");
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

    MLGP_option opt;
    
    processArgs_rMLGP(argc, argv, &opt);
    opt.co_stop_size = 30;
    opt.co_stop_level = 1000;
     // opt.conpar = 0;
    // opt.inipart = 11;
    opt.use_binary_input = 0;
    printf("Calling run_rMLGP from main\n");
    
    //run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, &vertex_name_string[0], vertexWeight, numrows, numcols, nrowblks, ncolblks, block_width);

    run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, (const char**) &vertex_name_string[0], vertexWeight,1,"msdoor");


    printf("run_rMLGP Finshed\n");
    //run_rMLGP(opt.file_name, opt);

    free(pseudo_tid_map);
    
    free_opt(&opt);

 //   return 0;

    

}




////////// secondloop function start ////////////////////////



void secondloop(int blocksize, int block_width, int argc, char *argv[])
{
    /*
        usage: ./lobpcg_gen_graph_v24.x  <nblk> <block_width> 
        $$ change the custom format sparse matrix file path on line 202 
    */

    //cout << "USHRT_MAX: " << USHRT_MAX << " INT_MAX: " << INT_MAX << " LONG_MAX: " << LONG_MAX << endl;
    //exit(1);

    int M, N, index = 0;
    int currentBlockSize;
    int iterationNumber = 2, maxIterations = 2;
    int i, j;

    double tstart, tend, total_time, t1, t2;

    ////initialize edgeCount and nodeCount to zero

    edgeCount = 0 ; 
    nodeCount = 0 ; 
    globalNodeCount = 0 ; 


    printf("Rows: %d, Cols: %d\n", M, N);
    printf("Block Size: %d Block Width: %d nthreads: %d nrowblks: %d ncolblks: %d\n", blocksize, block_width, nthreads, nrowblks, ncolblks);

    currentBlockSize = blocksize;

    tstart = omp_get_wtime();

    /* special node ids */
    int _lambda_id, activeMask_id, residualNorms_id, CONSTRUCTGA1_id, EIGEN_id, CONSTRUCTGA2_id;
    int CONVERGE_id, CONSTRUCTGB_id;
    char ary[150], i_string[8];

    for(iterationNumber = 2 ; iterationNumber <= maxIterations ; iterationNumber++)
    {
        // OP: blockVectorR = blockVectorAX - blockVectorX*spdiags(lambda,0,blockSize,blockSize); 
        // part-1: blockVectorR = blockVectorX*spdiags(lambda,0,blockSize,blockSize) 


        // #### Hier #####
        char main_task[100];
        char tmp_input1[100];
        char tmp_input2[100];
        memory_chunk temp_chunk;

        t1 = omp_get_wtime();

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "_lambda");
        vertexName[strdup(ary)] =  nodeCount;
        vertexWeight[nodeCount] = blocksize * blocksize * sizeof(double);

        _lambda_id = nodeCount;
        nodeCount++;
        
        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        t1 = omp_get_wtime();

        for(i = 0 ; i < nrowblks ; i++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "_X,");
            strcat(ary, nrowblksString[i]);
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
            nodeCount++;
            
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "_AX,");
            strcat(ary, nrowblksString[i]);
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
            nodeCount++;

            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "_P,");
            strcat(ary, nrowblksString[i]);
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
            nodeCount++;

            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "_AP,");
            strcat(ary, nrowblksString[i]);
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * blocksize * sizeof(double);
            nodeCount++;
        }

        t2 = omp_get_wtime();
        graphGenTime[11] += (t2 - t1);

        //_XY(blockVectorX, lambda, blockVectorR, M, blocksize, blocksize, block_width);
        //xy 1
        int xy_id = 1;
        _XY(1, "_X", "null", "null", "null", "null", -1,
            1, "_lambda", "null", "null", "null", "null", -1,
            "_X", "_lambda", "blockVectorR", M, blocksize, blocksize, block_width, xy_id);


        //sub 1
        // part-2: blockVectorR = blockVectorAX - blockVectorR
        int sub_id = 1;
        mat_sub(1, "_AX", "null", "null", "null", "null", -1,
                2, "null", "R", "XY", "_X", "_lambda", xy_id,
                "_AX", "blockVectorR", "blockVectorR", M, blocksize, block_width, sub_id);

        //OP: residualNorms = full(sqrt(sum(conj(blockVectorR).*blockVectorR)'));
        //complex task-1
        //node_file << "residualNorms_RESET(residualNorms);" << endl;

        t1 = omp_get_wtime();

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RESET,RN");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = blocksize * sizeof(double);
        residualNorms_id = nodeCount;
        nodeCount++;

        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);
        //mult 1
        mat_mult(2, "null", "R", "SUB", "AX", "R", sub_id,
                 2, "null", "R", "SUB", "AX", "R", sub_id,
                "blockVectorR", "blockVectorR", "newX", M, blocksize, block_width);

        //replacing mat_mult by dot_mm
        // dot_mm(2, "null", "R", "SUB", "AX", "R", sub_id,
        //          2, "null", "R", "SUB", "AX", "R", sub_id,
        //      "blockVectorR", "blockVectorR", "newX", M, blocksize, block_width);

        //sumsqrt 1
        //sum_sqrt_task newX, residualNorms, M, blocksize, block_width
        sum_sqrt("newX", "MULT", "R", "R", "RESET,RN", residualNorms_id,
            "newX", "RN", M, blocksize, block_width);

        // OP: activeMask = full(residualNorms > residualTolerance) & activeMask; 
        //update_activeMask_task(activeMask, residualNorms, residualTolerance, blocksize);
        
        t1 = omp_get_wtime();
        
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RESET,actMask");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = blocksize * sizeof(double);
        activeMask_id = nodeCount;
        nodeCount++;

        edgeU[edgeCount] = vertexName["SQRT,RN"];
        edgeV[edgeCount] = activeMask_id; //vertexName["activeMask_UPDATE(activeMask)"];
        edgeW[edgeCount] = blocksize * sizeof(double);
        edgeCount++;

        // #### Hier ####
        strcpy(tmp_input1, "residualNorms");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["RESET,actMask"]["SQRT,RN"] = temp_chunk;
        out_map["SQRT,RN"]["RESET,actMask"] = temp_chunk;

        //printf("input_map[RESET,actMask][SQRT,RN] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);

                
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "CONV,actMask");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = sizeof(double);
        CONVERGE_id = nodeCount;
        nodeCount++; 

        edgeU[edgeCount] = activeMask_id; 
        edgeV[edgeCount] = CONVERGE_id; 
        edgeW[edgeCount] = blocksize * sizeof(double);
        edgeCount++;


        // #### Hier ####
        strcpy(tmp_input1,"actMask");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["CONV,actMask"]["RESET,actMask"] = temp_chunk;
        out_map["RESET,actMask"]["CONV,actMask"] = temp_chunk;

        //printf("input_map[CONV,actMask][RESET,actMask] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        //if convergeFlag == 0 block starts here

        // OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask) - ...
        //                                  blockVectorX*(blockVectorX'*blockVectorR(:,activeMask)); 

        // partial c implementation 
        // blockVectorR(:,activeMask) -> activeBlockVectorR 
        //getActiveBlockVector_task(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width);
        
        //get 1
        int get_id = 1;

        getActiveBlockVector(2, "null", "R", "SUB", "AX", "R", sub_id,
            "CONV,actMask", CONVERGE_id,
            "blockVectorR", "actMask", "actR", M, blocksize, currentBlockSize, block_width, get_id);

        t1 = omp_get_wtime();

        //NOT needed in PCU 
        string dummyString = "";
        for(i = 0 ; i < nrowblks ; i++)
        {
            edgeU[edgeCount] = CONVERGE_id;
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "GET,");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",1");
        
            edgeV[edgeCount] = vertexName[ary];
            edgeW[edgeCount] = sizeof(double);
            edgeCount++;

            strcpy(tmp_input1,"actMask");


            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[ary]["CONV,actMask"] = temp_chunk;
            out_map["CONV,actMask"][ary] = temp_chunk;
            

            printf("input_map[%s][CONV,actMask] = %s %lf\n",ary,tmp_input1,edgeW[edgeCount-1]);
        }

        t2 = omp_get_wtime();
        graphGenTime[11] += (t2 - t1);

        //_XTY_v1(blockVectorX, activeBlockVectorR, temp2, M, blocksize, currentBlockSize, block_width);
        //xty 1
        int xty_id = 1;
        _XTY(1, "_X", "null", "null", "null", "null", -1,
                2, "null", "actR", "GET", "R", "actMask", get_id,
                "_X", "activeBlockVectorR", "temp2", M, blocksize, currentBlockSize, block_width, xty_id);

        /* OP: temp3 = blockVectorX * temp2 */
        //_XY(blockVectorX, temp2, temp3_R, M, blocksize, currentBlockSize, block_width); //temp2(0)_REDUCTION(temp2BUF);
        //xy 2
        xy_id = 2;
        _XY(1, "_X", "null", "null", "null", "null", -1,
            2, "RED,temp2BUF,0", "null", "null", "null", "null", -1,
            "_X", "temp2", "temp3", M, blocksize, currentBlockSize, block_width, xy_id);

        //mat_sub_task(activeBlockVectorR, temp3_R, activeBlockVectorR, M, currentBlockSize, block_width);
        //sub 2
        sub_id = 2;
        mat_sub(2, "null", "actR", "GET", "R", "actMask", get_id,
                2, "null", "temp3R",  "XY", "_X", "temp2", xy_id,
                "activeBlockVectorR", "temp3", "activeBlockVectorR", M, currentBlockSize, block_width, sub_id);

        //_XTY_v1(activeBlockVectorR, activeBlockVectorR, gramRBR, M, currentBlockSize, currentBlockSize, block_width);
        
        //change temp3R 
        //xty 2
        xty_id = 2;
        _XTY(2, "null", "actR", "SUB", "actR", "temp3R", sub_id,
            2, "null", "actR", "SUB", "actR", "temp3R", sub_id,
            "activeBlockVectorR", "activeBlockVectorR", "RBR", M, currentBlockSize, currentBlockSize, block_width, xty_id);

        //complex task 1
        t1 = omp_get_wtime();

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "CHOL,RBR");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "RED,RBRBUF,0");

        edgeU[edgeCount] = vertexName[ary];
        edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRBR(0)_CHOL(gramRBR)"]; -> inserted just above!
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // ##### Hier #####
        strcpy(tmp_input1,"RBR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["CHOL,RBR"][strdup(ary)] = temp_chunk;
        out_map[strdup(ary)]["CHOL,RBR"] = temp_chunk;

        //printf("input_map[CHOL,RBR][%s] = %s %lf\n", ary,tmp_input1,edgeW[edgeCount-1]);


        //complex task 2
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "INV,RBR");
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        edgeU[edgeCount] = nodeCount - 2; //vertexName["gramRBR(0)_CHOL(gramRBR)"]; -> 2nd last insert
        edgeV[edgeCount] = nodeCount - 1; //vertexName[nodeName, "gramRBR(0)_INV(gramRBR)"]; -> inserted just above
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // ##### Hier #####
        strcpy(tmp_input1,"RBR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["INV,RBR"]["CHOL,RBR"] = temp_chunk;
        out_map["CHOL,RBR"]["INV,RBR"] = temp_chunk;


        //printf("input_map[INV,RBR][CHOL,RBR] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);
        


        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        // OP: blockVectorR(:,activeMask) = blockVectorR(:,activeMask)/gramRBR;

        //_XY(activeBlockVectorR, gramRBR, temp3_R, M, currentBlockSize, currentBlockSize, block_width);
        //xy 3
        xy_id = 3;
        _XY(2, "null", "actR", "SUB", "actR", "temp3R", sub_id,
           2, "INV,RBR", "null", "null", "null", "null", -1,
           "activeBlockVectorR", "RBR", "temp3_R", M, currentBlockSize, currentBlockSize, block_width, xy_id);

        //pareparing to generate SPMM tasks
        t1 = omp_get_wtime();
        
        int taskwait_node_no = nodeCount; //-> not used anywhere
        
        t2 = omp_get_wtime(); 
        graphGenTime[12] += (t2 - t1);


        //last actR
        //custom_dlacpy_task(temp3_R, activeBlockVectorR, M, currentBlockSize, block_width);
        //dlacy 1
        int dlacpy_id = 1;
        
        int *actR_vertexNo = (int *) malloc(nrowblks * sizeof(int));

        custom_dlacpy_v1(2, "null", "temp3R", "XY", "actR", "RBR", xy_id, 
                    "temp3_R", "activeBlockVectorR", M, currentBlockSize, block_width, dlacpy_id, taskwait_node_no, actR_vertexNo);


        /*t1 = omp_get_wtime();
        //set activeBlockVectorAR for SPMM (activeBlockVectorAR is the actual ooutput of the SPMM task)

        int *SETZERO_SPMM_OUTPUT_vertexNo = (int *) malloc(nrowblks * sizeof(int)); 

        for(i = 0 ; i < nrowblks ; i ++)
        {
            memset(&ary[0], 0, sizeof(ary));
            strcat(ary, "SETZERO,");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",1");
            vertexName[strdup(ary)] = nodeCount;
            vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
            SETZERO_SPMM_OUTPUT_vertexNo[i] = nodeCount;
            nodeCount++;
        }

        t2 = omp_get_wtime();
        graphGenTime[11] += (t2 - t1);*/


        int spmm_id = 1;
        //spmm_blkcoord_v1<double>(numrows, numcols, currentBlockSize, nthrds, activeBlockVectorR, activeBlockVectorAR, matrixBlock, block_width);
        
        //spmm_blkcoord_v1(2, "null", "activeBlockVectorAR", "SETZERO", "activeBlockVectorAR",
        //                "A", "activeBlockVectorR", "activeBlockVectorAR", M, N, currentBlockSize, block_width, spmm_id);

        int **SPMM_vertexNo = (int **) malloc(nrowblks * sizeof(int *));
        for(i = 0 ; i < ncolblks ; i++)
            SPMM_vertexNo[i] = (int *) malloc(ncolblks * sizeof(int));
        
        for(i = 0 ; i < nrowblks ; i++)
            for(j = 0 ; j < ncolblks ; j++)
                SPMM_vertexNo[i][j] = -1;
    
        spmm_blkcoord_csbTask(2, "null", "actAR", "SETZERO", "actAR",
                         "A", "activeBlockVectorR", "activeBlockVectorAR", M, N, currentBlockSize, block_width, currentBlockSize, taskwait_node_no, actR_vertexNo, SPMM_vertexNo);

        /*for(i = 0 ; i < nrowblks ; i++)
        {
            for(j = 0 ; j < ncolblks ; j++)
            {
                printf("%d ", SPMM_vertexNo[i][j]);
            }
            printf("\n");
        }*/

        //can we do this update after SPMM tasks???
        //updateBlockVector_task(activeBlockVectorR, activeMask, blockVectorR, M, blocksize, currentBlockSize, block_width); 
        //update 1
        int update_id = 1;
        updateBlockVector(2, "null", "actR", "DLACPY", "temp3R", dlacpy_id,
            "CONV,actMask", CONVERGE_id,
            "activeBlockVectorR", "actMask", "activeBlockVectorR", M, blocksize, currentBlockSize, block_width, update_id);


        if(iterationNumber >= 1)
        {
            // OP : gramPBP=blockVectorP(:,activeMask)'*blockVectorP(:,activeMask); 
            //merged with _blockVectorX, _blockVectorAX up there
            
            //getActiveBlockVector_task(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width);
            //get 2
            get_id = 2;
            getActiveBlockVector(1, "_P", "null", "null", "null", "null", -1, 
                                "CONV,actMask", CONVERGE_id, 
                                "_P", "actMask", "actP", M, blocksize, currentBlockSize, block_width, get_id);

            //_XTY_v1(activeBlockVectorP, activeBlockVectorP, gramPBP, M, currentBlockSize, currentBlockSize, block_width);
            //xty 3
            xty_id = 3;
            _XTY(2, "null", "actP", "GET", "_P", "actMask", get_id,
                2, "null", "actP", "GET", "_P", "actMask", get_id,
                "activeBlockVectorP", "activeBlockVectorP", "PBP", M, currentBlockSize, currentBlockSize, block_width, xty_id);

            //complex task 3 gramPBP(0)_CHOL(gramPBP)
            t1 = omp_get_wtime();

            vertexName[strdup("CHOL,PBP")] = nodeCount;
            vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            nodeCount++;

            edgeU[edgeCount] = vertexName["RED,PBPBUF,0"];
            edgeV[edgeCount] = nodeCount - 1; //vertexName["gramPBP(0)_CHOL(gramPBP)"]; -> inserted up there! last insert
            edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            edgeCount++;

                // ##### Hier #####
            strcpy(tmp_input1,"PBP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CHOL,PBP"]["RED,PBPBUF,0"] = temp_chunk;
            out_map["RED,PBPBUF,0"]["CHOL,PBP"] = temp_chunk;

            //printf("input_map[CHOL,PBP][RED,PBPBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



            
            //complex task 4 gramPBP(0)_INV(gramPBP)
            vertexName[strdup("INV,PBP")] = nodeCount;
            vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            nodeCount++;

            edgeU[edgeCount] = nodeCount - 2; //vertexName["gramPBP(0)_CHOL(gramPBP)"]; -> second last insert
            edgeV[edgeCount] = nodeCount - 1; //vertexName["gramPBP(0)_INV(gramPBP)"]; -> last insert
            edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
            edgeCount++;

            strcpy(tmp_input1,"PBP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["INV,PBP"]["CHOL,PBP"] = temp_chunk;
            out_map["CHOL,PBP"]["INV,PBP"] = temp_chunk;

            //printf("input_map[INV,PBP][CHOL,PBP] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


            t2 = omp_get_wtime();
            graphGenTime[12] += (t2 - t1);

            //_XY(activeBlockVectorP, gramPBP, temp3_P, M, currentBlockSize, currentBlockSize, block_width);
            //xy 4
            xy_id = 4;
            _XY(2, "null", "actP", "GET", "_P", "actMask", get_id,
               2, "INV,PBP", "null", "null", "null", "null", -1,
               "activeBlockVectorP", "PBP", "temp3_P", M, currentBlockSize, currentBlockSize, block_width, xy_id);
            
            //last actP
            //custom_dlacpy_task(temp3_P, activeBlockVectorP, M, currentBlockSize, block_width);
            //dlacpy 2
            dlacpy_id = 2;
            custom_dlacpy(2, "null", "temp3P", "XY", "actP", "PBP", xy_id,
                        "temp3_P", "activeBlockVectorP", M, currentBlockSize, block_width, dlacpy_id);
            
            //updateBlockVector_task(activeBlockVectorP, activeMask, blockVectorP, M, blocksize, currentBlockSize, block_width);
            //update 2
            update_id = 2;
            updateBlockVector(2, "null", "actP", "DLACPY", "temp3P", dlacpy_id,
                "CONV,actMask", CONVERGE_id,
                "activeBlockVectorP", "actMask", "_P", M, blocksize, currentBlockSize, block_width, update_id);

            //getActiveBlockVector_task(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width);
            //merged with _blockVectorX, _blockVectorAX up there

            //get 3
            get_id = 3;
            getActiveBlockVector(1, "_AP", "null", "null", "null", "null", -1,
                                "CONV,actMask", CONVERGE_id,
                                "_AP", "actMask", "activeBlockVectorAP", M, blocksize, currentBlockSize, block_width, get_id);

            //_XY(activeBlockVectorAP, gramPBP, temp3_AP, M, currentBlockSize, currentBlockSize, block_width);
            //xy 5
            xy_id = 5;
            _XY(2, "null", "actAP", "GET", "_AP", "actMask", get_id,
               2, "INV,PBP", "null", "null", "null", "null", -1,
               "activeBlockVectorAP", "PBP", "temp3_AP", M, currentBlockSize, currentBlockSize, block_width, xy_id);
            //last actAP
            //custom_dlacpy_task(temp3_AP, activeBlockVectorAP, M, currentBlockSize, block_width);
            //dlacpy 3
            dlacpy_id = 3;
            custom_dlacpy(2, "null", "temp3AP", "XY", "actAP", "PBP", xy_id,
                        "temp3_AP", "activeBlockVectorAP", M, currentBlockSize, block_width, dlacpy_id);

            //blockVectorAP
            //updateBlockVector_task(activeBlockVectorAP, activeMask, blockVectorAP, M, blocksize, currentBlockSize, block_width);
            //update 3
            update_id = 3;
            updateBlockVector(2, "null", "actAP", "DLACPY", "temp3AP", dlacpy_id,
                "CONV,actMask", CONVERGE_id,
                "activeBlockVectorAP", "actMask", "_AP", M, blocksize, currentBlockSize, block_width, update_id);
        }

        // OP: gramXAR=full(blockVectorAX'*blockVectorR(:,activeMask));
        //     gramRAR=full(blockVectorAR(:,activeMask)'*blockVectorR(:,activeMask));
        //     gramRAR=(gramRAR'+gramRAR)*0.5;

        //_XTY_v1(blockVectorAX, activeBlockVectorR, gramXAR, M, blocksize, currentBlockSize, block_width);

        //different type of task format (2n one) 
        //xty 4
        xty_id = 4;
        
        //cout << "before calling xty 4: row:" << M << " col: " << blocksize << " currentBlockSize: " << currentBlockSize << " id: " << xty_id << endl;

        _XTY_v1(1, "_AX", "null", "null", "null", "null", -1,
                2, "null", "actR", "DLACPY", "temp3R", 1,
                "_AX", "activeBlockVectorR", "XAR", M, blocksize, currentBlockSize, block_width, xty_id);

        //_XTY_v1(activeBlockVectorAR, activeBlockVectorR, gramRAR, M, currentBlockSize, currentBlockSize, block_width); 
        //xty 5
        xty_id = 5;
        
        //from v10
        //_XTY_v1(2, "null", "activeBlockVectorAR", "SPMM", "A", "activeBlockVectorR", 1,
        //    2, "null", "activeBlockVectorR", "DLACPY", "temp3R", 1,
        //    "activeBlockVectorAR", "activeBlockVectorR", "gramRAR", M, currentBlockSize, currentBlockSize, block_width, xty_id);
        
        //from v12
        //_XTY_v2(2, "null", "activeBlockVectorAR", "SPMMREDUCTION", "tempactiveBlockVectorARSPMMBUF", 
        //2, "null", "activeBlockVectorR", "DLACPY", "temp3R",
        //"activeBlockVectorAR", "activeBlockVectorR", "gramRAR", M, currentBlockSize, currentBlockSize, block_width);

        //new -> takes task id for 2nd edge but not for SPMM
        _XTY_v3(2, "null", "actAR", "SPMMRED", "SPMMBUF", 
                2, "null", "actR", "DLACPY", "temp3R", 1,
                "activeBlockVectorAR", "activeBlockVectorR", "RAR", M, currentBlockSize, currentBlockSize, block_width, xty_id, SPMM_vertexNo);

        t1 = omp_get_wtime();

        vertexName[strdup("TRANS,RAR")] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        edgeU[edgeCount] = vertexName["RED,RARBUF,0"];
        edgeV[edgeCount] = nodeCount - 1; //vertexName["transGramRAR(0)_TRANS(gramRAR)"]; -> last insert
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // #### Hier ####
        strcpy(tmp_input1,"RAR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["TRANS,RAR"]["RED,RARBUF,0"] = temp_chunk;
        out_map["RED,RARBUF,0"]["TRANS,RAR"] = temp_chunk;

        //printf("input_map[TRANS,RAR][RED,RARBUF] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);

        
        vertexName[strdup("SPEUPDATE,RAR")] = nodeCount;
        vertexWeight[nodeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        nodeCount++;

        edgeU[edgeCount] = vertexName["RED,RARBUF,0"];
        edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"]; -> last insert
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        // #### Hier ####
        strcpy(tmp_input1,"RAR,0");


        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["SPEUPDATE,RAR"]["RED,RARBUF,0"] = temp_chunk;
        out_map["RED,RARBUF,0"]["SPEUPDATE,RAR"] = temp_chunk;

        //printf("input_map[SPEUPDATE,RAR][RED,RARBUF] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


        edgeU[edgeCount] = nodeCount - 2; //vertexName["transGramRAR(0)_TRANS(gramRAR)"]; -> 2nd last insert
        edgeV[edgeCount] = nodeCount - 1; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"]; -> last insert
        edgeW[edgeCount] = currentBlockSize * currentBlockSize * sizeof(double);
        edgeCount++;

        strcpy(tmp_input1,"transRAR,0");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["SPEUPDATE,RAR"]["TRANS,RAR"] = temp_chunk;
        out_map["TRANS,RAR"]["SPEUPDATE,RAR"] = temp_chunk;

        //printf("input_map[SPEUPDATE,RAR][TRANS,RAR] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        //inner for loop start here

        if(iterationNumber >= 2) //equivalent to if(restart == 0) 
        {
            // OP : gramXAP = full(blockVectorAX'*blockVectorP(:,activeMask)); 
            //_XTY_v1(blockVectorAX, activeBlockVectorP, gramXAP, M, blocksize, currentBlockSize, block_width);
            //xty 6
            xty_id = 6;
            _XTY_v1(1, "_AX", "null", "null", "null", "null", -1,
                2, "null", "actP", "DLACPY", "temp3P", 2,
                "_AX", "activeBlockVectorP", "XAP", M, blocksize, currentBlockSize, block_width, xty_id);
                                    
            // OP : gramRAP = full(blockVectorAR(:,activeMask)'*blockVectorP(:,activeMask)); 
            //_XTY_v1(activeBlockVectorAR, activeBlockVectorP, gramRAP, M, currentBlockSize, currentBlockSize, block_width);
            //xty 7
            xty_id = 7;
            //_XTY_v1(2, "null", "activeBlockVectorAR", "SPMM", "A", "activeBlockVectorR", 1,
            //        2, "null", "activeBlockVectorP", "DLACPY", "temp3P", 2,
            //        "activeBlockVectorAR", "activeBlockVectorP", "gramRAP", M, blocksize, currentBlockSize, block_width, xty_id);

            _XTY_v3(2, "null", "actAR", "SPMMRED", "SPMMBUF",
                2, "null", "actP", "DLACPY", "temp3P", 2,
                "activeBlockVectorAR", "activeBlockVectorP", "RAP", M, blocksize, currentBlockSize, block_width, xty_id, SPMM_vertexNo);
                                    
            // OP : gramPAP=full(blockVectorAP(:,activeMask)'*blockVectorP(:,activeMask));
            //2 diff types of task edge 
            //_XTY_v1(activeBlockVectorAP, activeBlockVectorP, gramPAP, M, currentBlockSize, currentBlockSize, block_width);
            //xty 8
            xty_id = 8;
            _XTY_v2(2, "null", "actAP", "DLACPY", "temp3AP", 3,
                    2, "null", "actP", "DLACPY", "temp3P", 2,
                    "activeBlockVectorAP", "activeBlockVectorP", "PAP", M, blocksize, currentBlockSize, block_width, xty_id);

            /* OP: gramPAP=(gramPAP' + gramPAP)*0.5; */

            t1 = omp_get_wtime();
            vertexName[strdup("SPEUPDATE,PAP")] = nodeCount;
            vertexWeight[nodeCount] = blocksize * currentBlockSize * sizeof(double);
            nodeCount++;

            edgeU[edgeCount] = vertexName["RED,PAPBUF,0"];
            edgeV[edgeCount] = nodeCount - 1; //vertexName[nodeName, "gramPAP(0)_UPDATE(gramPAP)"]; -> last insert
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"PAP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["SPEUPDATE,PAP"]["RED,PAPBUF"] = temp_chunk;
            out_map["RED,PAPBUF"]["SPEUPDATE,PAP"] = temp_chunk;

            //printf("input_map[SPEUPDATE,PAP][RED,PAPBUF] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);



            //CONSTRUCTGA1  //gramXAR, lambda, gramXAP
            vertexName[strdup("CONSTRUCTGA1")] = nodeCount;
            vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
            CONSTRUCTGA1_id = nodeCount;
            nodeCount++;
            
            //gramXAP -> gramXAP(0)_REDUCTION(gramXAPBUF);
            
            edgeU[edgeCount] = _lambda_id; 
            edgeV[edgeCount] = CONSTRUCTGA1_id; 
            edgeW[edgeCount] = blocksize * blocksize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"_lambda");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA1"]["_lambda"] = temp_chunk;
            out_map["_lambda"]["CONSTRUCTGA1"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA1][_lambda] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


            //gramXAR -> gramXAR(0)_REDUCTION(gramXARBUF);
            
            edgeU[edgeCount] = vertexName["RED,XAPBUF,0"];
            edgeV[edgeCount] =  CONSTRUCTGA1_id; //vertexName["CONSTRUCTGA1"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"XAP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA1"]["RED,XAPBUF,0"] = temp_chunk;
            out_map["RED,XAPBUF,0"]["CONSTRUCTGA1"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA1][RED,XAPBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);





            edgeU[edgeCount] = vertexName["RED,XARBUF,0"];
            edgeV[edgeCount] = CONSTRUCTGA1_id; //vertexName["CONSTRUCTGA1"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"XAR,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA1"]["RED,XARBUF,0"] = temp_chunk;
            out_map["RED,XARBUF,0"]["CONSTRUCTGA1"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA1][RED,XARBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);

            
            vertexName[strdup("CONSTRUCTGA2")] = nodeCount;
            CONSTRUCTGA2_id = nodeCount;
            vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
            nodeCount++;

            edgeU[edgeCount] = CONSTRUCTGA1_id;
            edgeV[edgeCount] = CONSTRUCTGA2_id; 
            edgeW[edgeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"gramA");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA2"]["CONSTRUCTGA1"] = temp_chunk;
            out_map["CONSTRUCTGA1"]["CONSTRUCTGA2"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA2][CONSTRUCTGA1] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


            //CONSTRUCTGA2 <- gramXAR  gramXAP gramRAR gramRAP           
            //gramXAP -> gramXAP(0)_REDUCTION(gramXAPBUF);
            //gramXAR -> gramXAR(0)_REDUCTION(gramXARBUF);
            
            edgeU[edgeCount] = vertexName["RED,XAPBUF,0"];
            edgeV[edgeCount] = CONSTRUCTGA2_id; //vertexName["CONSTRUCTGA2"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"XAP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA2"]["RED,XAPBUF,0"] = temp_chunk;
            out_map["RED,XARPBUF,0"]["CONSTRUCTGA2"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA2][RED,XAPBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




            //gramRAR - > gramRAR(0)_UPDATE(gramRAR_transGramRAR)
            edgeU[edgeCount] = vertexName["RED,XARBUF,0"];
            edgeV[edgeCount] = CONSTRUCTGA2_id; //vertexName["CONSTRUCTGA2"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"XAR,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA2"]["RED,XARBUF,0"] = temp_chunk;
            out_map["RED,XARBUF,0"]["CONSTRUCTGA2"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA2][RED,XARBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);





            //gramRAP - > gramRAP(0)_REDUCTION(gramRAPBUF); 

            edgeU[edgeCount] = vertexName["SPEUPDATE,RAR"]; //vertexName["gramRAR(0)_UPDATE(gramRAR_transGramRAR)"];
            edgeV[edgeCount] = CONSTRUCTGA2_id; //vertexName["CONSTRUCTGA2"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"RAR,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA2"]["SPEUPDATE,RAR"] = temp_chunk;
            out_map["SPEUPDATE,RAR"]["CONSTRUCTGA2"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA2][SPEUPDATE,RAR] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




            /* this was missing before */
            edgeU[edgeCount] = vertexName["SPEUPDATE,PAP"]; 
            edgeV[edgeCount] = CONSTRUCTGA2_id; //vertexName["CONSTRUCTGA2"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"PAP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA2"]["SPEUPDATE,PAP"] = temp_chunk;
            out_map["SPEUPDATE,PAP"]["CONSTRUCTGA2"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA2][SPEUPDATE,PAP] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




            edgeU[edgeCount] = vertexName["RED,RAPBUF,0"];
            edgeV[edgeCount] = CONSTRUCTGA2_id; //vertexName["CONSTRUCTGA2"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"RAP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGA2"]["RED,RAPBUF,0"] = temp_chunk;
            out_map["RED,RAPBUF,0"]["CONSTRUCTGA2"] = temp_chunk;

            //printf("input_map[CONSTRUCTGA2][RED,RAPBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


            t2 = omp_get_wtime();
            graphGenTime[12] += (t2 - t1);
            
            //gramB

            //_XTY_v1(blockVectorX, activeBlockVectorP, gramXBP, M, blocksize, currentBlockSize, block_width);
            //xty 9
            xty_id = 9;
            _XTY_v1(1, "_X", "null", "null", "null", "null", -1,
                2, "null", "actP", "DLACPY", "temp3P", 2,
                "_X", "activeBlockVectorP", "XBP", M, blocksize, currentBlockSize, block_width, xty_id);

            //_XTY_v1(activeBlockVectorR, activeBlockVectorP, gramRBP, M, currentBlockSize, currentBlockSize, block_width);
            //xty 10
            xty_id = 10;
            _XTY_v2(2, "null", "actR", "DLACPY", "temp3R", 1,
                    2, "null", "actP", "DLACPY", "temp3P", 2,
                    "activeBlockVectorR", "activeBlockVectorP", "RBP", M, currentBlockSize, currentBlockSize, block_width, xty_id);

            //gramXBP -> gramXBP(0)_REDUCTION(gramXBPBUF);
            //gramRBP -> gramRBP(0)_REDUCTION(gramRBPBUF);
            t1 = omp_get_wtime();

            //nodeName.push_back("CONSTRUCTGB");
            vertexName[strdup("CONSTRUCTGB")] = nodeCount;
            vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
            CONSTRUCTGB_id = nodeCount;
            nodeCount++;

            edgeU[edgeCount] = vertexName["RED,XBPBUF,0"];
            edgeV[edgeCount] = vertexName["CONSTRUCTGB"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"XBP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGB"]["RED,XBPBUF,0"] = temp_chunk;
            out_map["RED,XBPBUF,0"]["CONSTRUCTGB"] = temp_chunk;

            //printf("input_map[CONSTRUCTGB][RED,XBPBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);

            edgeU[edgeCount] = vertexName["RED,RBPBUF,0"];
            edgeV[edgeCount] = CONSTRUCTGB_id; //vertexName["CONSTRUCTGB"];
            edgeW[edgeCount] = blocksize * currentBlockSize * sizeof(double);
            edgeCount++;

            // ##### Hier #####
            strcpy(tmp_input1,"RBP,0");
            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map["CONSTRUCTGB"]["RED,RBPBUF,0"] = temp_chunk;
            out_map["RED,RBPBUF,0"]["CONSTRUCTGB"] = temp_chunk;

            //printf("input_map[CONSTRUCTGB][RED,RBPBUF,0] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




            t2 = omp_get_wtime();
            graphGenTime[12] += (t2 - t1);           
        }

        t1  = omp_get_wtime();

        vertexName[strdup("EIGEN")] = nodeCount;
        vertexWeight[nodeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                        (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
        EIGEN_id = nodeCount;
        nodeCount++;

        edgeU[edgeCount] = CONSTRUCTGA2_id; //vertexName["CONSTRUCTGA2"];
        edgeV[edgeCount] = EIGEN_id; //vertexName["EIGEN"];
        edgeW[edgeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                    (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
        edgeCount++;

         // ##### Hier #####
        strcpy(tmp_input1,"gramA");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["EIGEN"]["CONSTRUCTGA2"] = temp_chunk;
        out_map["CONSTRUCTGA2"]["EIGEN"] = temp_chunk;

        //printf("input_map[EIGEN][CONSTRUCTGA2] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);




        edgeU[edgeCount] = CONSTRUCTGB_id; //vertexName["CONSTRUCTGB"];
        edgeV[edgeCount] = EIGEN_id; //vertexName["EIGEN"];
        edgeW[edgeCount] = (blocksize + currentBlockSize + currentBlockSize) * 
                    (blocksize + currentBlockSize + currentBlockSize) * sizeof(double);
        edgeCount++;

         // ##### Hier #####
        strcpy(tmp_input1,"gramB");
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map["EIGEN"]["CONSTRUCTGB"] = temp_chunk;
        out_map["CONSTRUCTGB"]["EIGEN"] = temp_chunk;

        //printf("input_map[EIGEN][CONSTRUCTGB] = %s %lf\n",tmp_input1,edgeW[edgeCount-1]);


        t2 = omp_get_wtime();
        graphGenTime[12] += (t2 - t1);

        int add_id = -1;
        
        //last part after EIGEN
        if(iterationNumber >= 2)
        {
            // blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + blockVectorP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize+activeRSize+activePSize,:); 
            // partil result- part1:- blockVectorP =  blockVectorR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:)

            //_XY(activeBlockVectorR, coordX + (blocksize * blocksize), blockVectorP, M, currentBlockSize, blocksize, block_width);
            //xy 6
            xy_id = 6;
            _XY_v1(2, "null", "actR", "DLACPY", "temp3R", 1,
                    1, "EIGEN", "null", "null", "null", "null",
                    "activeBlockVectorR", "EIGEN", "blockVectorP", M, currentBlockSize, blocksize, block_width, xy_id);
            //_XY(activeBlockVectorP, coordX+((blocksize+currentBlockSize)*blocksize), newP, M, currentBlockSize, blocksize, block_width);
            //xy 7
            xy_id = 7;
            _XY_v1(2, "null", "actP", "DLACPY", "temp3P", 2,
                    1, "EIGEN", "null", "null", "null", "null",
                    "activeBlockVectorP", "EIGEN", "newP", M, currentBlockSize, blocksize, block_width, xy_id);
            
            //mat_addition_task(blockVectorP, newP, blockVectorP, M, blocksize, block_width);
            //add 1
            add_id = 1;
            mat_addition(2, "null", "P", "XY", "actR", "EIGEN", 6,
                        2, "null", "newP", "XY", "actP", "EIGEN",  xy_id,
                        "blockVectorP", "newP", "blockVectorP", M, blocksize, block_width, add_id);
            
            //  OP: blockVectorAP = blockVectorAR(:,activeMask)*coordX(blockSize+1:blockSize+activeRSize,:) + 
            //  blockVectorAP(:,activeMask)*coordX(blockSize+activeRSize+1:blockSize + activeRSize+activePSize,:);

            //_XY(activeBlockVectorAR, coordX+(blocksize*blocksize), blockVectorAP, M, currentBlockSize, blocksize, block_width);
            //xy 8
            xy_id = 8;
            //_XY(2, "null", "activeBlockVectorAR", "SPMM", "A", "activeBlockVectorR", spmm_id,
            //    1, "EIGEN", "null", "null", "null", "null", -1,
            //    "activeBlockVectorAR", "EIGEN", "blockVectorAP", M, currentBlockSize, blocksize, block_width, xy_id);

            _XY_v2(2, "null", "actAR", "SPMMRED", "SPMMBUF", 
                    "EIGEN",
                    "activeBlockVectorAR", "EIGEN", "_AP", M, currentBlockSize, blocksize, block_width, xy_id, SPMM_vertexNo);

            //_XY(activeBlockVectorAP, coordX+((blocksize+currentBlockSize)*blocksize), newAP, M, currentBlockSize, blocksize, block_width);
            //xy 9
            xy_id = 9;
            _XY_v1(2, "null", "actAP", "DLACPY", "temp3AP", 3,
                    1, "EIGEN", "null", "null", "null", "null",
                    "activeBlockVectorAP", "EIGEN", "newAP", M, currentBlockSize, blocksize, block_width, xy_id);

            //mat_addition_task(blockVectorAP, newAP, blockVectorAP, M, blocksize, block_width);
            //add 2
            add_id = 2;
            mat_addition(2, "null", "AP", "XY", "actAR", "EIGEN", 8,
                        2, "null", "newAP", "XY", "actAP", "EIGEN", xy_id,
                        "blockVectorAP", "newAP", "blockVectorAP", M, blocksize, block_width, add_id);
        }

        // OP: blockVectorX = blockVectorX*coordX(1:blockSize,:) + blockVectorP; 
        //_XY(blockVectorX, coordX, newX, M, blocksize, blocksize, block_width);
        //xy 10
        xy_id = 10;
        _XY(1, "_X", "null", "null", "null", "null", -1,
            1, "EIGEN", "null", "null", "null", "null", -1,
            "_X", "EIGEN", "newX", M, blocksize, blocksize, block_width, xy_id);

        //mat_addition_task(newX, blockVectorP, blockVectorX, M, blocksize, block_width);
        //add 3
        add_id = 3;
        mat_addition(2, "null", "newX", "XY", "X", "EIGEN", xy_id,
                    2, "null", "P", "ADD", "P", "newP", 1,
                    "newX", "blockVectorP", "_X", M, blocksize, block_width, add_id); //blockVectorP(0)_ADD(blockVectorP_newP_0);
        
        /* OP: blockVectorAX=blockVectorAX*coordX(1:blockSize,:) + blockVectorAP; */
        
        //_XY(blockVectorAX, coordX, newAX, M, blocksize, blocksize, block_width); 
        //xy 11
        xy_id = 11;
        _XY(1, "_AX", "null", "null", "null", "null", -1,
            1, "EIGEN", "null", "null", "null", "null", -1,
            "_AX", "EIGEN", "newAX", M, blocksize, blocksize, block_width, xy_id);

        //mat_addition_task(newAX, blockVectorAP, blockVectorAX, M, blocksize, block_width);
        //add 4
        add_id = 4;
        mat_addition(2, "null", "newAX", "XY", "AX", "EIGEN", xy_id,
                    2, "null", "AP", "ADD", "AP", "newAP", 2,
                    "newAX", "blockVectorAP", "_AX", M, blocksize, block_width, add_id);

    } // end for

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
        //  printf("%d = %s \n",aelement.second,element.first);
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

    MLGP_option opt;
    
    processArgs_rMLGP(argc, argv, &opt);
    opt.co_stop_size = 30;
    opt.co_stop_level = 1000;
     // opt.conpar = 0;
    // opt.inipart = 11;
    opt.use_binary_input = 0;
    printf("Calling run_rMLGP from main\n");
    
    //run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, &vertex_name_string[0], vertexWeight, numrows, numcols, nrowblks, ncolblks, block_width);
    run_rMLGP(opt.file_name, opt, edgeU, edgeV, edgeW, edgeCount, nodeCount, &vertex_name_string[0], vertexWeight, 2, "msdoor");
    printf("run_rMLGP Finshed\n");
    //run_rMLGP(opt.file_name, opt);

    free(pseudo_tid_map);
    
    free_opt(&opt);

//    return 0;
}







