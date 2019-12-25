#include "util.h"




//------ global parametes (block) ---------

long position = 0 ;
int *colptrs, *irem;
int nrows, ncols, nnz, numrows, numcols, nnonzero, nthrds = 32;
int wblk, nrowblks, ncolblks, nthreads;
int *nnzPerRow;
//block<double> *matrixBlock;
block *matrixBlock;




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

double total_memory_limit; 



//int **pseudo_tid_map;
//int small_block;

char **globalGraph;
int globalNodeCount;





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

    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;


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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;


        
        strcpy(main_task,ary);

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

            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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

            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


        }

        if(edge2Format == 1) // edge coming from a single matrix 
        {
            edgeU[edgeCount] = edge2_id;
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;

            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

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

            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

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

    //printf("edgeCount in _XY  = %d nodeCount = %d edge1Format = %d edge2Format = %d\n", edgeCount, nodeCount, edge1Format, edge2Format);

    if(edge2Format == 1) 
    {
        edge2_id = vertexName[edge2_var];
    }

    //printf("inside XY function edge1_var = %s edge2_var = %s\n", edge1_var,edge2_var);

    // #### Hier #####
    char main_task[100];
    char tmp_input1[100];
    char tmp_input2[100];
    memory_chunk temp_chunk;

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, xy_id_char);

        // #### Hier #####
        strcpy(main_task, ary);




        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * P * sizeof(double);
        nodeCount++;

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
            
            //printf("in XY edgeU[%d] = %d edgeV[%d] = %d \n", edgeCount,edgeU[edgeCount],edgeCount,edgeV[edgeCount]);


            edgeCount++;

            
            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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

            //printf("in XY edgeU[%d] = %d edgeV[%d] = %d \n", edgeCount,edgeU[edgeCount],edgeCount,edgeV[edgeCount]);

            edgeCount++;
            
            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        }

        if(edge2Format == 1) //whole edge2_var
        {   
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
       
            //printf("in XY edgeU[%d] = %d edgeV[%d] = %d \n", edgeCount,edgeU[edgeCount],edgeCount,edgeV[edgeCount]);


            edgeCount++; 
        
            // #### Hier #####
            strcpy(tmp_input2,input2);
            

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(edge2_var)] = temp_chunk;
            out_map[strdup(edge2_var)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2_var,tmp_input2,edgeW[edgeCount-1]);


        }
        else
        {   
            /*memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            strcat(ary, ",");
            strcat(ary, task_id2_char);
               
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N * sizeof(double);
            edgeCount++; 

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);*/
            //quick fix ==> Double check later
            edge2_id = vertexName[edge2_var];
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = N * P *  sizeof(double); //block_width * N * sizeof(double);
            edgeCount++; 

            // #### Hier #####
            strcpy(tmp_input2, input2);
            strcat(tmp_input2, ",0");
            strcpy(temp_chunk.memory_name, tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(edge2_var)] = temp_chunk;
            out_map[strdup(edge2_var)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2_var,tmp_input2,edgeW[edgeCount-1]);

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

    //printf("inside XY function edge1_var = %s edge2_var = %s\n", edge1_var,edge2_var);

    char main_task[100];
    char tmp_input1[100];
    char tmp_input2[100];
    memory_chunk temp_chunk;

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "XY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, xy_id_char);

        strcpy(main_task,ary);




        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * P * sizeof(double);
        nodeCount++;

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


            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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


            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        }

        if(edge2Format == 1) //whole edge2_var
        {
            edgeU[edgeCount] = edge2_id;
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * N  *sizeof(double);
            edgeCount++; 

            strcpy(tmp_input2,input2);
            

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(edge2_var)] = temp_chunk;
            out_map[strdup(edge2_var)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2_var,tmp_input2,edgeW[edgeCount-1]);


        }
        else
        {
            /*memset(&ary[0], 0, sizeof(ary));
            strcat(ary, edge2_func);
            strcat(ary, ",");
            strcat(ary, nrowblksString[i]);
            
            edgeU[edgeCount] = vertexName[ary];
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = N * P *sizeof(double);
            edgeCount++; 

            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);*/

            //quick fix ==> Double check later
            edge2_id = vertexName[edge2_var];
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = N * P *  sizeof(double); //block_width * N * sizeof(double);
            edgeCount++; 

            // #### Hier #####
            strcpy(tmp_input2, input2);
            strcat(tmp_input2, ",0");
            strcpy(temp_chunk.memory_name, tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(edge2_var)] = temp_chunk;
            out_map[strdup(edge2_var)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2_var,tmp_input2,edgeW[edgeCount-1]);

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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    char spmm_task[150];
    memory_chunk temp_chunk;

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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        // #### Hier #####
        strcpy(main_task,ary);

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


             

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

                    // #### Hier #####
                    strcpy(spmm_task, "SPMM,");
                    strcat(spmm_task, nrowblksString[i]);
                    strcat(spmm_task, ",");
                    strcat(spmm_task, nrowblksString[l]);
                    strcat(spmm_task, ",");
                    strcat(spmm_task, nrowblksString[pseudo_tid_map[i][l]]);

                    strcpy(tmp_input1,input1);
                    strcat(tmp_input1,",");
                    strcat(tmp_input1,nrowblksString[i]);

                    strcpy(temp_chunk.memory_name,tmp_input1);
                    temp_chunk.value = edgeW[edgeCount-1];

                    inp_map[strdup(main_task)][strdup(spmm_task)] = temp_chunk;
                    out_map[strdup(spmm_task)][strdup(main_task)] = temp_chunk;

                    //printf("input_map[%s][%s] = %s %lf\n", main_task,spmm_task,tmp_input1,edgeW[edgeCount-1]);


                }
            }      
        }

        //whole edge2_var, EIGEN
        edgeU[edgeCount] = edge2_id;
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = N * P * sizeof(double);
        edgeCount++;      

        // #### Hier #####
        strcpy(tmp_input1,input2);

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(main_task)][strdup(edge2_var)] = temp_chunk;
        out_map[strdup(edge2_var)][strdup(main_task)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2_var,tmp_input1,edgeW[edgeCount-1]);



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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
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

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        if( (i * block_width + block_width) > row)
            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);        
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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);      
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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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
            edgeW[edgeCount] = nthrds * col * p * sizeof(double); //=> changed //sizeof(double); 

            //printf("%s --> %s %lf nthrds = %d col = %d p = %d \n", main_task, ary, edgeW[edgeCount],nthrds,col,p);
            edgeCount++;

            // #### Hier #####
            strcpy(tmp_input1,output);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[l]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(ary)][strdup(main_task)] = temp_chunk;
            out_map[strdup(main_task)][strdup(ary)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", ary,main_task,tmp_input1,edgeW[edgeCount-1]);

        }
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

    // #### Hier #####
    char main_task[100];
    char tmp_input1[100];
    char tmp_input2[100];
    memory_chunk temp_chunk;

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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
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

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        if( (i * block_width + block_width) > row)
        {

            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
            //cout << "Hi " << i * block_width + block_width << " " << row << " " << vertexWeight[nodeCount] << endl;
        }
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);        
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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);      
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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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
            edgeW[edgeCount] = nthrds * col * p * sizeof(double); //sizeof(double);
            edgeCount++;

            // #### Hier #####
            strcpy(tmp_input1,output);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[l]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(ary)][strdup(main_task)] = temp_chunk;
            out_map[strdup(main_task)][strdup(ary)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", ary,main_task,tmp_input1,edgeW[edgeCount-1]);

        }
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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
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

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        
        if( (i * block_width + block_width) > row)
            vertexWeight[nodeCount] = (row - i * block_width) * p * sizeof(double);
        else
            vertexWeight[nodeCount] = block_width * p * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);        
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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);      
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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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
            edgeW[edgeCount] = nthrds * col * p * sizeof(double); //sizeof(double);
            edgeCount++;

            // #### Hier #####
            strcpy(tmp_input1,output);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[l]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(ary)][strdup(main_task)] = temp_chunk;
            out_map[strdup(main_task)][strdup(ary)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", ary,main_task,tmp_input1,edgeW[edgeCount-1]);

        }
    }

    tend = omp_get_wtime();
    graphGenTime[0] += (tend - tstart);   
}



void  _XTY_v3(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    char spmm_task[150];
    memory_chunk temp_chunk;

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


        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;
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

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        nodeCount++;

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

                    // #### Hier #####
                    strcpy(spmm_task, "SPMM,");
                    strcat(spmm_task, nrowblksString[i]);
                    strcat(spmm_task, ",");
                    strcat(spmm_task, nrowblksString[l]);
                    strcat(spmm_task, ",");
                    strcat(spmm_task, nrowblksString[pseudo_tid_map[i][l]]);

                    //printf("\n\nspmm_task  %s\n\n",spmm_task);

                    strcpy(tmp_input1,input1);
                    strcat(tmp_input1,",");
                    strcat(tmp_input1,nrowblksString[i]);

                    strcpy(temp_chunk.memory_name,tmp_input1);
                    temp_chunk.value = edgeW[edgeCount-1];

                    inp_map[strdup(main_task)][strdup(spmm_task)] = temp_chunk;
                    out_map[strdup(spmm_task)][strdup(main_task)] = temp_chunk;


                    //printf("input_map[%s][%s] = %s %lf\n",main_task,spmm_task, tmp_input1,edgeW[edgeCount-1]);



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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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

            // #### Hier #####
            strcpy(tmp_input1,input2);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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
            edgeW[edgeCount] = nthrds * col * p * sizeof(double); //sizeof(double);
            edgeCount++;

            // #### Hier #####
            strcpy(tmp_input1,output);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[l]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(ary)][strdup(main_task)] = temp_chunk;
            out_map[strdup(main_task)][strdup(ary)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", ary,main_task,tmp_input1,edgeW[edgeCount-1]);

        }
    }
    tend = omp_get_wtime();
    graphGenTime[0] += (tend - tstart);
}



void spmm_blkcoord_csbTask(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int currentBlockSize, int taskwait_node_no, int *actR_vertexNo, int **SPMM_vertexNo)
{
    //printf("inside spmm_blkcoord_csbTask function\n");
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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    char extra_task1[150];
    char extra_task2[150];
    memory_chunk temp_chunk;

    

    tstart = omp_get_wtime();

    for(i = 0 ; i < nrowblks ; i++)
    {
        //t1 = omp_get_wtime();   
     
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SETZERO,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",1");

        // #### Hier #####
        strcpy(extra_task2, ary);

        printf("%s\n", extra_task2 );
     
        //t2 = omp_get_wtime();
        //sprintf_time += (t2 - t1);
        //t1 = omp_get_wtime();

        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        buf_setzero_node_no = nodeCount; //saving it to use later in nested looop
        nodeCount++;


        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        //printf("vertexName set\n");

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

                //printf("nthreads = %d\n", nthreads);

                pseudo_tid = ( (((i * ncolblks) + j) % nbuf) > (nthreads - 1) ? 0 : ( ((i * ncolblks) + j) % nbuf ) );
                k = pseudo_tid;
                max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );
                
                //t1 = omp_get_wtime();

                //printf("inside spmm block calc k = %d\n", k);
                //printf(" %s sizeof ary %d i =%d j = %d\n", nrowblksString[k], sizeof(ary),i,j);
                
                memset(&ary[0], 0, sizeof(ary));
                strcat(ary, "SPMM,");
                strcat(ary, nrowblksString[i]);
                strcat(ary, ",");
                strcat(ary, nrowblksString[j]);
                strcat(ary, ",");
                strcat(ary, nrowblksString[k]);

                // #### Hier #####
                //printf("pseudo_tid_map[0][0] = %d\n", pseudo_tid_map[0][0]);
                pseudo_tid_map[i][j] = k;
                strcpy(main_task, ary);
                //strcat(ary, ")");

               // printf("%s\n", main_task);

                //t2 = omp_get_wtime();
                //sprintf_time += (t2 - t1);

                //t1 = omp_get_wtime();


                vertexName[strdup(ary)] = nodeCount;
                vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
                SPMM_vertexNo[i][j] = nodeCount; //saving SPMM node number for later use
                nodeCount++;

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

                // #### Hier #####
                strcpy(tmp_input1,input2);
                strcat(tmp_input1,",");
                strcat(tmp_input1,nrowblksString[j]);

                strcpy(temp_chunk.memory_name,tmp_input1);
                temp_chunk.value = edgeW[edgeCount-1];

                strcpy(extra_task1,"DLACPY,");  

                //char dla_id[4];
                //myitoa(j,dla_id);
                //strcat(extra_task1,dla_id);
                strcat(extra_task1,nrowblksString[j]);
                strcat(extra_task1,",1");


                inp_map[strdup(main_task)][strdup(extra_task1)] = temp_chunk;
                out_map[strdup(extra_task1)][strdup(main_task)] = temp_chunk;

                //printf("input_map[%s][%s] = %s %lf\n", main_task,extra_task1,tmp_input1,edgeW[edgeCount-1]);


                //SETZERO AR to SPMM
                edgeU[edgeCount] = buf_setzero_node_no; 
                edgeV[edgeCount] = nodeCount - 1; 
                edgeW[edgeCount] = block_width * p * sizeof(double); //single block of a particular partial buffer
                edgeCount++;

                strcpy(tmp_input1,output);
                strcat(tmp_input1,",");
                strcat(tmp_input1,nrowblksString[i]);

                strcpy(temp_chunk.memory_name, tmp_input1);
                temp_chunk.value = edgeW[edgeCount-1];

                inp_map[strdup(main_task)][strdup(extra_task2)] = temp_chunk;
                out_map[strdup(extra_task2)][strdup(main_task)] = temp_chunk;

                //printf("input_map[%s][%s] = %s %lf\n", main_task,extra_task2,tmp_input1,edgeW[edgeCount-1]);



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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;


    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DLACPY,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, dlacpy_id_char);

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);

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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        // #### Hier #####
        strcpy(main_task,ary);

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);


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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);

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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "GET,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, get_id_char);

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        }

        edgeU[edgeCount] = edge2_id;
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = col * sizeof(double);
        edgeCount++;

        // #### Hier #####
        strcpy(tmp_input1,input2);
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(main_task)][strdup(edge2)] = temp_chunk;
        out_map[strdup(edge2)][strdup(main_task)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2,tmp_input1,edgeW[edgeCount-1]);





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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;
    
    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "UPDATE,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, update_id_char);

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * currentBlockSize * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);



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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);

        }

        edgeU[edgeCount] = edge2_id; 
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = col * sizeof(double);
        edgeCount++; 

        // #### Hier #####
        strcpy(tmp_input1,input2);
        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(main_task)][strdup(edge2)] = temp_chunk;
        out_map[strdup(edge2)][strdup(main_task)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", main_task,edge2,tmp_input1,edgeW[edgeCount-1]);

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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "SUB,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, sub_id_char);

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount;
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;


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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
    
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

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

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

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

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

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

    for(i = 0 ; i < nrowblks ; i++)
    {
        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "MULT,");
        strcat(ary, nrowblksString[i]);

        // #### Hier #####
        strcpy(main_task,ary);
        
        vertexName[strdup(ary)] = nodeCount; 
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        
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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        
        }

        if(edge2Format == 1) // edge coming from a single matrix 
        {
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

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

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

        }
    }

    tend = omp_get_wtime();
    graphGenTime[5] += (tend - tstart);
}


void dot_mm(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int block_width)
{
    /* funciton code: 5 */
    int nbuf = 16, k, pseudo_tid, max_pesudo_tid = -1;

    string dummyString = "";
    double tstart, tend;
    int i, edge2_id;
    char i_string[8], task_id1_char[4], task_id2_char[4];
    char ary[150];
       
    myitoa(task_id_1, task_id1_char);
    myitoa(task_id_2, task_id2_char);

    //added for DOT [RNRED,RNBUF] is moved from sum_sqrt to here
    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RNRED,");
    strcat(ary, "RNBUF");
    
    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = col * sizeof(double);
    int sumsqrt_buf_id = nodeCount;
    printf("sumsqrt_buf_id : %d\n", sumsqrt_buf_id);
    nodeCount++;


        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

    char *tempRNRED = (char *) malloc(50 * sizeof(char));
    //Global Graph
    strcpy(tempRNRED, ary);

    //-------------
    
    tstart = omp_get_wtime();
    if(edge2Format == 1) // edge coming from a single matrix 
    {
        edge2_id = vertexName[edge2_var]; 
    }

    // #### Hier #####
    char main_task[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

    for(i = 0 ; i < nrowblks ; i++)
    {
        //memset(&ary[0], 0, sizeof(ary));
        //strcat(ary, "MULT,");
        //strcat(ary, nrowblksString[i]);

        pseudo_tid = ((i % nbuf) > (nthreads - 1) ? 0 : (i % nbuf) );
        k = pseudo_tid;
        max_pesudo_tid = ((max_pesudo_tid > pseudo_tid) ? max_pesudo_tid : pseudo_tid );

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, "DOT,");
        strcat(ary, nrowblksString[i]);
        strcat(ary, ",");
        strcat(ary, nrowblksString[k]);

        // #### Hier #####
        strcpy(main_task, ary);
        
        vertexName[strdup(ary)] = nodeCount; 
        vertexWeight[nodeCount] = block_width * col * sizeof(double);
        nodeCount++;

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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        
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

            // #### Hier #####
            strcpy(tmp_input1,input1);
            strcat(tmp_input1,",");
            strcat(tmp_input1,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input1);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input1,edgeW[edgeCount-1]);
        
        }

        if(edge2Format == 1) // edge coming from a single matrix 
        {
            edgeU[edgeCount] = edge2_id; 
            edgeV[edgeCount] = nodeCount - 1;
            edgeW[edgeCount] = block_width * col * sizeof(double);
            edgeCount++;

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

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

            // #### Hier #####
            strcpy(tmp_input2,input2);
            strcat(tmp_input2,",");
            strcat(tmp_input2,nrowblksString[i]);

            strcpy(temp_chunk.memory_name,tmp_input2);
            temp_chunk.value = edgeW[edgeCount-1];

            inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
            out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

            //printf("input_map[%s][%s] = %s %lf\n", main_task,ary,tmp_input2,edgeW[edgeCount-1]);

        }

        //DOT --> RNRED
        edgeU[edgeCount] = nodeCount - 1;
        edgeV[edgeCount] = sumsqrt_buf_id;
        edgeW[edgeCount] = nthreads * col * sizeof(double);
        edgeCount++;

        //need to add hier code here

        // #### Hier #####
        strcpy(tmp_input1, "RNBUF");
    

        strcpy(temp_chunk.memory_name,tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(tempRNRED)][strdup(main_task)] = temp_chunk;
        out_map[strdup(main_task)][strdup(tempRNRED)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", tempRNRED,main_task,tmp_input1,edgeW[edgeCount-1]);
    }
    
    //Global Graph
    //strcpy(globalGraph[globalNodeCount], tempRNRED);
    //globalNodeCount++;

    free(tempRNRED);

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

    char i_string[8], k_string[4], task_id2_char[4];
    char ary[150];

    // #### Hier #####
    char main_task[150];
    char extra_task1[150];
    char extra_task2[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "RNRED,");
    strcat(ary, "RNBUF");
    
    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = col * sizeof(double);
    int sumsqrt_buf_id = nodeCount;
    nodeCount++;

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

    // #### Hier #####
    strcpy(extra_task1, ary);

    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "SQRT,");
    strcat(ary, output);

    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = col * sizeof(double);
    int sqrt_id = nodeCount;
    nodeCount++;

            strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

    // #### Hier #####
    strcpy(extra_task2, ary);

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

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

        // #### Hier #####
        strcpy(main_task, ary);

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, edge1_func);
        strcat(ary, ",");
        strcat(ary, nrowblksString[i]);


        
        edgeU[edgeCount] = vertexName[ary];
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = block_width * col * sizeof(double);
        edgeCount++;

        // #### Hier #####
        strcpy(tmp_input1, input1);
        strcat(tmp_input1, ",");
        strcat(tmp_input1, nrowblksString[i]);

        strcpy(temp_chunk.memory_name, tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
        out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", main_task, ary, tmp_input1, edgeW[edgeCount-1]);

        
        //edge 2

        edgeU[edgeCount] = nodeCount - 1;
        edgeV[edgeCount] = sumsqrt_buf_id; 
        edgeW[edgeCount] = col * sizeof(double);
        edgeCount++;


        // #### Hier #####
        strcpy(tmp_input1, "RNBUF");


        strcpy(temp_chunk.memory_name, tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(extra_task1)][strdup(main_task)] = temp_chunk;
        out_map[strdup(main_task)][strdup(extra_task1)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", extra_task1, main_task, tmp_input1, edgeW[edgeCount-1]);




    }

    edgeU[edgeCount] = edge2_id; 
    edgeV[edgeCount] = sumsqrt_buf_id; 
    edgeW[edgeCount] = col * sizeof(double);
    edgeCount++;



    // #### Hier #####
    strcpy(tmp_input1, "residualNorms");
    

    strcpy(temp_chunk.memory_name, tmp_input1);
    temp_chunk.value = edgeW[edgeCount - 1];

    inp_map[strdup(extra_task1)][strdup(edge2)] = temp_chunk;
    out_map[strdup(edge2)][strdup(extra_task1)] = temp_chunk;

    //printf("input_map[%s][%s] = %s %lf\n", extra_task1, edge2, tmp_input1, edgeW[edgeCount-1]);




    edgeU[edgeCount] = sumsqrt_buf_id; 
    edgeV[edgeCount] = sqrt_id;
    edgeW[edgeCount] = col * sizeof(double);
    edgeCount++;

    tend = omp_get_wtime();
    graphGenTime[8] += (tend - tstart);

    // #### Hier #####
    strcpy(tmp_input1, "RNBUF");
    

    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map[strdup(extra_task2)][strdup(extra_task1)] = temp_chunk;
    out_map[strdup(extra_task1)][strdup(extra_task2)] = temp_chunk;

    //printf("input_map[%s][%s] = %s %lf\n", extra_task2,extra_task1,tmp_input1,edgeW[edgeCount-1]);


}

void sum_sqrt_dot(char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[],
            char edge2[], int edge2_id,
            char input1[], char output[], int row, int col, int block_width)
{
    /* funciton code: 3 */
    double tstart, tend;
    tstart = omp_get_wtime();

    int i, k;
    int nbuf = 16;
    int pseudo_tid, max_pesudo_tid = -1;

    char i_string[8], k_string[4], task_id2_char[4];
    char ary[150];

    // #### Hier #####
    char main_task[150];
    char extra_task1[150];
    char extra_task2[150];
    char tmp_input1[150];
    char tmp_input2[150];
    memory_chunk temp_chunk;

    //memset(&ary[0], 0, sizeof(ary));
    //strcat(ary, "RNRED,");
    //strcat(ary, "RNBUF");
    
    //vertexName[strdup(ary)] = nodeCount;
    //vertexWeight[nodeCount] = col * sizeof(double);
    //int sumsqrt_buf_id = nodeCount;
    //nodeCount++;

    // #### Hier #####
    //strcpy(extra_task1, ary);

    char *tempRNRED = (char *) malloc(50 * sizeof(char));
    //char *tempSQRT = (char *) malloc(50 * sizeof(char));

    memset(&tempRNRED[0], 0, sizeof(tempRNRED)); //RNRED,RNBUF already added in dot_mm
    strcat(tempRNRED, "RNRED,");
    strcat(tempRNRED, "RNBUF");

    // #### Hier #####
    strcpy(extra_task1, tempRNRED);

    memset(&ary[0], 0, sizeof(ary));
    strcat(ary, "SQRT,");
    strcat(ary, output);

    vertexName[strdup(ary)] = nodeCount;
    vertexWeight[nodeCount] = col * sizeof(double);
    int sqrt_id = nodeCount;
    nodeCount++;

        strcpy(globalGraph[globalNodeCount], ary);
        globalNodeCount++;

    //Global Graph
    //strcpy(tempSQRT, ary);

    // #### Hier #####
    strcpy(extra_task2, ary);

    /*for(i = 0, k = 0 ; i < nrowblks ; i++, k++)
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

        // #### Hier #####
        strcpy(main_task, ary);

        memset(&ary[0], 0, sizeof(ary));
        strcat(ary, edge1_func);
        strcat(ary, ",");
        strcat(ary, nrowblksString[i]);

        edgeU[edgeCount] = vertexName[ary];
        edgeV[edgeCount] = nodeCount - 1;
        edgeW[edgeCount] = block_width * col * sizeof(double);
        edgeCount++;

        // #### Hier #####
        strcpy(tmp_input1, input1);
        strcat(tmp_input1, ",");
        strcat(tmp_input1, nrowblksString[i]);

        strcpy(temp_chunk.memory_name, tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(main_task)][strdup(ary)] = temp_chunk;
        out_map[strdup(ary)][strdup(main_task)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", main_task, ary, tmp_input1, edgeW[edgeCount-1]);

        //edge 2
        edgeU[edgeCount] = nodeCount - 1;
        edgeV[edgeCount] = sumsqrt_buf_id; 
        edgeW[edgeCount] = col * sizeof(double);
        edgeCount++;

        // #### Hier #####
        strcpy(tmp_input1, "RNBUF");
        strcpy(temp_chunk.memory_name, tmp_input1);
        temp_chunk.value = edgeW[edgeCount-1];

        inp_map[strdup(extra_task1)][strdup(main_task)] = temp_chunk;
        out_map[strdup(main_task)][strdup(extra_task1)] = temp_chunk;

        //printf("input_map[%s][%s] = %s %lf\n", extra_task1, main_task, tmp_input1, edgeW[edgeCount-1]);
    }*/

    edgeU[edgeCount] = edge2_id; 
    edgeV[edgeCount] = vertexName[tempRNRED]; 
    edgeW[edgeCount] = col * sizeof(double);
    edgeCount++;

    // #### Hier #####
    //RESET,RN --> SQRT,RN
    strcpy(tmp_input1, "residualNorms");
    strcpy(temp_chunk.memory_name, tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map[strdup(ary)][strdup(edge2)] = temp_chunk;
    out_map[strdup(edge2)][strdup(ary)] = temp_chunk;

    //printf("input_map[%s][%s] = %s %lf\n", ary, edge2, tmp_input1, edgeW[edgeCount-1]);


    //RNRED,RNBUF --> SQRT,RN
    edgeU[edgeCount] = vertexName[tempRNRED];
    printf("tempRNRED : %d\n", vertexName[tempRNRED]);
    edgeV[edgeCount] = sqrt_id;
    edgeW[edgeCount] = nthreads * col * sizeof(double);
    edgeCount++;

    tend = omp_get_wtime();
    graphGenTime[8] += (tend - tstart);

    // #### Hier #####
    strcpy(tmp_input1, "RNBUF");
    strcpy(temp_chunk.memory_name,tmp_input1);
    temp_chunk.value = edgeW[edgeCount-1];

    inp_map[strdup(ary)][strdup(tempRNRED)] = temp_chunk;
    out_map[strdup(tempRNRED)][strdup(ary)] = temp_chunk;

    //printf("input_map[%s][%s] = %s %lf\n", ary,tempRNRED,tmp_input1,edgeW[edgeCount-1]);

    free(tempRNRED);
    //free(tempSQRT);

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

    fread(&nnonzero, sizeof(float), 1, fp);
    cout<<"non zero: "<<nnonzero<<endl;

    colptrs = new int[numcols + 1];
    irem = new int[nnonzero];
        //xrem = new T[nnonzero];
    xrem = new double[nnonzero];
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


    for(int i = 0 ; i < numcols+1 ; i++)
    {   
        colptrs[i]--;
    }

    for(int i = 0 ; i < nnonzero ; i++)
        irem[i]--;
    
    delete []txrem;
    tend = omp_get_wtime();
    cout << "Matrix is read in " << tend - tstart << " seconds." << endl;


}

//template<typename T>
//void csc2blkcoord(block<T> *&matrixBlock, double *xrem)
void csc2blkcoord(block *&matrixBlock, double *xrem)
{
    int i, j, r, c, k, k1, k2, blkr, blkc, tmp;
    int **top;
    nrowblks = ceil(numrows / (double)(wblk));
    ncolblks = ceil(numcols / (double)(wblk));
    cout << "wblk = " << wblk << endl;
    cout << "nrowblks = " << nrowblks << endl;
    cout << "ncolblks = " << ncolblks << endl;

    //matrixBlock = new block<T>[nrowblks * ncolblks];
    matrixBlock = new block[nrowblks * ncolblks];
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
        k1 = colptrs[c]+1;
        k2 = colptrs[c + 1] - 1+1;
        blkc = ceil((c + 1) / (double)wblk);
        //cout<<"K1: "<<k1<<" K2: "<<k2<<" blkc: "<<blkc<<endl;

        for(k = k1 - 1 ; k < k2 ; k++)
        {
            r = irem[k]+1;
            blkr = ceil(r/(double)wblk);
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

    //for(blkr=0;blkr<nrowblks;blkr++)
    //{
        //printf("nnzPerRow[%d] : %d\n", blkr, nnzPerRow[blkr]);
    //}
    //cout<<"end for"<<endl;

    for(c = 0 ; c < numcols ; c++)
    {
        k1 = colptrs[c]+1;
        k2 = colptrs[c + 1] - 1+1; 
        blkc = ceil((c + 1) / (float)wblk);

        //:wq
        //cout<<"K1: "<<k1<<" K2: "<<k2<<" blkc: "<<blkc<<endl;

        for(k = k1 - 1 ; k < k2 ; k++)
        {
            r = irem[k]+1;
            blkr = ceil(r / (float)wblk);

            matrixBlock[(blkr - 1) * ncolblks+blkc - 1].rloc[top[blkr-1][blkc-1]] = r - matrixBlock[(blkr - 1) * ncolblks + blkc - 1].roffset;
            matrixBlock[(blkr - 1) * ncolblks+blkc - 1].cloc[top[blkr-1][blkc-1]] = (c + 1) -  matrixBlock[(blkr - 1) * ncolblks + blkc - 1].coffset;
            matrixBlock[(blkr - 1) * ncolblks+blkc - 1].val[top[blkr-1][blkc-1]] = xrem[k];

            top[blkr-1][blkc-1]=top[blkr-1][blkc-1]+1;
        }
    }


    printf("allocated memory from each block\n\n");

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
    //block<double> *newMatrixBlock;
    block *newMatrixBlock;
    double* newXrem;
    //read_custom<double>(global_filename, newXrem);
    //csc2blkcoord<double>(newMatrixBlock, newXrem);

    //read_custom(global_filename, newXrem);
    csc2blkcoord(newMatrixBlock, newXrem);

    printf("new blks created in get new csb blk\n");

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




/////buildTaskinfoStruct functions//////////////////


void buildTaskInfoStruct_main(int nodeCount, char **graph , const char *loopType, int blksize , const char *matrixName)
{
    char taskName[200], ary[200];
    char buffer [33];
    char **splitParams;
    int partNo, tokenCount, priority;
    int vertexCount = 0, taskCounter = 0;


    struct  TaskInfo *taskInfo;

    int i;

    string numStr;

    char taskinfo_file_name[500];
    strcpy(taskinfo_file_name , "../dag_files/");
    strcat(taskinfo_file_name , matrixName);
    strcat(taskinfo_file_name , "_");
    strcat(taskinfo_file_name , loopType);
    strcat(taskinfo_file_name , "_");
    strcat(taskinfo_file_name , "taskinfo.txt");


    ofstream taskInfoFile(taskinfo_file_name);

    partNo = 0 ; 


    taskInfoFile << nodeCount << endl;

    for(i = 0 ; i < nodeCount ; i++)
    {
        strcpy(taskName , graph[i]);
        tokenCount = split(taskName, ',', &splitParams);

        if(taskName[0] == '_') 
        {
            strcpy(ary, "22,0,1,");
            strcat(ary, splitParams[0]);
            strcat(ary, ",-1,");
            myitoa (partNo, buffer);
            strcat(ary, buffer);
            strcat(ary, ",0");
            taskInfoFile << ary << endl;

            //printf("taskName: %s --> %s\n", taskName, ary);
            continue;
        }
        
        
        if(tokenCount == 1)
        {
            strcpy(ary, "21,0,1,");
            strcat(ary, splitParams[0]);
            strcat(ary, ",-1,");
            myitoa (partNo, buffer);
            strcat(ary, buffer);
            strcat(ary, ",0");
            taskInfoFile << ary << endl;
            
            //printf("taskName: %s --> %s\n", taskName, ary);
        }
        else
        {
            if(!strcmp(splitParams[0], "RESET")) /* taskName starts RESET */
            {
                strcpy(ary, "1,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "SPMM"))
            {
                strcpy(ary, "2,3,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                strcat(ary, splitParams[3]);
                strcat(ary, ",0");
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "XTY")) /* taskName starts XTY */
            {
                strcpy(ary, "3,2,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",");
                strcat(ary, splitParams[2]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[3]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "RED")) /* XTY partial sum reduction */
            {
                strcpy(ary, "4,1,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "XY")) /* taskName starts XY */
            {
                strcpy(ary, "5,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "ADD")) /* taskName starts ADD */
            {
                strcpy(ary, "6,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "DLACPY")) /* taskName starts DLACPY */
            {
                strcpy(ary, "7,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "UPDATE")) /* taskName starts UPDATE */
            {
                strcpy(ary, "8,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "SUB")) /* taskName starts SUB */
            {
                strcpy(ary, "9,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "MULT")) /* taskName starts MULT */
            {
                strcpy(ary, "10,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "COL")) /* taskName starts COL */
            {
                strcpy(ary, "11,2,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",");
                strcat(ary, splitParams[2]);
                strcat(ary, ",0,-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "RNRED")) /* taskName starts RNRED */
            {
                strcpy(ary, "12,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "SQRT")) /* taskName starts SQRT */
            {
                strcpy(ary, "13,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "GET")) /* taskName starts SUB */
            {
                strcpy(ary, "14,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "TRANS")) /* taskName starts RNRED */
            {
                strcpy(ary, "15,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "SPEUPDATE")) /* taskName starts RNRED */
            {
                strcpy(ary, "16,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "CHOL")) /* taskName starts RNRED */
            {
                strcpy(ary, "17,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;
                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "INV")) /* taskName starts RNRED */
            {
                strcpy(ary, "18,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;
                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "SETZERO")) /* taskName starts SUB */
            {
                strcpy(ary, "19,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;
                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "CONV")) /* taskName starts RNRED */
            {
                strcpy(ary, "20,0,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",-1,");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }


            ///for Power Iteration Method --> start task_id from 23
            else if(!strcmp(splitParams[0], "SPMV"))
            {
                strcpy(ary, "23,3,");
                strcat(ary, splitParams[1]); //row_id
                strcat(ary, ",");
                strcat(ary, splitParams[2]); //col_id
                strcat(ary, ",");
                strcat(ary, splitParams[3]); //buf_id
                strcat(ary, ",0"); //no string params
                strcat(ary, ",-1,"); //task_id, only one spmv
                myitoa (partNo, buffer); 
                strcat(ary, buffer); //partNo
                strcat(ary, ",0"); //priority
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            else if(!strcmp(splitParams[0], "SUBMAX")) /* taskName starts SUBMAX */
            {
                strcpy(ary, "24,2,");
                strcat(ary, splitParams[1]); //row_id
                strcat(ary, ",");
                strcat(ary, splitParams[2]); //buf_id
                strcat(ary, ",0,"); //no string params
                strcat(ary, splitParams[3]); //task_id
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer); //partNo
                strcat(ary, ",0"); //priority
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
            // else if(!strcmp(splitParams[0], "NORM")) /* taskName starts NORM */
   //          {
   //           strcpy(ary, "25,1,");
      //        strcat(ary, splitParams[1]); //block_id
      //        strcat(ary, ",0,"); //no string params
      //        strcat(ary, splitParams[2]); //task_id
      //        strcat(ary, ",");
      //        myitoa (partNo, buffer);
      //        strcat(ary, buffer); //partNo
      //        strcat(ary, ",0"); //priority
      //        taskInfoFile << ary << endl;

      //        //printf("taskName: %s --> %s\n", taskName, ary);
   //          }
            else if(!strcmp(splitParams[0], "DAXPY")) /* taskName starts DAXPY */
            {
                strcpy(ary, "27,1,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[2]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }

            else if(!strcmp(splitParams[0], "DGEMV")) /* taskName starts DGEMV */
            {
                strcpy(ary, "28,2,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",");
                strcat(ary, splitParams[2]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[3]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }

            else if(!strcmp(splitParams[0], "DOTV")) /* taskName starts DGEMV */
            {
                strcpy(ary, "29,2,");
                strcat(ary, splitParams[1]);
                strcat(ary, ",");
                strcat(ary, splitParams[2]);
                strcat(ary, ",0,");
                strcat(ary, splitParams[3]);
                strcat(ary, ",");
                myitoa (partNo, buffer);
                strcat(ary, buffer);
                strcat(ary, ",0");
                taskInfoFile << ary << endl;

                //printf("taskName: %s --> %s\n", taskName, ary);
            }
        }// end if(tokenCount == 1)
    } //end while (file read)

    taskInfoFile.close();

    buildTaskInfoStruct(taskInfo, "taskinfo.txt");

    
}




void buildTaskInfoStruct(struct TaskInfo *taskInfo, char *partFile)
{
    char taskName[200], ary[200], structToStr[2000];
    char buffer [33];
    char **splitParams;
    int partNo, tokenCount, priority, opCode, numParamsCount, strParamsCount;
    int vertexCount = 0, taskCounter = 0, index;
    string numStr;
    int i, j;

    //opening partition file
    ifstream partitionFile(partFile);

    partitionFile >> vertexCount;

    printf("vertexCount: %d\n", vertexCount);

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
                taskInfo[taskCounter].strParamsList[i] = (char *) malloc(strlen(splitParams[index + taskInfo[taskCounter].numParamsCount + 2 + i]) * sizeof(char));
                strcpy(taskInfo[taskCounter].strParamsList[i], splitParams[index + taskInfo[taskCounter].numParamsCount + 2 + i]);
            }   
        }
            
        taskInfo[taskCounter].taskID = atoi(splitParams[tokenCount - 3]);
        taskInfo[taskCounter].partitionNo = atoi(splitParams[tokenCount - 2]);
        taskInfo[taskCounter].priority = atoi(splitParams[tokenCount - 1]);
        
        // if(opCode == 15 || opCode == 16 || opCode == 17 || opCode == 18 || opCode == 20)
        // {
        //  if(taskInfo[taskCounter].numParamsCount == 0 && taskInfo[taskCounter].strParamsCount == 1)
        //      printf("%s ---> %d,%d,%d,%s,%d,%d,%d\n", taskName, taskInfo[taskCounter].opCode, taskInfo[taskCounter].numParamsCount, taskInfo[taskCounter].strParamsCount, taskInfo[taskCounter].strParamsList[0], taskInfo[taskCounter].taskID, taskInfo[taskCounter].partitionNo, taskInfo[taskCounter].priority);
            
            
        // }
        structToString(taskInfo[taskCounter], structToStr);

        //if(strcmp(taskName, structToStr) != 0)
        //    printf("%s --> %s\n", taskName, structToStr);

        taskCounter++;

    }//end while

    printf("Finish allocating taskInfo\n");

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







