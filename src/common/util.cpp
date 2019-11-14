#include <util.h>












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




