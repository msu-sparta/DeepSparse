#ifndef UTIL_H_
#define UTIL_H_


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
#include <iostream>

using namespace std;

#include <omp.h>


extern long position;
extern int *colptrs, *irem;

extern int nrows, ncols, nnz, numrows, numcols, nnonzero, nthrds ;



/* ##### For Sparse MTX DS ##### */
//template<typename T>
struct block
{   
    int nnz;
    int roffset, coffset;
    int *rloc, *cloc;
    //short int *rloc , *cloc;
    double *val;
};


/* graph data structure */
struct my_hash
{
    inline size_t operator()(const char* val) const
    {
        unsigned long h = 0;
        for (; *val; ++val)
            h = 5 * h + *val;
        return size_t(h);
    }
};
struct eqstr
{
    inline bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) == 0;
    }
};


////taskinfo struct
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
} ;


extern int wblk, nrowblks, ncolblks, nthreads;
extern int *nnzPerRow;
//extern block<double> *matrixBlock; // goes to lib_mtx
extern block *matrixBlock; // goes to lib_mtx

/* #### For DAG generator #### */
typedef std::unordered_map<const char * , int, my_hash, eqstr> VertexType;
extern stringstream convertToString;
extern VertexType vertexName;

extern int *edgeU, *edgeV;
extern double *edgeW;
extern double *vertexWeight;
extern char **nrowblksString;
extern int nodeCount, edgeCount;
/* gen timing breakdown */
extern int total_func; //not now
extern double *graphGenTime; //not now



extern char **globalGraph;
extern int globalNodeCount;


/* ##### For Hierarchical Partitioning DS ##### */
struct task_in_out
{
    char input1[500];
    char input2[500];
    char output[500];
    //int emnei;
    // char *inp1;
    // char *inp2;
    // char *output;
};


typedef std::unordered_map<const char * , task_in_out, my_hash, eqstr> TaskDetail;
typedef std::unordered_map<const char * , int, my_hash, eqstr> InOutVariable;
typedef std::unordered_map<const char * , std::unordered_map<const char * , double, my_hash, eqstr>, my_hash, eqstr> task_allinout_memory;


/*new try with input map and output map*/

struct memory_chunk
{
    char memory_name[100];
    double value;
};

typedef std::unordered_map<const char*, memory_chunk, my_hash, eqstr> internal_map;
typedef std::unordered_map<const char* , internal_map, my_hash, eqstr> input_map;
typedef std::unordered_map<const char* , internal_map , my_hash, eqstr> output_map;

///here
extern int** pseudo_tid_map;
extern int small_block;



int split(const char *str, char c, char ***arr);
void print_vector(vector<string> nodeName);
void print_map(VertexType mymap);



//template<typename T>
void read_custom(char* filename, double *&xrem);

//template<typename T>
//void csc2blkcoord(block<T> *&matrixBlock, double *xrem);
void csc2blkcoord(block *&matrixBlock, double *xrem);

void _XY(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int M, int N, int P, int block_width, int xy_id);

void _XY_v1(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], int task_id,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[],
            char input1[], char input2[], char output[], int M, int N, int P, int block_width, int xy_id);

void _XY_v2(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            char edge2_var[],
            char input1[], char input2[], char output[], int M, int N, int P, int block_width, int xy_id, int **SPMM_vertexNo);

void mat_sub(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int block_width, int sub_id);

void mat_addition(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int block_width, int add_id);

void mat_mult(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int block_width);

void sum_sqrt(char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[],
            char edge2[], int edge2_id,
            char input1[], char output[], int row, int col, int block_width);

void getActiveBlockVector(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            char edge2[], int edge2_id,
            char input1[], char input2[], char output[], int row, int col, int currentBlockSize, int block_width, int get_id );

void updateBlockVector(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], int task_id,
            char edge2[], int edge2_id,
            char input1[], char input2[], char output[], int row, int col, int currentBlockSize, int block_width, int update_id);

void _XTY(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], char edge2_part3[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id);

void _XTY_v1(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id);

void _XTY_v2(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], int task_id_1,
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id);

void _XTY_v3(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            int edge2Format, char edge2_var[], char edge2_part1[], char edge2_func[], char edge2_part2[], int task_id_2,
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int xty_id, int **SPMM_vertexNo);

void custom_dlacpy(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id,
            char input1[], char output[], int row, int col, int block_width, int dlacpy_id);

//for SPMM onlly
void custom_dlacpy_v1(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[], char edge1_part3[], int task_id,
            char input1[], char output[], int row, int col, int block_width, int dlacpy_id, int taskwait_node_no, int *actR_vertexNo);

void spmm_blkcoord_v1(int edge1Format, string edge1_var, string edge1_part1, string edge1_func, string edge1_part2,
            string input1, string input2, string output, int row, int col, int p, int block_width, int spmm_id);

void spmm_blkcoord_csbTask(int edge1Format, char edge1_var[], char edge1_part1[], char edge1_func[], char edge1_part2[],
            char input1[], char input2[], char output[], int row, int col, int p, int block_width, int currentBlockSize, int taskwait_node_no, int *actR_vertexNo, int **SPMM_vertexNo);


void strrev(char *input, int length);
void string_rev(char *p);
void str_reverse_in_place(char *str, int len);
void str_rev(char *str);
void myitoa(int x, char* dest);


void myprint();
void get_new_csb_block(int newWblk, int*** nnzBlock, int* nrowblocks, int* ncolblocks);

void get_input_output(const char* node, char* input1, char* input2, char* output);
void clear_InOutVariable();
int search_in_InOutVariable(const char* var);

void fill_allinout_memory_map(const char* task_name, const char* parent_output, double memory_amount);
void get_output_of_a_task(const char* task_name, char* output);
int get_internal_map_size(const char* task_name);
void get_all_the_keys_input(const char* task_name, char **keys);
void get_all_the_keys_output(const char* task_name, char **keys);
double get_task_one_input_memory(const char* task_name, const char* input_mem);

int get_inp_mem_size(const char* task);
int get_out_mem_size(const char* task);
void get_incoming_memory_name(const char* task_name, const char* parent_task_name, char* memory_name);
double get_incoming_memory_value(const char* task_name, const char* parent_task_name);
void get_outgoing_memory_name(const char* task_name, const char* child_task_name, char* memory_name);
double get_outgoing_memory_value(const char* task_name, const char* child_task_name);



///buildTaskinfoStruct functions
void buildTaskInfoStruct(struct TaskInfo *taskInfo, char *partFile);
void buildTaskInfoStruct_main(int nodeCount, char **graph , const char *loopType, int blksize , const char *matrixName);
void reverse(char str[], int length);
void structToString(struct TaskInfo taskInfo, char *structToStr);


#endif 



