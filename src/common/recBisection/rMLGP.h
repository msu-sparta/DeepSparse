#ifndef RMLGP_H_
#define RMLGP_H_


#include "utils.h"
#include "dgraph.h"
#include "dgraphReader.h"
#include "rvcycle.h"
 //#include "../lobpcg_gen_graph_v30.h"

//#include "../lobpcg_gen_graph_v28.h"
#include "../lobpcg_gen_graph_v29.h"

extern dgraph main_graph;
int processArgs_rMLGP(int argc, char **argv, MLGP_option* opt);
void run_rMLGP(char* file_name, MLGP_option opt, int *edge_u, int *edge_v, double *edge_weight, int edgeCount, int vertexCount,const char** vertexName,double* vertexWeight, int loopType, char* matrix_name);
void get_task_name(const char* node_name, char* task);
void make_new_table_file(rcoarsen* rcoarse, int *my_partition_topsort, char* loopType, char* matrix_name, int large_blk, int small_blk, int partCount);


#endif
