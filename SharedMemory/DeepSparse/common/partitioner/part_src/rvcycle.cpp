#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../part_inc/rvcycle.h"
#include "../part_inc/option.h"
#include "../part_inc/utils.h"
#include "../part_inc/info.h"
#include "../part_inc/dgraph.h"
#include "../part_inc/undirPartitioning.h"
#include "../part_inc/rMLGP.h"
#include "../part_inc/dgraphTraversal.h"


rcoarsen *initializeRCoarsen(coarsen* icoars)
{
    rcoarsen * rcoars = (rcoarsen*) umalloc(sizeof(rcoarsen), "rcoars");
    rcoars->coars = icoars;
    rcoars->previous_rcoarsen = NULL;
    rcoars->next_rcoarsen1 = NULL;
    rcoars->next_rcoarsen2 = NULL;
    rcoars->previous_index = (idxType * ) umalloc(sizeof(idxType) * (icoars->graph->nVrtx +1), "rcoars->previous_index");
    rcoars->next_index = (idxType * ) umalloc(sizeof(idxType) * (icoars->graph->nVrtx +1), "rcoars->next_index");
    return rcoars;
}


void freeRCoarsen(rcoarsen *rcoars)
{
    freeCoarsen(rcoars->coars);
    free(rcoars->previous_index);
    free(rcoars->next_index);
    free(rcoars);
}

void freeRCoarsenHeader(rcoarsen *rcoars)
{
    freeCoarsenHeader(rcoars->coars);
    free(rcoars->previous_index);
    free(rcoars->next_index);
    free(rcoars);
}


void initializeNextRCoarsen(rcoarsen* rcoars, rcoarsen* rcoars1, rcoarsen* rcoars2)
{
    dgraph* graph = rcoars->coars->graph;
    dgraph* graph1 = rcoars1->coars->graph;
    dgraph* graph2 = rcoars2->coars->graph;
    idxType i,j;
    idxType* part = rcoars->coars->part;
    idxType nVrtx = rcoars->coars->graph->nVrtx;
    idxType nEdge = rcoars->coars->graph->nEdge;
    idxType nVrtx1 = 0, nVrtx2 = 0, nEdge1 = 0, nEdge2 = 0;

    idxType* onetograph = rcoars1->previous_index;
    idxType* twotograph = rcoars2->previous_index;
    idxType* onetoedge = (idxType*) malloc((nEdge+1)*sizeof(idxType));
    idxType* twotoedge = (idxType*) malloc((nEdge+1)*sizeof(idxType));
    rcoars->next_index = (idxType*) malloc((nVrtx+1)*sizeof(idxType));
    idxType* graphto12 = rcoars->next_index;
    for (i=1; i<=nVrtx; i++) {
	if (part[i] == 0) {
	    onetograph[++nVrtx1] = i;
	    graphto12[i] = nVrtx1;
	}
	if (part[i] == 1) {
	    twotograph[++nVrtx2] = i;
	    graphto12[i] = nVrtx2;
	}
    }
    graph1->nVrtx = nVrtx1;
    graph2->nVrtx = nVrtx2;
    graph1->inStart = (idxType*) malloc((nVrtx1+2)*sizeof(idxType));
    graph1->inEnd = (idxType*) malloc((nVrtx1+2)*sizeof(idxType));
    graph1->outStart = (idxType*) malloc((nVrtx1+2)*sizeof(idxType));
    graph1->outEnd = (idxType*) malloc((nVrtx1+2)*sizeof(idxType));
    graph1->hollow  = (int * ) calloc(nVrtx1 + 2, sizeof(int));
    idxType* in1 = (idxType*) malloc((nEdge+1)*sizeof(idxType));
    graph2->inStart = (idxType*) malloc((nVrtx2+2)*sizeof(idxType));
    graph2->inEnd = (idxType*) malloc((nVrtx2+2)*sizeof(idxType));
    graph2->outStart = (idxType*) malloc((nVrtx2+2)*sizeof(idxType));
    graph2->outEnd = (idxType*) malloc((nVrtx2+2)*sizeof(idxType));
    graph2->hollow  = (int * ) calloc(nVrtx2 + 2, sizeof(int));
    idxType* in2 = (idxType*) malloc((nEdge+1)*sizeof(idxType));
    for (i=1; i<=nVrtx; i++) {
	if (part[i] == 0) {
	    graph1->inStart[graphto12[i]] = nEdge1;
	    if (i > 0)
		graph1->inEnd[graphto12[i]-1] = nEdge1-1;
	}
	if (part[i] == 1) {
	    graph2->inStart[graphto12[i]] = nEdge2;
	    if (i > 0)
		graph2->inEnd[graphto12[i]-1] = nEdge2-1;
	}
	for (j=graph->inStart[i]; j<=graph->inEnd[i]; j++) {
	    idxType innode = graph->in[j];
	    if ((part[i] == 0)&&(part[innode] == 0)) {
		in1[++nEdge1] = graphto12[innode];
		onetoedge[nEdge1] = j;
	    }
	    if ((part[i] == 1)&&(part[innode] == 1)) {
		in2[++nEdge2] = graphto12[innode];
		twotoedge[nEdge2] = j;
	    }
	}
    }
    graph1->nEdge = nEdge1;
    graph2->nEdge = nEdge2;
    graph1->inEnd[nVrtx1] = nEdge1-1;
    graph2->inEnd[nVrtx2] = nEdge2-1;
    graph1->inStart[nVrtx1+1] = nEdge1;
    graph1->inEnd[nVrtx1+1] = nEdge1-1;
    graph2->inStart[nVrtx2+1] = nEdge2;
    graph2->inEnd[nVrtx2+1] = nEdge2-1;
    graph1->in = (idxType*) malloc((nEdge1+1)*sizeof(idxType));
    graph2->in = (idxType*) malloc((nEdge2+1)*sizeof(idxType));
    graph1->out = (idxType*) malloc((nEdge1+1)*sizeof(idxType));
    graph2->out = (idxType*) malloc((nEdge2+1)*sizeof(idxType));
    for (i=1; i<=nEdge1; i++)
	    graph1->in[i] = in1[i];
    for (i=1; i<=nEdge2; i++)
	    graph2->in[i] = in2[i];
    graph1->vw = (vwType*) malloc((nVrtx1+1)*sizeof(vwType));
    graph2->vw = (vwType*) malloc((nVrtx2+1)*sizeof(vwType));
    graph1->totvw = 0;
    for (i=1; i<=nVrtx1; i++) {
	    graph1->vw[i] = graph->vw[onetograph[i]];
	    graph1->totvw += graph1->vw[i];
    }
    graph2->totvw = 0;
    for (i=1; i<=nVrtx2; i++) {
	    graph2->vw[i] = graph->vw[twotograph[i]];
	    graph2->totvw += graph2->vw[i];
    }
    graph1->totec = 0;
    graph1->ecIn = (ecType*) malloc((nEdge1+1)*sizeof(ecType));
    graph1->ecOut = (ecType*) malloc((nEdge1+1)*sizeof(ecType));
    for (i=1; i<=nEdge1; i++) {
	    graph1->ecIn[i] = graph->ecIn[onetoedge[i]];
	    graph1->totec += graph1->ecIn[i];
    }
    graph2->totec = 0;
    graph2->ecIn = (ecType*) malloc((nEdge2+1)*sizeof(ecType));
    graph2->ecOut = (ecType*) malloc((nEdge2+1)*sizeof(ecType));
    for (i=1; i<=nEdge2; i++) {
	    graph2->ecIn[i] = graph->ecIn[twotoedge[i]];
	    graph1->totec += graph2->ecIn[i];
    }
    graph1->frmt = graph->frmt;
    graph2->frmt = graph->frmt;
    free(in1);
    free(in2);
    free(onetoedge);
    free(twotoedge);
    fillOutFromIn(graph1);
    fillOutFromIn(graph2);
}


void check(coarsen* C)
{
  if (C->graph->ecIn == NULL)
    printf("!!!ECIN IS NULL!!!\n");
}

void splitCoarsen(rcoarsen* rcoars, dgraph* G1, dgraph* G2, int* part1, int* part2)
{
    /*Build G1 and G2 and fill rcoars->next_index based on rcoars->coars->part*/
    coarsen* C = rcoars->coars;
    idxType* part = C->part;
    idxType nVrtx = C->graph->nVrtx;
    idxType* inStart = C->graph->inStart;
    idxType* inEnd = C->graph->inEnd;
    idxType* in = C->graph->in;
    idxType* next_index = rcoars->next_index;
    dgraph* graph = C->graph;

    idxType nVrtx1 = 1, nVrtx2 = 1;
    idxType nEdge1 = 0, nEdge2 = 0;
    idxType vertexName1=0,vertexName2=0;
    idxType i,j;
    int degree;
    int partin = -1, partout = -1;

    for (i=1; i <= nVrtx; i++) {
        for (j=inStart[i]; j<=inEnd[i]; j++) {
            if (part[i] != part[C->graph->in[j]]) {
                partin = part[C->graph->in[j]];
                partout = part[i];
                break;
            }
        }
        if (partin != -1)
            break;
    }
    if ((partin == -1)||(partout == -1)) {
      partin = 0;
      partout = 1;
    }
    *part1 = partin;
    *part2 = partout;

    /*Fill rcoars->next_index and compute nVrtx1, nVrtx2, nEdge1 and nEdge2*/
    for (i=1; i <= nVrtx; i++) {
	    if (part[i] == partin) {
	        next_index[i] = nVrtx1;
	        nVrtx1++;
	    }
	    if (part[i] == partout) {
	        next_index[i] = nVrtx2;
	        nVrtx2++;
	    }
	    for (j=inStart[i]; j<=inEnd[i]; j++) {
	        if ((part[i] == partin)&&(part[in[j]] == partin))
		        nEdge1++;
	        if ((part[i] == partout)&&(part[in[j]] == partout))
		        nEdge2++;
	    }
    }
    nVrtx1--;
    nVrtx2--;

    // vertex name allocation
    G1->vertices = (char**)malloc(sizeof(char*)*(nVrtx1+1));
    for(i = 0;i<=nVrtx1;++i){
        G1->vertices[i] = (char*)malloc(100*sizeof(char));
    }

    G2->vertices = (char**)malloc(sizeof(char*)*(nVrtx2+1));
    for(i = 0;i<=nVrtx2;++i){
        G2->vertices[i] = (char*)malloc(100*sizeof(char));


    }



    for (i=1; i <= nVrtx; i++){
        if (part[i] == partin){
            strcpy(G1->vertices[vertexName1],C->graph->vertices[i-1]);
        //    printf("\nG1 %d = %s\n",vertexName1,G1->vertices[vertexName1]);
            vertexName1++;
        }
        if (part[i] == partout){
            strcpy(G2->vertices[vertexName2],C->graph->vertices[i-1]);
        //    printf("\nG2 %d = %s\n",vertexName2,G2->vertices[vertexName2]);
            vertexName2++;
        }
        
    }
    /*Allocate everything*/
    G1->inStart = (idxType*) malloc (sizeof(idxType) * (nVrtx1 + 2));
    G1->inEnd = (idxType*) malloc (sizeof(idxType) * (nVrtx1 + 2));
    G1->in = (idxType*) malloc (sizeof(idxType) * (nEdge1 + 2));
    G1->outStart = (idxType*) malloc (sizeof(idxType) * (nVrtx1 + 2));
    G1->outEnd = (idxType*) malloc (sizeof(idxType) * (nVrtx1 + 2));
    G1->out = (idxType*) malloc (sizeof(idxType) * (nEdge1 + 2));
    G1->vw = (idxType*) malloc (sizeof(idxType) * (nVrtx1 + 1));
    G1->ecIn = (ecType*) malloc (sizeof(ecType) * (nEdge1 + 2));
    G1->ecOut = (ecType*) malloc (sizeof(ecType) * (nEdge1 + 2));
    G1->hollow  = (int * ) calloc(nVrtx1 + 2, sizeof(int));
    G1->vWeight = (ecType*) malloc (sizeof(ecType) * (nVrtx1 + 1));
    G1->incoming_edge_weight = (ecType*)calloc((nVrtx1+1),sizeof(ecType));


    G2->inStart = (idxType*) malloc (sizeof(idxType) * (nVrtx2 + 2));
    G2->inEnd = (idxType*) malloc (sizeof(idxType) * (nVrtx2 + 2));
    G2->in = (idxType*) malloc (sizeof(idxType) * (nEdge2 + 2));
    G2->outStart = (idxType*) malloc (sizeof(idxType) * (nVrtx2 + 2));
    G2->outEnd = (idxType*) malloc (sizeof(idxType) * (nVrtx2 + 2));
    G2->out = (idxType*) malloc (sizeof(idxType) * (nEdge2 + 2));
    G2->vw = (idxType*) malloc (sizeof(idxType) * (nVrtx2 + 1));
    G2->ecIn = (ecType*) malloc (sizeof(ecType) * (nEdge2 + 2));
    G2->ecOut = (ecType*) malloc (sizeof(ecType) * (nEdge2 + 2));
    G2->hollow  = (int * ) calloc(nVrtx2 + 2, sizeof(int));
    G2->vWeight = (ecType*) malloc (sizeof(ecType) * (nVrtx2 + 1));
    G2->incoming_edge_weight = (ecType*)calloc((nVrtx2+1),sizeof(ecType));

    /*Fill inStart1, inStart2, inEnd1, inEnd2, in1 and in2*/
    idxType idx1 = 0, idx2 = 0, outnode = 0;
    ecType weight;
    for (i=1; i <= nVrtx; i++) {
	    if (part[i] == partin) {
	        G1->inStart[next_index[i]] = idx1;
	        if (next_index[i] > 0)
		        G1->inEnd[next_index[i]-1] = idx1-1;
        }
	    if (part[i] == partout) {
	        G2->inStart[next_index[i]] = idx2;
	        if (next_index[i] > 0)
		        G2->inEnd[next_index[i]-1] = idx2-1;
	    }
        G1->totec = 0.0;
        G2->totec = 0.0;
        for (j=inStart[i]; j<=inEnd[i]; j++) {
	        outnode = C->graph->in[j];
	        weight = C->graph->ecIn[j];
	        if ((part[i] == partin)&&(part[outnode] == partin)) {
                G1->in[idx1++] = next_index[outnode];
		        G1->ecIn[idx1-1] = weight;
                G1->totec += G1->ecIn[idx1-1];
	        }
	        if ((part[i] == partout)&&(part[outnode] == partout)) {
		        G2->in[idx2++] = next_index[outnode];
		        G2->ecIn[idx2-1] = weight;
                G2->totec += G2->ecIn[idx2-1];
            }
	    }
    }
    G1->inEnd[nVrtx1] = nEdge1-1;
    G1->inStart[nVrtx1+1] = nEdge1;
    G1->inEnd[nVrtx1+1] = nEdge1-1;
    G2->inEnd[nVrtx2] = nEdge2-1;
    G2->inStart[nVrtx2+1] = nEdge2;
    G2->inEnd[nVrtx2+1] = nEdge2-1;

    G1->maxindegree = 0;
    G2->maxindegree = 0;
    for (i=1; i <= nVrtx; i++) {
        if (part[i] == partin) {
            degree = G1->inEnd[next_index[i]] - G1->inStart[next_index[i]] + 1;
            G1->maxindegree = G1->maxindegree < degree ? degree : G1->maxindegree;
        }
        if (part[i] == partout) {
            degree = G2->inEnd[next_index[i]] - G2->inStart[next_index[i]] + 1;
            G2->maxindegree = G2->maxindegree < degree ? degree : G2->maxindegree;
        }
    }

    /*Fill vw1 and vw2*/
    G1->totvw = 0.0;
    G2->totvw = 0.0;
    G1->maxVW = 0;
    G2->maxVW = 0;
    for (i=1; i<=graph->nVrtx; i++) {
	    if (part[i] == partin) {
            G1->vw[next_index[i]] = C->graph->vw[i];
            G1->vWeight[next_index[i]] = C->graph->vWeight[i];
            G1->totvw += G1->vw[next_index[i]];
            G1->maxVW = G1->maxVW < C->graph->vw[i] ? C->graph->vw[i] : G1->maxVW;
        }
        if (part[i] == partout) {
            G2->vw[next_index[i]] = C->graph->vw[i];
            G2->vWeight[next_index[i]] = C->graph->vWeight[i];
            G2->totvw += G2->vw[next_index[i]];
            G2->maxVW = G2->maxVW < C->graph->vw[i] ? C->graph->vw[i] : G2->maxVW;
        }
    }
    /*Fill G1 and G2*/
    G1->frmt = graph->frmt;
    G1->nVrtx = nVrtx1;
    G1->nEdge = nEdge1;
    G2->frmt = graph->frmt;
    G2->nVrtx = nVrtx2;
    G2->nEdge = nEdge2;

    idx1 = 0 ;
    idx2 = 0 ;
    outnode = 0;

    double part_to_part = 0;

    for (i=1; i <= nVrtx; i++){
        G1->totec = 0.0;
        G2->totec = 0.0;

        if ((part[i] == partin)){
                G1->incoming_edge_weight[next_index[i]] = graph->incoming_edge_weight[i];
            }
        if ((part[i] == partout)){
                G2->incoming_edge_weight[next_index[i]] = graph->incoming_edge_weight[i];
            }
        for (j=inStart[i]; j<=inEnd[i]; j++){

            outnode = C->graph->in[j];
            weight = C->graph->ecIn[j];
            if ((part[i] == partin)&&(part[outnode] == partin)){
                G1->ecIn[idx1++] = weight;
                //printf("G1 ecIn[%d] = %lf\n",idx1-1,G1->ecIn[idx1-1]);
                G1->totec += G1->ecIn[idx1-1];
            }


            if ((part[i] == partout)&&(part[outnode] == partout)){
                G2->ecIn[idx2++] = weight;

                //printf("G2 ecIn[%d] = %lf\n",idx2-1,G2->ecIn[idx2-1]);
                G2->totec += G2->ecIn[idx2-1];
            }
            if ((part[i] == partout)&&(part[outnode] == partin)){
                G2->incoming_edge_weight[next_index[i]] +=  weight;
                part_to_part += weight;
            }
        }

    }

    double inc_G1=0, inc_G2=0;


    for(i=1; i <= G1->nVrtx ; i++)
        inc_G1 += G1->incoming_edge_weight[i];

    for(i=1; i <= G2->nVrtx ; i++)
        inc_G2 += G2->incoming_edge_weight[i];


    fillOutFromIn(G1);
    fillOutFromIn(G2);

    G1->sources  = (idxType * ) malloc(sizeof(idxType) * (nVrtx1 + 1));
    G1->targets  = (idxType * ) malloc(sizeof(idxType) * (nVrtx1 + 1));
    G1->nbsources = 0;
    G1->nbtargets = 0;
    for (i=1; i<=nVrtx1; i++){
        if (G1->inEnd[i] < G1->inStart[i])
            G1->sources[G1->nbsources++] = i;
        if (G1->outEnd[i] < G1->outStart[i])
            G1->targets[G1->nbtargets++] = i;
    }
    G2->sources  = (idxType * ) malloc(sizeof(idxType) * (nVrtx2 + 1));
    G2->targets  = (idxType * ) malloc(sizeof(idxType) * (nVrtx2 + 1));
    G2->nbsources = 0;
    G2->nbtargets = 0;
    for (i=1; i<=nVrtx2; i++){
        if (G2->inEnd[i] < G2->inStart[i])
            G2->sources[G2->nbsources++] = i;
        if (G2->outEnd[i] < G2->outStart[i])
            G2->targets[G2->nbtargets++] = i;
    }
}


rcoarsen* rVCycle(dgraph *G, MLGP_option opt, rMLGP_info* info, int *resultingNbPart)
{
    /*part is allocated and this function fills it*/
    info->timing_global -= u_wseconds();
    info->depth++;

    double my_weight = 0 ;

    double incoming_edge_weight_in_part_time;

    double spmm_memory,xy_memory,xty_memory,setzero_memory;

    idxType i,j;
    int oldnbPart;
    dgraph Gsingle;
    dgraph* igraph;

    int nbpart = opt.nbPart;
    ecType ratio = opt.ratio;
    double* targeted_weight = (double*)malloc((opt.nbPart)*sizeof(double));
    double* lb_save = (double*)malloc((opt.nbPart)*sizeof(double));
    int nVrtx = G->nVrtx, partin = -1, partout = -1;
    idxType m;

    ecType incoming_edge_weight_in_part = 0 ;
    double a ;
    ecType task_memory = 0;
    ecType total_memory = 0;

    for (i=0; i < opt.nbPart; i++){
        targeted_weight[i] = opt.ub[i];
        lb_save[i] = opt.lb[i];
    }




    idxType* my_toporder = (idxType*)malloc((G->nVrtx+1)*sizeof(idxType));
    DFSsort(G,my_toporder);

    idxType* my_bfs_traversal = (idxType*)malloc((G->nVrtx+2)*sizeof(idxType));
    my_BFS(G,my_bfs_traversal);


    idxType* my_dfs_traversal = (idxType*)malloc((G->nVrtx+1)*sizeof(idxType));
    my_DFS(G,my_dfs_traversal);

    char* input1 = (char*)malloc(200*sizeof(char));
    char* input2 = (char*)malloc(200*sizeof(char));
    char* output = (char*)malloc(200*sizeof(char));

    clear_InOutVariable();



    incoming_edge_weight_in_part_time = u_wseconds();
    double diff_part_weight = 0;

    int map_size = 0;
    char mem_name[100];
    double mem_value = 0;



    for(i=1;i<=nVrtx;++i){

        map_size = get_inp_mem_size(G->vertices[my_dfs_traversal[i]-1]);

        double already_counted = 0;

        char **input_values ;

        input_values = (char**)malloc(sizeof(char*)*(map_size+1));
        for(m = 0;m<=map_size;++m){
            input_values[m] = (char*)malloc(200*sizeof(char));

        }

        if(map_size > 0){
            get_all_the_keys_input(G->vertices[my_dfs_traversal[i]-1],input_values);
            for(m = 0;m<map_size;++m){
                get_incoming_memory_name(G->vertices[my_dfs_traversal[i]-1],input_values[m],mem_name);
                mem_value = get_incoming_memory_value(G->vertices[my_dfs_traversal[i]-1],input_values[m]);
                //printf("incoming mem name = %s value = %lf\n",mem_name,mem_value);
                if(strcmp(mem_name,"null") && search_in_InOutVariable(mem_name)){
                    already_counted += mem_value;
                }
                else {

                    incoming_edge_weight_in_part += mem_value;
                    //printf("mem name = %s mem value = %lf incoming_edge_weight_in_part = %lf\n", mem_name, mem_value, incoming_edge_weight_in_part);
                }
            }
        }

        map_size = get_out_mem_size(G->vertices[my_dfs_traversal[i]-1]);

        char **output_values ;

        output_values = (char**)malloc(sizeof(char*)*(map_size+1));
        for(m = 0;m<=map_size;++m){
            output_values[m] = (char*)malloc(200*sizeof(char));

        }

        if(map_size > 0){
            get_all_the_keys_output(G->vertices[my_dfs_traversal[i]-1],output_values);
            for(m = 0;m<map_size;++m){
                get_outgoing_memory_name(G->vertices[my_dfs_traversal[i]-1],output_values[m],mem_name);
                //printf("outgoing mem name = %s value = %lf\n",mem_name,mem_value);
                mem_value = get_outgoing_memory_value(G->vertices[my_dfs_traversal[i]-1],output_values[m]);
                if(strcmp(mem_name,"null") && search_in_InOutVariable(mem_name)){
                    already_counted += mem_value;
                }
                else {
                    incoming_edge_weight_in_part += mem_value;
                    //printf("mem name = %s mem value = %lf incoming_edge_weight_in_part = %lf\n", mem_name, mem_value, incoming_edge_weight_in_part);
                }
            }
        }

        diff_part_weight +=  G->incoming_edge_weight[i];
        //printf("incoming_edge_weight_in_part inside loop = %ld\n", incoming_edge_weight_in_part);
        a = incoming_edge_weight_in_part;
    }
    //incoming_edge_weight_in_part_time = u_wseconds() - incoming_edge_weight_in_part_time;

    //printf("before total memory incoming_edge_weight_in_part = %lf a = %lf\n", incoming_edge_weight_in_part,a);

    total_memory = incoming_edge_weight_in_part+task_memory;

    //printf("vertex count = %d total memory = %lf \n",G->nVrtx,total_memory );
    coarsen* C;



     
    if(total_memory < total_memory_limit) //32 MB for haswell L3 data cache
    //if(total_memory < 134217728) //128 MB 
    //if(total_memory < 536870912) // 512 MB
    {

        C = initializeCoarsen(G);
        //printf("coarse vertex count = %d\n",C->graph->nVrtx);
        for(i = 1 ; i<= C->graph->nVrtx ; ++i){
            //C->new_index[i] = i;
            C->part[i] = 0;
            //C->leader[i] = 0 ;
        }
        C->nbpart = 1;
        info->depth--;
        *resultingNbPart = 1;

        info->timing_global += u_wseconds();
        info->timing_coars = 0;
        info->timing_inipart = 0;
        info->timing_uncoars = 0;
        rcoarsen* rcoarse = initializeRCoarsen(C);
        //fclose(fp);
        return rcoarse;
    }

    idxType* flag = (idxType * ) calloc(G->nVrtx +3, sizeof(idxType));
    if(opt.nbPart != 1){
        //printf("flag assign loop entered conpar value = %d\n",opt.conpar);
        if ((opt.conpar > 0)||(opt.anchored > 0)) {
            double Gweight_ub1 = 0.5; // for round up effect
            double Gweight_ub2 = 0.5; // for round up effect
            for (i=0; i<opt.nbPart/2; i++) {
                Gweight_ub1 += opt.ub[i];
            }
            for (i=opt.nbPart/2; i<opt.nbPart; i++) {
                Gweight_ub2 += opt.ub[i];
            }
            MLGP_option* optbis = copyOpt(opt, 2);
            optbis->ub[0] = (Gweight_ub1 / opt.ratio) * (1 + 0.7* (opt.ratio-1) / log2(opt.nbPart));
            optbis->ub[1] = (Gweight_ub2 / opt.ratio) * (1 + 0.7* (opt.ratio-1) / log2(opt.nbPart));

            if (opt.debug > 0) {
                for (i=0; i<info->depth; i++)
                    printf("@");
                printf("In Constraint Partitioning nVrtx = %d, nbpart = %d, lb0 = %f, ub0 = %f, lb1 = %f, ub1 = %f\n",
                    G->nVrtx, optbis->nbPart, optbis->lb[0], optbis->ub[0], optbis->lb[1], optbis->ub[1]);
            }

            conpar(G, flag, *optbis, info->info);
            free_opt(optbis);
            free(optbis);

            //opt.nbPart = oldnbPart;
    	   if (opt.debug > 0) {
                if (opt.conpar_fix)
                    printf("Edgecut Metis + fix : %lld\n", (long long) edgeCut(G, flag));
                else
                    printf("Edgecut Metis : %lld\n", (long long) edgeCut(G, flag));
            }
        }
    }

    if (opt.anchored == 1) {
        copyDgraph(G, &Gsingle);
        addSingleSourceTarget(&Gsingle, flag);
    }
    if (opt.conpar == 0)
        for (i = 0; i <= G->nVrtx + 2; i++)
            flag[i] = 0;

    if (opt.anchored == 1)
        igraph = &Gsingle;
    else
        igraph = G;

    nbpart = opt.nbPart;
    nVrtx = igraph->nVrtx;
    partin = -1;
    partout = -1;

    if (opt.debug > 0)
     {
        for (i=0; i<info->depth; i++)
            printf("@");
        printf("In rVcycle nVrtx = %d, nbpart = %d, lb0 = %f, ub0 = %f, lb1 = %f, ub1 = %f\n", nVrtx, opt.nbPart, opt.lb[0], opt.ub[0], opt.lb[1], opt.ub[1]);
    }

    if ((nbpart == 1)||(nbpart == 2)) {

        // TODO: Compare with new version; update/enable as necessary
         for (i=0; i<opt.nbPart; i++) {
             opt.ub[i] = targeted_weight[i] * opt.ratio;
         //     opt.lb[i] = ((2.0 - opt.ratio)/opt.ratio) * opt.ub[i];
         }
        C = VCycle2way(igraph, flag, opt, info->info);

        if (nbpart == 2) {
            for (i = 1; i <= nVrtx; i++) {
                for (j = igraph->inStart[i]; j <= igraph->inEnd[i]; j++) {
                    if (C->part[i] != C->part[igraph->in[j]]) {
                        partin = C->part[igraph->in[j]];
                        partout = C->part[i];
                        break;
                    }
                }
                if (partin != -1)
                    break;
            }
	    if (partin != -1)
	      for (i = 1; i <= nVrtx; i++) {
                if (C->part[i] == partin)
                    C->part[i] = 0;
                else if (C->part[i] == partout)
                    C->part[i] = 1;
	      }
        }
	    info->depth--;
        free(targeted_weight);
        free(lb_save);

        *resultingNbPart = 2;

        info->timing_global += u_wseconds();
        info->timing_coars = info->info->timing_coars;
        info->timing_inipart = info->info->timing_inipart;
        info->timing_uncoars = info->info->timing_uncoars;

        return initializeRCoarsen(C);
    }

    //First we compute the bisection

    double Gweight_ub1 = 0.5; // for round up effect
    double Gweight_ub2 = 0.5; // for round up effect
    for (i=0; i<opt.nbPart/2; i++) {
        Gweight_ub1 += opt.ub[i];
    }
    for (i=opt.nbPart/2; i<opt.nbPart; i++) {
        Gweight_ub2 += opt.ub[i];
    }

    // TODO: Compare with old version; update as necessary
    MLGP_option* optbis = copyOpt(opt, 2);
    optbis->ub[0] = (Gweight_ub1 / opt.ratio) * (1 + 0.7* (opt.ratio-1) / log2(nbpart));
    optbis->ub[1] = (Gweight_ub2 / opt.ratio) * (1 + 0.7* (opt.ratio-1) / log2(nbpart));

    if (opt.debug > 0) {
        for (i=0; i<info->depth; i++)
	        printf("@");
        printf("In recursive nVrtx = %d, nbpart = %d, lb0 = %f, ub0 = %f, lb1 = %f, ub1 = %f\n", nVrtx, optbis->nbPart, optbis->lb[0], optbis->ub[0], optbis->lb[1], optbis->ub[1]);
    }

    C = VCycle2way(igraph, flag, *optbis, info->info);
    if (opt.debug > 0) check(C);
    rcoarsen *rcoars = initializeRCoarsen(C);
    free_opt(optbis);
    free(optbis);

    //We build the two subgraphes based on the bisection
    int part1, part2;
    dgraph *G1 = (dgraph*) malloc(sizeof(dgraph));
    dgraph *G2 = (dgraph*) malloc(sizeof(dgraph));
    splitCoarsen(rcoars, G1, G2, &part1, &part2);
    vwType G1weight = 0;
    for (i=1; i<=G1->nVrtx; i++)
        G1weight += G1->vw[i];
    vwType G2weight = 0;
    for (i=1; i<=G2->nVrtx; i++)
        G2weight += G2->vw[i];

    // TODO: Check correctness
     for (i=0; i<nbpart/2; i++) {
         opt.ub[i] = targeted_weight[i];
     }


    int partcount1 = -1,partcount2 = - 1;

    //We compute recursive bisection for G1
    MLGP_option* opt1 = copyOpt(opt, nbpart/2);
    info->rec1 = (rMLGP_info*) malloc (sizeof (struct rMLGP_info));
    initRInfoPart(info->rec1);
    rcoarsen *rcoars1 = rVCycle(G1, *opt1, info->rec1, &partcount1);
    free_opt(opt1);
    free(opt1);

    // TODO: Check correctness
     for (i=0; i<nbpart - nbpart/2; i++) {
         opt.ub[i] = targeted_weight[i+nbpart/2];
     }
    //We compute recursive bisection for G2
    MLGP_option* opt2 = copyOpt(opt, nbpart - nbpart/2);
    info->rec2 = (rMLGP_info*) malloc (sizeof (struct rMLGP_info));
    initRInfoPart(info->rec2);
    rcoarsen *rcoars2 = rVCycle(G2, *opt2, info->rec2, &partcount2);
    free_opt(opt2);
    free(opt2);

    //We rebuild and return final solution
    rcoars->next_rcoarsen1 = rcoars1;
    rcoars->next_rcoarsen2 = rcoars2;
    rcoars1->previous_rcoarsen = rcoars;
    rcoars2->previous_rcoarsen = rcoars;

    for (i=1; i<=nVrtx; i++) {
        if (C->part[i] == part1)
	        rcoars1->previous_index[rcoars->next_index[i]] = i;
	    if (C->part[i] == part2)
	        rcoars2->previous_index[rcoars->next_index[i]] = i;
    }

    for (i=1; i<=nVrtx; i++) {
	    if (C->part[i] == part1) {
	        if (rcoars->next_index[i] > rcoars1->coars->graph->nVrtx) {
                u_errexit("next_index %d > %d nVrtx1\n", rcoars->next_index[i], rcoars1->coars->graph->nVrtx);
            }
            C->part[i] = rcoars1->coars->part[rcoars->next_index[i]];
        }
	    else {
	        if (rcoars->next_index[i] > rcoars2->coars->graph->nVrtx) {
                u_errexit("next_index %d > %d nVrtx2\n", rcoars->next_index[i], rcoars2->coars->graph->nVrtx);
	        }
            C->part[i] = (partcount1) + rcoars2->coars->part[rcoars->next_index[i]];
        }
    }

    *resultingNbPart = partcount1+partcount2;

    //printf("Now free\n");
    free(targeted_weight);
    free(lb_save);
    freeRCoarsen(rcoars1);
    rcoars1 = (rcoarsen*) NULL;
    freeRCoarsen(rcoars2);
    rcoars2 = (rcoarsen*) NULL;
    info->depth--;
    info->timing_global += u_wseconds();
    info->timing_coars = info->rec1->timing_coars + info->rec2->timing_coars;
    info->timing_inipart = info->rec1->timing_inipart + info->rec2->timing_inipart;
    info->timing_uncoars = info->rec1->timing_uncoars + info->rec2->timing_uncoars;

    // printf("rcoars nvrtx from nVrtx = %d\n",rcoars->coars->graph->nVrtx);

    return rcoars;
}

