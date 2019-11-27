#ifndef DGRAPHTRAVERSAL_H
#define DGRAPHTRAVERSAL_H

#include "dgraph.h"
#include "utils.h"


void randTopSortOnParts(dgraph *G, idxType *part, idxType *toporder, idxType nbpart);
void topsortPriorities(dgraph *G, idxType *toporder, ecType *priorities);

void DFStopsort(dgraph *G, idxType *toporder);
void BFStopsort(dgraph *G, idxType *toporder);
void randDFStopsort(dgraph *G, idxType *toporder);
void randBFStopsort(dgraph *G, idxType *toporder);
void DFSsort(dgraph *G, idxType *toporder);
void BFSsort(dgraph *G, idxType *toporder);
void my_BFS(dgraph *G, idxType *visited);
void my_DFS(dgraph *G, idxType *visited);
void randDFSsort(dgraph *G, idxType *toporder);
void randBFSsort(dgraph *G, idxType *toporder);
void randTopsort(dgraph *G, idxType *toporder);
void mixtopsort(dgraph *G, idxType *toporder,int priority,int first);
void randTopsort(dgraph *G, idxType *toporder);


#endif
