#include "../part_inc/dgraph.h"
#include "../part_inc/utils.h"
#include "../part_inc/dgraphReader.h"

int computePartmatrix(dgraph *G, idxType *part, ecType** partmatrix, idxType node, idxType movetopart)
{
    idxType j;
    int flag = 0;
    for (j=G->outStart[node]; j<=G->outEnd[node]; j++) {
        idxType out = G->out[j];

        if (partmatrix[part[node]][part[out]] - G->ecOut[j]== 0 )
            flag=1;
        partmatrix[part[node]][part[out]] -= G->ecOut[j];

        if (partmatrix[movetopart][part[out]] == 0)
            flag=1;
        partmatrix[movetopart][part[out]] += G->ecOut[j];
    }
    for (j=G->inStart[node]; j<=G->inEnd[node]; j++) {
        idxType in = G->in[j];
        if (partmatrix[part[in]][part[node]] - G->ecIn[j] == 0)
            flag = 1;
        partmatrix[part[in]][part[node]] -= G->ecIn[j];
        if (partmatrix[part[in]][movetopart] == 0)
            flag =1;
        partmatrix[part[in]][movetopart] += G->ecIn[j];
    }
    return flag;
}

void buildPartmatrix(dgraph *G, idxType * part, idxType nbpart, ecType** partmatrix)
{
    idxType i,j;
    for (i=0; i< nbpart; i++)
        for (j=0; j<nbpart; j++)
            partmatrix[i][j] = 0;
    for (i=1; i<= G->nVrtx; i++){
        for (j=G->outStart[i]; j<=G->outEnd[i]; j++){
            idxType node = G->out[j];
            partmatrix[part[i]][part[node]] += G->ecOut[j];
        }
    }
}

int thereIsCycle(idxType nbpart, ecType** partmatrix)
{
    idxType* ready = (idxType*) malloc(nbpart * sizeof(idxType));
    idxType* traversed = (idxType*) calloc(nbpart, sizeof(idxType));
    idxType nbready = 0, i, j, nbtraversed = 0;

    for (i = 0; i < nbpart; i++){
        int nbin = 0;
        for (j = 0; j < nbpart; j++)
            if ((partmatrix[j][i] > 0)&&(j!=i))
                nbin++;
        if (nbin == 0)
            ready[nbready++] = i;
    }
    while (nbready > 0){
        idxType node = ready[--nbready];
        traversed[node] = 1;
        nbtraversed++;
        for (i = 0; i <nbpart; i++){
            if (partmatrix[node][i] == 0)
                continue;
            if (traversed[i] == 1)
                continue;
            int is_ready = 1;
            for (j = 0; j < nbpart; j++){
                if ((partmatrix[j][i] == 0)||(j==i))
                    continue;
                if (traversed[j] == 0)
                    is_ready = 0;
            }
            if (is_ready == 1)
                ready[nbready++] = i;
        }
    }
    free(ready);
    free(traversed);
    return (nbtraversed != nbpart);
}

int partSizeChecker(idxType* partsize, double ub_pw[], int nbpart, vwType maxvw){
    //as long as at least one of them is bigger than bound...
    int i;
    for ( i=0; i<nbpart; ++i )
        if (partsize[i] > ub_pw[i]+maxvw)
            return 1;
    return 0;
}


void allocateDGraphData(dgraph *G, idxType nVrtx, idxType nEdge, int frmt)
{
    /*assumes *G is allocated*/
    G->frmt = frmt;
    G->nVrtx = nVrtx;
    G->nEdge = nEdge;

    G->vw = NULL;
    G->ecIn = G->ecOut = NULL;

    if (nEdge <= 0)
        u_errexit("allocateDGraphData: empty edge set\n");

    if (nVrtx <=-1)
        u_errexit("allocateDGraphData: empty vertex set\n");

    G->inStart  = (idxType * ) umalloc(sizeof(idxType) * (nVrtx + 2), "G->inStart");
    G->inEnd= (idxType * ) umalloc(sizeof(idxType) * (nVrtx + 2), "G->inEnd");
    G->in = (idxType * ) umalloc(sizeof(idxType) * nEdge, "G->in");
    G->hollow = (int *) calloc(nVrtx+2, sizeof(int));
    G->outStart  = (idxType * ) umalloc(sizeof(idxType) * (nVrtx + 2), "G->outStart");
    G->outEnd  = (idxType * ) umalloc(sizeof(idxType) * (nVrtx + 2), "G->outEnd");
    G->out = (idxType * ) umalloc(sizeof(idxType) * nEdge, "G->out");
    G->sources = (idxType *) umalloc(sizeof(idxType) * (nVrtx+2), "G->sources");
    G->targets = (idxType *) umalloc(sizeof(idxType) * (nVrtx+2), "G->targets");

    if(G->hollow == NULL)
        u_errexit("Failed allocation of G->hollow");

    if (frmt == 1 || frmt == 3)
        G->vw = (vwType *) umalloc(sizeof(vwType) * (nVrtx+1), "G->vw");

    if (frmt == 2 || frmt == 3)
    {
        G->ecIn = (ecType *) umalloc(sizeof(ecType) * nEdge, "G->ecIn");
        G->ecOut = (ecType *) umalloc(sizeof(ecType) * nEdge, "G->ecOut");
    }
    G->maxVW = G->maxEC = -1;
}

void freeDGraphData(dgraph *G)
{
    free(G->inStart);
    free(G->inEnd);
    free(G->in);
    free(G->outStart);
    free(G->outEnd);
    free(G->out);
    free(G->hollow);
    free(G->sources);
    free(G->targets);

    if (G->frmt == 1 || G->frmt == 3)
    {
        if (!G->vw)
            u_errexit("free a graph with frmt %d, but vw is not allocated", G->frmt);
        else
            free(G->vw);
    }

    if (G->frmt == 2 || G->frmt == 3)
    {
        if (!G->ecIn) {
            u_errexit("free a graph with frmt %d, but ecIn is not allocated", G->frmt);
        }
        else
            free(G->ecIn);

        if (!G->ecOut)
            u_errexit("free a graph with frmt %d, but ecOut is not allocated", G->frmt);
        else
            free(G->ecOut);
    }

    G->frmt = -1;
    G->nVrtx = 0;
    G->nEdge = 0;
    G->maxVW = G->maxEC = -1;

    G->out = G->outEnd = G->outStart = G->in = G->inEnd = G->inStart = NULL;

    G->vw = NULL;
    G->ecIn = G->ecOut = NULL;
}

void dgraph_info(dgraph* G, int* maxindegree, int* minindegree, double* aveindegree, int* maxoutdegree, int* minoutdegree, double* aveoutdegree)
{
    idxType i;
    int indegree, outdegree;
    *aveindegree = 0.0;
    *aveoutdegree = 0.0;
    *maxindegree = 0;
    *minindegree = 0;
    *maxoutdegree = 0;
    *minoutdegree = 0;
    for (i=1; i<=G->nVrtx; i++) {
	indegree = G->inEnd[i]-G->inStart[i]+1;
	outdegree = G->outEnd[i]-G->outStart[i]+1;
	if (indegree > *maxindegree)
	    *maxindegree = indegree;
	if (indegree < *minindegree)
	    *minindegree = indegree;
	*aveindegree += indegree;
	if (outdegree > *maxoutdegree)
	    *maxoutdegree = outdegree;
	if (outdegree < *minoutdegree)
	    *minoutdegree = outdegree;
	*aveoutdegree += outdegree;
    }
    *aveindegree = *aveindegree / G->nVrtx;
    *aveoutdegree = *aveoutdegree / G->nVrtx;
}

void set_dgraph_info(dgraph* G)
{
    idxType i,j;

    double totvw = 0.0;
    double totec = 0.0;
    idxType maxindegree = 0;
    idxType maxoutdegree = 0;
    vwType maxVW = 0;
    ecType maxEC = 0;

    for (i=1; i<=G->nVrtx; i++) {
        if (maxindegree < G->inEnd[i] - G->inStart[i]+1)
            maxindegree = G->inEnd[i] - G->inStart[i]+1;

        totvw += G->vw[i];
        if (maxVW < G->vw[i])
            maxVW = G->vw[i];

        for (j=G->inStart[i]; j<=G->inEnd[i]; j++) {
            totec += G->ecIn[j];
            if (maxEC < G->ecIn[j])
                maxEC = G->ecIn[j];
        }
        if (maxoutdegree < G->outEnd[i] - G->outStart[i]+1)
            maxoutdegree = G->outEnd[i] - G->outStart[i]+1;

    }
    G->totvw = totvw;
    G->totec = totec;
    G->maxindegree = maxindegree;
    G->maxoutdegree = maxoutdegree;
    G->maxVW = maxVW;
    G->maxEC = maxEC;
}


typedef struct{
    idxType ngh;
    ecType cst;
} sortData;

int dgqsortfnct(const void *i1, const void *i2)
{
    idxType ii1=((sortData*)i1)->ngh, ii2 = ((sortData*)i2)->ngh;
    if ( ii1 < ii2) return -1;
    else if (ii1 > ii2) return 1;
    else return 0;
}

void sortNeighborLists(dgraph *G)
{
    /*sorts in and out in the increasing order of neighbor indices*/
    idxType i, j, at, nVrtx = G->nVrtx;
    idxType *out=G->out, *outStart = G->outStart, *outEnd = G->outEnd;
    idxType *in=G->in, *inStart = G->inStart, *inEnd = G->inEnd;
    ecType *ecIn=G->ecIn, *ecOut = G->ecOut;

    sortData *sd = (sortData*) umalloc(nVrtx * sizeof(sortData), "sd");

    /*sort in*/
    for (j=1; j <= nVrtx; j++)
    {
        at = 0;
        for (i = inStart[j]; i <= inEnd[j]; i++)
        {
            sd[at].ngh = in[i];
            if (G->frmt == 2 || G->frmt == 3)
                sd[at].cst = ecIn[i];
            at ++;
        }
        qsort(sd, inEnd[j]-inStart[j]+1, sizeof(sortData), dgqsortfnct);
        at = 0;
        for (i = inStart[j]; i <= inEnd[j]; i++)
        {
            in[i] =  sd[at].ngh ;
            if (G->frmt == 2 || G->frmt == 3)
                ecIn[i] = sd[at].cst ;
            at ++;
        }
    }

    /*sort out*/
    for (j=1; j <= nVrtx; j++)
    {
        at = 0;
        for (i = outStart[j]; i <= outEnd[j]; i++)
        {
            sd[at].ngh = out[i];
            if (G->frmt == 2 || G->frmt == 3)
                sd[at].cst = ecOut[i];
            at ++;
        }
        qsort(sd, outEnd[j]-outStart[j]+1, sizeof(sortData), dgqsortfnct);
        at = 0;
        for (i = outStart[j]; i <= outEnd[j]; i++)
        {
            out[i] =  sd[at].ngh ;
            if (G->frmt == 2 || G->frmt == 3)
                ecOut[i] = sd[at].cst ;
            at ++;
        }
    }

    free(sd);
}

void fillOutFromIn(dgraph *G)
{
    idxType i, at, j, nVrtx = G->nVrtx;
    idxType *out=G->out, *outStart = G->outStart, *outEnd=G->outEnd;
    idxType *in=G->in, *inStart = G->inStart, *inEnd = G->inEnd;
    ecType *ecIn=G->ecIn, *ecOut = G->ecOut;

    for (i = 0; i <= G->nVrtx+1; i++)
        outStart[i] = 0;

    /*count*/
    for (j = 1; j<= nVrtx; j++) {
        for (i = inStart[j]; i <= inEnd[j]; i++) {
		  outStart[G->in[i]]++;
	    }
    }
    /*prefix sum*/
    for (j = 1; j<= nVrtx+1; j++) {
        outStart[j] += outStart[j-1];
    }

    /*write*/
    for (j = 1; j<= nVrtx; j++) {
        for (i = inStart[j]; i <= inEnd[j]; i++) {
            at =  --outStart[in[i]];
            out[at] = j;
            if (G->frmt & DG_FRMT_EC)
                ecOut[at] = ecIn[i];
        }
    }
    G->maxoutdegree = 0;
    for (j = 1; j<= nVrtx; j++) {
        outEnd[j] = outStart[j+1]-1;
	    idxType degree = outEnd[j] - outStart[j] + 1;
	    G->maxoutdegree = G->maxoutdegree < degree ? degree : G->maxoutdegree;
    }

    if (inStart[1] != 0)
        u_errexit("fillOutFromIn: the first index not right");

    if (inStart[nVrtx+1] != G->nEdge)
        u_errexit("fillOutFromIn: the last index not right");
}

void fillInFromOut(dgraph *G)
{
    /*Fill G->in, G->inStart, G->inEnd, G->ecIn and G->maxindegree
     * based on G->out, G->outStart, G->outEnd, G->ecOut and G->maxoutdegree
     */
    idxType i, at, j, nVrtx = G->nVrtx;
    idxType *out=G->out, *outStart = G->outStart, *outEnd=G->outEnd;
    idxType *in=G->in, *inStart = G->inStart, *inEnd = G->inEnd;
    ecType *ecIn=G->ecIn, *ecOut = G->ecOut;
    for (i = 0; i <= G->nVrtx+1; i++)
        inStart[i] = 0;

    /*count*/
    for (j = 1; j<= nVrtx; j++) {
        for (i = outStart[j]; i <= outEnd[j]; i++) {
            inStart[G->out[i]]++;
        }
    }
    /*prefix sum*/
    for (j = 1; j<= nVrtx+1; j++) {
        inStart[j] += inStart[j-1];
    }

    /*write*/
    for (j = 1; j<= nVrtx; j++) {
        for (i = outStart[j]; i <= outEnd[j]; i++) {
            at =  --inStart[out[i]];
            in[at] = j;
            if (G->frmt & DG_FRMT_EC)
                ecIn[at] = ecOut[i];
        }
    }
    G->maxindegree = 0;

    for (j = 1; j<= nVrtx; j++) {
        inEnd[j] = inStart[j+1]-1;
        idxType degree = inEnd[j] - inStart[j] + 1;
        G->maxindegree = G->maxindegree < degree ? degree : G->maxindegree;
    }

    if (outStart[1] != 0)
        u_errexit("fillInFromOut: the first index not right");

    if (outStart[nVrtx+1] != G->nEdge)
        u_errexit("fillInFromOut: the last index not right");
}


void setVertexWeights(dgraph *G, vwType *vw)
{
    /*assuming n weights in vw*/
    idxType i, nVrtx = G->nVrtx;
    G->totvw = 0.0;
    G->maxVW = -1;

    if (G->frmt == 2 || G->frmt == 0)
    {
        if (G->vw == NULL)
            G->vw = (vwType * ) malloc(sizeof(vwType) * (nVrtx+1));
        G->frmt +=1;/*if ec, then ec+vw, if no w no cost, then vw*/
    }
    for (i = 1; i <= nVrtx; i++)
    {
        G->vw[i] = vw[i-1];
        G->totvw +=vw[i-1];
        G->maxVW = G->maxVW < vw[i-1] ? vw[i-1] : G->maxVW;
    }

}

int checkAcyclicity(dgraph *G, idxType *part, idxType nbpart)
{
    idxType i, j, k, ip;
    idxType** outpart = (idxType**) malloc(sizeof(idxType*)*nbpart);
    idxType* nbout = (idxType*) malloc(sizeof(idxType)*nbpart);
    idxType** inpart = (idxType**) malloc(sizeof(idxType*)*nbpart);
    idxType* nbin = (idxType*) malloc(sizeof(idxType)*nbpart);
    for (i=0; i<nbpart; i++) {
        outpart[i] = (idxType*) malloc(sizeof(idxType)*nbpart);
        nbout[i] = 0;
        inpart[i] = (idxType*) malloc(sizeof(idxType)*nbpart);
        nbin[i] = 0;
    }
    int isAcyclic = 1;

    for (i=1; i<=G->nVrtx; i++) {
        for (j=G->outStart[i]; j<=G->outEnd[i]; j++) {
            idxType outnode = G->out[j];
            if (part[i] == part[outnode])
                continue;

            int is_new = 1;
            for (k=0; k<nbout[part[i]]; k++)
                if (outpart[part[i]][k] == part[outnode]) {
                    is_new = 0;
                    break;
                }

            if (is_new == 1) {
                outpart[part[i]][nbout[part[i]]] = part[outnode];
                nbout[part[i]]++;
            }

            is_new = 1;
            for (k=0; k<nbin[part[outnode]]; k++)
                if (inpart[part[outnode]][k] == part[i]) {
                    is_new = 0;
                    break;
            }

            if (is_new == 1) {
                inpart[part[outnode]][nbin[part[outnode]]] = part[i];
                nbin[part[outnode]]++;
            }
        }
    }

    idxType* ready = (idxType*) malloc(sizeof(idxType)*(nbpart));
    idxType* nbinleft = (idxType*) malloc(sizeof(idxType)*(nbpart));
    int nbready = 0;
    for (i=0; i<nbpart; i++) {
        nbinleft[i] = nbin[i];
        if (nbin[i] == 0)
            ready[nbready++] = i;
    }

    int to = 0;
    while (nbready > 0) {
        idxType pno = ready[nbready-1];
        nbready--;
        to++;
        for (ip = 0; ip < nbout[pno]; ip++) {
            idxType succ = outpart[pno][ip];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0) {
                ready[nbready++] = succ;
            }
        }
    }
    for (i=0; i<nbpart; i++) {
        free(outpart[i]);
        free(inpart[i]);
    }
    free(nbinleft);
    free(ready);
    free(nbin);
    free(nbout);
    free(inpart);
    free(outpart);
    if (to != nbpart) {
        isAcyclic = 0;
    }
    return isAcyclic;
}

void removeMarkedEdges(dgraph* G, int* marked, idxType nbmarked){
    /* Modify G to remove marked edges
     */
    idxType i,j, outidx = 0;
    G->maxEC = G->totec = 0;

    for (i = 1; i <= G->nVrtx; i++){
        idxType formerstart = G->outStart[i];
        G->outStart[i] = outidx;
        G->outEnd[i-1] = outidx-1;
        for (j = formerstart; j <= G->outEnd[i]; j++){
            if (marked[j] == 1) {
                continue;
            }
            G->out[outidx] = G->out[j];
            G->ecOut[outidx] = G->ecOut[j];
            G->maxEC = G->maxEC < G->ecOut[j] ? G->ecOut[j] : G->maxEC;
            G->totec += G->ecOut[j];
            outidx++;
        }
    }
    G->outEnd[G->nVrtx] = outidx-1;
    G->outStart[G->nVrtx+1] = G->nEdge - nbmarked;
    G->outEnd[G->nVrtx+1] = G->nEdge - nbmarked - 1;
    G->maxoutdegree = 0;
    for (i = 1; i <= G->nVrtx; i++) {
        idxType degree = G->outEnd[i] - G->outStart[i] + 1;
        G->maxoutdegree = G->maxoutdegree < degree ? degree : G->maxoutdegree;
    }
    G->ecOut = (ecType*) realloc(G->ecOut, (G->nEdge - nbmarked + 1)*sizeof(ecType));
    G->out = (idxType*) realloc(G->out, (G->nEdge - nbmarked + 1)*sizeof(idxType));
    G->in = (idxType*) realloc(G->in, (G->nEdge - nbmarked + 1)*sizeof(idxType));
    G->ecIn = (ecType*) realloc(G->ecIn, (G->nEdge - nbmarked + 1)*sizeof(ecType));
    G->nEdge = G->nEdge - nbmarked;

    fillInFromOut(G);
}

idxType findPath(dgraph* G, idxType source, int* istarget, idxType* path, idxType* toplevel, idxType maxtoplevel){
    /* Find a path in G from the source to one of the target and return its length
     * Source is not considered as one of the targets in this function
     */
    if (toplevel[source] >= maxtoplevel)
        return 0;

    for (idxType i = G->outStart[source]; i <= G->outEnd[source]; i++){
        idxType outnode = G->out[i];
        if (istarget[outnode]){
            path[0] = i;
            return 1;
        }
        idxType length = findPath(G, outnode, istarget, path+1, toplevel, maxtoplevel);
        if (length > 0){
            path[0] = i;
            return length + 1;
        }
    }
    return 0;
}

void transitiveReduction(dgraph* G){
    /*Modify G to make it transitive irreductible
     * (It uses toplevel to optimize the traversal)
     */
    idxType* toplevel = (idxType*) malloc(sizeof(idxType) * (G->nVrtx + 1));
    computeToplevels(G, toplevel);

    int* toRemove = (int*) calloc(G->nEdge+1, sizeof(int));
    int* istarget = (int*) calloc(G->nVrtx+1, sizeof(int));
    idxType* path = (idxType*) malloc(sizeof(idxType)*(G->nEdge+1));
    idxType i,j, nbmarked = 0;

    for (i = 1; i <= G->nVrtx; i++){
        idxType maxtoplevel = -1, length = 0;
        for (j = G->outStart[i]; j<= G->outEnd[i]; j++){
            istarget[G->out[j]] = 1;
            maxtoplevel = maxtoplevel < toplevel[G->out[j]] ? toplevel[G->out[j]] : maxtoplevel;
        }
        for (j = G->outStart[i]; j<= G->outEnd[i]; j++) {
            length = findPath(G, G->out[j], istarget, path, toplevel, maxtoplevel);
            if (length > 0)
                break;
        }
        for (j = G->outStart[i]; j<= G->outEnd[i]; j++)
            istarget[G->out[j]] = 0;
        if (length > 0 ){
            idxType edgeweigth = -1;
            for (j = G->outStart[i]; j<= G->outEnd[i]; j++)
                if (G->out[j] == G->out[path[length-1]]) {
                    toRemove[j] = 1;
                    nbmarked++;
                    edgeweigth = G->ecOut[j];
                    break;
                }
            for (j = 0; j <= length-1; j++)
                G->ecOut[j] += edgeweigth;
        }
    }

    removeMarkedEdges(G, toRemove, nbmarked);

    free(toRemove);
    free(istarget);
    free(path);
    free(toplevel);
}


idxType sourcesList(dgraph* G, idxType* sources)
{
    /*Assume that sources is already allocated and fill the tab sources with the sources nodes of G
      Return the number of sources in G*/
    idxType i, ind = 0;
    for (i=1; i<=G->nVrtx; i++)
        if (G->inStart[i] > G->inEnd[i])
            sources[ind++] = i;
    return ind;
}

idxType sourcesListPart(dgraph* G, idxType* sources, idxType *part, int part_idx)
{
    /*Assume that sources is already allocated and fill the tab sources with the sources nodes of part_idx
      Return the number of sources in G*/
    idxType i, ind = 0, is_source, j;
    for (i=1; i<=G->nVrtx; i++) {
        if (part[i] != part_idx)
            continue;
        is_source = 1;
        for (j=G->inStart[i]; j <= G->inEnd[i]; j++)
            if (part[G->in[j]] == part_idx)
                is_source = 0;
        if (is_source == 1)
            sources[ind++] = i;
    }
    return ind;
}

idxType outputList(dgraph* G, idxType* output)
{
    /*Assume that output is already allocated and fill the tab output with the output nodes of G
      Return the number of output in G*/
    idxType i, ind = 0;
    for (i=1; i<=G->nVrtx; i++)
        if (G->outStart[i] > G->outEnd[i])
            output[ind++] = i;
    return ind;
}

void oneDegreeFirst(dgraph* G, idxType* order)
{
    idxType i,j;
    u_errexit("oneDegreeFirst: this function, as implemented, should not be called at all\n");
    for (i=1; i<=G->nVrtx; i++) {
	    idxType node = order[i];
	    int nboutnodes = G->outEnd[node] - G->outStart[node] + 1;
        int nbinnodes = G->inEnd[node] - G->inStart[node] + 1;
	    if ((nboutnodes == 1)&&(nbinnodes == 1)) {
	        for (j=i; j>=2; j--)
		        order[j] = order[j-1];
	        order[1] = node;
	    }
    }
}

idxType farthestNode(dgraph* G, idxType startnode)
{
    idxType to = 1, succ, node, i;
    idxType* ready = (idxType*) malloc(sizeof(idxType) * (G->nVrtx+1));
    int* touched = (int*) calloc(G->nVrtx+1, sizeof(int));
    ready[0] = startnode;
    touched[startnode] = 1;
    idxType endready = 0;
    idxType beginready = 0;

    while (endready >= beginready) {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        node = ready[beginready];
        beginready++;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            succ = G->out[i];
            if (touched[succ] == 0) {
                ready[++endready] = succ;
                touched[succ] = 1;
            }
        }
        for (i = G->inStart[node]; i <= G->inEnd[node]; i++) {
            succ = G->in[i];
            if (touched[succ] == 0) {
                ready[++endready] = succ;
                touched[succ] = 1;
            }
        }
    }
    return node;
}

void topsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1;
    idxType* ready = (idxType*) malloc(sizeof(idxType) * (G->nVrtx+1));
    idxType i;
    idxType nbready = G->nbsources;
    for (i = 0; i < nbready; i++)
        ready[i] = G->sources[i];
    sourcesList(G, ready);
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    while (nbready > 0) {
        if (to > G->nVrtx)
            u_errexit("Proof topsort to = %d\n", to);
        idxType node = ready[nbready-1];
        nbready--;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                ready[nbready++] = succ;
            else if (nbinleft[succ]<0)
                u_errexit("topsort: negative indegree\n");
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("topsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(ready);
}

void computeToplevels(dgraph* G, idxType* toplevels)
{
    /*We assume that toplevels is already allocated*/
    int i,j;
    idxType* toporder = (idxType*) malloc(sizeof(idxType)*(G->nVrtx+1));
    topsort(G, toporder);
    for (i=1; i<=G->nVrtx; i++) {
        idxType node = toporder[i];
        idxType tl = 0;

        for (j=G->inStart[node]; j<= G->inEnd[node]; j++)
            tl = tl > toplevels[G->in[j]]+1 ? tl : toplevels[G->in[j]]+1;
        toplevels[node] = tl;
    }
    free(toporder);
}
void computeWeightedToplevels(dgraph* G, ecType* toplevels)
{
    /*We assume that toplevels is already allocated*/
    int i,j;
    idxType* toporder = (idxType*) malloc(sizeof(idxType)*(G->nVrtx+1));
    topsort(G, toporder);
    for (i=1; i<=G->nVrtx; i++) {
        idxType node = toporder[i];
        idxType tl = 0;

        for (j=G->inStart[node]; j<= G->inEnd[node]; j++)
            // tl = tl > toplevels[G->in[j]] + G->vw[G->in[j]] + G->ecIn[j] ? tl : toplevels[G->in[j]] + G->vw[G->in[j]] + G->ecIn[j];
            if (tl < toplevels[G->in[j]] + G->vw[G->in[j]] + G->ecIn[j])
                tl = toplevels[G->in[j]] + G->vw[G->in[j]] + G->ecIn[j];
        toplevels[node] = tl;
    }
    free(toporder);
}

void computeBottomlevels(dgraph* G, idxType* bottomlevels)
{
    /*We assume that bottomlevels is already allocated*/
    int i,j;
    idxType* toporder = (idxType*) malloc(sizeof(idxType)*(G->nVrtx+1));
    topsort(G, toporder);
    for (i=1; i<=G->nVrtx; i++) {
        idxType node = toporder[G->nVrtx-i+1];
        idxType bl = 0;

        for (j=G->outStart[node]; j<= G->outEnd[node]; j++)
            bl = bl > bottomlevels[G->out[j]]+1 ? bl : bottomlevels[G->out[j]]+1;
        bottomlevels[node] = bl;
    }
    free(toporder);
}

void computeWeightedBottomlevels(dgraph* G, idxType* bottomlevels)
{
    /*We assume that bottomlevels is already allocated*/
    int i,j;
    idxType* toporder = (idxType*) malloc(sizeof(idxType)*(G->nVrtx+1));
    topsort(G, toporder);
    for (i=1; i<=G->nVrtx; i++) {
        idxType node = toporder[G->nVrtx-i+1];
        idxType bl = 0;

        for (j=G->outStart[node]; j<= G->outEnd[node]; j++)
            bl = bl > bottomlevels[G->out[j]]+ G->ecOut[j] ? bl : bottomlevels[G->out[j]]+ G->ecOut[j];
        bottomlevels[node] = bl + G->vw[node];
    }
    free(toporder);
}

void computeToplevelsWithTopOrder(dgraph* G, idxType* toplevels, idxType* toporder)
{
    /*We assume that toplevels is already computed and toprder allocated*/
    idxType i,j;
    for (i=1; i<=G->nVrtx; i++)
    {
        idxType node = toporder[i];
        int tl = 0;
        for (j=G->inStart[node]; j<= G->inEnd[node]; j++)
            tl = tl > toplevels[G->in[j]]+1 ? tl : toplevels[G->in[j]]+1;
        toplevels[node] = tl;
    }
}


ecType nbCommunications(dgraph* G, idxType* part)
{
    /*Return the number of communication in the partition
      that is the edge cut by counting the edges coming from
      a same node toward a partition only once*/

    ecType nbcomm = 0;
    int frmt = G->frmt;
    idxType i,j,k;
    for (i = 1; i<=G->nVrtx; i++)
    for (j=G->outStart[i]; j<=G->outEnd[i]; j++) {
        if (part[G->out[j]] != part[i]) {
        int is_new = 1;
        for (k=G->outStart[i]; k<j; k++)
            if (part[G->out[j]] == part[G->out[k]])
            is_new = 0;
        if (is_new){
            if (frmt & DG_FRMT_EC)
            nbcomm += G->ecOut[j];
            else
            nbcomm++;
        }
        }
    }
    return nbcomm;      
}


double computeLatency(dgraph* G, idxType* part, double l1, double l2)
{
    int i,j;
    double max = 0.0;
    double* latencies = (double*) malloc(sizeof(double)*(G->nVrtx + 1));
    idxType* toporder = (idxType*) malloc(sizeof(idxType)*(G->nVrtx+1));
    topsort(G, toporder);
    for (i=1; i<=G->nVrtx; i++)
    {
        idxType node = toporder[i];
	double lat = 1.0;
	for (j=G->inStart[node]; j<= G->inEnd[node]; j++) {
	  double currentLat = part[node] == part[G->in[j]] ? l1 : l2;
	  lat = lat > latencies[G->in[j]]+currentLat+1.0 ? lat : latencies[G->in[j]]+currentLat+1.0;
	}
	latencies[node] = lat;
	max = max > lat ? max : lat;
    }
    free(latencies);
    free(toporder);
    return max;
}

void computeDistances(dgraph* G, idxType sourceNode, idxType* dist)
{
    /*Fill the distance array with the distance toward sourceNode in G*/
    idxType i,j;
    for (i=1; i<=G->nVrtx; i++)
        dist[i] = G->nVrtx+1;
    idxType* ready = (idxType*) malloc(sizeof(idxType)*(G->nVrtx + 1));
    ready[0] = sourceNode;
    idxType nbready = 1;
    dist[sourceNode] = 0;
    while(nbready > 0){
        idxType node = ready[nbready-1];
        nbready--;
        for (j=G->inStart[node]; j<= G->inEnd[node]; j++) {
            idxType inNode = G->in[j];
            if (dist[inNode] == G->nVrtx+1){
                dist[inNode] = dist[node] + 1;
                ready[nbready++] = inNode;
            }
        }
        for (j=G->outStart[node]; j<= G->outEnd[node]; j++) {
            idxType outNode = G->out[j];
            if (dist[outNode] == G->nVrtx+1){
                dist[outNode] = dist[node] + 1;
                ready[nbready++] = outNode;
            }
        }
    }
    free(ready);
}

void topSortOnParts(dgraph *G, idxType *part, idxType *toporder, idxType nbpart){
    /*Fill toporder assuming it's already allocated*/

    printf("topSortOnParts called\n");
    idxType i, j, k;
    idxType** outpart = (idxType**) malloc(sizeof(idxType*)*nbpart);
    idxType* nbout = (idxType*) malloc(sizeof(idxType)*nbpart);
    idxType** inpart = (idxType**) malloc(sizeof(idxType*)*nbpart);
    idxType* nbin = (idxType*) malloc(sizeof(idxType)*nbpart);


    for (i=0; i<nbpart; i++){
        outpart[i] = (idxType*) malloc(sizeof(idxType)*nbpart);
        nbout[i] = 0;
        inpart[i] = (idxType*) malloc(sizeof(idxType)*nbpart);
        nbin[i] = 0;
    }



    printf("topSortOnParts before main kernel\n");
    // for (i = 1 ; i <= G->nVrtx ; i++){
    //     printf("%s %d\n",G->vertices[i-1],part[i]);
    // }
    for (i=1; i<=G->nVrtx; i++){
        for (j=G->outStart[i]; j<=G->outEnd[i]; j++){


            idxType outnode = G->out[j];
            //printf("%s(%d) --> %s(%d)\n", G->vertices[i-1],part[i],G->vertices[outnode-1],part[outnode]);
            if (part[i] == part[outnode])
                continue;
            int is_new = 1;
            for (k=0; k<nbout[part[i]]; k++)
                if (outpart[part[i]][k] == part[outnode]){

                    is_new = 0;
                    break;
                }
            if (is_new == 1){
                outpart[part[i]][nbout[part[i]]] = part[outnode];
                nbout[part[i]]++;
            }
            is_new = 1;
            //printf("i = %d, nVrtx = %d, nbpart = %d, outnode = %d, part[outnode] = %d\n", i, G->nVrtx, nbpart, outnode, part[outnode]);
            //printf("nbin[%d] = %d\n", part[outnode], nbin[part[outnode]]);
            for (k=0; k<nbin[part[outnode]]; k++)
                if (inpart[part[outnode]][k] == part[i]){
                    is_new = 0;
                    break;
                }
            if (is_new == 1){
                inpart[part[outnode]][nbin[part[outnode]]] = part[i];
                nbin[part[outnode]]++;
            }
        }
    }
    printf("topSortOnParts right after main kernel\n");
    idxType* ready = (idxType*) malloc(sizeof(idxType)*(nbpart));
    idxType* nbinleft = (idxType*) malloc(sizeof(idxType)*(nbpart));
    int nbready = 0;
    for (i=0; i<nbpart; i++) {
        nbinleft[i] = nbin[i];
        if (nbin[i] == 0)
            ready[nbready++] = i;
    }
    
    int to = 0;
    printf("In topsortpart, nbready = %d\n", nbready);
    while (nbready > 0) {
        idxType part = ready[nbready-1];
        nbready--;
        toporder[to++] = part;
    //printf("For topsortpart, nbready = %d, to = %d, part = %d\n", nbready, to, part);
        for (i = 0; i < nbout[part]; i++) {
            idxType succ = outpart[part][i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                ready[nbready++] = succ;
        }
    }

    //print topsort of parts

   //  //FILE* topsort_part  = fopen("topsort_of_parts.txt","w");
   // // FILE* part_in = fopen("partin.txt","w");
   // // FILE* part_out = fopen("partout.txt","w");
   //  for(i=0;i<nbpart;i++){
   //      //printf( "%d\n", toporder[i]);
   //      //fprintf(topsort_part, "%d\n", toporder[i]);
   //  }
   //  for(i = 0; i<nbpart ; i++){
   //      //fprintf(part_in, "%d <--- ", i);

   //      for(j = 0 ; j < nbin[i] ; j++){
   //          //fprintf(part_in, "%d ", inpart[i][j]);
   //      }
   //      //fprintf(part_in, "\n" );
   //  }



   //  fclose(topsort_part);
   //  fclose(part_in);
   //  fclose(part_out);


    if (to != nbpart){
        u_errexit("In topsortPart, not all sort: to = %d, nbpart = %d\n", to, nbpart);
    }
    for (i=0; i<nbpart; i++){
        free(outpart[i]);
        free(inpart[i]);
    }
    free(ready);
    free(nbinleft);
    free(nbin);
    free(nbout);
    free(inpart);
    free(outpart);
}

void DFStopsort_with_part(dgraph* G, idxType *part, int nbpart, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS respecting the convexity of the partitioning
    (every nodes of one part, then another part...)*/
    printf("DFStopsort_with_part called\n");
    idxType i,j;
    idxType* toporderpart = (idxType*) malloc (sizeof(idxType) * nbpart);
    topSortOnParts(G, part, toporderpart, nbpart);

    printf("DFStopsort_with_part topsort on parts done\n");
    idxType* topsortpart = (idxType*) malloc (sizeof(idxType) * nbpart);

    for (i = 0; i<nbpart; i++){
        topsortpart[toporderpart[i]] = i;
    }

    idxType to = 1;
    int current_part = 0;
    idxType* ready;
    ready = (idxType*) malloc(sizeof(idxType) * G->nVrtx);
    idxType nbready = sourcesList(G, ready);

    idxType* nbinleft;
    nbinleft = (idxType*) malloc(sizeof(idxType) * (G->nVrtx+1));
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    while (nbready > 0)
    {
        idxType node = -1;
        for (i = 1; i<=nbready; i++){
            if (part[ready[nbready-i]] == toporderpart[current_part]){
                node = ready[nbready-i];
                break;
            }
        }
        if (node == -1){
            current_part++;
            continue;
        }
        for (j = nbready-i; j<nbready-1; j++){
            ready[j] = ready[j+1]; /*BU2JH: a regarder*/
        }

        nbready--;
        toporder[to++] = node;

        //printf("Loop nbready = %d, node = %d, to = %d, part = %d\n", (int) nbready, (int) node, (int) to, part[node]);
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                ready[nbready++] = succ;
        }
    }
    if (to < G->nVrtx)
        u_errexit("DFStopsort_with_part: not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(toporderpart);
    free(topsortpart);
    free(ready);
}

int get_row_num(const char* node){
    
    int i = 0, row_num = 0, len = strlen(node);
    
    while(i < len && node[i] != ',')
    {
        i++;
    }
    i++;
    while(i < len && node[i] != ',')
    {
        row_num = row_num*10 + node[i] - 48;
        i++;
    }
    
    return row_num;
}

int get_spmm_col_num(const char* node){
    
    int i = 0, col_num = 0, len = strlen(node);
    while(i < len && node[i] != ',')
    {
        i++;
    }
    i++;
    while(i < len && node[i] != ',')
    {
        i++;
    }
    i++;
    while(i < len && node[i] != ',')
    {
        col_num = col_num*10 + node[i] - 48;
        i++;

    }

    return col_num;
}

void task_name(const char *node_name, char** task){
    int i = 0, len = strlen(node_name);
    while(i < len && node_name[i]!= ',')
    {
        (*task)[i] = node_name[i];
        i++;
    }
    (*task)[i] = '\0';
    if(i > 30)
    {
        u_errexit("ERROR, task_name hit %d with %s\n", i, node_name);
    }
    return;
}


int node_type(const char* node){
    
    char* task = (char*)malloc(50*sizeof(char));
    //strcpy(temp_node_name,node);
    task_name(node,&task);

    //if(!strcmp(task_name(node),"SPMM"))return 3;
    //if(node[0] == 'S' && node[0] == 'P' && node[0] == 'M' && node[0] == 'M' && node[0] == ',')return 3;
    if(!strcmp(task,"SPMM")) {free(task);return 3;}
    else if(!strcmp(task,"SPMV")) {free(task);return 3;}


    else if(!strcmp(node,"_lambda")) {free(task);return 0;}
    else if(!strcmp(node,"RNRED,RNBUF")) {free(task);return 0;}
    else if(!strcmp(node,"RESET,RN")) {free(task);return 0;}
    else if(!strcmp(node,"SQRT,RN")) {free(task);return 0;}
    else if(!strcmp(node,"RESET,actMask")) {free(task);return 0;}
    else if(!strcmp(node,"CONV,actMaks")) {free(task);return 0;}
    else if(!strcmp(node,"RED,temp2BUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,RBRBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"CHOL,RBR")) {free(task);return 0;}
    else if(!strcmp(node,"INV,RBR")) {free(task);return 0;}
    else if(!strcmp(node,"RED,PBPBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"CHOL,PBP")) {free(task);return 0;}
    else if(!strcmp(node,"INV,PBP")) {free(task);return 0;}
    else if(!strcmp(node,"EIGEN")) {free(task);return 0;}
    else if(!strcmp(node,"CONSTRUCTGB")) {free(task);return 0;}
    else if(!strcmp(node,"CONSTRUCTGA1")) {free(task);return 0;}
    else if(!strcmp(node,"CONSTRUCTGA2")) {free(task);return 0;}
    else if(!strcmp(node,"RED,XAPBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"SPEUPDATE,RAR")) {free(task);return 0;}
    else if(!strcmp(node,"SPEUPDATE,PAP")) {free(task);return 0;}
    else if(!strcmp(node,"TRANS,RAR")) {free(task);return 0;}
    else if(!strcmp(node,"RED,RAPBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,XARBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,RARBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,PAPBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,XBPBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,RBPBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"MAX")) {free(task);return 0;}

//these are constant nodes for nonloop version
    else if(!strcmp(node,"RED,XAXBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"RED,XBXBUF,0")) {free(task);return 0;}
    else if(!strcmp(node,"CHOL,XBX")) {free(task);return 0;}
    else if(!strcmp(node,"INV,XBX")) {free(task);return 0;}
    else if(!strcmp(node,"DLACPY,0,20")) {free(task);return 0;}


    else if(!strcmp(task,"_X")){free(task);return 1;}
    else if(!strcmp(task,"_AX")){free(task);return 1;}
    else if(!strcmp(task,"_P")){free(task);return 1;}
    else if(!strcmp(task,"_AP")){free(task);return 1;}
    else if(!strcmp(task,"MULT")){free(task);return 1;}
    else if(!strcmp(task,"SPMMRED")){free(task);return 1;}
    else if(!strcmp(task,"NORM")){free(task);return 1;}
    else if(!strcmp(task,"_Y")){free(task);return 1;}


    else if(!strcmp(task,"XTY")){free(task);return 2;}
    else if(!strcmp(task,"XY")){free(task);return 2;}
    else if(!strcmp(task,"DLACPY")){free(task);return 2;}
    else if(!strcmp(task,"SUB")){free(task);return 2;}
    else if(!strcmp(task,"COL")){free(task);return 2;}
    else if(!strcmp(task,"DOT")){free(task);return 2;}
    else if(!strcmp(task,"GET")){free(task);return 2;}
    else if(!strcmp(task,"UPDATE")){free(task);return 2;}
    else if(!strcmp(task,"SETZERO")){free(task);return 2;}
    else if(!strcmp(task,"ADD")){free(task);return 2;}
    else if(!strcmp(task,"SUBMAX")){free(task);return 2;}


    //else if(node[0]=='A' && node[1] == 'D')return 2;

    //printf("task %s ended in node_type\n",task);
    //fflush(stdout);
    free(task);
    //printf("task is freed\n");
    //fflush(stdout);

    
    return 0;
}


void my_generate_smallgraph(dgraph *G, char* file_name, int use_binary_input, int *edge_u, int *edge_v, double *edge_weight, int edgeCount, int vertexCount, const char** vertexName,double* vertexWeight, int block_divisor){
        printf("\n\n\n\ngenerate small Dgraph called\n\n\n\n");
        fflush(stdout);

        //printf("\n\n%s\n\n",vertexName[0]);

        char fnbin[100];
        strcpy(fnbin,file_name);
        strcat(fnbin,".bin");

        if(use_binary_input){
            FILE *fp = fopen(fnbin,"rb");
            if(fp!=NULL){
                fclose(fp);
                // printf("Yes-ish\n");
                readDGraphBinary(G,fnbin);
                return;
            }
            else if (strstr(file_name,".bin")){
                readDGraphBinary(G,file_name);
                return;
            }
            // printf("No bin file found for input!\n");
        }

        char line[1000];
        char token[100];
        idxType nVrtx = 0, nEdge = 0;
        idxType i, j, k = 0 ;
        idxType* nbneighbors; //how many neighbours of each vertex
        idxType** inneighbors; //adjacency list (in-neighbours) , parent tracking

        printf("vertexCount = %d\n",vertexCount);

        //idxType my_nVrtx = 45*block_divisor+block_divisor*block_divisor+27 , my_nEdge = 0 ;
        idxType my_nVrtx = vertexCount, my_nEdge = 0;
        idxType* my_nbneighbors;
        idxType** my_inneighbors;
        //char vertices[12000][100]; //vertex name
        //double vweights[12000]; 
       // ecType my_inedgeweights[12000][10]; //ecType double incoming edge weights
        ecType** my_inedgeweights;

        my_nbneighbors = (idxType*) calloc((vertexCount+1), sizeof(idxType));
        //my_nbneighbors = (idxType*) malloc((vertexCount+1) * sizeof(idxType));
        my_inneighbors = (idxType**) malloc((vertexCount+1) * sizeof(idxType*));
        my_inedgeweights = (ecType**) malloc((my_nVrtx+1) * sizeof(ecType*));
        for (i = 0; i <= my_nVrtx; i++)
        {
            my_inneighbors[i] = (idxType*) malloc(100 * sizeof(idxType)); //say ever vertex has max 100 parents
            my_inedgeweights[i] = (ecType*) malloc(100 * sizeof(ecType));
        }
        while(k < edgeCount)
        {
      
            int my_v1, my_v2; //(my_v1, my_v2) edge
            double edge_w;
           
            
            my_v2 = edge_u[k];
            my_v1 = edge_v[k];
            edge_w = edge_weight[k];

            my_nEdge++;

            if ((my_nbneighbors[my_v1+1] + 1) % 100 == 0)
            {
                my_inneighbors[my_v1+1] = (idxType*) realloc (my_inneighbors[my_v1+1], (my_nbneighbors[my_v1+1] + 101) * sizeof(idxType));
                my_inedgeweights[my_v1+1] = (ecType*) realloc (my_inedgeweights[my_v1+1], (my_nbneighbors[my_v1+1] + 101) * sizeof(ecType));
            }
    

            my_inneighbors[my_v1+1][my_nbneighbors[my_v1+1]] = my_v2+1;
            my_inedgeweights[my_v1+1][my_nbneighbors[my_v1+1]] = edge_w;
            my_nbneighbors[my_v1+1]++;
            
            k++;
        }
        printf("my vertex count = %d my edge count= %d\n", vertexCount, my_nEdge);

        //exit(1);

        G->frmt = 2;
        G->nVrtx = my_nVrtx;
        G->nEdge = my_nEdge;
        G->totvw = my_nVrtx;
        G->maxindegree = 0;
        G->hollow  = (int * ) calloc(my_nVrtx + 2, sizeof(int));
        G->inStart  = (idxType * ) calloc(my_nVrtx + 2, sizeof(idxType));
        G->inEnd= (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 2));
        G->in = (idxType * ) malloc(sizeof(idxType) * my_nEdge);

        // TODO: Update these two accordingly.
        G->maxVW = 1;
        G->maxEC = -1;

        //anik add
        G->vertices = (char**)malloc(sizeof(char*)*(my_nVrtx+1));
        for(i = 0;i<my_nVrtx;++i){
            G->vertices[i] = (char*)malloc(100*sizeof(char));
        }
        G->ecIn = (ecType *) malloc(sizeof(ecType) * my_nEdge);
        G->ecOut = (ecType *) malloc(sizeof(ecType) * my_nEdge);

        idxType idx = 0, degree;
        for (i=1; i<=my_nVrtx; i++){
            G->inStart[i] = idx;
            G->inEnd[i-1] = idx-1;
            if (i>1){
                degree = G->inEnd[i-1] - G->inStart[i-1] + 1;
                G->maxindegree = G->maxindegree < degree ? degree : G->maxindegree;
            }
            for (j=0; j< my_nbneighbors[i]; j++){
                G->in[idx] = my_inneighbors[i][j];
                G->ecIn[idx] = my_inedgeweights[i][j];
                if (G->maxEC < G->ecIn[idx])
                    G->maxEC = G->ecIn[idx];
                idx++;
            }

        }
        G->inStart[0] = 0;
        G->inEnd[0] = -1;
        G->inEnd[my_nVrtx] = idx-1;
        G->inEnd[my_nVrtx+1] = my_nEdge;
        G->inStart[my_nVrtx+1] = my_nEdge;
        degree = G->inEnd[my_nVrtx] - G->inStart[my_nVrtx] + 1;
        G->maxindegree = G->maxindegree < degree ? degree : G->maxindegree;
        if (my_nEdge <= 0)
            u_errexit("allocateDGraphData: empty edge set\n");

        if (my_nVrtx <=-1)
            u_errexit("allocateDGraphData: empty vertex set\n");

        G->outStart  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 2));
        G->outEnd  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 2));
        G->out = (idxType * ) malloc(sizeof(idxType) * my_nEdge);

        // TODO: Verify vw values for the partitioner
        G->vw = (vwType *) malloc(sizeof(vwType) * (my_nVrtx+1));
        G->vWeight = (ecType*)malloc(sizeof(ecType)*(my_nVrtx+1));
    //    G->totvw = 0;
        for (i=1; i<=my_nVrtx; i++){
      //      G->vw[i] = vweights[i-1];
            G->vw[i] = 1.0;
     //       G->totvw += G->vw[i];
    //      printf("G.vw[%d] = %lf totvw = %lf\n",i,G->vw[i],G->totvw);
        }

        //for (i=0; i< my_nEdge; i++){
            //G->ecIn[i] = 1;
            //G->ecOut[i] = 1;
            //G->ecIn[i] = edge_weights[i];
            //G->ecOut[i] = edge_weights[i];
        //}

        for(i=0;i<my_nVrtx;++i){
     
            strcpy(G->vertices[i],vertexName[i]);
            //printf("vertices %d = %s\n",i,G->vertices[i]);
            G->vWeight[i+1] = vertexWeight[i+1];
            //printf("vWeight[%d] = %lf\n",i+1,G->vWeight[i+1]);
        }

//      for(i = 0 ; i < G->nEdge ; i++){
//          printf("ecIn[%d] = %lf\n",i,G->ecIn[i]);
//      }

     //   printf("before fillout for min\n");
    //    for(i = 0 ; i < G->nEdge ; i++){
    //      printf("out[%d] = %d\n",i,G->out[i]);
    //    }

        // FILE* indegree_file;
        // indegree_file = fopen("indegree.txt","w");
        // for(i = 1 ; i <= my_nVrtx ; i++){
        //     //fprintf(indegree_file, "%s %d\n", G->vertices[i-1],G->inEnd[i] - G->inStart[i]);
        // }


        // fclose(indegree_file);

        printf("before calling filloutfrom\n");

        /*FILE* entire_small_graph;
        entire_small_graph = fopen("entire_small_graph.txt","w");
        for(i = 1 ; i <= my_nVrtx ; i++){
            for (j = G->inStart[i]; j <= G->inEnd[i]; ++j)
            {
                //code
                fprintf(entire_small_graph, "%s ---> %s;\n", G->vertices[G->in[j]-1],G->vertices[i-1]);
            }
        }
        fclose(entire_small_graph);*/



        
        fillOutFromIn(G);

    // FILE *initial_graph;
    // initial_graph = fopen("initial_graph.dot","w");
    // fprintf(initial_graph,"digraph {\n");

    // for (i = 1;i<=G->nVrtx ; i++){
    //     for(j = G->inStart[i] ; j <= G->inEnd[i]; j++){
    //         //printf("%d ---> %d\n",G->in[j],i);
    //         //fprintf(initial_graph,"%d -> %d;\n",G->in[j],i);
    //     }
    // }
    // fprintf(initial_graph, "}\n" );

    // fclose(initial_graph);



        printf("filloutfrom in done\n");



        G->sources  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 1));
        G->targets  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 1));
        G->nbsources = 0;
        G->nbtargets = 0;
        for (i=1; i<=nVrtx; i++){
            if (G->inEnd[i] < G->inStart[i])
                G->sources[G->nbsources++] = i;
            if (G->outEnd[i] < G->outStart[i])
                G->targets[G->nbtargets++] = i;
        }



        //exit(1);
       // printf("total vweight = %lf\n",G->totvw); 
        
        // if(use_binary_input){
        //     FILE *fp = fopen(fnbin,"wb");

        //     if(fp == NULL){
        //        u_errexit("Cannot open file");
        //     }

        //     fwrite(&G->frmt,sizeof(G->frmt),1,fp);
        //     fwrite(&G->nVrtx,sizeof(G->nVrtx),1,fp);
        //     fwrite(&G->nEdge,sizeof(G->nEdge),1,fp);
        //     fwrite(&G->totvw,sizeof(G->totvw),1,fp);
        //     fwrite(&G->maxindegree,sizeof(G->maxindegree),1,fp);
        //     fwrite(&G->maxoutdegree,sizeof(G->maxoutdegree),1,fp);
        //     fwrite(G->inStart,sizeof(G->inStart[0]),(nVrtx+2),fp);
        //     fwrite(G->inEnd,sizeof(G->inEnd[0]),(nVrtx+2),fp);
        //     fwrite(G->in,sizeof(G->in[0]),nEdge,fp);
        //     fwrite(G->vw,sizeof(G->vw[0]),(nVrtx+1),fp);
        //     fwrite(G->outStart,sizeof(G->outStart[0]),(nVrtx+2),fp);
        //     fwrite(G->outEnd,sizeof(G->outEnd[0]),(nVrtx+2),fp);
        //     fwrite(G->out,sizeof(G->out[0]),nEdge,fp);
        //     fwrite(G->ecIn,sizeof(ecType),nEdge,fp);
        //     fwrite(G->ecOut,sizeof(ecType),nEdge,fp);

        //     //anik add
        //     fwrite(G->vertices,sizeof(G->vertices[0]),(nVrtx+1),fp);
        //     fwrite(G->vWeight,sizeof(G->vWeight[0]),(nVrtx+1),fp);


        //     fclose(fp);
        // }




        // for (i = 1; i <= nVrtx; i++)
        //     free(inneighbors[i]);
        // free(nbneighbors);
        // free(inneighbors);



}


// generate small graph by breaking up the large graph
void create_smallgraph_datastructure_sparse(int *edge_u, int *edge_v, double *edge_weight, int edgeCount, int vertexCount, const char** vertexName,double* vertexWeight, int block_divisor, char*** newVertexName, int new_vertexcount,
                                            int new_edgeCount, int **newEdge_u, int **newEdge_v, double** newEdge_weight, int** prev_vertex , double** newVertex_weight, int** nnzblock_matrix, int newRblock, int newCblock,
                                            int* updatedVertexCount,int *updatedEdgeCount, int ***vmap){

    int v1,v2;
    //printf("edgecount = %d\n",edgeCount);
    printf("block divisor = %d\n",block_divisor);
    int k = 0,i,j;
    
    int type=0;
    int global_vertex_index = 0; 
    int local_row = 0;
    int local_col = 0;
    char loc_row_str_type1[10];
    char task_id[10];
    int vmap_index = 0;

    int **vertex_map;
    vertex_map = (int**)calloc((vertexCount+1),sizeof(int*));

    
    for(i = 0;i< (vertexCount);++i)
    {
        vertex_map[i] = (int*)calloc((block_divisor*block_divisor+1),sizeof(int));
    }

    printf("new_vertexcount = %d\n",new_vertexcount);
    int buff = 0;
    while (k < vertexCount){
       // printf("%s %d\n", vertexName[k],global_vertex_index);
        type = node_type(vertexName[k]);

        char* task = (char*)malloc(50*sizeof(char));
        task_name(vertexName[k],&task);

      //  printf("k = %d node = %s type %d global_vertex = %d\n", k,vertexName[k],type,global_vertex_index);
        //k++;
        //continue;
        
        if(type == 3){
            buff = 0;
            //k++;
            //continue;
            //printf("node= %s\n", vertexName[k]);
            vmap_index = 0;
            int spmm_row =0,spmm_col =0;
            int s = 5;
            while(vertexName[k][s]!= ','){
                spmm_row = spmm_row*10+vertexName[k][s++]-48;
            }
            s++;
            while(vertexName[k][s]!= ','){
                spmm_col = spmm_col*10+vertexName[k][s++]-48;
            }
            //printf("spmm_row = %d spmm_col = %d\n",spmm_row,spmm_col);

            //if(spmm_row == 7 && spmm_col == 0)
            
            for(i=0;i<block_divisor;i++){
                char loc_row[10];
                local_row = spmm_row*block_divisor+i;
                sprintf(loc_row,"%d",local_row);
                //printf("local row = %d\n",local_row);
                for(j=0;j<block_divisor;j++){

                    char loc_col[10];
                    local_col = spmm_col*block_divisor+j;

                    //if(!nnzblock_matrix[local_row][local_col]) continue;
                    //printf("nnzblock_matrix[%d][%d] = %d\n",local_row,local_col,nnzblock_matrix[local_row][local_col]);

                    if(local_row < newRblock && local_col < newCblock && nnzblock_matrix[local_row][local_col]) 
                    {
                        sprintf(loc_col,"%d",local_col);
                        char total_node[1000];

                        strcpy(total_node,task);
                        strcat(total_node,",");
                        strcat(total_node,loc_row);
                        strcat(total_node,",");
                        strcat(total_node,loc_col);
                        strcat(total_node,",");


                        char buff_c[2];
                        //sprintf(buff_c,"%d",local_row%16);
                        buff++;
                        strcat(total_node,"0");
                        //printf("totalnode made\n");
                        strcpy((*newVertexName)[global_vertex_index],total_node);
                        //printf("%s %d\n", (*newVertexName)[global_vertex_index],global_vertex_index);
                        //(*newVertexName)[global_vertex_index] = strdup(total_node);
                        (*prev_vertex)[global_vertex_index] = k;
                        //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
                        vertex_map[k][i*block_divisor+j] = global_vertex_index;
                        //printf("vertex_map[%d][%d] = %d(%s)\n",k,i*block_divisor+j , global_vertex_index,(*newVertexName)[global_vertex_index]);
                        (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
                        vmap_index++;
                        global_vertex_index++;
                        //printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);
                    }
                }
            }

        }

        else if(type == 0) {
            //printf("global_vertex_index %d vertex %s\n",global_vertex_index,vertexName[k]);
            vmap_index = 0;
            strcpy((*newVertexName)[global_vertex_index],vertexName[k]);
            for(i = 0 ; i < block_divisor ; i++)
                vertex_map[k][i] = global_vertex_index;
            (*prev_vertex)[global_vertex_index] = k;
            //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
            (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
            global_vertex_index++;
          // printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);

        }

        else if(type == 1) {
            //k++;
            //continue;
            vmap_index = 0;
            int row_num = 0;
            int s = strlen(task)+1;
            //printf("type 1 task %s s value %d char %c\n",task,s,vertexName[k][s]);
            while(vertexName[k][s]!= '\0'){
                row_num = row_num*10+vertexName[k][s]-48;
                s++;
            }
            //printf("row num = %d\n",row_num);
            for(i=0;i<block_divisor ;i++){
                int loc;
                loc= row_num*block_divisor+i;
                if(loc >= newRblock) break;
                //char* loc_row = (char*)malloc(10*sizeof(char));
                
                sprintf(loc_row_str_type1,"%d",loc);
              //  printf("loc row str %s\n", loc_row_str_type1);
                char total_node[1000];
                strcpy(total_node,task);
                strcat(total_node,",");
                strcat(total_node,loc_row_str_type1);

                
              //  printf("type 1 total node %s\n",total_node);

                vertex_map[k][i] = global_vertex_index;
                vmap_index++;

                (*prev_vertex)[global_vertex_index] = k;


                strcpy((*newVertexName)[global_vertex_index],total_node);
                //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
                (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
              //  printf("strdup e problem????\n");
                global_vertex_index++;
                //printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);

            }
         
        }
        
        else if(type == 2){

            //special case col node
            if(!strcmp(task,"COL")){

                vmap_index = 0;

                int row_num = 0;
                buff =0;
                int s = strlen(task)+1;
                //printf("type 1 task %s s value %d char %c\n",task,s,vertexName[k][s]);
                while(vertexName[k][s]!= ','){
                    row_num = row_num*10+vertexName[k][s]-48;
                    s++;
                }
                //printf("row num = %d\n",row_num);
                for(i=0;i<block_divisor;i++){
                    int loc;
                    loc= row_num*block_divisor+i;
                    if(loc >= newRblock) break;
                    //char* loc_row = (char*)malloc(10*sizeof(char));
                    
                    sprintf(loc_row_str_type1,"%d",loc);
                    //printf("loc row str %s\n", loc_row_str_type1);
                    char total_node[1000];
                    strcpy(total_node,task);
                    strcat(total_node,",");
                    strcat(total_node,loc_row_str_type1);
                    strcat(total_node,",");


                    char buff_c[2];
                        sprintf(buff_c,"%d",loc%16);
                        buff++;
                        strcat(total_node,buff_c);
                    //printf("type 1 total node %s\n",total_node);

                    vertex_map[k][i] = global_vertex_index;
                        vmap_index++;
                    (*prev_vertex)[global_vertex_index] = k;

                    strcpy((*newVertexName)[global_vertex_index],total_node);
                    //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
                    (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
                    //printf("strdup e problem????\n");
                    global_vertex_index++;
                    //printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);

                }

            }

            //special case col node
            else if(!strcmp(task,"DOT")){

                vmap_index = 0;

                int row_num = 0;
                buff =0;
                int s = strlen(task)+1;
                //printf("type 1 task %s s value %d char %c\n",task,s,vertexName[k][s]);
                while(vertexName[k][s]!= ','){
                    row_num = row_num*10+vertexName[k][s]-48;
                    s++;
                }
                //printf("row num = %d\n",row_num);
                for(i=0;i<block_divisor;i++){
                    int loc;
                    loc= row_num*block_divisor+i;
                    if(loc >= newRblock) break;
                    //char* loc_row = (char*)malloc(10*sizeof(char));
                    
                    sprintf(loc_row_str_type1,"%d",loc);
                    //printf("loc row str %s\n", loc_row_str_type1);
                    char total_node[1000];
                    strcpy(total_node,task);
                    strcat(total_node,",");
                    strcat(total_node,loc_row_str_type1);
                    strcat(total_node,",");


                    char buff_c[2];
                        sprintf(buff_c,"%d",loc%16);
                        buff++;
                        strcat(total_node,buff_c);
                    //printf("type 1 total node %s\n",total_node);

                    vertex_map[k][i] = global_vertex_index;
                        vmap_index++;
                    (*prev_vertex)[global_vertex_index] = k;

                    strcpy((*newVertexName)[global_vertex_index],total_node);
                    //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
                    (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
                    //printf("strdup e problem????\n");
                    global_vertex_index++;
                    //printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);

                }

            }

            else if(!strcmp(task,"XTY")){

                vmap_index = 0;
                int task_id_index=strlen(vertexName[k])-1;
                while(vertexName[k][task_id_index]!= ',')
                    task_id_index--;
                task_id_index++;
                int t = 0;
                while(vertexName[k][task_id_index]!= '\0')
                    task_id[t++] = vertexName[k][task_id_index++];
                task_id[t] = '\0';

                int row_num = 0;
                buff =0;
                int s = strlen(task)+1;
                //printf("type 1 task %s s value %d char %c\n",task,s,vertexName[k][s]);
                while(vertexName[k][s]!= ','){
                    row_num = row_num*10+vertexName[k][s]-48;
                    s++;
                }
                //printf("row num = %d\n",row_num);
                for(i=0;i<block_divisor;i++){
                    int loc;
                    loc= row_num*block_divisor+i;
                    if(loc >= newRblock) break;
                    //char* loc_row = (char*)malloc(10*sizeof(char));
                    
                    sprintf(loc_row_str_type1,"%d",loc);
                    //printf("loc row str %s\n", loc_row_str_type1);
                    char total_node[1000];
                    strcpy(total_node,task);
                    strcat(total_node,",");
                    strcat(total_node,loc_row_str_type1);
                    strcat(total_node,",");


                    char buff_c[2];
                    sprintf(buff_c,"%d",loc%16);
                    buff++;
                    strcat(total_node,buff_c);
                    strcat(total_node,",");
                    strcat(total_node,task_id);

                    vertex_map[k][i] = global_vertex_index;
                        vmap_index++;
                    //printf("type 1 total node %s\n",total_node);
                    (*prev_vertex)[global_vertex_index] = k;

                    strcpy((*newVertexName)[global_vertex_index],total_node);
                    (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
                    //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
                    //printf("strdup e problem????\n");
                    global_vertex_index++;
                    //printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);

                }
            }

            else{
                vmap_index = 0;

                int task_id_index=strlen(vertexName[k])-1;
                while(vertexName[k][task_id_index]!= ',')
                    task_id_index--;
                task_id_index++;
                int t = 0;
                while(vertexName[k][task_id_index]!= '\0')
                    task_id[t++] = vertexName[k][task_id_index++];
                task_id[t] = '\0';

                int row_num = 0;
                buff =0;
                int s = strlen(task)+1;
                //printf("type 1 task %s s value %d char %c\n",task,s,vertexName[k][s]);
                while(vertexName[k][s]!= ','){
                    row_num = row_num*10+vertexName[k][s]-48;
                    s++;
                }
                //printf("row num = %d\n",row_num);
                for(i=0;i<block_divisor;i++){
                    int loc;
                    loc= row_num*block_divisor+i;
                    if(loc >= newRblock) break;
                    //char* loc_row = (char*)malloc(10*sizeof(char));
                    
                    sprintf(loc_row_str_type1,"%d",loc);
                    //printf("loc row str %s\n", loc_row_str_type1);
                    char total_node[1000];
                    strcpy(total_node,task);
                    strcat(total_node,",");
                    strcat(total_node,loc_row_str_type1);
                    strcat(total_node,",");


                    
                    strcat(total_node,task_id);
                    //printf("type 1 total node %s\n",total_node);


                    vertex_map[k][i] = global_vertex_index;
                        vmap_index++;
                    (*prev_vertex)[global_vertex_index] = k;

                    strcpy((*newVertexName)[global_vertex_index],total_node);
                    //printf("prev_vertex[%s] = %s\n",(*newVertexName)[global_vertex_index],vertexName[k]);
                    (*newVertex_weight)[global_vertex_index] = vertexWeight[k]/block_divisor;
                    //printf("strdup e problem????\n");
                    global_vertex_index++;
                    //printf("added %s prev %d\n",(*newVertexName)[global_vertex_index-1],(*prev_vertex)[global_vertex_index-1]);

                }

            }
        }

        k++;
        free(task); 
    }
                
    printf("updated vertex count = %d\n",global_vertex_index);
    *updatedVertexCount = global_vertex_index;

    //for(i = 0; i < global_vertex_index ; i++){
        //printf("new node %d = %s %d\n",i,(*newVertexName)[i],(*prev_vertex)[i]);
    //}

    k = 0;
    int global_edge_index = 0;
    //char* tsk1 = (char*)malloc(100*sizeof(char));
    //char* tsk2 = (char*)malloc(100*sizeof(char));
    while(k < edgeCount){
        //printf("k = %d %d %d\n",k,edge_u[k],edge_v[k]);

        v1 = edge_u[k];
        v2 = edge_v[k];

        //printf("k = %d %d %d\n",k,edge_u[k],edge_v[k]);

        //task_name(vertexName[v1],&task1);
        //task_name(vertexName[v2],&task2);

        //printf("k = %d %d %d \n",k,v1,v2);

        int nd_type1 = node_type(vertexName[v1]);
        int nd_type2 = node_type(vertexName[v2]);

        //printf("k = %d %s(%d) %s(%d)\n",k,vertexName[v1],nd_type1,vertexName[v2],nd_type2);

        int sp_row,sp_col;


        //if(node_type(vertexName[v1]) == 3 )
            //printf("k = %d node 1 type = %s(%d) node 2 = %s(%d)\n",k,vertexName[v1],nd_type1, vertexName[v2],nd_type2);

        if(nd_type1 == 0 && nd_type2 == 0){

            (*newEdge_u)[global_edge_index] = vertex_map[v1][0];
            (*newEdge_v)[global_edge_index] = vertex_map[v2][0];
            (*newEdge_weight)[global_edge_index] = edge_weight[k];
            //printf("%s --> %s new added %s --> %s\n",vertexName[v1],vertexName[v2],(*newVertexName)[(*newEdge_u)[global_edge_index]],(*newVertexName)[(*newEdge_v)[global_edge_index]]);
            global_edge_index++;
        }
        else if(nd_type1 == 3 || nd_type2 == 3){

            if(nd_type1 == 3){

                for(i = 0 ; i < block_divisor ; i++){
                    sp_row = get_row_num(vertexName[v1])*block_divisor+i;


                    for(j = 0 ; j < block_divisor ; j++){
                        sp_col = get_spmm_col_num(vertexName[v1])*block_divisor+j;
                        if(sp_row < newRblock && sp_col < newCblock && vertex_map[v1][i*block_divisor+j])
                        {
                            (*newEdge_u)[global_edge_index] = vertex_map[v1][i*block_divisor+j];
                            (*newEdge_v)[global_edge_index] = vertex_map[v2][i];
                            (*newEdge_weight)[global_edge_index] = edge_weight[k]/block_divisor;
                            //printf("added %s --> %s\n",(*newVertexName)[(*newEdge_u)[global_edge_index]],(*newVertexName)[(*newEdge_v)[global_edge_index]]);
                            global_edge_index++;
                        }
                    }
                }
            }
            else if(vertexName[v1][0] == 'S'){
            //printf("%s --> %s\n",vertexName[v1],vertexName[v2]);
                for(i = 0 ; i < block_divisor ; i++){
                    sp_row = get_row_num(vertexName[v2])*block_divisor+i;
                    for(j = 0 ; j < block_divisor ; j++){
                        sp_col = get_spmm_col_num(vertexName[v2])*block_divisor+j;
                        if(sp_row < newRblock && sp_col < newCblock && vertex_map[v2][i*block_divisor+j])
                        {
                            (*newEdge_u)[global_edge_index] = vertex_map[v1][i];
                            (*newEdge_v)[global_edge_index] = vertex_map[v2][i*block_divisor+j];
                            (*newEdge_weight)[global_edge_index] = edge_weight[k]/block_divisor;
                            //printf("added %s --> %s\n",(*newVertexName)[(*newEdge_u)[global_edge_index]],(*newVertexName)[(*newEdge_v)[global_edge_index]]);
                            global_edge_index++;
                        }
                    }
                }   
            }
            else if(vertexName[v1][0] == 'D'){
            //printf("%s --> %s\n",vertexName[v1],vertexName[v2]);
              for(j = 0 ; j < block_divisor ; j++){
                    sp_col = get_spmm_col_num(vertexName[v2])*block_divisor+j;
                    for(i = 0 ; i < block_divisor ; i++){
                        sp_row = get_row_num(vertexName[v2])*block_divisor+i;
                        //if(!strcmp((*newVertexName)[vertex_map[v1][j]],"DLACPY,28,1"))
                        //printf("%s\n",vertexName[])
                        //printf("sp_row = %d sp_col = %d vmap[%d][%d] = %d\n", sp_row,sp_col,v2,i*block_divisor+j,vertex_map[v2][i*block_divisor+j]);
                        if(sp_row < newRblock && sp_col < newCblock && vertex_map[v2][i*block_divisor+j] > 0 )//&& vertex_map[v2][i*block_divisor+j] <= global_vertex_index)
                        {
                            (*newEdge_u)[global_edge_index] = vertex_map[v1][j];
                            (*newEdge_v)[global_edge_index] = vertex_map[v2][i*block_divisor+j];
                            (*newEdge_weight)[global_edge_index] = edge_weight[k]/block_divisor;
                            //if(!strcmp((*newVertexName)[vertex_map[v1][j]],"DLACPY,28,1"))
                            //printf("added %s --> %s\n",(*newVertexName)[(*newEdge_u)[global_edge_index]],(*newVertexName)[(*newEdge_v)[global_edge_index]]);
                            global_edge_index++;
                        }
                    }
                }  
            }

        }

        //else if(node_type(vertexName[v1]) == 1 && node_type(vertexName[v2]) == 2){

        //}
        
        else {

            for(i = 0 ; i < block_divisor ; i++){
                // if(!strcmp(task1,"SETZERO") && !strcmp(task2,"SPMMRED")){
                if(nd_type1 == 0)
                    sp_row = get_row_num(vertexName[v2])*block_divisor+i;
                else 
                    sp_row = get_row_num(vertexName[v1])*block_divisor+i;
            
                //printf("%s --> %s  %d\n",vertexName[v1],vertexName[v2], sp_row);

                if(vertexName[v1][0] == 'S' && vertexName[v1][1] == 'E' && vertexName[v1][2] == 'T' && vertexName[v2][3] == 'M' &&
                        vertexName[v2][4] == 'R' && vertexName[v2][5] == 'E' && vertexName[v2][6] == 'D' ){
                    for(j = 0 ; j < block_divisor ; j++){


                        sp_col = get_spmm_col_num(vertexName[edge_u[k-1]])*block_divisor+j;

                        //printf("spmm node = %s col no = %d cal col no = %d\n",vertexName[edge_u[k-1]],get_spmm_col_num(vertexName[edge_u[k-1]]),sp_col);
                        if(sp_row < newRblock && sp_col < newCblock && vertex_map[edge_u[k-1]][i*block_divisor+j])
                        {
                            (*newEdge_u)[global_edge_index] = vertex_map[v1][i];
                            (*newEdge_v)[global_edge_index] = vertex_map[v2][i];
                            (*newEdge_weight)[global_edge_index] = edge_weight[k]/block_divisor;
                            //printf("added %s --> %s\n",(*newVertexName)[(*newEdge_u)[global_edge_index]],(*newVertexName)[(*newEdge_v)[global_edge_index]]);
                            global_edge_index++;
                        }

                    }
                }
                else{
                    if(sp_row < newRblock){
                        (*newEdge_u)[global_edge_index] = vertex_map[v1][i];
                        (*newEdge_v)[global_edge_index] = vertex_map[v2][i];
                        (*newEdge_weight)[global_edge_index] = edge_weight[k]/block_divisor;
                        //printf("%s --> %s new added %s --> %s\n",vertexName[v1],vertexName[v2],(*newVertexName)[(*newEdge_u)[global_edge_index]],(*newVertexName)[(*newEdge_v)[global_edge_index]]);
                        global_edge_index++;
                    }
                }
            }
        }

        k++;
    }
                
    printf("updated edge count = %d\n", global_edge_index);
    *updatedEdgeCount = global_edge_index;

    for(i = 0 ; i < vertexCount ; i++)
    {
        for(j = 0 ; j < block_divisor*block_divisor ; j++)
        {
            (*vmap)[i][j] = vertex_map[i][j];
        }
        if(node_type(vertexName[i])==0)
        {
            (*vmap)[i][1] = -1;
        }
        else if((node_type(vertexName[i]) == 1) || (node_type(vertexName[i]) == 2)) 
        {
            (*vmap)[i][block_divisor] = -1;
        }
    }

}

void my_generate_graph_fazlay(dgraph *G, char* file_name, int use_binary_input, int *edge_u, int *edge_v, double *edge_weight, int edgeCount, int vertexCount,const char** vertexName,double* vertexWeight)
{
        printf("\n\n\n\ngenerateDgraph called\n\n\n\n");

        char fnbin[100];
        strcpy(fnbin,file_name);
        strcat(fnbin,".bin");
#if 0
        if(use_binary_input){
            FILE *fp = fopen(fnbin,"rb");
            if(fp!=NULL){
                fclose(fp);
                generateDGraphFromBinaryDot(G,fnbin);
                return;
            }
            else if (strstr(file_name,".bin")){
                generateDGraphFromBinaryDot(G,file_name);
                return;
            }
        }
#endif
        char line[1000];
        char token[200];
        idxType nVrtx = 0, nEdge = 0;
        idxType i, j, k = 0 ;
        // idxType* nbneighbors; //how many neighbours of each vertex
        // idxType** inneighbors; //adjacency list (in-neighbours) , parent tracking


        idxType my_nVrtx = vertexCount , my_nEdge = 0 ;
        idxType* my_nbneighbors;
        idxType** my_inneighbors;
        ecType** my_inedgeweights;

        my_nbneighbors = (idxType*) calloc((vertexCount+1), sizeof(idxType));
        my_inneighbors = (idxType**) malloc((vertexCount+1) * sizeof(idxType*));
        my_inedgeweights = (ecType**) malloc((my_nVrtx+1) * sizeof(ecType*));
        

        for (i = 0; i <= my_nVrtx; i++)
        {
            my_inneighbors[i] = (idxType*) malloc(100 * sizeof(idxType)); //say ever vertex has max 100 parents
            my_inedgeweights[i] = (ecType*) malloc(100 * sizeof(ecType));
        }

        while(k < edgeCount)
        {
      
            int my_v1, my_v2; //(my_v1, my_v2) edge
            double edge_w;
           
            
            my_v2 = edge_u[k];
            my_v1 = edge_v[k];
            edge_w = edge_weight[k];



            my_nEdge++;

            if ((my_nbneighbors[my_v1+1] + 1) % 100 == 0)
            {
                my_inneighbors[my_v1+1] = (idxType*) realloc (my_inneighbors[my_v1+1], (my_nbneighbors[my_v1+1] + 101) * sizeof(idxType));
                my_inedgeweights[my_v1+1] = (ecType*) realloc (my_inedgeweights[my_v1+1], (my_nbneighbors[my_v1+1] + 101) * sizeof(ecType));
            }
    

            my_inneighbors[my_v1+1][my_nbneighbors[my_v1+1]] = my_v2+1;
            my_inedgeweights[my_v1+1][my_nbneighbors[my_v1+1]] = edge_w;
            my_nbneighbors[my_v1+1]++;
            
            k++;
        }
        printf("my vertex count = %d my edge count= %d\n", vertexCount, my_nEdge);

        G->frmt = 2;
        G->nVrtx = my_nVrtx;
        G->nEdge = my_nEdge;
        G->totvw = my_nVrtx;
        G->maxindegree = 0;
        G->hollow  = (int * ) calloc(my_nVrtx + 2, sizeof(int));
        G->inStart  = (idxType * ) calloc(my_nVrtx + 2, sizeof(idxType));
        G->inEnd= (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 2));
        G->in = (idxType * ) malloc(sizeof(idxType) * my_nEdge);

        //anik add
        G->vertices = (char**)malloc(sizeof(char*)*(my_nVrtx+1));
        for(i = 0;i<=my_nVrtx;++i){
            G->vertices[i] = (char*)malloc(100*sizeof(char));
        }
        G->ecIn = (ecType *) malloc(sizeof(ecType) * my_nEdge);
        G->ecOut = (ecType *) malloc(sizeof(ecType) * my_nEdge);

        idxType idx = 0, degree;

        for (i=1; i<=my_nVrtx; i++){
            G->inStart[i] = idx;
            G->inEnd[i-1] = idx-1;
            if (i>1){
                degree = G->inEnd[i-1] - G->inStart[i-1] + 1;
                G->maxindegree = G->maxindegree < degree ? degree : G->maxindegree;
            }
            for (j=0; j< my_nbneighbors[i]; j++){
                G->in[idx] = my_inneighbors[i][j];
                G->ecIn[idx] = my_inedgeweights[i][j];
                idx++;
            }

        }
        G->inStart[0] = 0;
        G->inEnd[0] = -1;
        G->inEnd[my_nVrtx] = idx-1;
        G->inEnd[my_nVrtx+1] = my_nEdge;
        G->inStart[my_nVrtx+1] = my_nEdge;
        degree = G->inEnd[my_nVrtx] - G->inStart[my_nVrtx] + 1;
        G->maxindegree = G->maxindegree < degree ? degree : G->maxindegree;
        if (my_nEdge <= 0)
            u_errexit("allocateDGraphData: empty edge set\n");

        if (my_nVrtx <=-1)
            u_errexit("allocateDGraphData: empty vertex set\n");

        G->outStart  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 2));
        G->outEnd  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 2));
        G->out = (idxType * ) malloc(sizeof(idxType) * my_nEdge);

        G->vw = (vwType *) malloc(sizeof(vwType) * (my_nVrtx+1));
        G->vWeight = (ecType*)malloc(sizeof(ecType)*(my_nVrtx+1));
        //G->totvw = 0;
        for (i=1; i<=my_nVrtx; i++){
            //G->vw[i] = vweights[i-1];
            G->vw[i] = 1.0;
            //G->totvw += G->vw[i];
            //printf("G.vw[%d] = %lf totvw = %lf\n",i,G->vw[i],G->totvw);
        }


        ////// incoming edge weight needed for partitioning
        G->incoming_edge_weight = (ecType*)calloc((my_nVrtx+1),sizeof(ecType));

        //for (i=0; i< my_nEdge; i++){
            //G->ecIn[i] = 1;
            //G->ecOut[i] = 1;
            //G->ecIn[i] = edge_weights[i];
            //G->ecOut[i] = edge_weights[i];
        //}



        for(i=0;i<my_nVrtx;++i){
     
            strcpy(G->vertices[i],vertexName[i]);
            //printf("vertices %d = %s\n",i,G->vertices[i]);
            G->vWeight[i+1] = vertexWeight[i];
            //printf("vWeight[%d] = %lf\n",i+1,G->vWeight[i+1]);
        }


        printf("before calling filloutfrom\n");

        fillOutFromIn(G);


        printf("filloutfrom in done\n");



        G->sources  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 1));
        G->targets  = (idxType * ) malloc(sizeof(idxType) * (my_nVrtx + 1));
        G->nbsources = 0;
        G->nbtargets = 0;
        for (i=1; i<=my_nVrtx; i++){
            if (G->inEnd[i] < G->inStart[i])
                G->sources[G->nbsources++] = i;
            if (G->outEnd[i] < G->outStart[i])
                G->targets[G->nbtargets++] = i;
        }


#if 0        
        if(use_binary_input){
            FILE *fp = fopen(fnbin,"wb");

            if(fp == NULL){
               u_errexit("Cannot open file");
            }

            fwrite(&G->frmt,sizeof(G->frmt),1,fp);
            fwrite(&G->nVrtx,sizeof(G->nVrtx),1,fp);
            fwrite(&G->nEdge,sizeof(G->nEdge),1,fp);
            fwrite(&G->totvw,sizeof(G->totvw),1,fp);
            fwrite(&G->maxindegree,sizeof(G->maxindegree),1,fp);
            fwrite(&G->maxoutdegree,sizeof(G->maxoutdegree),1,fp);
            fwrite(G->inStart,sizeof(G->inStart[0]),(nVrtx+2),fp);
            fwrite(G->inEnd,sizeof(G->inEnd[0]),(nVrtx+2),fp);
            fwrite(G->in,sizeof(G->in[0]),nEdge,fp);
            fwrite(G->vw,sizeof(G->vw[0]),(nVrtx+1),fp);
            fwrite(G->outStart,sizeof(G->outStart[0]),(nVrtx+2),fp);
            fwrite(G->outEnd,sizeof(G->outEnd[0]),(nVrtx+2),fp);
            fwrite(G->out,sizeof(G->out[0]),nEdge,fp);
            fwrite(G->ecIn,sizeof(ecType),nEdge,fp);
            fwrite(G->ecOut,sizeof(ecType),nEdge,fp);

            //anik add
            fwrite(G->vertices,sizeof(G->vertices[0]),(nVrtx+1),fp);
            fwrite(G->vWeight,sizeof(G->vWeight[0]),(nVrtx+1),fp);


            fclose(fp);
        }
#endif

        // for (i = 1; i <= nVrtx; i++)
        //     free(inneighbors[i]);
        // free(nbneighbors);
        // free(inneighbors);



}

void connectedComponents(dgraph* G, idxType** components, idxType* sizes, idxType* nbcomp){
    /*Fill the components[i] array with the list of nodes in this connected component*/
    idxType i,j;
    int* marked = (int*) calloc(G->nVrtx + 1, sizeof(int));
    int nbmarked = 0;
    idxType* ready = (idxType*) malloc(sizeof(idxType)*(G->nVrtx + 1));
    idxType nbready = 1;
    int lastsource = 0;
    idxType source = 1;

    *nbcomp = 0;

    while (nbmarked < G->nVrtx){
        components[*nbcomp] = (idxType*) malloc((G->nVrtx - nbmarked+1)*sizeof(idxType));
        sizes[*nbcomp] = 0;
        while (marked[source] == 1)
            source++;
        marked[source] = 1;
        nbmarked++;
        ready[0] = source;
        nbready = 1;
        while(nbready > 0){
            idxType node = ready[--nbready];
            components[*nbcomp][sizes[*nbcomp]++] = node;
            for (j=G->inStart[node]; j<= G->inEnd[node]; j++) {
                idxType inNode = G->in[j];
                if (marked[inNode] == 0){
                    marked[inNode] = 1;
                    nbmarked++;
                    ready[nbready++] = inNode;
                }
            }
            for (j=G->outStart[node]; j<= G->outEnd[node]; j++) {
                idxType outNode = G->out[j];
                if (marked[outNode] == 0){
                    marked[outNode] = 1;
                    nbmarked++;
                    ready[nbready++] = outNode;
                }
            }
        }
        *nbcomp = *nbcomp + 1;
    }

    free(ready);
    free(marked);
}

void checkGraphAcyclicity(dgraph* G)
{
    /*Try to run a topolotical order. If cannot, then error.*/
    idxType to = 1;
    idxType* ready = (idxType*) malloc(sizeof(idxType) * (G->nVrtx+1));
    idxType i;
    idxType nbready = sourcesList(G, ready);
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    while (nbready > 0) {
        idxType node = ready[nbready-1];
        nbready--;
        to++;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                ready[nbready++] = succ;
            else if (nbinleft[succ]<0)
                u_errexit("Graph is not acyclic\n");
        }
    }
    to = 1;
    nbready = outputList(G, ready);
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->outEnd[i] - G->outStart[i] + 1;
    while (nbready > 0) {
        idxType node = ready[nbready-1];
        nbready--;
        to++;
        for (i = G->inStart[node]; i <= G->inEnd[node]; i++) {
            idxType pred = G->in[i];
            nbinleft[pred]--;
            if (nbinleft[pred] == 0)
                ready[nbready++] = pred;
            else if (nbinleft[pred]<0)
                u_errexit("Graph is not acyclic\n");
        }
    }

    if (to != G->nVrtx+1)
        u_errexit("Graph is not acyclic to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(ready);
}

void addSingleSourceTarget(dgraph* G, idxType* flag) {
    /*Let's add single source and single target*/
    /*And we change flag*/
    idxType i,j;
    idxType flagin = 0;
    idxType flagout = 1;
    for (i = 0; i < G->nVrtx; i++)
        for (j = G->inStart[i]; j <= G->inEnd[i]; j++)
            if (flag[i] != flag[G->in[j]]){
                flagin = flag[G->in[j]];
                flagout = flag[i];
            }

    //We compute sources and issource
    idxType *sources = G->sources;
    idxType nbsources = G->nbsources;
    int *issource = (int *) calloc(G->nVrtx + 3, sizeof(int));
    for (i = 0; i < nbsources; i++)
        issource[sources[i]] = 1;
    //We compute targets and istargets
    idxType *targets = G->targets;
    idxType nbtargets = G->nbtargets;
    int *istarget = (int *) calloc(G->nVrtx + 1, sizeof(int));
    for (i = 0; i < nbtargets; i++)
        istarget[targets[i]] = 1;
    //Index for the source and target
    idxType source = G->nVrtx + 1;
    idxType target = G->nVrtx + 2;
    //Update nVrtx, nEdge, totvw and maxindegree
    G->nVrtx += 2;
    G->nEdge += nbsources + nbtargets;
    G->totvw += 2;
    G->maxindegree = G->maxindegree < nbsources ? nbsources : G->maxindegree;
    //Realloc pointers
    idxType* newinStart = (idxType *) malloc((G->nVrtx + 2)*sizeof(idxType));
    int* newhollow = (int *) calloc(G->nVrtx + 2, sizeof(int));
    idxType* newinEnd = (idxType *) malloc((G->nVrtx + 2)*sizeof(idxType));
    idxType* newin = (idxType *) malloc(G->nEdge * sizeof(idxType));
    ecType* newecIn = (ecType *) malloc(G->nEdge * sizeof(ecType));
    vwType* newvw = (vwType *) malloc((G->nVrtx+1)*sizeof(vwType));
    //Fill new vw
    for (i = 1; i <= G->nVrtx - 2; i++)
        newvw[i] = G->vw[i];
    newvw[source] = 1;
    newvw[target] = 1;
    //Fil new inStart
    newinStart[0] = 0;
    idxType sourceidx = 0;
    for (i = 1; i <= G->nVrtx - 2; i++){
        newinStart[i] = G->inStart[i] + sourceidx;
        if (issource[i] == 1)
            sourceidx++;
    }
    newinStart[source] = G->inStart[source] + sourceidx;
    newinStart[target] = G->inStart[source] + sourceidx;
    newinStart[target+1] = G->nEdge;
    //Fill new inEnd
    newinEnd[0] = -1;
    for (i = 1; i <= G->nVrtx; i++)
        newinEnd[i] = newinStart[i+1] - 1;
    //newinEnd[G->nVrtx] = G->nEdge;
    newinEnd[G->nVrtx + 1] = G->nEdge;
    //Fill new in and ecIn
    idxType oldidx = 0;
    idxType newidx = 0;
    for (i = 1; i <= G->nVrtx-2; i++){
        if (issource[i] == 1) {
            if (flag[i] == flagin)
                newecIn[newidx] = G->nEdge / 2;
            else
                newecIn[newidx] = 1;
            newin[newidx++] = source;
        }
        for (j = G->inStart[i]; j <= G->inEnd[i]; j++) {
            newecIn[newidx] = G->ecIn[j];
            newin[newidx++] = G->in[j];
        }
    }
    for (j = newinStart[target]; j <= newinEnd[target]; j++) {
        if (flag[targets[j - newinStart[target]]] == flagout)
            newecIn[j] = G->nEdge / 2;
        else
            newecIn[j] = 1;
        newin[j] = targets[j - newinStart[target]];
    }
    //Realloc outStart, outEnd and out to fill them

    idxType* newoutStart  = (idxType * ) malloc((G->nVrtx + 2)*sizeof(idxType));
    idxType* newoutEnd  = (idxType * ) malloc((G->nVrtx + 2)*sizeof(idxType));
    idxType* newout = (idxType * ) malloc(G->nEdge * sizeof(idxType));
    ecType* newecOut = (ecType * ) malloc(G->nEdge * sizeof(ecType));

    //free everything
    free(G->hollow);
    G->hollow = newhollow;
    free(G->in);
    G->in = newin;
    free(G->ecIn);
    G->ecIn = newecIn;
    free(G->inEnd);
    G->inEnd = newinEnd;
    free(G->inStart);
    G->inStart = newinStart;
    free(G->out);
    G->out = newout;
    free(G->ecOut);
    G->ecOut = newecOut;
    free(G->outEnd);
    G->outEnd = newoutEnd;
    free(G->outStart);
    G->outStart = newoutStart;
    free(G->vw);
    G->vw = newvw;

    checkDgraph(G);
    fillOutFromIn(G);

    G->sources[0] = source;
    G->targets[0] = target;
    G->nbsources = 1;
    G->nbtargets = 1;

    free(issource);
    free(istarget);

    flag = (idxType *) realloc(flag, (G->nVrtx + 1)*sizeof(idxType));
    flag[source] = flagin;
    flag[target] = flagout;
}


void copyDgraph(dgraph* Gcopy, dgraph* G) {
    idxType i;
    allocateDGraphData(G, Gcopy->nVrtx, Gcopy->nEdge, Gcopy->frmt);
    G->nVrtx = Gcopy->nVrtx;
    G->nEdge = Gcopy->nEdge;
    G->frmt = Gcopy->frmt;
    G->nbsources = Gcopy->nbsources;
    G->nbtargets = Gcopy->nbtargets;
    G->totvw = Gcopy->totvw;
    G->totec = Gcopy->totec;
    G->maxindegree = Gcopy->maxindegree;
    G->maxoutdegree = Gcopy->maxoutdegree;
    G->maxVW = Gcopy->maxVW;
    G->maxEC = Gcopy->maxEC;

    for(i = 0; i <= G->nVrtx + 1; i++)
        G->inStart[i] = Gcopy->inStart[i];

    for(i = 0; i <= G->nVrtx + 1; i++)
        G->inEnd[i] = Gcopy->inEnd[i];

    for(i = 0; i <= G->nEdge -1; i++)
        G->in[i] = Gcopy->in[i];

    for(i = 0; i <= G->nVrtx + 1; i++)
        G->hollow[i] = Gcopy->hollow[i];

    for(i = 0; i <= G->nVrtx + 1; i++)
        G->outStart[i] = Gcopy->outStart[i];

    for(i = 0; i <= G->nVrtx + 1; i++)
        G->outEnd[i] = Gcopy->outEnd[i];

    for(i = 0; i <= G->nEdge -1; i++)
        G->out[i] = Gcopy->out[i];


    if (G->frmt == 1 || G->frmt == 3)
        for(i = 0; i <= G->nVrtx; i++)
            G->vw[i] = Gcopy->vw[i];


    if (G->frmt == 2 || G->frmt == 3)
        for(i = 0; i <= G->nEdge - 1; i++) {
            G->ecIn[i] = Gcopy->ecIn[i];
            G->ecOut[i] = Gcopy->ecOut[i];
        }

    for(i = 0; i <= G->nVrtx; i++)
        G->sources[i] = Gcopy->sources[i];

    for(i = 0; i <= G->nVrtx; i++)
        G->targets[i] = Gcopy->targets[i];
}

void checkDgraph(dgraph* G){
    idxType i;
    for (i = 1; i <= G->nVrtx-1; i++) {
        if (G->inStart[i] > G->inStart[i + 1])
            u_errexit("inStart not correct");
        if (G->inEnd[i] > G->inEnd[i + 1])
            u_errexit("inEnd not correct");
    }
    for (i = 1; i <= G->nVrtx; i++) {
        if (G->inStart[i] > G->inEnd[i] + 1)
            u_errexit("inEnd and inStart not coherent (inStart[%d] = %d, inEnd[%d] = %d", i, G->inStart[i], i, G->inEnd[i]);
        if (G->inStart[i] > G->nEdge + 1)
            u_errexit("inStart not coherent");
        if (G->inEnd[i] > G->nEdge + 1)
            u_errexit("inEnd not coherent");
    }
    if (G->inStart[1] != 0)
        u_errexit("First index of inStart not right");

    if (G->inStart[G->nVrtx+1] != G->nEdge)
        u_errexit("Last index of inStart not right");
}

void analyzeDGraph(dgraph *G)
{
    idxType i, j, l, w, nlvls, nVrtx = G->nVrtx;
    idxType *toplevels, *histogram;
    ecType *ecLvls; /*we will store from the cost from the level i to level i+1 in ecLvls[i]; for i=0, u < nlvls*/

    idxType wndwLngth = 25 ;

    /*Graph's pointers*/
    idxType *outStart = G->outStart;
    idxType *outEnd = G->outEnd;
    idxType *out = G->out;
    ecType *ecOut = G->ecOut;

    toplevels = (idxType*) malloc(sizeof(idxType) * (nVrtx+1));

    computeToplevels(G, toplevels);

    nlvls = 0;
    for (i=1; i<= nVrtx; i++)
        nlvls = nlvls < toplevels[i] ? toplevels[i] : nlvls;

    nlvls = nlvls+1;

    histogram = (idxType*) calloc(nlvls, sizeof(idxType));
    ecLvls =     (ecType*) calloc(nlvls, sizeof(ecType));/*nlvls is enough for ecLvls but let us keep it parallel*/
    if (nlvls >= 1000)
        wndwLngth = (idxType) ceil(nlvls/25.0);
    else if (nlvls>=500)
        wndwLngth = 12;
    else if (nlvls>=250)
        wndwLngth = 10;
    else if (nlvls>=125)
        wndwLngth = 5;
    else if (nlvls>=65)
        wndwLngth = 2;
    else
        wndwLngth = 1;

    for (i=1; i<= nVrtx; i++) {
      histogram[toplevels[i]]+=G->vw[i];
        for (j=outStart[i]; j<= outEnd[i]; j++) {
            idxType ng = out[j];
            if (toplevels[ng]<= toplevels[i])
                u_errexit("analyzeDGraph: did not like the graph\n");
            if (((int)toplevels[ng]/wndwLngth)!=((int)toplevels[i]/wndwLngth))
		ecLvls[toplevels[i]] += ecOut[j];
        }
    }

    printf("\t\t%ld vertices in %ld levels (S=%ld, T=%ld) \n", (long) nVrtx, (long) nlvls,
           (long) histogram[0], (long)  histogram[nlvls-1]);
    for (w = 0; w < nlvls; w += wndwLngth) {
        idxType nVrtxWnd = 0;
        ecType ecWnd = 0;
        for (l = w; l < nlvls && l < w + wndwLngth; l++) {
            nVrtxWnd += histogram[l];
            ecWnd += ecLvls[l];
        }
        printf("L%ld-%ld [%ld]: ec-> [%.0f]\n", (long) w, (long) l-1, (long) nVrtxWnd, (1.0)*ecWnd);

    }
    free(ecLvls);
    free(histogram);
    free(toplevels);
}

void printNodeInfo(dgraph *G,idxType* part,idxType node)
{
    int j=0;
        printf("Node: %d (%d): [", node, part[node]);
        for (j=G->inStart[node]; j<=G->inEnd[node]; j++){
            idxType father = G->in[j];
            printf("%d (%d), ", father,part[father] );
        }
        printf("]<" );
        for (j=G->outStart[node]; j<=G->outEnd[node]; j++){
            idxType child = G->out[j];
            printf("%d (%d),", child,part[child] );
        }
        printf(">\n");
}

int printPartWeights(dgraph* G, idxType* part)
{
    int nbpart = 0;

    idxType maxpart = part[1];
    idxType i;
    for (i = 1; i<= G->nVrtx; i++) {
        maxpart = part[i] > maxpart  ? part[i] : maxpart;
    }
    nbpart = maxpart+1;

    idxType* partsize = (idxType*) calloc(nbpart, sizeof(idxType));
    idxType minsize = INT_MAX, maxsize = 0;

    for (i = 1; i <= G->nVrtx; i++) {
        partsize[part[i]] += G->vw[i];
    }

    for (i = 0; i < nbpart; i++) {
        minsize = minsize < partsize[i] ? minsize : partsize[i];
        maxsize = maxsize < partsize[i] ? partsize[i] : maxsize;
    }

    free(partsize);
    return (int) maxsize;
}



ecType edgeCut(dgraph* G, idxType* part)
{
    int frmt = G->frmt;
    /*Return the edge cut of the partition*/
    ecType edgecut = 0;
    idxType i,j;
    for (i = 1; i<=G->nVrtx; i++)
        for (j=G->outStart[i]; j<=G->outEnd[i]; j++) {
            if (part[G->out[j]] != part[i]) {
                if (frmt & DG_FRMT_EC)
                    edgecut += G->ecOut[j];
                else
                    edgecut++;
            }
        }
    return edgecut;
}

void reverseGraph(dgraph *G)
{

    idxType *inStart = G->inStart;
    idxType *inEnd = G->inEnd;
    idxType *in = G->in;
    ecType *ecIn = G->ecIn;

    idxType *outStart = G->outStart;
    idxType *outEnd = G->outEnd;
    idxType *out = G->out;
    ecType *ecOut = G->ecOut;

    G->inStart = outStart;
    G->inEnd = outEnd;
    G->in = out;
    G->ecIn = ecOut;

    G->outStart = inStart;
    G->outEnd = inEnd;
    G->out = in;
    G->ecOut = ecIn;
}

void randomizeWeights(dgraph *G, vwType vmin, vwType vmax, ecType emin, ecType emax){
    idxType i,j;
    G->totvw = 0;
    G->maxVW = 0;
    G->totec = 0;
    G->maxEC = 0;
    for (i = 1; i <= G->nVrtx; i++) {
        G->vw[i] = vmin + uRandom((int) (vmax - vmin + 1));
        G->totvw += G->vw[i];
        G->maxVW = G->maxVW < G->vw[i] ? G->vw[i] : G->maxVW;
        //printf("Weight %d : %d\n", (int) i, (int) G->vw[i]);
        for (j = G->inStart[i]; j <= G->inEnd[i]; j++) {
            G->ecIn[j] = emin + uRandom((int) (emax - emin + 1));
            G->totec += G->ecIn[j];
            G->maxEC = G->maxEC < G->ecIn[j] ? G->ecIn[j] : G->maxEC;
            //printf("Weight %d -> %d : %d\n", (int) G->in[j], (int) i, (int) G->ecIn[j]);
        }
    }
    fillOutFromIn(G);
}

void applyCCR(dgraph *G, double CCR){
    double mult = CCR * G->totvw / G->totec;
    idxType i,j;
    G->totec = 0;
    for (i = 1; i <= G->nVrtx; i++) {
        for (j = G->inStart[i]; j <= G->inEnd[i]; j++) {
            G->ecIn[j] = (ecType) round(mult * G->ecIn[j]);
            G->totec += G->ecIn[j];
        }
    }
    fillOutFromIn(G);
}
