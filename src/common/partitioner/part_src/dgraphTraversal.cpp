#include "dgraphTraversal.h"

void randTopSortOnParts(dgraph *G, idxType *part, idxType *toporder, idxType nbpart)
{
    /*Fill toporder assuming it's already allocated*/
    idxType i, j, k, ip;
    idxType** outpart = (idxType**) umalloc(sizeof(idxType*)*nbpart, "outpart");
    idxType* nbout = (idxType*) umalloc(sizeof(idxType)*nbpart, "nbout");
    idxType** inpart = (idxType**) umalloc(sizeof(idxType*)*nbpart, "inpart");
    idxType* nbin = (idxType*) umalloc(sizeof(idxType)*nbpart, "nbin");
    for (i=0; i<nbpart; i++) {
        outpart[i] = (idxType*) umalloc(sizeof(idxType)*nbpart, "outpart[i]");
        nbout[i] = 0;
        inpart[i] = (idxType*) umalloc(sizeof(idxType)*nbpart, "inpart[i]");
        nbin[i] = 0;
    }
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

    idxType* ready = (idxType*) umalloc(sizeof(idxType)*(nbpart), "ready");
    idxType* nbinleft = (idxType*) umalloc(sizeof(idxType)*(nbpart), "nbinleft");
    int nbready = 0;
    for (i=0; i<nbpart; i++) {
        nbinleft[i] = nbin[i];
        if (nbin[i] == 0)
            ready[nbready++] = i;
    }

    int to = 0;
    idxType* shuffle = (idxType *) umalloc(nbpart * sizeof(idxType), "shuffle");
    while (nbready > 0) {
        idxType pno = ready[nbready-1];
        nbready--;
        toporder[to++] = pno;
        if (nbout[pno]) {
            shuffleTab(0, nbout[pno]-1, shuffle);
            for (ip = 0; ip < nbout[pno]; ip++) {
                i = shuffle[ip];
                idxType succ = outpart[pno][i];
                nbinleft[succ]--;
                if (nbinleft[succ] == 0) {
                    ready[nbready++] = succ;
                }
            }
        }
    }
    for (i=0; i<nbpart; i++) {
        free(outpart[i]);
        free(inpart[i]);
    }
    free(nbinleft);
    free(shuffle);
    free(ready);
    free(nbin);
    free(nbout);
    free(inpart);
    free(outpart);
    if (to != nbpart) {
        u_errexit("In topsortPart, not every nodes are sorted: to = %d, nbpart = %d\n", to, nbpart);
    }
}

void topsortPriorities(dgraph *G, idxType *toporder, ecType *priority)
{
    /*Assume that toporder is already allocated an priorities filled
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1, i, node;
    // printf("in topsortPriorities G->nVrtx %d\n", G->nVrtx);
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL)
        u_errexit("nbinleft alloc error\n");
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    //Build heap of ready tasks
    idxType* heap = (idxType*) umalloc(sizeof(idxType)*(G->nVrtx+1), "heap");
    idxType* inheap = (idxType*) calloc(G->nVrtx+1, sizeof(idxType));
    if (inheap == NULL)
        u_errexit("inheap alloc error\n");
    idxType hsize = 0;
    for (i =0; i < G->nbsources; i++){
        hsize++;
        heap[hsize] = G->sources[i];
        inheap[G->sources[i]] = hsize;
    }
    heapBuild(heap, priority, hsize, inheap);

    while (hsize > 0) {
        if (to > G->nVrtx)
            u_errexit("Proof topsortPrioritiesto = %d\n", to);
        node = heapExtractMax(heap, priority, &hsize, inheap);
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                heapInsert(heap, priority, &hsize, succ, inheap);
            else if (nbinleft[succ]<0)
                u_errexit("topsortPriorities: negative indegree\n");
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("topsortPriorities : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(heap);
    free(inheap);
}

void DFStopsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1;
    idxType* ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i;
    idxType nbready = G->nbsources;
    for (i = 0; i < nbready; i++)
        ready[i] = G->sources[i];
    sourcesList(G, ready);
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL)
        u_errexit("nbinleft alloc error\n");

    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    while (nbready > 0) {
        if (to > G->nVrtx)
            u_errexit("Proof DFStopsort to = %d\n", to);
        idxType node = ready[nbready-1];
        nbready--;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                ready[nbready++] = succ;
            else if (nbinleft[succ]<0)
                u_errexit("DFStopsort: negative indegree\n");
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("DFStopsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(ready);
}

void BFStopsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i;
    idxType endready = sourcesList(G, ready) - 1;
    idxType beginready = 0;
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL)
        u_errexit("nbinleft alloc error\n");

    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    while (endready >= beginready) {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        idxType node = ready[beginready];
        beginready++;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0) {
                ready[++endready] = succ;
            }
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("BFStopsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(ready);
}

void DFSsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1;
    idxType* ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i;
    idxType nbready = sourcesList(G, ready);
    idxType* done = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (done == NULL)
        u_errexit("done alloc error\n");
    while (nbready > 0)
    {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        idxType node = ready[nbready-1];
        nbready--;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++)
        {
            idxType succ = G->out[i];
            if (done[succ] == 1)
                continue;
            ready[nbready++] = succ;
            done[succ] = 1;
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("DFSsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(done);
    free(ready);
}

void my_BFS(dgraph* G,idxType* visited){
    int head,tail;
    idxType* active_nodes = (idxType*)malloc(G->nVrtx*sizeof(idxType));
    int i;
    int active_node_count,visited_head = 1;
    active_node_count = sourcesList(G,active_nodes);
    int out_vertex;
    idxType* already_visited = (idxType*)calloc(G->nVrtx+1,sizeof(idxType));
    head = 0;
    tail = active_node_count - 1;
    //printf("inside my BFS\n");
    while(head<=tail){
        int curr = active_nodes[head++];
        if(tail>G->nVrtx)
            printf("something wrong, tail > vertex count\n");
        visited[visited_head++] = curr;
//      printf("visited[%d] = %d head = %d\n",visited_head-1,visited[visited_head-1],head-1);
        for(i = G->outStart[curr] ; i <= G->outEnd[curr] ; ++i){
            out_vertex = G->out[i];

            if(already_visited[out_vertex]==1)
                continue;
            active_nodes[++tail] = out_vertex;
            already_visited[out_vertex] = 1;
    //      printf("out vertex = %d tail = %d\n",out_vertex,tail);

        }

    }
    if(visited_head != G->nVrtx+1)
    {
        printf("graph is not connected\n");
    }
    free(active_nodes);
    free(already_visited);
    //printf("\nfreed\n");
}


///// added by anik

void my_DFS(dgraph* G, idxType *visited){
    idxType* active_nodes = (idxType*)malloc(sizeof(idxType) * G->nVrtx);
    int i;
    int active_nodes_count;
    int out_vertex;
    int* already_visited = (int*)calloc(G->nVrtx+1,sizeof(int));


    active_nodes_count = sourcesList(G,active_nodes);

    int visited_head = 1;

    while(active_nodes_count > 0){
        if(visited_head>G->nVrtx){
            printf("head > vertex count, something wrong\n");
            break;
        }

        int curr = active_nodes[active_nodes_count-1];
        active_nodes_count--;
        visited[visited_head++] = curr;
//      printf("visited[%d] = %d\n",visited_head-1,visited[visited_head-1]);
        for(i = G->outStart[curr] ; i<=G->outEnd[curr] ; ++i){
            out_vertex = G->out[i];
            if(already_visited[out_vertex] == 1)
                continue;
            active_nodes[active_nodes_count++] = out_vertex;
            already_visited[out_vertex] = 1;

        }

    }
    if(visited_head != (G->nVrtx+1))
        printf("graph is disconnected\n");

    free(active_nodes);
    free(already_visited);



}
void BFSsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
     with DFS*/
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i;
    idxType endready = sourcesList(G, ready) - 1;
    idxType beginready = 0;
    idxType* done = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (done == NULL)
        u_errexit("done alloc error\n");

    while (endready >= beginready)
    {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        idxType node = ready[beginready];
        beginready++;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            if (done[succ] == 1)
                continue;
            ready[++endready] = succ;
            done[succ] = 1;
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("BFSsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(done);
    free(ready);
}

void randDFStopsort(dgraph *G, idxType *toporder)
{
    /*Assume that toporder is already allocated
      Fill toporder to have a random topological order of nodes in G*/
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i,itmp;
    idxType nbready = sourcesList(G, ready);
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL)
        u_errexit("nbinleft alloc error\n");
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    idxType* shuffle = (idxType*) umalloc(sizeof(idxType)*(G->maxoutdegree), "shuffle");
    while (nbready > 0)
    {
        idxType node = ready[nbready-1];
        nbready--;
        toporder[to++] = node;
        shuffleTab(G->outStart[node], G->outEnd[node], shuffle);
        for (itmp = 0; itmp <= G->outEnd[node]-G->outStart[node]; itmp++) {
            i = shuffle[itmp];
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0)
                ready[nbready++] = succ;
        }
    }
    free(shuffle);
    free(nbinleft);
    free(ready);
}

void randBFStopsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i,itmp;
    idxType endready = sourcesList(G, ready) - 1;
    idxType beginready = 0;
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL)
        u_errexit("nbinleft alloc error\n");
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    idxType* shuffle = (idxType*) umalloc(sizeof(idxType)*(G->maxoutdegree), "shuffle");
    while (endready >= beginready) {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        idxType node = ready[beginready];
        beginready++;
        toporder[to++] = node;
        shuffleTab(G->outStart[node], G->outEnd[node], shuffle);
        for (itmp = 0; itmp <= G->outEnd[node]-G->outStart[node]; itmp++) {
            i = shuffle[itmp];
            idxType succ = G->out[i];
            assert(succ <= G->nVrtx);
            nbinleft[succ]--;
            if (nbinleft[succ] == 0) {
                ready[++endready] = succ;
            }
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("randBFStopsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(shuffle);
    free(ready);
}

void randDFSsort(dgraph *G, idxType *toporder)
{
    /*Assume that toporder is already allocated
      Fill toporder to have a random topological order of nodes in G*/
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i,itmp;
    idxType nbready = sourcesList(G, ready);
    idxType* done = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (done == NULL)
        u_errexit("done alloc error\n");
    idxType* shuffle = (idxType*) umalloc(sizeof(idxType)*(G->maxoutdegree+1), "shuffle");
    while (nbready > 0) {
        idxType node = ready[nbready-1];
        nbready--;
        toporder[to++] = node;
        shuffleTab(G->outStart[node], G->outEnd[node], shuffle);
        for (itmp = 0; itmp <= G->outEnd[node]-G->outStart[node]; itmp++) {
            i = shuffle[itmp];
            idxType succ = G->out[i];
            if (done[succ] == 1)
                continue;
            ready[nbready++] = succ;
            done[succ] = 1;
        }
    }
    free(shuffle);
    free(done);
    free(ready);
}

void randBFSsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a topological order of nodes in G
    with DFS*/
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i,itmp;
    idxType endready = sourcesList(G, ready) - 1;
    idxType beginready = 0;
    idxType* done = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (done == NULL)
        u_errexit("done alloc error\n");
    idxType* shuffle = (idxType*) umalloc(sizeof(idxType)*(G->maxoutdegree), "shuffle");
    while (endready >= beginready)
    {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        idxType node = ready[beginready];
        beginready++;
        toporder[to++] = node;
        shuffleTab(G->outStart[node], G->outEnd[node], shuffle);
        for (itmp = 0; itmp <= G->outEnd[node]-G->outStart[node]; itmp++) {
            i = shuffle[itmp];
            idxType succ = G->out[i];
            if (done[succ] == 1)
                continue;
            ready[++endready] = succ;
            done[succ] = 1;
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("randBFSsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(done);
    free(shuffle);
    free(ready);
}


void randTopsort(dgraph* G, idxType *toporder)
{
    /*Assume that toporder is already allocated
     Fill toporder to have a random topological order of nodes in G
    */
    idxType to = 1;
    idxType* ready;
    ready = (idxType*) umalloc(sizeof(idxType) * (G->nVrtx+1), "ready");
    idxType i;
    idxType nbready = sourcesList(G, ready);
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL)
        u_errexit("nbinleft alloc error\n");
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;
    while (nbready > 0) {
        if (to > G->nVrtx)
            u_errexit("Proof to = %d\n", to);
        idxType node_idx = uRandom(nbready);
        idxType node = ready[node_idx];
        ready[node_idx] = ready[nbready-1];
        nbready--;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0) {
                ready[nbready++] = succ;
            }
        }
    }
    if (to != G->nVrtx + 1)
        u_errexit("randTopsort : Not every node concerned! to = %d, nVrtx = %d\n", to, G->nVrtx);
    free(nbinleft);
    free(ready);
}


int gcd ( int a, int b )
{
    int c;
    while ( a != 0 ) {
        c = a; a = b%a;  b = c;
    }
    return b;
}

void mixtopsort(dgraph* G, idxType* toporder, int priority, int first)
{
    //first==0 --> bfs first
    //first==1 --> dfs first
    idxType* ready = (idxType*) umalloc(sizeof(idxType)*(G->nVrtx+1), "ready");
    idxType* mark = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    idxType* nbinleft = (idxType *) calloc(G->nVrtx+1, sizeof(idxType));
    if (nbinleft == NULL || mark == NULL)
        u_errexit("nbinleft or mark alloc error\n");
    idxType readyend = sourcesList(G,ready) - 1;
    idxType readybegin = 0;
    idxType bfrac=0, dfrac=0;
    idxType biter=0, diter=0;
    idxType node=-1;
    idxType i,j,to=1;
    for (i = 1; i <= G->nVrtx; i++)
        nbinleft[i] = G->inEnd[i] - G->inStart[i] + 1;

    i=gcd(priority,100-priority);
    if (i==0) {
        bfrac=priority/100;
        dfrac=(100-priority)/100;
    }
    else{
        bfrac=priority/i;
        dfrac=(100-priority)/i;
    }
    if (first)
        diter=dfrac;
    else
        biter=bfrac;

    while (readyend>=readybegin) {
        if (to > G->nVrtx+1) {
            printf("Proof to = %d\n", to);
            break;
        }

        //pick from bfs priority
        if (biter>=0) {
            node = ready[readybegin];
            readybegin++;
            biter--;
            if (biter<0)
                diter=dfrac;
        }
        else { //pick from dfs priority
            node = ready[readyend];
            diter--;
            readyend--;
            if (diter<0)
                biter=bfrac;
        }
        mark[node]=1;
        toporder[to++] = node;
        for (i = G->outStart[node]; i <= G->outEnd[node]; i++) {
            idxType succ = G->out[i];
            nbinleft[succ]--;
            if (nbinleft[succ] == 0) {
                readyend++;
                ready[readyend] = succ;
            }
        }
    }
    if (to != G->nVrtx+1)
        u_errexit("Mixtopsort : Not every node concerned!  to = %d, nVrtx = %d\n", to, G->nVrtx);

    free(nbinleft);
    free(mark);
    free(ready);
}
