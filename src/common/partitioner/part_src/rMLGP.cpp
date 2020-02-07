#include "../part_inc/utils.h"
#include "../part_inc/dgraph.h"
#include "../part_inc/dgraphReader.h"
#include "../part_inc/rvcycle.h"
#include "../part_inc/rMLGP.h"

dgraph main_graph;


int processArgs_rMLGP(int argc, char **argv, MLGP_option* opt)
{
    if (argc < 3) {
        printMLGPusage(argv[0]);
        u_errexit("%s: There is a problem with the arguments.\n", argv[0]);
    }

//    initMLGPoptions(opt, atoi(argv[2]));
    initMLGPoptions(opt, atoi(argv[6]));
    
    processMLGPargs(argc, argv, opt);

    // if ((opt->nbPart<1) || !isPowOfTwo(opt->nbPart)) {
    if ((opt->nbPart<1)) {
        printMLGPusage(argv[0]);
        u_errexit("%s: This number of partitions is not supported for now. Try a power of 2.\n", argv[0]);
    }

    return 0;
}

void make_new_table_file(rcoarsen* rcoarse, int *my_partition_topsort, char* loopname, char* matrix_name, int large_blk, int small_blk, int partCount){

    printf("tasktable started\n");
    int i,j;

    


    char* task = (char*) malloc(150*sizeof(char));
    char* strparam = (char*) malloc(150*sizeof(char));

    int opcode;
    int numparamsCount;
    int strparamsCount;
    int task_id;
    int partition_no;
    int priority;

    //    FILE* vertexName_file ;
    //    vertexName_file = fopen("vertexName.txt","w");


    //FILE* vertexName_file ;
    char vertex_filename[150];

    sprintf(vertex_filename,"%s_%dk_%dk_part%d_%s.txt",matrix_name,large_blk/1024,small_blk/1024,partCount,loopname);

    //vertexName_file = fopen(vertex_filename,"w");


    char taskinfo_file_name[200];

    sprintf(taskinfo_file_name,"%s_%dk_%dk_part%d_%s_taskinfo.txt",matrix_name,large_blk/1024,small_blk/1024,partCount,loopname);

    FILE* task_table = fopen(taskinfo_file_name,"w");


    fprintf(task_table,"%d\n",rcoarse->coars->graph->nVrtx);

    for(i = 1, j = rcoarse->coars->graph->nVrtx ; i <= rcoarse->coars->graph->nVrtx ; i++, j--)
    {
        

        //non-priority
        //fprintf(vertexName_file,"%s 0\n",rcoarse->coars->graph->vertices[my_partition_topsort[i]-1]);
        //printf("%d = %s\n",i,rcoarse->coars->graph->vertices[my_partition_topsort[i]-1]);
        //get_task_name(rcoarse->coars->graph->vertices[my_partition_topsort[i]-1],task);
        
        task = strtok(rcoarse->coars->graph->vertices[my_partition_topsort[i]-1],",");
    
        if(task == NULL)
        {
            fprintf(stderr, "One of the tasks is NULLL, cannot proceed...\n");
            exit(-1);
        }
        
        else if(!strcmp(task,"RESET")){
            opcode = 1;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"SPMM")){
            opcode = 2;
            numparamsCount = 3 ;
            strparam = strtok(NULL,"/");//ttttsav the whole row,col,buf of SPMM in this string and just print that 
            strparamsCount = 0;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%s,%d,%d,%d,%d\n",opcode,numparamsCount,strparam,strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"XTY")){
            opcode = 3;
            numparamsCount = 2 ;
            int rowblk = atoi(strtok(NULL,","));
            int buffid = atoi(strtok(NULL,","));

            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk,buffid, strparamsCount,task_id,partition_no,priority);
        }

        else if(!strcmp(task,"RED")){
            opcode = 4;
            numparamsCount = 1 ;
            strparam = strtok(NULL,",");
            int rowblk = atoi(strtok(NULL,","));

            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,strparam, task_id,partition_no,priority);
        }
        else if(!strcmp(task,"XY")){
            opcode = 5;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));
            
            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"ADD")){
            opcode = 6;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));
            
            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"DLACPY")){
            opcode = 7;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));
            
            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"UPDATE")){
            opcode = 8;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));

            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"SUB")){
            opcode = 9;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));

            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"MULT")){
            opcode = 10;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));
            
            strparamsCount = 0;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"COL")){
            opcode = 11;
            numparamsCount = 2 ;
            int rowblk = atoi(strtok(NULL,","));
            int buffid = atoi(strtok(NULL,","));

            strparamsCount = 0;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;

            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk,buffid, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"RNRED")){
            opcode = 12;
            numparamsCount = 0; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"SQRT")){
            opcode = 13;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"GET")){
            opcode = 14;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));

            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;

            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }

        else if(!strcmp(task,"TRANS")){
            opcode = 15;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"SPEUPDATE")){
            opcode = 16;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"CHOL")){
            opcode = 17;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"INV")){
            opcode = 18;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"SETZERO")){
            opcode = 19;
            numparamsCount = 1 ;
            int rowblk = atoi(strtok(NULL,","));
            
            strparamsCount = 0;
            task_id = atoi(strtok(NULL,","));
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            
            fprintf(task_table,"%d,%d,%d,%d,%d,%d,%d\n",opcode,numparamsCount,rowblk, strparamsCount,task_id,partition_no,priority);
        }
        else if(!strcmp(task,"CONV")){
            opcode = 20;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = strtok(NULL,",");

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(task[0] == '_'){
            opcode = 22;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = task;

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
        else if(strtok(NULL,",")==NULL){
            opcode = 21;
            numparamsCount = 0 ; 
            strparamsCount = 1;
            task_id = -1;
            partition_no = rcoarse->coars->part[my_partition_topsort[i]];
            priority = j;
            strparam = task;

            fprintf(task_table,"%d,%d,%d,%s,%d,%d,%d\n",opcode,numparamsCount,strparamsCount,strparam,task_id,partition_no,priority);
        }
    }

    fclose(task_table);
    //fclose(vertexName_file);
    printf("tasktable done\n");

}


void run_rMLGP(char* file_name, MLGP_option opt, int *edge_u, int *edge_v, double *edge_weight, int edgeCount, int vertexCount,const char** vertexName,double* vertexWeight, int loopType, char* matrix_name)
{
    dgraph G;
    int partcount = -1;
    int i, j;

 //   ecType* memory_from_out;
 //   ecType* memory_from_in;

    my_generate_graph_fazlay(&G, file_name,opt.use_binary_input, edge_u, edge_v, edge_weight, edgeCount, vertexCount,vertexName,vertexWeight);

    


    ////////////////partinfo file ////////////////

    
    FILE* partInfo ;
    int prev_part = -1;
    char new_fname[100];
    char loopname[100];
    char partinfo_file_name[200];



    if(loopType == 0 ) strcpy(loopname,"nonloop");
    else if(loopType == 1 )strcpy(loopname,"firstloop");
    else if(loopType == 2 )strcpy(loopname,"secondloop");


    ////////////////////////////////////////////////


    for(int i = 0 ; i < G.nEdge ; i++){
    //  printf("out[%d] = %d\n",i,G.out[i]);
    }

    printf("run_rMLGP function edge = %d node = %d \n",edgeCount,vertexCount);

    set_dgraph_info(&G);



    main_graph = G;

    printf("G.nvrtx = %d\n",G.nVrtx);
    G.totvw = G.nVrtx;  
    printf("set_dgraph_info done\n");

    int maxindegree, minindegree, maxoutdegree, minoutdegree;
    double aveindegree, aveoutdegree;
    dgraph_info(&G, &maxindegree, &minindegree, &aveindegree, &maxoutdegree, &minoutdegree, &aveoutdegree);
    printf("dgraph_info done\n");
    G.maxindegree = maxindegree;
    G.maxoutdegree = maxoutdegree;
    printf("maxIndegree = %d\t minindegree =  %d\n maxoutdegree =  %d\t  minoutdegree = %d\n avgindegree = %lf\t avgutdegree =  %lf\n", maxindegree,minindegree, maxoutdegree,minoutdegree,aveindegree,aveoutdegree);

    char* input1 = (char*)malloc(200*sizeof(char));
    char* input2 = (char*)malloc(200*sizeof(char));
    char* output = (char*)malloc(200*sizeof(char));

    // get_input_output(G.vertices[100],input1,input2,output);

    // printf("\n\n\n node %s inp_1 %s inp_2 %s output %s \n\n",G.vertices[100],input1,input2,output);
    //FILE* large_graph = fopen("large_graph.txt","w");
    for (i = 1;i<=G.nVrtx ; i++){
        for(j = G.inStart[i] ; j <= G.inEnd[i]; j++){
            //printf("%d ---> %d\n",G->in[j],i);
            fill_allinout_memory_map(G.vertices[i-1],G.vertices[G.in[j]-1],G.ecIn[j]);
            //fprintf(large_graph,"%s --> %s;\n",vertexName[G.in[j]-1],vertexName[i-1]);	
            

        }
    }

    //fclose(large_graph);


/////////outgoing edge count for large graph//////////
	//FILE* out_edge_part_large_graph = fopen("out_edge_part_large_graph.txt","w");
	long int tot_out_edge_large = 0;
	for (i = 1;i<=G.nVrtx ; i++){
		//fprintf(out_edge_part_large_graph,"(%d)%s --> %d\n",i,G.vertices[i-1],G.outEnd[i]-G.outStart[i]+1);
		tot_out_edge_large += G.outEnd[i]-G.outStart[i]+1;
	}
	//fclose(out_edge_part_large_graph);

	printf("\ntotal outgoing edge coarse graph = %ld\n",tot_out_edge_large);



    printf("graph total weight = %lf\n",G.totvw);
    for(i = 0 ; i < G.nEdge ; i++){
    //  printf("out[%d] = %d\n",i,G.out[i]);
    }


    for (i=0; i<opt.nbPart; i++) {
        if (opt.lb[i] < 0)
            opt.lb[i] = 1;
        if (opt.ub[i] < 0) {
            // // anik's version had this line
            // opt.ub[i] = G.totvw/(opt.nbPart*1.0);
            // new version have this line
            opt.ub[i] = opt.ratio * (double)G.totvw/(double)opt.nbPart; 
        }
        if ((floor(opt.lb[i]) < opt.lb[i])&&(floor(opt.lb[i]) == floor(opt.ub[i]))) {
            printf("WARNING: The balance can not be matched!!!\n");
            opt.lb[i] = floor(opt.lb[i]);
            opt.ub[i] = floor(opt.ub[i])+1;
        }
    }

    double* save_ub = (double*) malloc(sizeof(double) * opt.nbPart);
    for (i=0; i<opt.nbPart; i++)
        save_ub[i] = opt.ub[i];

    printf("Nb node: %d\nNb Edges: %d\nNb part: %d\nLower Bound[0]: %f\nTargeted Bound[0]: %f\n\n", G.nVrtx, G.nEdge, opt.nbPart, opt.lb[0], opt.ub[0]);

    // printf("Graph Information:\n\tNb node: %d\n\tNb Edges: %d\n\tMax in-degree: %d\n\tMax out-degree: %d\n\tAv. in-degree: %.2f\n\tAv. out-degree: %.2f\nProblem Information:\n\tNb part: %d\n\tLower Bound[0]: %.1f\n\tUpper Bound[0]: %.1f\n\n", G.nVrtx, G.nEdge, maxindegree, maxoutdegree,aveindegree,aveoutdegree,opt.nbPart, opt.lb[0], opt.ub[0]);
    ecType * edgecut = (ecType *) malloc(sizeof(ecType) * opt.runs);
    ecType* nbcomm = (ecType*) malloc(sizeof(ecType) * opt.runs);
    double* latencies = (double*) malloc(sizeof(double) * opt.runs);
    int r;
    rcoarsen * rcoars;
    idxType nbcomp = 0;

    if(opt.seed == 0) {
        usRandom((int) time(NULL));
    }
    else
        usRandom(opt.seed);

    for (r = 0; r<opt.runs; r++) {
        int isAcyclic;
        rMLGP_info* info = (rMLGP_info*)  malloc (sizeof (rMLGP_info));
        initRInfoPart(info);
        printf("########################## RUN %d (seed=%d) ########################\n", r, uGetSeed());

        //dummy try, make conpar value 0
        // opt.conpar = 0;

        rcoars =  rVCycle(&G, opt, info, &partcount);

        printf("\n\n\nresulting part count = %d\n\n",partcount);

//         for (i=1; i<=rcoars->coars->graph->nVrtx; i++){
// //  fprintf(file, "%s = part %d\n",G.vertices[i], (int) rcoars->coars->part[i]);
//   //fprintf(original_graph_partition, "%s = part %d\n",rcoars->coars->graph->vertices[i-1], (int) rcoars->coars->part[i]);
//   printf("%s = part %d\n",rcoars->coars->graph->vertices[i-1], (int) rcoars->coars->part[i]);
//     }



        edgecut[r] = edgeCut(&G, rcoars->coars->part);
        nbcomm[r] = nbCommunications(&G, rcoars->coars->part);
        latencies[r] = computeLatency(&G, rcoars->coars->part, 1.0, 11.0);

        printf("Final run %d\t%d\t%d\t%lf\t%lf\t%lf\n", r, G.nVrtx, G.nEdge, edgecut[r], nbcomm[r], latencies[r]);

        if (opt.debug) {
            isAcyclic = checkAcyclicity(&G, rcoars->coars->part, opt.nbPart);
            if (isAcyclic == 0)
                u_errexit("rMLGP: the partition obtained is not acyclic\n");
        }

        for (i=0; i<opt.nbPart; i++)
            opt.ub[i] = save_ub[i];
        opt.seed++;

        // print the parts to file.
        if(opt.write_parts>0)
        {
            char name_tmp[200];
            sprintf(name_tmp,".partsfile.part_%d.seed_%d.txt", opt.nbPart, opt.seed);
            char res_name[200];
            strcpy(res_name, file_name);
            strcat(res_name, name_tmp);
            writePartsToTxt(&G,res_name,rcoars->coars->part);
        }

        printf("######################## END RUN %d ########################\n\n", r);
        if (opt.print > 0) {
            printf("######################## OUTPUT %d ########################\n\n", r);
            printRInfoPart(info, G.nVrtx, G.nEdge, opt.print);
            printf("\n###########################################################\n\n");
        }
        //freeRCoarsenHeader(rcoars);
        //rcoars = (rcoarsen*) NULL;
        //freeRInfoPart(info);
    }

    ecType edgecutave = 0.0, nbcommave = 0.0, edgecutsd = 0.0, nbcommsd = 0.0;
    for (r = 0; r<opt.runs; r++) {
        edgecutave += edgecut[r];
        nbcommave += nbcomm[r];
    }
    edgecutave = edgecutave / opt.runs;
    nbcommave = nbcommave / opt.runs;
    for (r = 0; r<opt.runs; r++) {
        edgecutsd = (edgecut[r] - edgecutave) < 0 ? edgecutave - edgecut[r] : edgecut[r] - edgecutave;
        nbcommsd = (nbcomm[r] - nbcommave) < 0 ? nbcommave - nbcomm[r] : nbcomm[r] - nbcommave;
    }
    edgecutsd = edgecutsd / opt.runs;
    nbcommsd = nbcommsd / opt.runs;
    printf("Average Edgecut:%lld\tStandard Deviation: %d\n", (int) edgecutave, (int) edgecutsd);





    //FILE *file;
    //FILE *different_part;
    //FILE *same_part;
    //FILE *graph_lookup_100;
    //FILE *graph_lookup_99;
    //char name_tmp[200];
    //sprintf(name_tmp,".rMLGP.part.%d.seed.%d", opt.nbPart, opt.seed);
    //char res_name[200];
    //strcpy(res_name, file_name);
    //strcat(res_name, name_tmp);
    //strcat(res_name,".txt");
    //file = fopen(res_name, "w");
    //different_part = fopen("different_part.txt","w");
    //same_part = fopen("same_part.txt","w");
    //graph_lookup_100 = fopen("graph_lookup_100.txt","w");
    //graph_lookup_99 = fopen("graph_lookup_99.txt","w");
//    idxType i;



    //FILE *original_graph_partition  = fopen("original_graph_partition.txt","w");
    //for (i=1; i<=rcoars->coars->graph->nVrtx; i++){
//  fprintf(file, "%s = part %d\n",G.vertices[i], (int) rcoars->coars->part[i]);
//  fprintf(original_graph_partition, "%s = part %d\n",rcoars->coars->graph->vertices[i-1], (int) rcoars->coars->part[i]);
    //}

    //fprintf(file, "%d\n", partcount);
    int p_part = -1;
    int distinct_part_count = 0;

    

    printf("inside rMLGP before partition topsort call result vertex count = %d\n",rcoars->coars->graph->nVrtx);


    idxType* my_partition_topsort = (idxType*)malloc((rcoars->coars->graph->nVrtx+1)*sizeof(idxType));
    //DFStopsort_with_part(&G,rcoars->coars->part,opt.nbPart,my_partition_topsort);
    DFStopsort_with_part(&G,rcoars->coars->part,partcount,my_partition_topsort);

    printf("topsort is done\n");
    for(i=1;i<=rcoars->coars->graph->nVrtx;i++){
        if(rcoars->coars->part[my_partition_topsort[i]] != p_part){
            distinct_part_count++;
            p_part = rcoars->coars->part[my_partition_topsort[i]];
        }
         //fprintf(original_graph_partition,"%s %d %d\n",vertexName[my_partition_topsort[i]-1],rcoars->coars->part[my_partition_topsort[i]],rcoars->coars->graph->nVrtx-i);
    }
    //fclose(original_graph_partition);

    int* processed_coarsened = (int*) calloc(rcoars->coars->graph->nVrtx+1 , sizeof(int));

    for(int i = 1 ; i <= rcoars->coars->graph->nVrtx ; i++){
    		processed_coarsened[my_partition_topsort[i]] = 1;

		for(j = rcoars->coars->graph->outStart[my_partition_topsort[i]] ; j <= rcoars->coars->graph->outEnd[my_partition_topsort[i]] ; j++){
		
		if(processed_coarsened[rcoars->coars->graph->out[j]] == 1){
			printf("coarsened graph %s --> %s , %s already processed\n",vertexName[my_partition_topsort[i]-1],rcoars->coars->graph->vertices[rcoars->coars->graph->out[j]-1],rcoars->coars->graph->vertices[rcoars->coars->graph->out[j]-1]);
		
			}
		}
    
    }
    

	printf("\n\ncoarse graph checking is done\n\n\n");


    //////memory calculation for each partition/////////////////
//    memory_from_out = (ecType*)calloc(partcount,sizeof(ecType));
//    memory_from_in = (ecType*)calloc(partcount,sizeof(ecType));

//    for(i=0;i<partcount;++i){
//        memory_from_in[i] = 0;
//        memory_from_out[i] = 0;
//    }




    //FILE* mem_out = fopen("mem_out.txt","w");
    //FILE* mem_in = fopen("mem_in.txt","w");



/////////////// if large blk and small blk are same , dont refine the graph/////////



    if(wblk == small_block){


//////// write partinfo file for large blk == small blk/////////



    

    sprintf(partinfo_file_name,"%s_%dk_%dk_part%d_%s_partinfo.txt",matrix_name,wblk/1024,wblk/1024,opt.nbPart,loopname);




    partInfo = fopen(partinfo_file_name,"w");

    fprintf(partInfo, "%d\n", distinct_part_count+1);


    for(i = 1, j = rcoars->coars->graph->nVrtx-1 ; i <= rcoars->coars->graph->nVrtx ; i++, j--)
    {
        
        //non-priority
        //fprintf(refined_graph_partition,"%s %d\n",newVertexName[my_small_partition_topsort[i]-1],small_rcoarse->coars->part[my_small_partition_topsort[i]]);

        //priority
        if(rcoars->coars->part[my_partition_topsort[i]] != prev_part){

            prev_part = rcoars->coars->part[my_partition_topsort[i]];
            fprintf(partInfo,"%d\n", i-1);


        }
    }

    fprintf(partInfo, "%d\n", rcoars->coars->graph->nVrtx);
    fclose(partInfo);




    printf("rcoarse\n");
    if(loopType == 0)
        make_new_table_file(rcoars, my_partition_topsort,"nonloop",matrix_name,wblk,wblk,opt.nbPart);

    if(loopType == 1)
        make_new_table_file(rcoars, my_partition_topsort,"firstloop",matrix_name,wblk,wblk,opt.nbPart);


    if(loopType == 2)
        make_new_table_file(rcoars, my_partition_topsort,"secondloop",matrix_name,wblk,wblk,opt.nbPart);



    return ;
}
    myprint();

    //get the new block info
    int **nnzblock_matrix = NULL;
    int nrowblocks,ncolblocks; 

    printf("wblk = %d\n", wblk);
    int starting_block_size = wblk;



    get_new_csb_block(small_block,&nnzblock_matrix, &nrowblocks, &ncolblocks);

    for(i = 0 ; i < nrowblocks ; i++){
        for(j = 0 ; j < ncolblocks ; j++){
            //printf("%d ",nnzblock_matrix[i][j]);
        }
        //printf("\n");
    }

/*temporary placement, move to the last of the function later*/
    //break large blocked graph into small blocked graph
    //my_generate_smallgraph(&G, file_name,opt.use_binary_input, edge_u, edge_v, edge_weight, edgeCount, vertexCount,vertexName,vertexWeight,8);
    int block_divisor = starting_block_size/small_block;
    //int old_blocksize = (sqrt(1917+4*vertexCount)-45)/2;
    int old_blocksize = nrowblocks;

    printf("old blocksize = %d\n",old_blocksize);
    int new_blocksize = nrowblocks;

    //printf("vertexcount = %d edgeCount\n", );

    printf("new_blocksize = %d\n", new_blocksize);

    int newVertexCount = new_blocksize*45+new_blocksize*new_blocksize+27;
    int newEdgeCount = 90*new_blocksize+25+new_blocksize*new_blocksize*4; /// replace 90 with 84 if needed
    char **newVertexName;
    printf("new vertexcount = %d newEdgeCount = %d\n",newVertexCount,newEdgeCount);

    int* prev_vertex = ((int*)malloc((newVertexCount+1)*sizeof(int)));
    double* newVWeight = (double*)malloc((newVertexCount+1)*sizeof(double));

    newVertexName = (char**)malloc((newVertexCount+1)*sizeof(char*));

    
    for(i = 0;i< (newVertexCount);++i){
            newVertexName[i] = (char*)malloc(100*sizeof(char));
        }


    int *newEdge_u = ((int*)malloc((newEdgeCount+1)*sizeof(int)));
    int *newEdge_v = ((int*)malloc((newEdgeCount+1)*sizeof(int)));
    double *newEdge_weight = ((double*)malloc((newEdgeCount+1)*sizeof(double)));

    printf("all allocation done\n");

    int updatedVertexCount, updatedEdgeCount;


    // getting the vmap from dgraph
    int **v_map;
    v_map = (int**)calloc((vertexCount+1),sizeof(int*));
    
    for(i = 0;i< (vertexCount);++i){
            v_map[i] = (int*)calloc((block_divisor*block_divisor+1),sizeof(int));
    }
    printf("vmap all allocation done with vertex count = %d and block divisor %d\n", vertexCount, block_divisor);
    fflush(stdout);

/////////////////////////////
////////////CHECK////////////
    create_smallgraph_datastructure_sparse(edge_u, edge_v, edge_weight, edgeCount, vertexCount,vertexName,vertexWeight,block_divisor,&newVertexName,newVertexCount, newEdgeCount,
                                    &newEdge_u,&newEdge_v,&newEdge_weight, &prev_vertex, &newVWeight, nnzblock_matrix,nrowblocks,ncolblocks,&updatedVertexCount,&updatedEdgeCount,&v_map);
/////////////////////////////
/////////////////////////////
    //create_smallgraph_datastructure(edge_u, edge_v, edge_weight, edgeCount, vertexCount,vertexName,vertexWeight,block_divisor,&newVertexName,newVertexCount, newEdgeCount,
    //                                &newEdge_u,&newEdge_v,&newEdge_weight, &prev_vertex, &newVWeight, nnzblock_matrix,nrowblocks,ncolblocks);
    printf("returned %s\n",newVertexName[0]);
    printf("rmlgp te asche abar\n");
    printf("updatedVertexCount = %d updatedEdgeCount = %d\n",updatedVertexCount,updatedEdgeCount);
    fflush(stdout);

    //for(i = 0 ; i < updatedVertexCount ; i++){
        //printf("prev[%s] = %s\n",newVertexName[i], G.vertices[prev_vertex[i]]);
    //}


    dgraph small_G;
/////////////////////////////
////////////CHECK////////////
    my_generate_smallgraph(&small_G, file_name, opt.use_binary_input, newEdge_u, newEdge_v, newEdge_weight, updatedEdgeCount, updatedVertexCount, (const char**) newVertexName,newVWeight,block_divisor);
/////////////////////////////
/////////////////////////////


    /////////outgoing edge count for small graph//////////
    //FILE* out_edge_part_small_graph = fopen("out_edge_part_small_graph.txt","w");
    long int tot_out_edge_small = 0 ;
    for (i = 1;i<=small_G.nVrtx ; i++){
        //fprintf(out_edge_part_small_graph,"(%d)%s --> %d\n",i,small_G.vertices[i-1],small_G.outEnd[i]-small_G.outStart[i]+1);
		tot_out_edge_small += small_G.outEnd[i]-small_G.outStart[i]+1;
    }
    //fclose(out_edge_part_small_graph);

	printf("\ntotal outgoing edge refined graph = %ld\n",tot_out_edge_small);

////////////////////////////////////////////////////////////


    coarsen* small_C;

    small_C = initializeCoarsen(&small_G);
    rcoarsen* small_rcoarse;
    small_rcoarse = initializeRCoarsen(small_C);

    //FILE *refined_graph_partition = fopen("refined_graph_partition.txt","w");

    for(i = 1 ; i <= small_rcoarse->coars->graph->nVrtx ; i++){
        small_rcoarse->coars->part[i] = rcoars->coars->part[prev_vertex[i-1]+1];
        //fprintf(refined_graph_partition,"old = %s(%d) new = %s(%d)  \n",G.vertices[prev_vertex[i-1]],rcoars->coars->part[prev_vertex[i-1]], small_rcoarse->coars->graph->vertices[i-1],small_rcoarse->coars->part[i]);
    }
    //fclose(refined_graph_partition);

    //for(i = 0 ; i < newEdgeCount ; i++){
        //printf("edge %d %s(%d) --> %s(%d)\n",i,newVertexName[newEdge_u[i]],newEdge_u[i],newVertexName[newEdge_v[i]],newEdge_v[i]);
    //}


    //FILE* small_file = fopen("small_file.txt","w");
    //fprintf(small_file, "%d\n", distinct_part_count);


    // TODO: update with appropriate function
    // print_info_part(small_rcoarse->coars->graph, small_rcoarse->coars->part, opt);


    idxType* my_small_partition_topsort = (idxType*)malloc((small_rcoarse->coars->graph->nVrtx+1)*sizeof(idxType));
    //DFStopsort_with_part(&G,rcoars->coars->part,opt.nbPart,my_partition_topsort);

    //for (i = 1 ; i <= small_G.nVrtx ; i++){
        //printf("%s %d\n",small_G.vertices[i-1],small_rcoarse->coars->part[i]);
    //}


    ///////// calling dfstop sort with part using refined graph////////////
    DFStopsort_with_part(&small_G,small_rcoarse->coars->part,partcount,my_small_partition_topsort);


    printf("\n\nloop type = %d large_block = %d small block %d\n", loopType,starting_block_size,small_block);
    FILE* refined_new;
    prev_part = -1;

    //if(loopType == 0 ) strcpy(loopname,"nonloop");
    //else if(loopType == 1 )strcpy(loopname,"firstloop");
    //else if(loopType == 2 )strcpy(loopname,"secondloop");
    //refined_new = fopen("refined_new.txt","w");
    //FILE* part_topsort = fopen("part_topsort.txt","w");

//////small blk partinfo file///////////////


    sprintf(partinfo_file_name,"%s_%dk_%dk_part%d_%s_partinfo.txt",matrix_name,starting_block_size/1024,wblk/1024,opt.nbPart,loopname);


    partInfo = fopen(partinfo_file_name,"w");


    fprintf(partInfo, "%d\n", distinct_part_count+1);

    for(i = 1, j = small_rcoarse->coars->graph->nVrtx-1 ; i <= small_rcoarse->coars->graph->nVrtx ; i++, j--)
    {
        
        //non-priority
        //fprintf(refined_graph_partition,"%s %d\n",newVertexName[my_small_partition_topsort[i]-1],small_rcoarse->coars->part[my_small_partition_topsort[i]]);

        //priority
        if(small_rcoarse->coars->part[my_small_partition_topsort[i]] != prev_part){

            prev_part = small_rcoarse->coars->part[my_small_partition_topsort[i]];
            fprintf(partInfo,"%d\n", i-1);
            //fprintf(part_topsort,"%d\n",prev_part);
            //sprintf(new_fname,"refined_new_graphs/refined_new_%d.txt",prev_part);
            //fclose(refined_new);
            //refined_new = fopen(new_fname,"w");

        }
        //fprintf(refined_graph_partition,"%s %d %d\n",newVertexName[my_small_partition_topsort[i]-1],small_rcoarse->coars->part[my_small_partition_topsort[i]], j);
        //fprintf(refined_new,"%s %d %d\n",newVertexName[my_small_partition_topsort[i]-1],small_rcoarse->coars->part[my_small_partition_topsort[i]], j);
    }


	/////////////////processed node check /////////////////
	int* processed_coarsened_small = (int*) calloc(small_rcoarse->coars->graph->nVrtx+1 , sizeof(int));

    for(int i = 1 ; i <= small_rcoarse->coars->graph->nVrtx ; i++)
    {
        processed_coarsened_small[my_small_partition_topsort[i]] = 1;
        for(j = small_rcoarse->coars->graph->outStart[my_small_partition_topsort[i]] ; j <= small_rcoarse->coars->graph->outEnd[my_small_partition_topsort[i]] ; j++)
        {
            if(processed_coarsened_small[small_rcoarse->coars->graph->out[j]] == 1)
            {
                printf("refined graph %s --> %s , %s already processed\n",newVertexName[my_partition_topsort[i]-1],small_rcoarse->coars->graph->vertices[rcoars->coars->graph->out[j]-1],small_rcoarse->coars->graph->vertices[small_rcoarse->coars->graph->out[j]-1]);
            }
        }
    }

	printf("\n\nrefined graph checking is done\n\n\n");

    fprintf(partInfo, "%d\n", small_rcoarse->coars->graph->nVrtx);
    //fclose(refined_new);
    //fclose(part_topsort);
    fclose(partInfo);


    //fclose(refined_graph_partition);

    printf("small_rcoarse\n");
    make_new_table_file(small_rcoarse, my_small_partition_topsort,loopname,matrix_name,starting_block_size,wblk,opt.nbPart);
     
    printf("make new table file is done\n");

  //  free(my_small_partition_topsort);
  //  free(my_partition_topsort);


        //////memory calculation for each partition/////////////////
//    ecType* small_memory_from_out = (ecType*)calloc(partcount,sizeof(ecType));
//    ecType* small_memory_from_in = (ecType*)calloc(partcount,sizeof(ecType));

//    for(i=0;i<partcount;++i){
//        small_memory_from_in[i] = 0;
//        small_memory_from_out[i] = 0;
//    }


    // for(i=1;i<=small_rcoarse->coars->graph->nVrtx;i++){


    //             //printf("%s %d\n",vertexName[my_partition_topsort[i]-1],rcoars->coars->part[my_partition_topsort[i]]);
    //             //fprintf(same_part, "%s\n", vertexName[my_partition_topsort[i]-1]);
    //             small_memory_from_in[small_rcoarse->coars->part[i]] += small_rcoarse->coars->graph->vWeight[i]; //at this moment 1
    //             for(j=small_rcoarse->coars->graph->inStart[i];j<=small_rcoarse->coars->graph->inEnd[i];++j)
    //             {
    //                 //printf("vertex = %s(%d) j = %s(%d) part = %d\n",vertexName[my_partition_topsort[i]-1],my_partition_topsort[i]-1,vertexName[rcoars->coars->graph->in[j]-1],j,rcoars->coars->part[rcoars->coars->graph->in[j]-1]);
    //                 //incoming_edge_weight_in_part += igraph->ecIn[j];
    //                 if(small_rcoarse->coars->part[small_rcoarse->coars->graph->in[j]] != small_rcoarse->coars->part[i])
    //                 {   
    //                     //fprintf(different_part, "%s %d\n", vertexName[rcoars->coars->graph->in[j]-1],rcoars->coars->part[rcoars->coars->graph->in[j]-1]);
    //                     small_memory_from_out[small_rcoarse->coars->part[i]] += small_rcoarse->coars->graph->ecIn[j];
    //                 }
    //             }
            


    //     }
    //FILE* small_mem_out = fopen("small_mem_out.txt","w");
    //FILE* small_mem_in = fopen("small_mem_in.txt","w");

    // for(i = 0 ; i < partcount ; i++){
    //     fprintf(small_mem_out, "%d %lf\n",i,small_memory_from_out[i] );
    //     fprintf(small_mem_in, "%d %lf\n",i,small_memory_from_in[i] );

    // }

    /*FILE* small_vname = fopen("small_vname.txt","w");
    for(i = 0 ; i < small_G.nVrtx ; i++){
        fprintf(small_vname,"%s\n",small_G.vertices[i]);
    }
    fclose(small_vname);
*/






    //fclose(file);
    //fclose(different_part);
    //fclose(same_part);
    //fclose(mem_out);
    //fclose(mem_in);
    //fclose(small_mem_out);
    //fclose(small_mem_in);
    //fclose(graph_lookup_100);
    //fclose(graph_lookup_99);
    //fclose(small_file);

    free(save_ub);
    free(edgecut);
    free(nbcomm);
    free(latencies);
    freeDGraphData(&G);

}
#if 0
int main(int argc, char *argv[])
{
    MLGP_option opt;

    processArgs_rMLGP(argc, argv, &opt);
    run_rMLGP(opt.file_name, opt);

    free_opt(&opt);
    return 0;
}
#endif
