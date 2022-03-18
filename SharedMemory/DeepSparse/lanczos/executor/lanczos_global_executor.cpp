#include "../../common/exec_util.h"
#include "../../common/matrix_ops.h"
#include "../../common/vector_ops.h"

int main(int argc, char *argv[])
{
    (void)argc;

    int i, j;
    int blocksize, block_width;
    int iterationNumber, eig_wanted;
    char *filename;

    double *xrem;
    block *matrixBlock;

    double *q, *qq;
    double norm_q;
    double *Q;
    double *z;
    double *alpha, *beta, *QpZ, *QQpZ;

    double *QpZBUFF, *AlphaBUFF, *normBUFF;

    struct TaskInfo *taskInfo;
    int structIterator, taskCount;
    int block_id, row_id, col_id, buf_id, task_id, blksz;
    
    double tstart, tend;
    double total_time;

    filename = argv[1];
    block_width = atoi(argv[2]);
    eig_wanted = atoi(argv[3]);
    taskCount = buildTaskInfoStruct(taskInfo, argv[4]);
    printf("taskCount: %d\n", taskCount);
    
    wblk = block_width; 
    
    read_custom(filename, xrem);
    csc2blkcoord(matrixBlock, xrem);

    #pragma omp parallel
    #pragma omp master
    {
        nthrds = omp_get_num_threads();
    }
    printf("nthrds: %d\n", nthrds);

    delete [] colptrs;
    delete [] irem;
    delete [] xrem;

    q = (double *) malloc(numcols * sizeof(double));
    qq = (double *) malloc(numcols * sizeof(double));
    z = (double *) malloc(numcols * sizeof(double));
    alpha = (double *) malloc(eig_wanted * sizeof(double));
    beta = (double *) malloc(eig_wanted * sizeof(double));
    QpZ = (double *) malloc(eig_wanted * sizeof(double));
    QQpZ = (double *) malloc(numcols * sizeof(double));
    Q = (double*) calloc(numcols*(eig_wanted + 1),sizeof(double));
    
    QpZBUFF = (double *) malloc (nthrds * eig_wanted * sizeof(double));
    AlphaBUFF = (double *) malloc (nthrds * sizeof(double));
    normBUFF = (double *) malloc (nthrds * sizeof(double));

    #pragma omp parallel for default(shared)
    for(i = 0; i < numcols; ++i)
    {
        q[i] = 1.0;
    }

    norm_q = sqrt(numcols);

    #pragma omp parallel for default(shared)
    for(i = 0 ; i < numcols; ++i)
    {
        qq[i] = q[i]/norm_q;
    }

    #pragma omp parallel for default(shared)
    for(i = 0 ; i < numcols; ++i)
    {
        Q[i*(eig_wanted)] = qq[i];
    }


    blocksize = 1;

    #pragma omp parallel
    {
        #pragma omp master
        {
            for(iterationNumber = 0; iterationNumber < eig_wanted; ++iterationNumber)
            {
                tstart = omp_get_wtime();

#ifdef EPYC
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, AlphaBUFF)\
                depend(in: nthrds)\
                depend(inout: AlphaBUFF[0 * blocksize : blocksize], AlphaBUFF[1 * blocksize : blocksize], AlphaBUFF[2 * blocksize : blocksize],\
                        AlphaBUFF[3 * blocksize : blocksize], AlphaBUFF[4 * blocksize : blocksize], AlphaBUFF[5 * blocksize : blocksize], AlphaBUFF[6 * blocksize : blocksize],\
                        AlphaBUFF[7 * blocksize : blocksize], AlphaBUFF[8 * blocksize : blocksize], AlphaBUFF[9 * blocksize : blocksize], AlphaBUFF[10 * blocksize : blocksize],\
                        AlphaBUFF[11 * blocksize : blocksize], AlphaBUFF[12 * blocksize : blocksize], AlphaBUFF[13 * blocksize : blocksize],\
                        AlphaBUFF[14 * blocksize : blocksize], AlphaBUFF[15 * blocksize : blocksize], AlphaBUFF[16 * blocksize : blocksize],\
                        AlphaBUFF[17 * blocksize : blocksize], AlphaBUFF[18 * blocksize : blocksize], AlphaBUFF[19 * blocksize : blocksize],\
                        AlphaBUFF[20 * blocksize : blocksize], AlphaBUFF[21 * blocksize : blocksize], AlphaBUFF[22 * blocksize : blocksize],\
                        AlphaBUFF[23 * blocksize : blocksize], AlphaBUFF[24 * blocksize : blocksize], AlphaBUFF[25 * blocksize : blocksize],\
                        AlphaBUFF[26 * blocksize : blocksize], AlphaBUFF[27 * blocksize : blocksize], AlphaBUFF[28 * blocksize : blocksize],\
                        AlphaBUFF[29 * blocksize : blocksize], AlphaBUFF[30 * blocksize : blocksize], AlphaBUFF[31 * blocksize : blocksize],\
                        AlphaBUFF[32 * blocksize : blocksize], AlphaBUFF[33 * blocksize : blocksize], AlphaBUFF[34 * blocksize : blocksize],\
                        AlphaBUFF[35 * blocksize : blocksize], AlphaBUFF[36 * blocksize : blocksize], AlphaBUFF[37 * blocksize : blocksize],\
                        AlphaBUFF[38 * blocksize : blocksize], AlphaBUFF[39 * blocksize : blocksize], AlphaBUFF[40 * blocksize : blocksize],\
                        AlphaBUFF[41 * blocksize : blocksize], AlphaBUFF[42 * blocksize : blocksize], AlphaBUFF[43 * blocksize : blocksize],\
                        AlphaBUFF[44 * blocksize : blocksize], AlphaBUFF[45 * blocksize : blocksize], AlphaBUFF[46 * blocksize : blocksize],\
                        AlphaBUFF[47 * blocksize : blocksize], AlphaBUFF[48 * blocksize : blocksize], AlphaBUFF[49 * blocksize : blocksize],\
                        AlphaBUFF[50 * blocksize : blocksize], AlphaBUFF[51 * blocksize : blocksize], AlphaBUFF[52 * blocksize : blocksize],\
                        AlphaBUFF[53 * blocksize : blocksize], AlphaBUFF[54 * blocksize : blocksize], AlphaBUFF[55 * blocksize : blocksize],\
                        AlphaBUFF[56 * blocksize : blocksize], AlphaBUFF[57 * blocksize : blocksize], AlphaBUFF[58 * blocksize : blocksize],\
                        AlphaBUFF[59 * blocksize : blocksize], AlphaBUFF[60 * blocksize : blocksize], AlphaBUFF[61 * blocksize : blocksize],\
                        AlphaBUFF[62 * blocksize : blocksize], AlphaBUFF[63 * blocksize : blocksize], AlphaBUFF[64 * blocksize : blocksize],\
                        AlphaBUFF[65 * blocksize : blocksize], AlphaBUFF[66 * blocksize : blocksize], AlphaBUFF[67 * blocksize : blocksize],\
                        AlphaBUFF[68 * blocksize : blocksize], AlphaBUFF[69 * blocksize : blocksize], AlphaBUFF[70 * blocksize : blocksize],\
                        AlphaBUFF[71 * blocksize : blocksize], AlphaBUFF[72 * blocksize : blocksize], AlphaBUFF[73 * blocksize : blocksize],\
                        AlphaBUFF[74 * blocksize : blocksize], AlphaBUFF[75 * blocksize : blocksize], AlphaBUFF[76 * blocksize : blocksize],\
                        AlphaBUFF[77 * blocksize : blocksize], AlphaBUFF[78 * blocksize : blocksize], AlphaBUFF[79 * blocksize : blocksize],\
                        AlphaBUFF[80 * blocksize : blocksize], AlphaBUFF[81 * blocksize : blocksize], AlphaBUFF[82 * blocksize : blocksize],\
                        AlphaBUFF[83 * blocksize : blocksize], AlphaBUFF[84 * blocksize : blocksize], AlphaBUFF[85 * blocksize : blocksize],\
                        AlphaBUFF[86 * blocksize : blocksize], AlphaBUFF[87 * blocksize : blocksize], AlphaBUFF[88 * blocksize : blocksize],\
                        AlphaBUFF[89 * blocksize : blocksize], AlphaBUFF[90 * blocksize : blocksize], AlphaBUFF[91 * blocksize : blocksize],\
                        AlphaBUFF[92 * blocksize : blocksize], AlphaBUFF[93 * blocksize : blocksize], AlphaBUFF[94 * blocksize : blocksize],\
                        AlphaBUFF[95 * blocksize : blocksize], AlphaBUFF[96 * blocksize : blocksize], AlphaBUFF[97 * blocksize : blocksize],\
                        AlphaBUFF[98 * blocksize : blocksize], AlphaBUFF[99 * blocksize : blocksize], AlphaBUFF[100 * blocksize : blocksize],\
                        AlphaBUFF[101 * blocksize : blocksize], AlphaBUFF[102 * blocksize : blocksize], AlphaBUFF[103 * blocksize : blocksize],\
                        AlphaBUFF[104 * blocksize : blocksize], AlphaBUFF[105 * blocksize : blocksize], AlphaBUFF[106 * blocksize : blocksize],\
                        AlphaBUFF[107 * blocksize : blocksize], AlphaBUFF[108 * blocksize : blocksize], AlphaBUFF[109 * blocksize : blocksize],\
                        AlphaBUFF[110 * blocksize : blocksize], AlphaBUFF[111 * blocksize : blocksize], AlphaBUFF[112 * blocksize : blocksize],\
                        AlphaBUFF[113 * blocksize : blocksize], AlphaBUFF[114 * blocksize : blocksize], AlphaBUFF[115 * blocksize : blocksize],\
                        AlphaBUFF[116 * blocksize : blocksize], AlphaBUFF[117 * blocksize : blocksize], AlphaBUFF[118 * blocksize : blocksize],\
                        AlphaBUFF[119 * blocksize : blocksize], AlphaBUFF[120 * blocksize : blocksize], AlphaBUFF[121 * blocksize : blocksize],\
                        AlphaBUFF[122 * blocksize : blocksize], AlphaBUFF[123 * blocksize : blocksize], AlphaBUFF[124 * blocksize : blocksize],\
                        AlphaBUFF[125 * blocksize : blocksize], AlphaBUFF[126 * blocksize : blocksize], AlphaBUFF[127 * blocksize : blocksize])
#else
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, AlphaBUFF)\
                depend(in: nthrds)\
                depend(inout: AlphaBUFF[0 * blocksize : blocksize], AlphaBUFF[1 * blocksize : blocksize], AlphaBUFF[2 * blocksize : blocksize],\
                        AlphaBUFF[3 * blocksize : blocksize], AlphaBUFF[4 * blocksize : blocksize], AlphaBUFF[5 * blocksize : blocksize], AlphaBUFF[6 * blocksize : blocksize],\
                        AlphaBUFF[7 * blocksize : blocksize], AlphaBUFF[8 * blocksize : blocksize], AlphaBUFF[9 * blocksize : blocksize], AlphaBUFF[10 * blocksize : blocksize],\
                        AlphaBUFF[11 * blocksize : blocksize], AlphaBUFF[12 * blocksize : blocksize], AlphaBUFF[13 * blocksize : blocksize])
#endif
                {
                    tstart = omp_get_wtime();
                    for(i = 0 ; i < nthrds * blocksize; ++i)
                    {
                        AlphaBUFF[i] = 0.0;
                    }
                }

#ifdef EPYC
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, normBUFF)\
                depend(in: nthrds)\
                depend(inout: normBUFF[0 * blocksize : blocksize], normBUFF[1 * blocksize : blocksize], normBUFF[2 * blocksize : blocksize],\
                        normBUFF[3 * blocksize : blocksize], normBUFF[4 * blocksize : blocksize], normBUFF[5 * blocksize : blocksize], normBUFF[6 * blocksize : blocksize],\
                        normBUFF[7 * blocksize : blocksize], normBUFF[8 * blocksize : blocksize], normBUFF[9 * blocksize : blocksize], normBUFF[10 * blocksize : blocksize],\
                        normBUFF[11 * blocksize : blocksize], normBUFF[12 * blocksize : blocksize], normBUFF[13 * blocksize : blocksize],\
                        normBUFF[14 * blocksize : blocksize], normBUFF[15 * blocksize : blocksize], normBUFF[16 * blocksize : blocksize],\
                        normBUFF[17 * blocksize : blocksize], normBUFF[18 * blocksize : blocksize], normBUFF[19 * blocksize : blocksize],\
                        normBUFF[20 * blocksize : blocksize], normBUFF[21 * blocksize : blocksize], normBUFF[22 * blocksize : blocksize],\
                        normBUFF[23 * blocksize : blocksize], normBUFF[24 * blocksize : blocksize], normBUFF[25 * blocksize : blocksize],\
                        normBUFF[26 * blocksize : blocksize], normBUFF[27 * blocksize : blocksize], normBUFF[28 * blocksize : blocksize],\
                        normBUFF[29 * blocksize : blocksize], normBUFF[30 * blocksize : blocksize], normBUFF[31 * blocksize : blocksize],\
                        normBUFF[32 * blocksize : blocksize], normBUFF[33 * blocksize : blocksize], normBUFF[34 * blocksize : blocksize],\
                        normBUFF[35 * blocksize : blocksize], normBUFF[36 * blocksize : blocksize], normBUFF[37 * blocksize : blocksize],\
                        normBUFF[38 * blocksize : blocksize], normBUFF[39 * blocksize : blocksize], normBUFF[40 * blocksize : blocksize],\
                        normBUFF[41 * blocksize : blocksize], normBUFF[42 * blocksize : blocksize], normBUFF[43 * blocksize : blocksize],\
                        normBUFF[44 * blocksize : blocksize], normBUFF[45 * blocksize : blocksize], normBUFF[46 * blocksize : blocksize],\
                        normBUFF[47 * blocksize : blocksize], normBUFF[48 * blocksize : blocksize], normBUFF[49 * blocksize : blocksize],\
                        normBUFF[50 * blocksize : blocksize], normBUFF[51 * blocksize : blocksize], normBUFF[52 * blocksize : blocksize],\
                        normBUFF[53 * blocksize : blocksize], normBUFF[54 * blocksize : blocksize], normBUFF[55 * blocksize : blocksize],\
                        normBUFF[56 * blocksize : blocksize], normBUFF[57 * blocksize : blocksize], normBUFF[58 * blocksize : blocksize],\
                        normBUFF[59 * blocksize : blocksize], normBUFF[60 * blocksize : blocksize], normBUFF[61 * blocksize : blocksize],\
                        normBUFF[62 * blocksize : blocksize], normBUFF[63 * blocksize : blocksize], normBUFF[64 * blocksize : blocksize],\
                        normBUFF[65 * blocksize : blocksize], normBUFF[66 * blocksize : blocksize], normBUFF[67 * blocksize : blocksize],\
                        normBUFF[68 * blocksize : blocksize], normBUFF[69 * blocksize : blocksize], normBUFF[70 * blocksize : blocksize],\
                        normBUFF[71 * blocksize : blocksize], normBUFF[72 * blocksize : blocksize], normBUFF[73 * blocksize : blocksize],\
                        normBUFF[74 * blocksize : blocksize], normBUFF[75 * blocksize : blocksize], normBUFF[76 * blocksize : blocksize],\
                        normBUFF[77 * blocksize : blocksize], normBUFF[78 * blocksize : blocksize], normBUFF[79 * blocksize : blocksize],\
                        normBUFF[80 * blocksize : blocksize], normBUFF[81 * blocksize : blocksize], normBUFF[82 * blocksize : blocksize],\
                        normBUFF[83 * blocksize : blocksize], normBUFF[84 * blocksize : blocksize], normBUFF[85 * blocksize : blocksize],\
                        normBUFF[86 * blocksize : blocksize], normBUFF[87 * blocksize : blocksize], normBUFF[88 * blocksize : blocksize],\
                        normBUFF[89 * blocksize : blocksize], normBUFF[90 * blocksize : blocksize], normBUFF[91 * blocksize : blocksize],\
                        normBUFF[92 * blocksize : blocksize], normBUFF[93 * blocksize : blocksize], normBUFF[94 * blocksize : blocksize],\
                        normBUFF[95 * blocksize : blocksize], normBUFF[96 * blocksize : blocksize], normBUFF[97 * blocksize : blocksize],\
                        normBUFF[98 * blocksize : blocksize], normBUFF[99 * blocksize : blocksize], normBUFF[100 * blocksize : blocksize],\
                        normBUFF[101 * blocksize : blocksize], normBUFF[102 * blocksize : blocksize], normBUFF[103 * blocksize : blocksize],\
                        normBUFF[104 * blocksize : blocksize], normBUFF[105 * blocksize : blocksize], normBUFF[106 * blocksize : blocksize],\
                        normBUFF[107 * blocksize : blocksize], normBUFF[108 * blocksize : blocksize], normBUFF[109 * blocksize : blocksize],\
                        normBUFF[110 * blocksize : blocksize], normBUFF[111 * blocksize : blocksize], normBUFF[112 * blocksize : blocksize],\
                        normBUFF[113 * blocksize : blocksize], normBUFF[114 * blocksize : blocksize], normBUFF[115 * blocksize : blocksize],\
                        normBUFF[116 * blocksize : blocksize], normBUFF[117 * blocksize : blocksize], normBUFF[118 * blocksize : blocksize],\
                        normBUFF[119 * blocksize : blocksize], normBUFF[120 * blocksize : blocksize], normBUFF[121 * blocksize : blocksize],\
                        normBUFF[122 * blocksize : blocksize], normBUFF[123 * blocksize : blocksize], normBUFF[124 * blocksize : blocksize],\
                        normBUFF[125 * blocksize : blocksize], normBUFF[126 * blocksize : blocksize], normBUFF[127 * blocksize : blocksize])
#else
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, normBUFF)\
                depend(in: nthrds)\
                depend(inout: normBUFF[0 * blocksize : blocksize], normBUFF[1 * blocksize : blocksize], normBUFF[2 * blocksize : blocksize],\
                        normBUFF[3 * blocksize : blocksize], normBUFF[4 * blocksize : blocksize], normBUFF[5 * blocksize : blocksize], normBUFF[6 * blocksize : blocksize],\
                        normBUFF[7 * blocksize : blocksize], normBUFF[8 * blocksize : blocksize], normBUFF[9 * blocksize : blocksize], normBUFF[10 * blocksize : blocksize],\
                        normBUFF[11 * blocksize : blocksize], normBUFF[12 * blocksize : blocksize], normBUFF[13 * blocksize : blocksize])
#endif
                {
                    tstart = omp_get_wtime();
                    for(i = 0 ; i < nthrds * blocksize; ++i)
                    {
                        normBUFF[i] = 0.0;
                    }
                }
                
#ifdef EPYC
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, QpZBUFF)\
                depend(in: nthrds)\
                depend(inout: QpZBUFF[0 * blocksize : eig_wanted], QpZBUFF[1 * blocksize : eig_wanted], QpZBUFF[2 * blocksize : eig_wanted],\
                        QpZBUFF[3 * blocksize : eig_wanted], QpZBUFF[4 * blocksize : eig_wanted], QpZBUFF[5 * blocksize : eig_wanted], QpZBUFF[6 * blocksize : eig_wanted],\
                        QpZBUFF[7 * blocksize : eig_wanted], QpZBUFF[8 * blocksize : eig_wanted], QpZBUFF[9 * blocksize : eig_wanted], QpZBUFF[10 * blocksize : eig_wanted],\
                        QpZBUFF[11 * blocksize : eig_wanted], QpZBUFF[12 * blocksize : eig_wanted], QpZBUFF[13 * blocksize : eig_wanted],\
                        QpZBUFF[14 * blocksize : eig_wanted], QpZBUFF[15 * blocksize : eig_wanted], QpZBUFF[16 * blocksize : eig_wanted],\
                        QpZBUFF[17 * blocksize : eig_wanted], QpZBUFF[18 * blocksize : eig_wanted], QpZBUFF[19 * blocksize : eig_wanted],\
                        QpZBUFF[20 * blocksize : eig_wanted], QpZBUFF[21 * blocksize : eig_wanted], QpZBUFF[22 * blocksize : eig_wanted],\
                        QpZBUFF[23 * blocksize : eig_wanted], QpZBUFF[24 * blocksize : eig_wanted], QpZBUFF[25 * blocksize : eig_wanted],\
                        QpZBUFF[26 * blocksize : eig_wanted], QpZBUFF[27 * blocksize : eig_wanted], QpZBUFF[28 * blocksize : eig_wanted],\
                        QpZBUFF[29 * blocksize : eig_wanted], QpZBUFF[30 * blocksize : eig_wanted], QpZBUFF[31 * blocksize : eig_wanted],\
                        QpZBUFF[32 * blocksize : eig_wanted], QpZBUFF[33 * blocksize : eig_wanted], QpZBUFF[34 * blocksize : eig_wanted],\
                        QpZBUFF[35 * blocksize : eig_wanted], QpZBUFF[36 * blocksize : eig_wanted], QpZBUFF[37 * blocksize : eig_wanted],\
                        QpZBUFF[38 * blocksize : eig_wanted], QpZBUFF[39 * blocksize : eig_wanted], QpZBUFF[40 * blocksize : eig_wanted],\
                        QpZBUFF[41 * blocksize : eig_wanted], QpZBUFF[42 * blocksize : eig_wanted], QpZBUFF[43 * blocksize : eig_wanted],\
                        QpZBUFF[44 * blocksize : eig_wanted], QpZBUFF[45 * blocksize : eig_wanted], QpZBUFF[46 * blocksize : eig_wanted],\
                        QpZBUFF[47 * blocksize : eig_wanted], QpZBUFF[48 * blocksize : eig_wanted], QpZBUFF[49 * blocksize : eig_wanted],\
                        QpZBUFF[50 * blocksize : eig_wanted], QpZBUFF[51 * blocksize : eig_wanted], QpZBUFF[52 * blocksize : eig_wanted],\
                        QpZBUFF[53 * blocksize : eig_wanted], QpZBUFF[54 * blocksize : eig_wanted], QpZBUFF[55 * blocksize : eig_wanted],\
                        QpZBUFF[56 * blocksize : eig_wanted], QpZBUFF[57 * blocksize : eig_wanted], QpZBUFF[58 * blocksize : eig_wanted],\
                        QpZBUFF[59 * blocksize : eig_wanted], QpZBUFF[60 * blocksize : eig_wanted], QpZBUFF[61 * blocksize : eig_wanted],\
                        QpZBUFF[62 * blocksize : blocksize], QpZBUFF[63 * blocksize : blocksize], QpZBUFF[64 * blocksize : blocksize],\
                        QpZBUFF[65 * blocksize : blocksize], QpZBUFF[66 * blocksize : blocksize], QpZBUFF[67 * blocksize : blocksize],\
                        QpZBUFF[68 * blocksize : blocksize], QpZBUFF[69 * blocksize : blocksize], QpZBUFF[70 * blocksize : blocksize],\
                        QpZBUFF[71 * blocksize : blocksize], QpZBUFF[72 * blocksize : blocksize], QpZBUFF[73 * blocksize : blocksize],\
                        QpZBUFF[74 * blocksize : blocksize], QpZBUFF[75 * blocksize : blocksize], QpZBUFF[76 * blocksize : blocksize],\
                        QpZBUFF[77 * blocksize : blocksize], QpZBUFF[78 * blocksize : blocksize], QpZBUFF[79 * blocksize : blocksize],\
                        QpZBUFF[80 * blocksize : blocksize], QpZBUFF[81 * blocksize : blocksize], QpZBUFF[82 * blocksize : blocksize],\
                        QpZBUFF[83 * blocksize : blocksize], QpZBUFF[84 * blocksize : blocksize], QpZBUFF[85 * blocksize : blocksize],\
                        QpZBUFF[86 * blocksize : blocksize], QpZBUFF[87 * blocksize : blocksize], QpZBUFF[88 * blocksize : blocksize],\
                        QpZBUFF[89 * blocksize : blocksize], QpZBUFF[90 * blocksize : blocksize], QpZBUFF[91 * blocksize : blocksize],\
                        QpZBUFF[92 * blocksize : blocksize], QpZBUFF[93 * blocksize : blocksize], QpZBUFF[94 * blocksize : blocksize],\
                        QpZBUFF[95 * blocksize : blocksize], QpZBUFF[96 * blocksize : blocksize], QpZBUFF[97 * blocksize : blocksize],\
                        QpZBUFF[98 * blocksize : blocksize], QpZBUFF[99 * blocksize : blocksize], QpZBUFF[100 * blocksize : blocksize],\
                        QpZBUFF[101 * blocksize : blocksize], QpZBUFF[102 * blocksize : blocksize], QpZBUFF[103 * blocksize : blocksize],\
                        QpZBUFF[104 * blocksize : blocksize], QpZBUFF[105 * blocksize : blocksize], QpZBUFF[106 * blocksize : blocksize],\
                        QpZBUFF[107 * blocksize : blocksize], QpZBUFF[108 * blocksize : blocksize], QpZBUFF[109 * blocksize : blocksize],\
                        QpZBUFF[110 * blocksize : blocksize], QpZBUFF[111 * blocksize : blocksize], QpZBUFF[112 * blocksize : blocksize],\
                        QpZBUFF[113 * blocksize : blocksize], QpZBUFF[114 * blocksize : blocksize], QpZBUFF[115 * blocksize : blocksize],\
                        QpZBUFF[116 * blocksize : blocksize], QpZBUFF[117 * blocksize : blocksize], QpZBUFF[118 * blocksize : blocksize],\
                        QpZBUFF[119 * blocksize : blocksize], QpZBUFF[120 * blocksize : blocksize], QpZBUFF[121 * blocksize : blocksize],\
                        QpZBUFF[122 * blocksize : blocksize], QpZBUFF[123 * blocksize : blocksize], QpZBUFF[124 * blocksize : blocksize],\
                        QpZBUFF[125 * blocksize : blocksize], QpZBUFF[126 * blocksize : blocksize], QpZBUFF[127 * blocksize : blocksize])
#else
                #pragma omp task private(i, tstart, tend)\
                firstprivate(nthrds, QpZBUFF)\
                depend(in: nthrds)\
                depend(inout: QpZBUFF[0 * blocksize : eig_wanted], QpZBUFF[1 * blocksize : eig_wanted], QpZBUFF[2 * blocksize : eig_wanted],\
                        QpZBUFF[3 * blocksize : eig_wanted], QpZBUFF[4 * blocksize : eig_wanted], QpZBUFF[5 * blocksize : eig_wanted], QpZBUFF[6 * blocksize : eig_wanted],\
                        QpZBUFF[7 * blocksize : eig_wanted], QpZBUFF[8 * blocksize : eig_wanted], QpZBUFF[9 * blocksize : eig_wanted], QpZBUFF[10 * blocksize : eig_wanted],\
                        QpZBUFF[11 * blocksize : eig_wanted], QpZBUFF[12 * blocksize : eig_wanted], QpZBUFF[13 * blocksize : eig_wanted])
#endif
                {
                    tstart = omp_get_wtime();
                    for(i = 0; i < nthrds * eig_wanted; ++i)
                    {
                        QpZBUFF[i] = 0.0;
                    }
                }
                for(structIterator = 0 ; structIterator < taskCount; ++structIterator) 
                {
                    if(taskInfo[structIterator].opCode == 19) // taskName starts SETZERO 
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        if(task_id == 1)
                        {
                            i = block_id * block_width; // starting point of the block
                            blksz = block_width;
                            if(i + blksz > numcols)
                            {
                                blksz = numcols - i;
                            }
                                
                            #pragma omp task default(shared) private(j, tstart, tend)\
                            firstprivate(z, blksz, i, numrows, block_width)\
                            depend(inout : z[i : blksz])
                            {
                                for(j = i; j < i + blksz; ++j)
                                {
                                    z[j] = 0.0;
                                }
                            }
                        }
                    }
                    else if(taskInfo[structIterator].opCode == 7) // taskName starts DLACPY
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        i = block_id * block_width; // starting point of the block
                        blksz = block_width;
                        if(i + blksz > numcols)
                        {
                            blksz = numcols - i;
                        }
                        #pragma omp task default(shared) private(j, tstart, tend)\
                        firstprivate( qq,Q,blksz, i, numrows, block_width)\
                        depend(in : qq[i:blksz])\
                        depend(out : Q[i*eig_wanted : blksz*eig_wanted])
                        {
                            for(j = i; j < i + blksz; ++j)
                            {
                                Q[j*eig_wanted+iterationNumber+1] = qq[j];
                            }
                        } //end task
                    }
                    else if(taskInfo[structIterator].opCode == 23) //SPMV
                    {
                        row_id = taskInfo[structIterator].numParamsList[0]; 
                        col_id = taskInfo[structIterator].numParamsList[1]; 
                        //buf_id = taskInfo[structIterator].numParamsList[2]; 
                        spmv_blkcoord_task(numcols, z, matrixBlock, qq, row_id, col_id, block_width);
                    }
                        
                    else if(taskInfo[structIterator].opCode == 21) // task name without , in it ==> opCode = 21 
                    {
                        if(!strcmp(taskInfo[structIterator].strParamsList[0], "NORM")) 
                        {
                            //norm_task_kk(normBUFF, numcols, beta, iterationNumber, eig_wanted);
                            norm_task(normBUFF, beta, iterationNumber);
                        }
                    }
                    
                    else if(taskInfo[structIterator].opCode == 27) // taskName DAXPY
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        divide_task(z, qq, beta, numcols, block_id, block_width, iterationNumber);
                    }

                    else if(taskInfo[structIterator].opCode == 9) // taskName SUB
                    {
                        block_id = taskInfo[structIterator].numParamsList[0]; 
                        task_id = taskInfo[structIterator].taskID;
                        sub_task(z, QQpZ, z, numcols, block_id, block_width);
                    }

                    else if(taskInfo[structIterator].opCode == 28) // taskName DGEMV
                    {
                        block_id = taskInfo[structIterator].numParamsList[0];
                        buf_id = taskInfo[structIterator].numParamsList[1];
                        task_id = taskInfo[structIterator].taskID;
                        if (task_id == 1 )
                        {
                            //dgemv_task_xty(Q, z, QpZBUFF, numcols, eig_wanted, 1, block_width, block_id, buf_id);
                            _XTY_v1_exe(Q, z, QpZBUFF, numcols, eig_wanted, 1, block_width, block_id, buf_id);
                        }
                        else if(task_id == 2)
                        {
                            //dgemv_task_xy(Q, QpZ, QQpZ ,numcols, eig_wanted, 1, block_width, block_id);
                            _XY_exe(Q, QpZ, QQpZ ,numcols, eig_wanted, 1, block_width, block_id);
                        }
                    }

                    else if(taskInfo[structIterator].opCode == 29) // taskName DOTV
                    {
                        block_id = taskInfo[structIterator].numParamsList[0];
                        buf_id = taskInfo[structIterator].numParamsList[1];
                        task_id = taskInfo[structIterator].taskID;
                        dotV(z, numcols , normBUFF, block_id, buf_id, block_width);			
                    }

                    else if(taskInfo[structIterator].opCode == 3) // taskName starts XTY 
                    {
                        block_id = taskInfo[structIterator].numParamsList[0];
                        buf_id = taskInfo[structIterator].numParamsList[1];
                        task_id = taskInfo[structIterator].taskID;
                        if(task_id == 1)
                        {
                            _XTY_v1_exe(qq, z, AlphaBUFF,numcols, 1, 1, block_width, block_id, buf_id);
                        }
                    }

                    else if(taskInfo[structIterator].opCode == 4) // RED
                    {
                        if(!strcmp(taskInfo[structIterator].strParamsList[0], "alpha")) //xty 1 reduction
                        {
                            //_XTY_v1_RED(AlphaBUFF, (alpha+iterationNumber), 1, 1, block_width);
                            reduce_task(AlphaBUFF, alpha, iterationNumber);
                        }
                        else if(!strcmp(taskInfo[structIterator].strParamsList[0], "QpZ")) //xty 2 reduction
                        {
                            //RED_QpZ(QpZBUFF, QpZ, eig_wanted, 1, block_width);
                            _XTY_v1_RED(QpZBUFF, QpZ, eig_wanted, 1, block_width);
                        }
                    }
                    else if(taskInfo[structIterator].opCode != 22) // undefined taskName
                    {
                        printf("NANI? WRONG opCOde %d\n", taskInfo[structIterator].opCode);
                        exit(1);
                    }
                }
                #pragma omp taskwait
                tend = omp_get_wtime();
                printf("%.4lf,",tend-tstart);
                total_time += (tend - tstart);
            }
        }
    }

    printf("%.4lf\n",total_time/eig_wanted);

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");
    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", beta[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");

    LAPACKE_dsterf(eig_wanted,alpha,beta);

    for(i = 0; i < eig_wanted; ++i)
    {
        printf("%.4lf", alpha[i]);
        if(i != eig_wanted - 1)
            printf(",");
    }
    printf("\n");

    //deallocation
    for(i = 0; i < nrowblks; ++i)
    {
        for(j = 0; j < ncolblks; ++j)
        {
            if(matrixBlock[i * ncolblks + j].nnz > 0)
            {
                delete [] matrixBlock[i * ncolblks + j].rloc;
                delete [] matrixBlock[i * ncolblks + j].cloc;
                delete [] matrixBlock[i * ncolblks + j].val;
            }
        }
    }
    delete [] matrixBlock;


    free(q);
    free(qq);
    free(z);
    free(alpha);
    free(beta);
    free(QpZ);
    free(QQpZ);
    free(Q);

    free(QpZBUFF);
    free(AlphaBUFF);
    free(normBUFF);

    return 0;
}
