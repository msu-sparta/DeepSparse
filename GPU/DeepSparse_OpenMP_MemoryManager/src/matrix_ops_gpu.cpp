#ifndef MATRIX_OPS_GPU_H
#define MATRIX_OPS_GPU_H
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <fstream>
#include <utility> 
using namespace std;

#include <omp.h>
#include "../inc/util.h"
#include "../inc/matrix_ops_gpu_v6.h"
#include "../inc/memory_manager_v6.h"

#if defined(USE_CUBLAS) || defined(USE_DEVICE)

// =========================== Memory Manager Managed Kernels ===========================

void SPMM_GPU_MM(int *row_ptr, int *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, long iterationNumber)
                         
{
    //This is not used in the main code
    // Z = A * X ==> A[numrows * numcols] Y[numrows * nvec] Z[numrows * nvec], A is the sparse matrix
    
    int blksz = block_width;
    int offset = block_id * block_width;

    if(offset + blksz > numrows)
        blksz = numrows - offset;
    
    int i, j, k, start, end;
    int r, c;
    double xcoef;

    int status = -1;
    pair<double *, int> prZ = make_pair(Z, block_id);

    if(!isOnDevice(prZ))
    {
        status = copyToDevice(Z, numrows, nvec, block_id, block_id * block_width * nvec, blksz * nvec, 1.0, iterationNumber);
        if(status != 0)
        {
            // cout << "Z copy is failed on SpMM :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    unsigned long int z_offset = (unsigned long int) mp[prZ][0];

    #pragma omp target firstprivate(block_id, block_width, nvec, offset, z_offset) is_device_ptr(d_Y, d_memory)\
    map(to: row_ptr[offset : blksz + 1], col_index[row_ptr[offset] : nnz_per_tile[block_id]], value[row_ptr[offset] : nnz_per_tile[block_id]])\
    depend(in: row_ptr[offset : blksz + 1], col_index[row_ptr[offset] : nnz_per_tile[block_id]], value[row_ptr[offset] : nnz_per_tile[block_id]])\
    depend(in: d_Y[0 : numrows * nvec]) depend(inout: Z[offset * nvec : blksz  * nvec])
    #pragma omp teams distribute parallel for private(start, end, r, c, xcoef, i, j, k) firstprivate(block_id, block_width, nvec, offset, nnz_per_tile, row_ptr, col_index, value, d_Y, Z)
    for(i = 0 ; i < blksz ; i++)
    {
        start = row_ptr[offset + i];
        end = row_ptr[offset + i + 1];
        for(j = start ; j < end ; j++)
        {
            r = offset + i;
            c = col_index[j];
            xcoef = value[j];  
            #pragma omp simd
            for(k = 0 ; k < nvec ; k++)
            {
                d_memory[z_offset + (r - offset) * nvec + k] = d_memory[z_offset + (r - offset) * nvec + k] + xcoef * d_Y[c * nvec + k];
            }
        }
    }
}

void SPMM_GPU_MM_v1(int *row_ptr, int *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, int *ia, int *ja, long iterationNumber)
                         
{
    // Z = A * X ==> A[numrows * numcols] Y[numrows * nvec] Z[numrows * nvec], A is the sparse matrix

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "SpMM: blockid - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    int blksz = block_width;
    int offset = block_id * block_width;

    if(offset + blksz > numrows)
        blksz = numrows - offset;
    
    int i, j, k, start, end;
    int r, c;
    double xcoef;

    double tstart = omp_get_wtime();

    int status = -1;
    pair<double *, int> prZ = make_pair(Z, block_id);
    pair<double *, int> prA = make_pair(value, block_id);

    if(!isOnDevice(prZ))
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && evictionQueue.front() == prA)
            {
                cout << "   SpMM: YES prA -- cann't be evicted" << endl;
                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(Z, numrows, nvec, block_id, block_id * block_width * nvec, blksz * nvec, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SpMM: Z copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prZ][5] = 1.0;
    }
    
    unsigned long int z_offset = (unsigned long int) mp[prZ][0] * num_per_blk;
    int value_offset = ia[offset];
    
    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << endl;
    #endif
    // map(value[value_offset : nnz_per_tile[block_id]])
    if(!isOnDevice(prA))
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && evictionQueue.front() == prZ)
            {
                cout << "   SpMM: YES prA -- cann't be evicted" << endl;
                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(value, 1, nnonzero, block_id, value_offset, nnz_per_tile[block_id], 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SpMM: Sparse matrix copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prA][5] = 0.0;
    }
    unsigned long int v_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    
    taskTiming[iterationNumber - 1][1][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    // map(value[value_offset : nnz_per_tile[block_id]])

    #pragma omp target is_device_ptr(d_Y, d_memory, row_ptr, col_index)\
    firstprivate(block_id, block_width, nvec, offset, nnz_per_tile, z_offset, value_offset, v_offset)\
    depend(in: d_Y[0 : numrows * nvec]) depend(inout: Z[offset * nvec : blksz  * nvec])
    #pragma omp teams distribute parallel for private(start, end, r, c, xcoef, i, j, k)
    for(i = 0 ; i < blksz ; i++)
    {
        start = row_ptr[offset + i];
        end = row_ptr[offset + i + 1];
        for(j = start ; j < end ; j++)
        {
            r = offset + i;
            c = col_index[j];
            // xcoef = value[j];
            xcoef = d_memory[v_offset + (j - value_offset)]; //value[j];  
            #pragma omp simd
            for(k = 0 ; k < nvec ; k++)
            {
                d_memory[z_offset + (r - offset) * nvec + k] = d_memory[z_offset + (r - offset) * nvec + k] + xcoef * d_Y[c * nvec + k];
            }
        }
    }
}

void SPMM_GPU_MM_v3(int *row_ptr, double *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, int *ia, int *ja, long iterationNumber)
                         
{
    // Z = A * X ==> A[numrows * numcols] Y[numrows * nvec] Z[numrows * nvec], A is the sparse matrix

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "SpMM: blockid - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    int blksz = block_width;
    int offset = block_id * block_width;

    if(offset + blksz > numrows)
        blksz = numrows - offset;
    
    int i, j, k, start, end;
    int r, c;
    double xcoef;

    double tstart = omp_get_wtime();

    int status = -1;
    pair<double *, int> prZ = make_pair(Z, block_id);
    pair<double *, int> prA = make_pair(value, block_id);
    // pair<double *, int> prRow = make_pair(row_ptr, block_id);
    pair<double *, int> prCol = make_pair(col_index, block_id);
    pair<double *, int> tempPair;

    if(!isOnDevice(prZ)) // output --> required 1 single block in V5
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && (evictionQueue.front() == prA ||  evictionQueue.front() == prCol)) // || evictionQueue.front() == prRow 
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   SpMM: YES prA -- cann't be evicted" << endl;
                #endif

                tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(Z, numrows, nvec, block_id, block_id * block_width * nvec, blksz * nvec, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SpMM: Z copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prZ][5] = 1.0;
    }
    
    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << endl;
    #endif

    unsigned long int z_offset = (unsigned long int) mp[prZ][0] * num_per_blk;
    int value_offset = ia[offset];
    unsigned long int z_index = (unsigned long int) mp[prZ][0];
    unsigned long int required_blk, top_index;
    
    // map(value[value_offset : nnz_per_tile[block_id]])
    if(!isOnDevice(prA))
    {

        // if(inEvictionMode)
        // {
        //     while(!evictionQueue.empty() && (evictionQueue.front() == prZ || evictionQueue.front() == prCol)) //  || evictionQueue.front() == prRow
        //     {
        //         cout << "   SpMM: YES prZ/prRow/prCol -- cann't be evicted" << endl;
        //         pair<double *, int> tempPair = evictionQueue.front();
        //         evictionQueue.pop();
        //         evictionQueue.push(tempPair);
        //     }
        // }

        required_blk = requiredDeviceBlocks(nnz_per_tile[block_id] * sizeof(double), memGranularity);
        tempPair = evictionQueue.front();
        // top_index = (unsigned long int)mp[tempPair][0];
        // cout << "   SpMM: prA required_blk : " << required_blk << " block_id : " << block_id << " iterationNumber: " << iterationNumber << endl;

        if(inEvictionMode)
        {
            while(!isOnDevice(tempPair) || (!evictionQueue.empty() && (tempPair == prZ || tempPair == prCol ||
                availableBlocksOnRight((unsigned long int)mp[tempPair][0]) < required_blk || (z_index >= (unsigned long int)mp[tempPair][0] && z_index <= (unsigned long int)mp[tempPair][0] + required_blk - 1)))) //  || evictionQueue.front() == prRow
            {
                // cout << "   SpMM: YES prZ/prRow/prCol -- cann't be evicted" << endl;
                
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
                tempPair = evictionQueue.front();
                // top_index = (unsigned long int)mp[tempPair][0];
            }
        }

        status = copyToDevice(value, 1, nnonzero, block_id, value_offset, nnz_per_tile[block_id], 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SpMM: Sparse matrix copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prA][5] = 0.0;
    }

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << endl;
    #endif

    unsigned long int v_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    unsigned long int v_index = (unsigned long int) mp[prA][0];

    // cout << "here : v_index - " << v_index << endl;
    // move col_ptr to device
    if(!isOnDevice(prCol))
    {
        // if(inEvictionMode)
        // {
        //     while(!evictionQueue.empty() && (evictionQueue.front() == prZ ||  evictionQueue.front() == prA)) //  || evictionQueue.front() == prRow 
        //     {
        //         cout << "   SpMM: YES prZ/prRow/prCol -- cann't be evicted" << endl;
        //         pair<double *, int> tempPair = evictionQueue.front();
        //         evictionQueue.pop();
        //         evictionQueue.push(tempPair);
        //     }
        // }
        // cout << "here : calculating required_blk" << endl;
        required_blk = requiredDeviceBlocks(nnz_per_tile[block_id] * sizeof(double), memGranularity);
        // cout << "here : required_blk - " << required_blk << endl;
        tempPair = evictionQueue.front();
        // if(!isOnDevice(tempPair))
        //     cout << "not on device - top: (" << matrixName[tempPair.first] << ", " << tempPair.second << ") - " << top_index << endl;
        // top_index = (unsigned long int)mp[tempPair][0];

        // cout << "   SpMM: prCol required_blk : " << required_blk << " block_id : " << block_id << " iterationNumber: " << iterationNumber << " top: (" << matrixName[tempPair.first] << ", " << tempPair.second << ")" << endl;
        
        if(inEvictionMode)
        {
            while(!isOnDevice(tempPair) || (!evictionQueue.empty() && (tempPair == prZ ||  tempPair == prA ||
                availableBlocksOnRight((unsigned long int)mp[tempPair][0]) < required_blk || (z_index >= (unsigned long int)mp[tempPair][0] && z_index <= (unsigned long int)mp[tempPair][0] + required_blk - 1) || 
               (v_index >= (unsigned long int)mp[tempPair][0] && v_index <= (unsigned long int)mp[tempPair][0] + required_blk - 1)))) //  || evictionQueue.front() == prRow
            {
                // cout << "   SpMM: YES prZ/prRow/prCol -- cann't be evicted" << endl;
                
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
                tempPair = evictionQueue.front();
                // top_index = (unsigned long int)mp[tempPair][0];
            }
        }

        status = copyToDevice(col_index, 1, nnonzero, block_id, value_offset, nnz_per_tile[block_id], 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SpMM: Sparse matrix copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
            cout << "   SpMM: prCol is on device : block_id : " << block_id << " iterationNumber: " << iterationNumber << endl;
        #endif

        mp[prCol][5] = 0.0;
    }
    unsigned long int c_offset = (unsigned long int) mp[prCol][0] * num_per_blk;

    // move row_ptr to device

    // unsigned long int row_offset = offset;
    // if(!isOnDevice(prRow))
    // {
    //     if(inEvictionMode)
    //     {
    //         while(!evictionQueue.empty() && (evictionQueue.front() == prZ || evictionQueue.front() == prA || evictionQueue.front() == prCol))
    //         {
    //             cout << "   SpMM: YES prZ/prRow/prCol -- cann't be evicted" << endl;
    //             pair<double *, int> tempPair = evictionQueue.front();
    //             evictionQueue.pop();
    //             evictionQueue.push(tempPair);
    //         }
    //     }

    //     status = copyToDevice(row_ptr, 1, numrows + 1, block_id, row_offset, blksz + 1, 0.0);
    //     if(status != 0)
    //     {
    //         cout << "   SpMM: Sparse matrix copy is failed on block_id : " << block_id << endl;
    //         errorFlag = true;
    //     }
    // }
    // else
    // {
    //     mp[prRow][5] = 0.0;
    // }
    // unsigned long int r_offset = (unsigned long int) mp[prRow][0] * num_per_blk;

    // cudaDeviceSynchronize();
    
    taskTiming[iterationNumber - 1][1][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prZ)) //this is output, 
    { 
        cout << "actAR is not on device"  << " block_id : " << block_id << " iterationNumber: " << iterationNumber << endl;
    }
    if(!isOnDevice(prA)) //full matrixB needs to be on device memory
    { 
        cout << "acsr is not on device"  << " block_id : " << block_id << " iterationNumber: " << iterationNumber << endl;
    }
    if(!isOnDevice(prCol)) //full matrixB needs to be on device memory
    { 
        cout << "col_index is not on device"  << " block_id : " << block_id << " iterationNumber: " << iterationNumber << endl;
    }
    #endif

    // map(value[value_offset : nnz_per_tile[block_id]])

    // map(to: row_ptr[offset : blksz + 1], col_index[row_ptr[offset] : nnz_per_tile[block_id]], value[row_ptr[offset] : nnz_per_tile[block_id]])

    #pragma omp target is_device_ptr(d_Y, d_memory, row_ptr)\
    firstprivate(block_id, block_width, nvec, offset, nnz_per_tile, z_offset, value_offset, v_offset, c_offset)\
    depend(in: d_Y[0 : numrows * nvec]) depend(inout: Z[offset * nvec : blksz  * nvec])
    #pragma omp teams distribute parallel for private(start, end, r, c, xcoef, i, j, k)
    for(i = 0 ; i < blksz ; i++)
    {
        start = row_ptr[offset + i];
        end = row_ptr[offset + i + 1];
        for(j = start ; j < end ; j++)
        {
            r = offset + i;
            c = (int) d_memory[c_offset + (j - value_offset)]; // col_index[j];
            // xcoef = value[j];
            xcoef = d_memory[v_offset + (j - value_offset)]; //value[j];  
            #pragma omp simd
            for(k = 0 ; k < nvec ; k++)
            {
                d_memory[z_offset + (r - offset) * nvec + k] = d_memory[z_offset + (r - offset) * nvec + k] + xcoef * d_Y[c * nvec + k];
            }
        }
    }
}

void SPMM_GPU_MM_v2(int *row_ptr, int *col_index, double *d_xrem, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile, double *d_memory, int *ia, int *ja, double *acsr, long iterationNumber)
                         
{
    // Z = A * X ==> A[numrows * numcols] Y[numrows * nvec] Z[numrows * nvec], A is the sparse matrix
    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "SpMM: blockid - " << block_id << endl;
    #endif

    int blksz = block_width;
    int offset = block_id * block_width;

    if(offset + blksz > numrows)
        blksz = numrows - offset;
    
    int i, j, k, start, end;
    int r, c;
    double xcoef;

    int status = -1;
    pair<double *, int> prZ = make_pair(Z, block_id);
    // pair<double *, int> prA = make_pair(value, block_id);
    
    if(!isOnDevice(prZ))
    {
        // if(inEvictionMode)
        // {
        //     while(!evictionQueue.empty() && evictionQueue.front() == prA)
        //     {
        //         cout << "   SpMM: YES prA -- cann't be evicted" << endl;
        //         pair<double *, int> tempPair = evictionQueue.front();
        //         evictionQueue.pop();
        //         evictionQueue.push(tempPair);
        //     }
        // }

        status = copyToDevice(Z, numrows, nvec, block_id, block_id * block_width * nvec, blksz * nvec, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SpMM: Z copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prZ][5] = 1.0;
    }
    
    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << endl;
    #endif

    unsigned long int z_offset = (unsigned long int) mp[prZ][0] * num_per_blk;
    int value_offset = ia[offset];
    // cudaDeviceSynchronize();
    // #pragma omp taskwait
    status = omp_target_memcpy(d_xrem, acsr + ia[offset], nnz_per_tile[block_id] * sizeof(int), 0, 0, device_id, host_id);
    if( status != 0 ){ printf("omp_target_memcpy failed d_xrem in SpMM"); errorFlag = true; }
    // map(value[value_offset : nnz_per_tile[block_id]])
    // if(!isOnDevice(prA))
    // {
    //     if(inEvictionMode)
    //     {
    //         while(!evictionQueue.empty() && evictionQueue.front() == prZ)
    //         {
    //             cout << "   SpMM: YES prA -- cann't be evicted" << endl;
    //             pair<double *, int> tempPair = evictionQueue.front();
    //             evictionQueue.pop();
    //             evictionQueue.push(tempPair);
    //         }
    //     }

    //     status = copyToDevice(value, 1, nnonzero, block_id, value_offset, nnz_per_tile[block_id], 0.0);
    //     if(status != 0)
    //     {
    //         cout << "   SpMM: Sparse matrix copy is failed on block_id : " << block_id << endl;
    //         errorFlag = true;
    //     }
    // }
    // else
    // {
    //     mp[prA][5] = 0.0;
    // }
    // unsigned long int v_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    
    // map(value[value_offset : nnz_per_tile[block_id]])

    #pragma omp target is_device_ptr(d_Y, d_xrem, d_memory, row_ptr, col_index)\
    firstprivate(block_id, block_width, nvec, offset, nnz_per_tile, z_offset, value_offset)\
    depend(in: d_Y[0 : numrows * nvec]) depend(inout: Z[offset * nvec : blksz  * nvec])
    #pragma omp teams distribute parallel for private(start, end, r, c, xcoef, i, j, k)
    for(i = 0 ; i < blksz ; i++)
    {
        start = row_ptr[offset + i];
        end = row_ptr[offset + i + 1];
        for(j = start ; j < end ; j++)
        {
            r = offset + i;
            c = col_index[j];
            xcoef = d_xrem[j - value_offset];
            // xcoef = d_memory[v_offset + (j - value_offset)]; //value[j];  
            #pragma omp simd
            for(k = 0 ; k < nvec ; k++)
            {
                d_memory[z_offset + (r - offset) * nvec + k] = d_memory[z_offset + (r - offset) * nvec + k] + xcoef * d_Y[c * nvec + k];
            }
        }
    }
}


void XTY_GPU_MM(double *X, double *Y, double *result_buf ,int M, int N, int P, int block_width, int block_id, int buf_id, double *d_memory, int task_id, long iterationNumber,
                double alpha, double beta, bool X_deviceptr, bool Y_deviceptr)
{
	// X[M * N] , Y[M * P], but[nbuf * N * P]

	int blksz = block_width;
	if(block_id * block_width + blksz > M)
		blksz = M - block_id * block_width;

	int i, j, k, status, rowOffset = block_id * block_width;
	double total = 0.0;

    double tstart = omp_get_wtime();

    unsigned long int X_offset, Y_offset; 
    pair<double *, int> prX = make_pair(X, block_id); 
    pair<double *, int> prY = make_pair(Y, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "XTY : task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prX) && !X_deviceptr)  
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && evictionQueue.front() == prY))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XTY: YES prY -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(X, M, N, block_id, block_id * block_width * N, blksz * N, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   XTY: X copy is failed on XTY :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }

    if(!isOnDevice(prY) && !Y_deviceptr)  
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && evictionQueue.front() == prX))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XTY: YES prX -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(Y, M, P, block_id, block_id * block_width * P, blksz * P, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   XTY: Y copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    
    if(!X_deviceptr)
        X_offset = (unsigned long int) mp[prX][0] * num_per_blk;
    if(!Y_deviceptr)
        Y_offset = (unsigned long int) mp[prY][0] * num_per_blk;
    
    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prX) && !X_deviceptr) //this is output, 
    { 
        cout << "prX is not on device"  << " block_id : " << block_id << endl;
    }
    if(!isOnDevice(prY) && !Y_deviceptr) //full matrixB needs to be on device memory
    { 
        cout << "matrixB is not on device"  << " block_id : 0" << endl;
    }
    #endif

    taskTiming[iterationNumber - 1][2][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    if(X_deviceptr && Y_deviceptr) //both deviceptr
	{
        #pragma omp target is_device_ptr(result_buf, d_memory, X, Y) firstprivate(rowOffset, buf_id, block_id, block_width)\
        depend(in: M, N, P) depend(in: X[rowOffset * N : blksz * N], Y[rowOffset * P : blksz * P])\
        depend(inout: result_buf[buf_id * N * P : N * P])
        // #pragma omp parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        #pragma omp teams distribute parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        for(i = 0 ; i < N ; i++)   
        {
            for(j = 0 ; j < P ; j++) 
            {   
                total = 0.0;
                #pragma omp simd
                for (k = 0 ; k < blksz ; k++) // # of rows in X or Y (should be same)
                {
                    total += X[rowOffset * N + k * N + i] * Y[rowOffset * P + k * P + j];
                }
                result_buf[buf_id * N * P + i * P + j] += total;
            }
        }
    }
    else if(X_deviceptr) //1st one device pointer
	{
        #pragma omp target is_device_ptr(result_buf, d_memory, X) firstprivate(rowOffset, buf_id, block_id, block_width, X_offset, Y_offset)\
        depend(in: M, N, P) depend(in: X[rowOffset * N : blksz * N], Y[rowOffset * P : blksz * P])\
        depend(inout: result_buf[buf_id * N * P : N * P])
        // #pragma omp parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        #pragma omp teams distribute parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        for(i = 0 ; i < N ; i++)   
        {
            for(j = 0 ; j < P ; j++) 
            {   
                total = 0.0;
                for (k = 0 ; k < blksz ; k++) // # of rows in X or Y (should be same)
                {
                    total += X[rowOffset * N + k * N + i] * d_memory[Y_offset + k * P + j];
                }
                result_buf[buf_id * N * P + i * P + j] += total;
            }
        }
    }
    else if(Y_deviceptr) //2nd one device ptr
	{
        #pragma omp target is_device_ptr(result_buf, d_memory, Y) firstprivate(rowOffset, buf_id, block_id, block_width, X_offset, Y_offset)\
        depend(in: M, N, P) depend(in: X[rowOffset * N : blksz * N], Y[rowOffset * P : blksz * P])\
        depend(inout: result_buf[buf_id * N * P : N * P])
        // #pragma omp parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        #pragma omp teams distribute parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        for(i = 0 ; i < N ; i++)   
        {
            for(j = 0 ; j < P ; j++) 
            {   
                total = 0.0;
                #pragma omp simd
                for (k = 0 ; k < blksz ; k++) // # of rows in X or Y (should be same)
                {
                    total += d_memory[X_offset + k * N + i] * Y[rowOffset * P + k * P + j];
                }
                result_buf[buf_id * N * P + i * P + j] += total;
            }
        }
    }
    else
	{
        #pragma omp target is_device_ptr(result_buf, d_memory) firstprivate(buf_id, block_id, block_width, X_offset, Y_offset)\
        depend(in: M, N, P) depend(in: X[rowOffset * N : blksz * N], Y[rowOffset * P : blksz * P])\
        depend(inout: result_buf[buf_id * N * P : N * P])
        // #pragma omp parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        #pragma omp teams distribute parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
        for(i = 0 ; i < N ; i++)   
        {
            for(j = 0 ; j < P ; j++) 
            {   
                total = 0.0;
                #pragma omp simd
                for (k = 0 ; k < blksz ; k++) // # of rows in X or Y (should be same)
                {
                    total += d_memory[X_offset + k * N + i] * d_memory[Y_offset + k * P + j];
                }
                result_buf[buf_id * N * P + i * P + j] += total;
            }
        }
    }	
}

void XTY_GPU_RED_MM(double *buf, double *result, int N, int P, int block_width, double *d_memory, long iterationNumber)
{
    /*
    _XTY_v1_RED: adding partial sums block by block, not row by row
    Input: buf: nthds * [N * P]
    Output: result[N * P]
    nthrds : global variable, total # of threads
    buf : how to free/deallocate corresponding memory location
    */
    
    int i, j, k, l, blksz, tid, nthreads, length;
    double sum, tend;
    
    int nbuf = 128;

    int status, n = N * P;

    double tstart = omp_get_wtime();

    unsigned long int result_offset; 
    pair<double *, int> prResult = make_pair(result, 0);

    if(!isOnDevice(prResult)) //this is the result on device, buf is already in device memory 
    { 
        status = reserveOnDevice(result, N, P, 0, 0, N * P, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   XTYRED: result reservation is failed on XTYRED" << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prResult][5] = 1.0;
    }
    
    result_offset = (unsigned long int) mp[prResult][0] * num_per_blk;

    // mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;    
    taskTiming[0][2][0] += omp_get_wtime() - tstart;

    for(i = 0 ; i < N ; i = i + block_width)
    {
        blksz = block_width;
        if(i + blksz > N)
            blksz = N - i;

        #pragma omp target is_device_ptr(buf, d_memory) private(sum, k, l, j)\
        firstprivate(i, n, nbuf, blksz, block_width, result_offset)\
        depend(in: N, P) depend(out: result[i * P : blksz * P]) depend(in: N, P)\
        depend(in : buf[0 * n : n], buf[1 * n : n], buf[2 * n : n], buf[3 * n : n], buf[4 * n : n], buf[5 * n : n], buf[6 * n : n], buf[7 * n : n], buf[8 * n : n], buf[9 * n : n], buf[10 * n : n],\
        buf[11 * n : n], buf[12 * n : n], buf[13 * n : n], buf[14 * n : n], buf[15 * n : n], buf[16 * n : n], buf[17 * n : n], buf[18 * n : n], buf[19 * n : n], buf[20 * n : n],\
        buf[21 * n : n], buf[22 * n : n], buf[23 * n : n], buf[24 * n : n], buf[25 * n : n], buf[26 * n : n], buf[27 * n : n], buf[28 * n : n], buf[29 * n : n], buf[30 * n : n],\
        buf[31 * n : n], buf[32 * n : n], buf[33 * n : n], buf[34 * n : n], buf[35 * n : n], buf[36 * n : n], buf[37 * n : n], buf[38 * n : n], buf[39 * n : n], buf[40 * n : n],\
        buf[41 * n : n], buf[42 * n : n], buf[43 * n : n], buf[44 * n : n], buf[45 * n : n], buf[46 * n : n], buf[47 * n : n], buf[48 * n : n], buf[49 * n : n], buf[50 * n : n],\
        buf[51 * n : n], buf[52 * n : n], buf[53 * n : n], buf[54 * n : n], buf[55 * n : n], buf[56 * n : n], buf[57 * n : n], buf[58 * n : n], buf[59 * n : n], buf[60 * n : n],\
        buf[61 * n : n], buf[62 * n : n], buf[63 * n : n], buf[64 * n : n], buf[65 * n : n], buf[66 * n : n], buf[67 * n : n], buf[68 * n : n], buf[69 * n : n], buf[70 * n : n],\
        buf[71 * n : n], buf[72 * n : n], buf[73 * n : n], buf[74 * n : n], buf[75 * n : n], buf[76 * n : n], buf[77 * n : n], buf[78 * n : n], buf[79 * n : n], buf[80 * n : n],\
        buf[81 * n : n], buf[82 * n : n], buf[83 * n : n], buf[84 * n : n], buf[85 * n : n], buf[86 * n : n], buf[87 * n : n], buf[88 * n : n], buf[89 * n : n], buf[90 * n : n],\
        buf[91 * n : n], buf[92 * n : n], buf[93 * n : n], buf[94 * n : n], buf[95 * n : n], buf[96 * n : n], buf[97 * n : n], buf[98 * n : n], buf[99 * n : n], buf[100 * n : n],\
        buf[101 * n : n], buf[102 * n : n], buf[103 * n : n], buf[104 * n : n], buf[105 * n : n], buf[106 * n : n], buf[107 * n : n], buf[108 * n : n], buf[109 * n : n], buf[110 * n : n],\
        buf[111 * n : n], buf[112 * n : n], buf[113 * n : n], buf[114 * n : n], buf[115 * n : n], buf[116 * n : n], buf[117 * n : n], buf[118 * n : n], buf[119 * n : n], buf[120 * n : n],\
        buf[121 * n : n], buf[122 * n : n], buf[123 * n : n], buf[124 * n : n], buf[125 * n : n], buf[126 * n : n], buf[127 * n : n])
        //{
        #pragma omp teams distribute parallel for    
            for(l = i ; l < (i + blksz) ; l++) //for each row in the block
            {
                for(k  = 0 ; k < P ; k++) //each col
                {
                    sum = 0.0;
                    #pragma omp simd
                    for(j = 0 ; j < nbuf ; j++) //for each thread access corresponding N*P matrix
                    {
                        sum += buf[j * N * P + l * P + k];
                    }
                    d_memory[result_offset + l * P + k] = sum;
                }
                
            }
        //}// end of task 
    }//end outer for
}

void XY_GPU_MM(double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, 
                                double alpha, double beta, bool A_deviceptr, bool B_deviceptr, bool C_deviceptr)
{
	int i, j, k;
	double total;
    int status = -1;
	int blksz = block_id * block_width + block_width > N ? N - block_id * block_width : block_width;
    int offset = block_id * block_width * M;
    
    int c_offset = block_id * block_width * P;

    double tstart = omp_get_wtime();

    unsigned long int matrixA_offset, matrixB_offset, matrixC_offset; 
    pair<double *, int> prA = make_pair(matrixA, block_id); 
    pair<double *, int> prB = make_pair(matrixB, 0);
    pair<double *, int> prC = make_pair(matrixC, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "XY : " << "task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prA) && !A_deviceptr) //this is input, 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prB || evictionQueue.front() == prC)))
            {
                pair<double *, int> tempPair = evictionQueue.front();
                
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XY: YES 1 -- (" << matrixName[tempPair.first] << ", " << tempPair.second << ") cann't be evicted task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
                #endif

                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(matrixA, N, M, block_id, block_id * block_width * M, blksz * M, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   XY: matrixA copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }

    if(!A_deviceptr)
        matrixA_offset = (unsigned long int) mp[prA][0] * num_per_blk;

    if(!isOnDevice(prB) && !B_deviceptr) //full matrixB needs to be on device memory
    { 
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prC)))
            {
                pair<double *, int> tempPair = evictionQueue.front();

                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XY: YES 2 -- (" << matrixName[tempPair.first] << ", " << tempPair.second << ") cann't be evicted task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
                #endif

                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
                
            }
        }

        status = copyToDevice(matrixB, M, P, 0, 0, M * P, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   XY: matrixB copy is failed on XY :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }

    if(!isOnDevice(prC) && !C_deviceptr) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prB)))
            {
                pair<double *, int> tempPair = evictionQueue.front();

                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XY: YES 3 -- (" << matrixName[tempPair.first] << ", " << tempPair.second << ")  cann't be evicted task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
                #endif 

                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = reserveOnDevice(matrixC, N, P, block_id, block_id * block_width * P, blksz * P, 1.0, iterationNumber); 
        if(status != 0)
        {
            cout << "   XY: matrixC copy is failed on XY :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!C_deviceptr)
    {
        mp[prC][5] = 1.0;
    }
    
    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prA) && !A_deviceptr) //this is output, 
    { 
       
        // cout << "matrixA is not on device"  << " block_id : " << block_id << endl;
        cout << "   XY - A: (" << matrixName[matrixA] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl; 
        errorFlag = true;
    }
    if(!isOnDevice(prB) && !B_deviceptr) //full matrixB needs to be on device memory
    { 
        //cout << "matrixB is not on device"  << " block_id : 0" << endl;
        cout << "   XY - B: (" << matrixName[matrixB] << ", " << 0 << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl;
        errorFlag = true;
    }
    if(!isOnDevice(prC) && !C_deviceptr) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    { 
        // cout << "matrixC is not on device"  << " block_id : " << block_id << endl;
        cout << "XY - C: (" << matrixName[matrixC] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl;
        errorFlag = true;
    }
    #endif

    
    matrixB_offset = (unsigned long int) mp[prB][0] * num_per_blk;
    
    if(!C_deviceptr)
        matrixC_offset = (unsigned long int) mp[prC][0] * num_per_blk;

    taskTiming[iterationNumber - 1][3][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    // depend(in: d_memory[matrixA_offset : blksz * M], d_memory[matrixB_offset : M * P])\
    // depend(inout: d_memory[matrixC_offset : blksz * P])
    // cout << "Lunching XY kernel on GPU -- block_id: " << block_id << endl;

    if(A_deviceptr)
    {
        #pragma omp target firstprivate(block_id, block_width, offset, matrixB_offset, matrixC_offset)\
        is_device_ptr(d_memory, matrixA) depend(in: N, M, P)\
        depend(in: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P])\
        depend(inout: matrixC[block_id * block_width * P : blksz * P])
        #pragma omp teams distribute parallel for private(i, j, k, total) //collapse(2) --> adding collpase() slows down with CCE compiler -- weird
        for(i = 0 ; i < blksz ; i++)    
        {
            for(j = 0 ; j < P ; j++)
            {
                total = 0.0;
                #pragma omp simd
                for (k = 0 ; k < M ; k++)
                {
                    total += matrixA[offset + i * M + k] * d_memory[matrixB_offset + k * P + j];
                }
                d_memory[matrixC_offset + i * P + j] = total + beta * d_memory[matrixC_offset + i * P + j];
            }
        }
    }
    else if(C_deviceptr)
    {
        #pragma omp target firstprivate(block_id, block_width, c_offset, matrixA_offset, matrixB_offset)\
        is_device_ptr(d_memory, matrixC) depend(in: N, M, P)\
        depend(in: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P])\
        depend(inout: matrixC[block_id * block_width * P : blksz * P])
        #pragma omp teams distribute parallel for private(i, j, k, total) //collapse(2) --> adding collpase() slows down with CCE compiler -- weird
        for(i = 0 ; i < blksz ; i++)    
        {
            for(j = 0 ; j < P ; j++)
            {
                total = 0.0;
                #pragma omp simd
                for (k = 0 ; k < M ; k++)
                {
                    total += d_memory[matrixA_offset + i * M + k] * d_memory[matrixB_offset + k * P + j];
                }
                matrixC[c_offset + i * P + j] = total + beta * matrixC[c_offset + i * P + j];
            }
        }
    }
    else
    {
        #pragma omp target firstprivate(block_id, block_width, matrixA_offset, matrixB_offset, matrixC_offset)\
        is_device_ptr(d_memory) depend(in: N, M, P)\
        depend(in: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P])\
        depend(inout: matrixC[block_id * block_width * P : blksz * P])
        #pragma omp teams distribute parallel for private(i, j, k, total) //collapse(2) --> adding collpase() slows down with CCE compiler -- weird
        for(i = 0 ; i < blksz ; i++)    
        {
            for(j = 0 ; j < P ; j++)
            {
                total = 0.0;
                #pragma omp simd
                for (k = 0 ; k < M ; k++)
                {
                    total += d_memory[matrixA_offset + i * M + k] * d_memory[matrixB_offset + k * P + j];
                }
                d_memory[matrixC_offset + i * P + j] = total + beta * d_memory[matrixC_offset + i * P + j];;
            }
        }
    }
}


void XY_GPU_MM_v1(double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, cublasHandle_t handle, double alpha, double beta)
{
	int i, j, k;
	double total;
    int status = -1;
    unsigned long int matrixA_offset, matrixB_offset, matrixC_offset; 
    int blksz = block_id * block_width + block_width > N ? N - block_id * block_width : block_width;
    cudaError_t cuberror;
    cublasStatus_t cubstat;

    double tstart = omp_get_wtime();
    
    pair<double *, int> prA = make_pair(matrixA, block_id); 
    pair<double *, int> prB = make_pair(matrixB, 0);
    pair<double *, int> prC = make_pair(matrixC, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "XY : " << "task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prA)) //this is input, 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && (evictionQueue.front() == prB || evictionQueue.front() == prC))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XY: YES 1 -- cann't be evicted task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(matrixA, N, M, block_id, block_id * block_width * M, blksz * M, 0.0, iterationNumber);
        
        if(status != 0)
        {
            cout << "   XY: matrixA copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }

    if(!isOnDevice(prB)) //full matrixB needs to be on device memory
    { 
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prC))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XY: YES 2 -- cann't be evicted task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(matrixB, M, P, 0, 0, M * P, 0.0, iterationNumber);
        
        if(status != 0)
        {
            cout << "   XY: matrixB copy is failed on XY :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }

    if(!isOnDevice(prC)) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prB))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   XY: YES 3 -- cann't be evicted task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = reserveOnDevice(matrixC, N, P, block_id, block_id * block_width * P, blksz * P, 1.0, iterationNumber); 
        
        if(status != 0)
        {
            cout << "   XY: matrixC copy is failed on XY :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prC][5] = 1.0;
    }
    
    if(!isOnDevice(prA)) 
        cout << "matrixA is not on device"  << " block_id : " << block_id << endl;
    if(!isOnDevice(prB))
        cout << "matrixB is not on device"  << " block_id : 0" << endl;
    if(!isOnDevice(prC))
        cout << "matrixC is not on device"  << " block_id : " << block_id << endl;

    matrixA_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    matrixB_offset = (unsigned long int) mp[prB][0] * num_per_blk;
    matrixC_offset = (unsigned long int) mp[prC][0] * num_per_blk;

    taskTiming[iterationNumber - 1][3][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    // depend(in: d_memory[matrixA_offset : blksz * M], d_memory[matrixB_offset : M * P])\
    // depend(inout: d_memory[matrixC_offset : blksz * P])
    // cout << "Lunching XY kernel on GPU -- block_id: " << block_id << endl;

    /*#pragma omp target firstprivate(block_id, block_width, matrixA_offset, matrixB_offset, matrixC_offset)\
    is_device_ptr(d_memory) depend(in: N, M, P)\
    depend(in: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P])\
    depend(inout: matrixC[block_id * block_width * P : blksz * P])
    #pragma omp teams distribute parallel for private(i, j, k, total) //collapse(2) --> adding collpase() slows down with CCE compiler -- weird
    for(i = 0 ; i < blksz ; i++)    
    {
        for(j = 0 ; j < P ; j++)
        {
            total = 0.0;
            for (k = 0 ; k < M ; k++)
            {
                total += d_memory[matrixA_offset + i * M + k] * d_memory[matrixB_offset + k * P + j];
            }
            d_memory[matrixC_offset + i * P + j] = total;
        }
    }*/

    #pragma omp task private(cubstat)\
    firstprivate(block_id, block_width, matrixA_offset, matrixB_offset, matrixC_offset, d_memory, errorFlag, blksz)\
    depend(in: N, M, P)\
    depend(in: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P])\
    depend(inout: matrixC[block_id * block_width * P : blksz * P])
    {
        cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P, blksz, M,
	              &alpha, d_memory + matrixB_offset, P, d_memory + matrixA_offset, M, &beta, d_memory + matrixC_offset, P);
	    cudaDeviceSynchronize();
	    if(cubstat != CUBLAS_STATUS_SUCCESS){ printf("cublasDgemm Failed in Tiling\n"); errorFlag = true; }
    }
}

void cublasDgemm_xy_block(cublasHandle_t handle, double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id)
{
    // Operating on a single block
    // matrixC = matrixA * matrixB
    // matrixA - blksz * M
    // matrixB - M * P
    // matrixC - blksz * P
    

    int i, blksz;
    double alpha = 1.0;
	double beta  = 0.0;
    cudaError_t cuberror;
    cublasStatus_t cubstat;
	
	// blksz = block_width;

    blksz = (block_id * block_width + block_width) > N ? N - block_id * block_width : block_width;
    // cout << "xy: " << " block_id: " << block_id  << " blksz: " << blksz << endl;

	cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P, blksz, M,
	              &alpha, matrixB, P, matrixA, M, &beta, matrixC, P);
	cudaDeviceSynchronize();
	if(cubstat != CUBLAS_STATUS_SUCCESS){ printf("cublasDgemm Failed in Tiling\n"); return; }

	
}


void mat_addition_GPU_MM(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber,
                        bool src1_deviceptr, bool src2_deviceptr, bool dst_deviceptr)
{
    // C = A - B
    // dst = src1 - src2

    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;
    
    int offset = block_id * block_width * col;

    double tstart = omp_get_wtime();

    unsigned long int src1_offset, src2_offset, dst_offset; 
    int status = -1;
    
    pair<double *, int> prA = make_pair(src1, block_id); 
    pair<double *, int> prB = make_pair(src2, block_id);
    pair<double *, int> prC = make_pair(dst, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "ADD : task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prA) && !src1_deviceptr) //this is output, 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prB || evictionQueue.front() == prC)))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   ADD: YES 1 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src1, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   ADD: src1 copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    // else
    // {
    //     mp[prA][5] = 0.0;
    // }
    if(!isOnDevice(prB)) //full matrixB needs to be on device memory
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || !evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prC))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   ADD: YES 2 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src2, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   ADD: src2 copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!isOnDevice(prC)) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prB)))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   ADD: YES 3 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = reserveOnDevice(dst, row, col, block_id, block_id * block_width * col, blksz * col, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   ADD: dst copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prC][5] = 1.0;
    }

    if(!src1_deviceptr)
        src1_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    
    src2_offset = (unsigned long int) mp[prB][0] * num_per_blk;
    dst_offset = (unsigned long int) mp[prC][0] * num_per_blk;

    // depend(in: d_memory[src1_offset : blksz * col], d_memory[src2_offset : blksz * col])\
    // depend(inout: d_memory[dst_offset : blksz * col])
    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prA) && !src1_deviceptr) //this is output, 
    { 
       
        // cout << "matrixA is not on device"  << " block_id : " << block_id << endl;
        cout << "   ADD - A: (" << matrixName[src1] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl; 
    }
    if(!isOnDevice(prB)) //full matrixB needs to be on device memory
    { 
        // cout << "matrixB is not on device"  << " block_id : 0" << endl;
        cout << "   ADD - B: (" << matrixName[src2] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl; 
    }
    if(!isOnDevice(prC)) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    { 
        // cout << "matrixC is not on device"  << " block_id : " << block_id << endl;
        cout << "   ADD - C: (" << matrixName[dst] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl; 
    }
    #endif

    taskTiming[iterationNumber - 1][4][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    if(src1_deviceptr)
    {
        #pragma omp target firstprivate(block_id, block_width, offset, src2_offset, dst_offset)\
        is_device_ptr(d_memory, src1) depend(in: row, col)\
        depend(in: src1[block_id * block_width * col : blksz * col])\
        depend(in: src2[block_id * block_width * col : blksz * col]) depend(inout: dst[block_id * block_width * col : blksz * col])
        #pragma omp teams distribute parallel for private(j) collapse(2)
        for(i = 0 ; i < blksz ; i++)
        {
            for(j = 0 ; j < col ; j++)
            {
                d_memory[dst_offset + i * col + j] = src1[offset + i * col + j] + d_memory[src2_offset + i * col + j];
            }
        }
    }
    else
    {
        #pragma omp target firstprivate(block_id, block_width, src1_offset, src2_offset, dst_offset)\
        is_device_ptr(d_memory) depend(in: row, col)\
        depend(in: src1[block_id * block_width * col : blksz * col])\
        depend(in: src2[block_id * block_width * col : blksz * col]) depend(inout: dst[block_id * block_width * col : blksz * col])
        #pragma omp teams distribute parallel for private(j) collapse(2)
        for(i = 0 ; i < blksz ; i++)
        {
            for(j = 0 ; j < col ; j++)
            {
                d_memory[dst_offset + i * col + j] = d_memory[src1_offset + i * col + j] + d_memory[src2_offset + i * col + j];
            }
        }
    }
}

void mat_sub_GPU_MM(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber,
                    bool src1_deviceptr, bool src2_deviceptr, bool dst_deviceptr)
{
    // C = A - B
    // dst = src1 - src2

    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;
    int offset = block_id * block_width * col;

    double tstart = omp_get_wtime();

    unsigned long int src1_offset, src2_offset, dst_offset; 
    int status = -1;
    
    pair<double *, int> prA = make_pair(src1, block_id); 
    pair<double *, int> prB = make_pair(src2, block_id);
    pair<double *, int> prC = make_pair(dst, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "SUB : task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prA) && !src1_deviceptr) //this is output, 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prB || evictionQueue.front() == prC)))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   SUB : YES 1 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src1, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SUB : src1 copy is failed on SUB :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!isOnDevice(prB) && !src2_deviceptr) //full matrixB needs to be on device memory
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) ||  (!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prC)))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   SUB : YES 2 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src2, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SUB : src2 copy is failed on SUB :"  << " block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!isOnDevice(prC) && !dst_deviceptr) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prB)))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   SUB : YES 3 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
        }

        status = reserveOnDevice(dst, row, col, block_id, block_id * block_width * col, blksz * col, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   SUB : dst copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!dst_deviceptr)
    {
        mp[prC][5] = 1.0;
    }

    if(!src1_deviceptr)
        src1_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    if(!src2_deviceptr)
        src2_offset = (unsigned long int) mp[prB][0] * num_per_blk;
    if(!dst_deviceptr)
        dst_offset = (unsigned long int) mp[prC][0] * num_per_blk;
    
    // depend(in: d_memory[src1_offset : blksz * col], d_memory[src2_offset : blksz * col])\
    // depend(inout: d_memory[dst_offset : blksz * col])
    // cout << "   SUB: " << (unsigned long int) mp[prA][0] << " " << (unsigned long int) mp[prB][0] << " " << (unsigned long int) mp[prC][0]  << " " << num_per_blk << endl;
    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prA) && !src1_deviceptr) //this is output, 
    { 
        // cout << "matrixA is not on device"  << " block_id : " << block_id << endl;
        cout << "   SUB - A: (" << matrixName[src1] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl; 
    }
    if(!isOnDevice(prB) && !src2_deviceptr) //full matrixB needs to be on device memory
    { 
        // cout << "matrixB is not on device"  << " block_id : 0" << endl;
        cout << "   SUB - B: (" << matrixName[src2] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl;
    }
    if(!isOnDevice(prC) && !dst_deviceptr) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    { 
        // cout << "matrixC is not on device"  << " block_id : " << block_id << endl;
        cout << "   SUB - C: (" << matrixName[dst] << ", " << block_id << ") not is device - task_id - " << task_id << " iterationNumber - " <<  iterationNumber << endl;
    }
    #endif

    taskTiming[iterationNumber - 1][6][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    if(src1_deviceptr && dst_deviceptr)
    {
        #pragma omp target firstprivate(block_id, block_width, src2_offset, offset)\
        is_device_ptr(d_memory, src1, dst) depend(in: row, col)\
        depend(in: src1[block_id * block_width * col : blksz * col])\
        depend(in: src2[block_id * block_width * col : blksz * col]) depend(inout: dst[block_id * block_width * col : blksz * col])
        #pragma omp teams distribute parallel for private(j) collapse(2)
        for(i = 0; i < blksz ; i++)
        {
            for(j = 0 ; j < col ; j++)
            {
                dst[offset + i * col + j] = src1[offset + i * col + j] - d_memory[src2_offset + i * col + j];
            }
        }
    }
    else
    {
        #pragma omp target firstprivate(block_id, block_width, src1_offset, src2_offset, dst_offset)\
        is_device_ptr(d_memory) depend(in: row, col)\
        depend(in: src1[block_id * block_width * col : blksz * col])\
        depend(in: src2[block_id * block_width * col : blksz * col]) depend(inout: dst[block_id * block_width * col : blksz * col])
        #pragma omp teams distribute parallel for private(j) collapse(2)
        for(i = 0; i < blksz ; i++)
        {
            for(j = 0 ; j < col ; j++)
            {
                d_memory[dst_offset + i * col + j] = d_memory[src1_offset + i * col + j] - d_memory[src2_offset + i * col + j];
            }
        }
    }
}

void mat_mult_GPU_MM(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id, double *d_memory, long iterationNumber)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;
    int offset = block_id * block_width * col;
    double tstart = omp_get_wtime();

    unsigned long int src1_offset, src2_offset, dst_offset; 
    int status = -1;
    
    pair<double *, int> prA = make_pair(src1, block_id); 
    pair<double *, int> prB = make_pair(src2, block_id);
    pair<double *, int> prC = make_pair(dst, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "MULT :" << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prA)) //this is output, 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && (evictionQueue.front() == prB || evictionQueue.front() == prC))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   MULT: YES 1 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src1, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   MULT: src1 copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!isOnDevice(prB)) //full matrixB needs to be on device memory
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prC))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   YES 2 -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src2, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   MULT: src2 copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    // if(!isOnDevice(prC)) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    // {
    //     if(inEvictionMode)
    //     {
    //         while(!evictionQueue.empty() && (evictionQueue.front() == prA || evictionQueue.front() == prB))
    //         {
    //             cout << "   MULT: YES 3 -- cann't be evicted" << endl;
    //             pair<double *, int> tempPair = evictionQueue.front();
    //             evictionQueue.pop();
    //             evictionQueue.push(tempPair);
    //         }
    //     }

    //     status = reserveOnDevice(dst, row, col, block_id, block_id * block_width * col, blksz * col, 1.0);
    //     if(status != 0)
    //     {
    //         cout << "   MULT: dst copy is failed on block_id : " << block_id << endl;
    //         errorFlag = true;
    //     }
    // }
    // else
    // {
    //     mp[prC][5] = 1.0;
    // }

    src1_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    src2_offset = (unsigned long int) mp[prB][0] * num_per_blk;
    // dst_offset = (unsigned long int) mp[prC][0] * num_per_blk;

    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prA)) //this is output, 
    { 
       
        // cout << "matrixA is not on device"  << " block_id : " << block_id << endl;
        cout << "   MULT - A: (" << matrixName[src1] << ", " << block_id << ") not is device - iterationNumber - " <<  iterationNumber << endl;
    }
    if(!isOnDevice(prB)) //full matrixB needs to be on device memory
    { 
        // cout << "matrixB is not on device"  << " block_id : 0" << endl;
        cout << "   MULT - B: (" << matrixName[src2] << ", " << block_id << ") not is device - iterationNumber - " <<  iterationNumber << endl;
    }
    #endif
    // if(!isOnDevice(prC)) //this is output ==> do not need to be copied to device memory, but we need to reserve its space on device 
    // { 
    //     cout << "matrixC is not on device"  << " block_id : " << block_id << endl;
    // }
    taskTiming[iterationNumber - 1][7][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    #pragma omp target firstprivate(block_id, block_width, src1_offset, src2_offset, offset)\
    is_device_ptr(d_memory, dst) depend(in: row, col)\
    depend(in: src1[block_id * block_width * col : blksz * col])\
    depend(in: src2[block_id * block_width * col : blksz * col]) depend(out: dst[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[offset + i * col + j] = d_memory[src1_offset + i * col + j] * d_memory[src2_offset + i * col + j];
        }
    }
}

void getActiveBlockVector_GPU_MM(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                                    int M, int blocksize, int currentBlockSize, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, bool actR_deviceptr)
{
    //activeBlockVectorR -> M * currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize

    int i, j, k = 0, blksz;
    blksz = (block_id * block_width + block_width) > M ? M - block_id * block_width : block_width;
    // cout << "GET: " << " block_id: " << block_id << " blksz: " << blksz << endl;
    int offset = block_id * block_width * currentBlockSize;

    double tstart = omp_get_wtime();

    unsigned long int actR_offset, R_offset; 
    int status = -1;
    
    pair<double *, int> prActR = make_pair(activeBlockVectorR, block_id);  
    pair<double *, int> prR = make_pair(blockVectorR, block_id);
    // pair<double *, int> prC = make_pair(dst, block_id);
    
    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "GET : task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    // cout << "Processing prActR" << endl;
    if(!isOnDevice(prActR) && !actR_deviceptr) //this is output (actR) 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && evictionQueue.front() == prR))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   GET: YES 1 prR -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
            // cout << "evictionQueue taken care of" << endl;
        }

        status = reserveOnDevice(activeBlockVectorR, M, currentBlockSize, block_id, block_id * block_width * currentBlockSize, blksz * currentBlockSize, 1.0, iterationNumber);
        
        if(status != 0)
        {
            cout << "   GET: activeBlockVectorR copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!actR_deviceptr)
    {
        mp[prActR][5] = 1.0;
    }

    // cout << "Processing prR" << endl;

    if(!isOnDevice(prR)) //this is input (R) 
    {
        if(inEvictionMode)
        {
            while((!evictionQueue.empty() && !isOnDevice(evictionQueue.front())) || (!evictionQueue.empty() && evictionQueue.front() == prActR))
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   GET: YES 1 prActR -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                if(isOnDevice(tempPair))
                    evictionQueue.push(tempPair);
            }
            // cout << "evictionQueue taken care of" << endl;
        }

        status = copyToDevice(blockVectorR, M, blocksize, block_id, block_id * block_width * blocksize, blksz * blocksize, 0.0, iterationNumber);
        
        if(status != 0)
        {
            cout << "   GET: blockVectorR copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!actR_deviceptr)
        actR_offset = (unsigned long int) mp[prActR][0] * num_per_blk;
    R_offset = (unsigned long int) mp[prR][0] * num_per_blk;
    
    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prActR) && !actR_deviceptr) //this is output, 
    { 
        cout << "   GET: prActR is not on device"  << " block_id : " << block_id << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
    }
    if(!isOnDevice(prR)) //full matrixB needs to be on device memory
    { 
        cout << "   prR is not on device"  << " block_id : " << block_id << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
    }
    #endif

    taskTiming[iterationNumber - 1][9][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    // cout << "Lunching GET Kernel" << endl;
    if(actR_deviceptr)
    {
        #pragma omp target firstprivate(offset, R_offset) is_device_ptr(activeMask, d_memory, activeBlockVectorR)\
        depend(inout: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize])\
        depend(in : blockVectorR[block_id * block_width * blocksize : blksz * blocksize], activeMask[0 : blocksize], currentBlockSize)
        #pragma omp teams distribute parallel for private(j, k) 
        for(i = 0 ; i < blksz ; i++)
        {
            k = 0;
            for(j = 0 ; j < blocksize ; j++)
            {
                if(activeMask[j] == 1)
                {
                    activeBlockVectorR[offset + i * currentBlockSize + k] = d_memory[R_offset + i * blocksize + j];
                    k++;
                }
            }
        }
    }
    else
    {
        #pragma omp target firstprivate(actR_offset, R_offset) is_device_ptr(activeMask, d_memory)\
        depend(inout: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize])\
        depend(in : blockVectorR[block_id * block_width * blocksize : blksz * blocksize], activeMask[0 : blocksize], currentBlockSize)
        #pragma omp teams distribute parallel for private(j, k) 
        for(i = 0 ; i < blksz ; i++)
        {
            k = 0;
            for(j = 0 ; j < blocksize ; j++)
            {
                if(activeMask[j] == 1)
                {
                    d_memory[actR_offset + i * currentBlockSize + k] = d_memory[R_offset + i * blocksize + j];
                    k++;
                }
            }
        }
    }
}

void updateBlockVector_GPU_MM(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                                int M, int blocksize, int currentBlockSize, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber, bool actR_deviceptr)
{
    // activeBlockVectorR -> M*currentBlockSize
    // blockVectorR -> M*blocksize
    // activeMask-> blocksize
    int i, j, k = 0, blksz;
    blksz = (block_id * block_width + block_width) > M ? M - block_id * block_width : block_width;
    // cout << "UPDATE: " << " block_id: " << block_id << " blksz: " << blksz << endl;
    int offset = block_id * block_width * currentBlockSize;
    double tstart = omp_get_wtime();

    unsigned long int actR_offset, R_offset; 
    int status = -1;
    
    pair<double *, int> prActR = make_pair(activeBlockVectorR, block_id);  
    pair<double *, int> prR = make_pair(blockVectorR, block_id);
    // pair<double *, int> prC = make_pair(dst, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "UPDATE : task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prActR) && !actR_deviceptr) //this is input (actR) 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && evictionQueue.front() == prR)
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   UPDATE: YES 1 prR -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(activeBlockVectorR, M, currentBlockSize, block_id, block_id * block_width * currentBlockSize, blksz * currentBlockSize, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   UPDATE: activeBlockVectorR copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!isOnDevice(prR)) //this is output (R) 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && evictionQueue.front() == prActR)
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   UPDATE: YES 1 prActR -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(blockVectorR, M, blocksize, block_id, block_id * block_width * blocksize, blksz * blocksize, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   UPDATE: blockVectorR copy is failed on UPDATE :"  << " block_id : " << block_id << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
            errorFlag = true;
        }
    }
    else
    {
        mp[prR][5] = 1.0;
    }
    if(!actR_deviceptr)
        actR_offset = (unsigned long int) mp[prActR][0] * num_per_blk;
    R_offset = (unsigned long int) mp[prR][0] * num_per_blk;

    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prActR) && !actR_deviceptr)
    { 
        cout << "   UPDATE: prActR is not on device"  << " block_id : " << block_id << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
    }
    if(!isOnDevice(prR)) 
    { 
        cout << " UPDATE: prR is not on device"  << " block_id : 0" << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
    }
    #endif
    
    taskTiming[iterationNumber - 1][10][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    if(actR_deviceptr)
    {
        #pragma omp target firstprivate(blksz, block_id, block_width, offset, R_offset) is_device_ptr(d_memory, activeMask, activeBlockVectorR)\
        depend(in: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize], activeMask[0 : blocksize], currentBlockSize)\
        depend(inout : blockVectorR[block_id * block_width * blocksize : blksz * blocksize])
        #pragma omp teams distribute parallel for private(k, j)
        for(i = 0 ; i < blksz ; i++)
        {
            k = 0 ;
            for(j = 0 ; j < blocksize ; j++)
            {
                if(activeMask[j] == 1)
                {
                    d_memory[R_offset + i * blocksize + j] = activeBlockVectorR[offset + i * currentBlockSize + k];
                    k++;
                }
            }
        }
    }
    else
    {
        #pragma omp target firstprivate(blksz, block_id, block_width, actR_offset, R_offset) is_device_ptr(d_memory, activeMask)\
        depend(in: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize], activeMask[0 : blocksize], currentBlockSize)\
        depend(inout : blockVectorR[block_id * block_width * blocksize : blksz * blocksize])
        #pragma omp teams distribute parallel for private(k, j)
        for(i = 0 ; i < blksz ; i++)
        {
            k = 0 ;
            for(j = 0 ; j < blocksize ; j++)
            {
                if(activeMask[j] == 1)
                {
                    d_memory[R_offset + i * blocksize + j] = d_memory[actR_offset + i * currentBlockSize + k];
                    k++;
                }
            }
        }
    }
}

void custom_dlacpy_GPU_MM(double *src, double *dst, int row, int col, int block_width, int block_id, double *d_memory, int task_id, long iterationNumber,
                            bool src_devicptr, bool dst_deviceptr)
{
    //src[m*n] and dst[m*n]
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;
    int offset = block_id * block_width * col;

    double tstart = omp_get_wtime();

    unsigned long int src_offset, dst_offset; 
    int status = -1;
    
    pair<double *, int> prSrc = make_pair(src, block_id);  
    pair<double *, int> prDst = make_pair(dst, block_id);
    // pair<double *, int> prC = make_pair(dst, block_id);

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "DLACPY : task_id - " << task_id << " block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    if(!isOnDevice(prSrc)) //this is input 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && evictionQueue.front() == prDst)
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   DLACPY: YES 1 prDst -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = copyToDevice(src, row, col, block_id, block_id * block_width * col, blksz * col, 0.0, iterationNumber);
        if(status != 0)
        {
            cout << "   DLACPY: src copy is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!isOnDevice(prDst) && !dst_deviceptr) //this is output ==> do not need to copy to device, reserve is OK 
    {
        if(inEvictionMode)
        {
            while(!evictionQueue.empty() && evictionQueue.front() == prSrc)
            {
                #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
                    cout << "   DLACPY: YES 1 prSrc -- cann't be evicted" << endl;
                #endif

                pair<double *, int> tempPair = evictionQueue.front();
                evictionQueue.pop();
                evictionQueue.push(tempPair);
            }
        }

        status = reserveOnDevice(dst, row, col, block_id, block_id * block_width * col, blksz * col, 1.0, iterationNumber);
        if(status != 0)
        {
            cout << "   DLACPY: dst reservation is failed on block_id : " << block_id << endl;
            errorFlag = true;
        }
    }
    if(!dst_deviceptr)
    {
        mp[prDst][5] = 1.0;
    }

    src_offset = (unsigned long int) mp[prSrc][0] * num_per_blk;
    if(!dst_deviceptr)
        dst_offset = (unsigned long int) mp[prDst][0] * num_per_blk;

    #if defined(DEBUG) || defined(DEBUG3)
    if(!isOnDevice(prSrc)) //this is output, 
    { 
        cout << " DLACPY: prSrc is not on device"  << " block_id : " << block_id << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
    }
    if(!isOnDevice(prDst) && !dst_deviceptr) //full matrixB needs to be on device memory
    { 
        cout << "   DLACPY: prDst is not on device"  << " block_id : 0" << " task_id - " << task_id << " iterationNumber - " << iterationNumber << endl;
    }
    #endif
    
    taskTiming[iterationNumber - 1][5][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    if(dst_deviceptr)
    {
        #pragma omp target is_device_ptr(d_memory, dst)\
        firstprivate(block_id, block_width, blksz, src_offset, offset)\
        depend(in: row, col) depend(in: src[block_id * block_width * col : blksz * col])\
        depend(inout: dst[block_id * block_width * col : blksz * col])  
        #pragma omp teams distribute parallel for collapse(2)
        for(i = 0 ; i < blksz ; i++) //each row
        {
            for(j = 0 ; j < col ; j++) //each column
            {
                dst[offset + i * col + j] = d_memory[src_offset + i * col + j];
            }
        }
    }
    else
    {
        #pragma omp target is_device_ptr(d_memory)\
        firstprivate(block_id, block_width, blksz, src_offset, dst_offset)\
        depend(in: row, col) depend(in: src[block_id * block_width * col : blksz * col])\
        depend(inout: dst[block_id * block_width * col : blksz * col])  
        #pragma omp teams distribute parallel for collapse(2)
        for(i = 0 ; i < blksz ; i++) //each row
        {
            for(j = 0 ; j < col ; j++) //each column
            {
                d_memory[dst_offset + i * col + j] = d_memory[src_offset + i * col + j];
            }
        }
    }
}

void sum_sqrt_GPU_COL_MM(double *src, double *dst, int row, int col, int block_width, int block_id, int buf_id, double *buf, double *d_memory, long iterationNumber)
{
    int i, j, blksz;

    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    double tstart = omp_get_wtime();

    unsigned long int src_offset = block_id * block_width * col; 
    int status = -1;
    
    pair<double *, int> prA = make_pair(src, block_id); 

    #if defined(DEBUG) || defined(DEBUG3) || defined(DEBUG4)
        cout << "COL : block_id - " << block_id << " iterationNumber - " << iterationNumber << endl;
    #endif

    // if(!isOnDevice(prA)) //this is output, 
    // { 
    //     status = copyToDevice(src, row, col, block_id, block_id * block_width * col, blksz * col, 0.0);
    //     if(status != 0)
    //     {
    //         cout << "   COL: src1 copy is failed on block_id : " << block_id << endl;
    //         errorFlag = true;
    //     }
    // }

    // src_offset = (unsigned long int) mp[prA][0] * num_per_blk;
    // mp[prA][5] = 0.0; //newX we don't need to copy back
    taskTiming[iterationNumber - 1][8][0] += omp_get_wtime() - tstart;
    mmTiming[iterationNumber - 1] += omp_get_wtime() - tstart;

    #pragma omp target firstprivate(buf_id, block_id, block_width, blksz, src_offset) is_device_ptr(buf, d_memory, src)\
    depend(inout: buf[buf_id * col : col]) depend(in: src[block_id * block_width * col : blksz * col]) 
    #pragma omp teams distribute parallel for //private(i)) collapse(2) //- not working
    for(i = 0 ; i < col ; i++) //i->col
    {
        #pragma omp simd
        for(j = 0 ; j < blksz ; j++) //j->row
        {
            buf[buf_id * col + i] += src[src_offset + j * col + i];
        }
    }
}

// =========================== Memory Manager Managed Kernels Ends ===========================

void update_activeMask_GPU(int *activeMask, int *d_activeMask, double *d_residualNorms, double residualTolerance, int blocksize)
{
    int i;

    #pragma omp target defaultmap(tofrom: scalar) is_device_ptr(d_residualNorms, d_activeMask)\
    map(tofrom: activeMask[0 : blocksize]) depend(inout: d_activeMask[0 : blocksize])\
    depend(inout: activeMask[0 : blocksize]) depend(in: d_residualNorms[0 : blocksize])
    #pragma omp teams distribute parallel for
    for(i = 0 ; i < blocksize ; i++)
    {
        if((d_residualNorms[i] > residualTolerance) && activeMask[i] == 1)
            d_activeMask[i] = activeMask[i] = 1;
        else
            d_activeMask[i] = activeMask[i] = 0;
    }
}

void SPMM_GPU_TILE(int *row_ptr, int *col_index, double *value, double *d_Y, double *Z, 
                        int numrows, int numcols, int nvec, int block_width, int block_id, int *nnz_per_tile)
                         
{
    // Z = A * X ==> A[numrows * numcols] Y[numrows * nvec] Z[numrows * nvec], A is the sparse matrix
    
    int blksz = block_width;
    int offset = block_id * block_width;

    if(offset + blksz > numrows)
        blksz = numrows - offset;
    
    int i, j, k, start, end;
    int r, c;
    double xcoef;

    #pragma omp target firstprivate(block_id, block_width, nvec, offset) is_device_ptr(d_Y)\
    map(to: row_ptr[offset : blksz + 1], col_index[row_ptr[offset] : nnz_per_tile[block_id]], value[row_ptr[offset] : nnz_per_tile[block_id]])\
    map(tofrom: Z[offset * nvec : blksz  * nvec])\
    depend(in: row_ptr[offset : blksz + 1], col_index[row_ptr[offset] : nnz_per_tile[block_id]], value[row_ptr[offset] : nnz_per_tile[block_id]])\
    depend(in: d_Y[0 : numrows * nvec]) depend(inout: Z[offset * nvec : blksz  * nvec])
    #pragma omp teams distribute parallel for private(start, end, r, c, xcoef, i, j, k) firstprivate(block_id, block_width, nvec, offset, nnz_per_tile, row_ptr, col_index, value, d_Y, Z)
    for(i = 0 ; i < blksz ; i++)
    {
        start = row_ptr[offset + i];
        end = row_ptr[offset + i + 1];
        for(j = start ; j < end ; j++)
        {
            r = offset + i;
            c = col_index[j];
            xcoef = value[j];  
            for(k = 0 ; k < nvec ; k++)
            {
                Z[r * nvec + k] = Z[r * nvec + k] + xcoef * d_Y[c * nvec + k];
            }
        }
    }
}

void XTY_OPENMP_GPU(double *X, double *Y, double *result_buf ,int M, int N, int P, int block_width, int block_id, int buf_id)
{
	// X[M * N] , Y[M * P], but[nbuf * N * P]

	int blksz = block_width;
	if(block_id * block_width + blksz > M)
		blksz = M - block_id * block_width;

	int i, j, k, rowOffset = block_id * block_width;
	double total = 0.0;

	#pragma omp target is_device_ptr(result_buf) firstprivate(buf_id, block_id, block_width)\
    map(to: X[rowOffset * N : blksz * N], Y[rowOffset * P : blksz * P])\
	depend(in: M, N, P) depend(in: X[rowOffset * N : blksz * N], Y[rowOffset * P : blksz * P])\
	depend(inout: result_buf[buf_id * N * P : N * P])
    // #pragma omp parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
	#pragma omp teams distribute parallel for firstprivate(rowOffset, blksz, block_id, block_width, buf_id) private(i, j, k, total) shared(X, Y, result_buf) collapse(2)
    for(i = 0 ; i < N ; i++)   
    {
        for(j = 0 ; j < P ; j++) 
        {   
			total = 0.0;
            for (k = rowOffset ; k < rowOffset + blksz ; k++) // # of rows in X or Y (should be same)
            {
                total += X[k * N + i] * Y[k * P + j];
            }
			result_buf[buf_id * N * P + i * P + j] += total;
        }
    }	
}

void XY_OPENMP_GPU_tiled(double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id)
{
	int i, j, k;
	double total;

	int blksz = block_id * block_width + block_width > N ? N - block_id * block_width : block_width;
    
    #pragma omp target firstprivate(block_id, block_width) map(to: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P])\
    map(tofrom: matrixC[block_id * block_width * P : blksz * P]) depend(in: N, M, P)\
    depend(in: matrixA[block_id * block_width * M : blksz * M], matrixB[0 : M * P]) depend(inout: matrixC[block_id * block_width * P : blksz * P])
    #pragma omp teams distribute parallel for private(i, j, k, total) collapse(2) //shared(matrixA, matrixB, matrixC, block_id, block_width)
    for(i = 0 ; i < blksz ; i++)    
    {
        for(j = 0 ; j < P ; j++)
        {
            total = 0.0;
            for (k = 0 ; k < M ; k++)
            {
                total += matrixA[block_id * block_width * M + i * M + k] * matrixB[k * P + j];
            }
            matrixC[block_id * block_width * P + i * P + j] = total;
        }
    }
}




void cublasDgemm_xy_block_betaOne(cublasHandle_t handle, double *matrixA, double *matrixB, double *matrixC,
                                int N, int M, int P, int block_width, int block_id)
{
    // cout << "in cublasDgemm_xy_block " << endl;
    /*
    Operating on a single block
    matrixC = matrixA * matrixB
    matrixA - block_width * M
    matrixB - M * P
    matrixC - block_width * P
    */

    int i, blksz;
    double alpha = 1.0;
	double beta  = 1.0;
    // int nrowblk = ceil(1.0 * N/block_width);

    cudaError_t cuberror;
    cublasStatus_t cubstat;
	
	// blksz = block_width;

    blksz = (block_id * block_width + block_width) > N ? N - block_id * block_width : block_width;
    // cout << "xy: " << " block_id: " << block_id  << " blksz: " << blksz << endl;

	cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P, blksz, M,
	              &alpha, matrixB, P, matrixA, M, &beta, matrixC, P);
	cudaDeviceSynchronize();
	if(cubstat != CUBLAS_STATUS_SUCCESS){ printf("cublasDgemm Failed in Tiling\n"); return; }

	
}

void cublasDgemm_xty_tiling(cublasHandle_t handle, double *devPtrA, double *devPtrB, double *devPtrC, 
                            int N, int blocksize, int currentBlockSize, int block_width, int block_id)
{
    /*
    matrixC = matrixA' * matrixB

    matrixA - N * b
    matrixB - N * cb
    matrixC - b * cb
    */
    int i, blksz;
    double alpha = 1.0, beta  = 0.0;
    cublasStatus_t cubstat;
    cudaError_t cuberror;
	
    blksz = (block_id * block_width + block_width) > N ? N - block_id * block_width : block_width;
    beta = 1.0;
	cubstat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, currentBlockSize, blocksize, blksz, 
					&alpha, devPtrB, currentBlockSize, devPtrA, blocksize, &beta, devPtrC, currentBlockSize); 
	cudaDeviceSynchronize();
        
    if(cubstat != CUBLAS_STATUS_SUCCESS)
		printf("cublasDgemm status: %d\n",cubstat);
}

void XTY_GPU_RED(double *buf, double *result, int N, int P, int block_width)
{
    /*
    _XTY_v1_RED: adding partial sums block by block, not row by row
    Input: buf: nthds * [N * P]
    Output: result[N * P]
    nthrds : global variable, total # of threads
    buf : how to free/deallocate corresponding memory location
    */
    
    int i, j, k, l, blksz, tid, nthreads, length;
    double sum, tstart, tend;
    
    int nbuf = 128;

    int n = N * P;

    for(i = 0 ; i < N ; i = i + block_width)
    {
        blksz = block_width;
        if(i + blksz > N)
            blksz = N - i;

        #pragma omp target is_device_ptr(buf) private(sum, k, l, j)\
        firstprivate(i, n, nbuf, blksz, block_width)\
        map(tofrom: result[i * P  : blksz * P])\
        depend(in: N, P) depend(out: result[i * P : blksz * P]) depend(in: N, P) depend(in: buf[0 * n : n])
        {
            for(l = i ; l < (i + blksz) ; l++) //for each row in the block
            {
                for(k  = 0 ; k < P ; k++) //each col
                {
                    sum = 0.0;
                    for(j = 0 ; j < nbuf ; j++) //for each thread access corresponding N*N matrix
                    {
                        sum += buf[j * N * P + l * P + k];
                    }
                    result[l * P + k] = sum;
                }
                
            }
        }// end of task 
    }//end outer for
}

void mat_sub_GPU_block(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    #pragma omp target firstprivate(block_id, block_width, blksz) map(to: src1[block_id * block_width * col : blksz * col])\
    map(to: src2[block_id * block_width * col : blksz * col]) map(tofrom: dst[block_id * block_width * col : blksz * col])\
    depend(in: row, col) depend(in: src1[block_id * block_width * col : blksz * col])\
    depend(in: src2[block_id * block_width * col : blksz * col]) depend(inout: dst[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) //collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        // #pragma omp parallel for
        for(j = 0 ; j < col ; j++)
        {
            dst[block_id * block_width * col + i * col + j] = src1[block_id * block_width * col + i * col + j] - src2[block_id * block_width * col + i * col + j];
        }
    }
}

void mat_sub_GPU_block_v1(double *src1, double *src2, int row, int col, int block_width, int block_id)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    #pragma omp target firstprivate(block_id, block_width) map(tofrom: src1[block_id * block_width * col : blksz * col])\
    map(to: src2[block_id * block_width * col : blksz * col])\
    depend(in: row, col) depend(inout: src1[block_id * block_width * col : blksz * col])\
    depend(in: src2[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            src1[block_id * block_width * col + i * col + j] = src1[block_id * block_width * col + i * col + j] - src2[block_id * block_width * col + i * col + j];
        }
    }
}
void mat_sub_GPU_block_v2(double *src1, double *src2, int row, int col, int block_width, int block_id)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    #pragma omp target firstprivate(block_id, block_width, blksz) map(to: src1[block_id * block_width * col : blksz * col])\
    map(tofrom: src2[block_id * block_width * col : blksz * col])\
    depend(in: row, col) depend(in: src1[block_id * block_width * col : blksz * col])\
    depend(inout: src2[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            src2[block_id * block_width * col + i * col + j] = src1[block_id * block_width * col + i * col + j] - src2[block_id * block_width * col + i * col + j];
        }
    }
}

void mat_addition_GPU_block_v1(double *src1, double *dst, int row, int col, int block_width, int block_id)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    #pragma omp target map(to: src1[block_id * block_width * col : blksz * col])\
    map(tofrom: dst[block_id * block_width * col : blksz * col])\
    depend(in: src1[block_id * block_width * col : blksz * col])\
    depend(inout: dst[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[block_id * block_width * col + i * col + j] = src1[block_id * block_width * col + i * col + j] + dst[block_id * block_width * col + i * col + j];
        }
    }
}

void mat_addition_GPU_block(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    #pragma omp target map(to: src1[block_id * block_width * col : blksz * col])\
    map(to: src2[block_id * block_width * col : blksz * col]) map(tofrom: dst[block_id * block_width * col : blksz * col])\
    depend(in: src1[block_id * block_width * col : blksz * col])\
    depend(in: src2[block_id * block_width * col : blksz * col]) depend(inout: dst[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[block_id * block_width * col + i * col + j] = src1[block_id * block_width * col + i * col + j] + src2[block_id * block_width * col + i * col + j];
        }
    }
}

void mat_mult_GPU_block(double *src1, double *src2, double *dst, int row, int col, int block_width, int block_id)
{
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;

    #pragma omp target firstprivate(block_id, block_width) map(to: src1[block_id * block_width * col : blksz * col])\
    map(to: src2[block_id * block_width * col : blksz * col]) map(tofrom: dst[block_id * block_width * col : blksz * col])\
    depend(in: row, col) depend(in: src1[block_id * block_width * col : blksz * col])\
    depend(in: src2[block_id * block_width * col : blksz * col]) depend(out: dst[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for private(j) collapse(2)
    for(i = 0; i < blksz ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[block_id * block_width * col + i * col + j] = src1[block_id * block_width * col + i * col + j] * src2[block_id * block_width * col + i * col + j];
        }
    }
}

void sum_sqrt_GPU_COL(double *src, double *dst, int row, int col, int block_width, int block_id, int buf_id, double *buf)
{
    int i, j, blksz;

    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;
    
    #pragma omp target firstprivate(buf_id, block_id, block_width, blksz)\
    is_device_ptr(buf) map(to: src[block_id * block_width * col : blksz * col])\
    depend(inout: buf[buf_id * col : col]) depend(in: src[block_id * block_width * col : blksz * col])
    #pragma omp teams distribute parallel for //private(i)) collapse(2) //- not working
    for(i = 0 ; i < col ; i++) //i->col
    {
        for(j = 0 ; j < blksz ; j++) //j->row
        {
            buf[buf_id * col + i] += src[block_id * block_width * col + j * col + i];
        }
    }
}

void custom_dlacpy_task_CPU(double *src, double *dst, int row, int col, int block_width, int block_id)
{
    int i, j, k, blksz, tid;
    double tstart, tend;
    
    k = block_id * block_width; //starting point of the block 
    blksz = block_width;
        
    if(k + blksz > row)
        blksz = row - k;
    
    #pragma omp task private(i, j, tid, tstart, tend)\
    firstprivate(blksz, block_width, k)\
    depend(in: src[k * col : blksz * col], row, col)\
    depend(out: dst[k * col : blksz * col])
    {
        for(i = k; i < k + blksz ; i++) //each row
        {
            for(j = 0 ; j < col ; j++) //each column
            {
                dst[i * col + j] = src[i * col + j];
            }
        }
    }  //end task
}

void custom_dlacpy_GPU_block(double *src, double *dst, int row, int col, int block_width, int block_id)
{
    //src[m*n] and dst[m*n]
    int i, j, blksz;
    blksz = (block_id * block_width + block_width) > row ? row - block_id * block_width : block_width;
    
    #pragma omp target firstprivate(block_id, block_width, blksz)\
    map(to: src[block_id * block_width * col : blksz * col]) map(tofrom: dst[block_id * block_width * col : blksz * col])\
    depend(in: row, col) depend(in: src[block_id * block_width * col : blksz * col])\
    depend(inout: dst[block_id * block_width * col : blksz * col])  
    #pragma omp teams distribute parallel for collapse(2)
    for(i = 0 ; i < blksz ; i++) //each row
    {
        for(j = 0 ; j < col ; j++) //each column
        {
            dst[block_id * block_width * col + i * col + j] = src[block_id * block_width * col + i * col + j];
        }
    }
}

void sum_sqrt_GPU_SQRT(double *dst, int col)
{
    int i, j;
    #pragma omp target map(tofrom : dst[0 : col]) depend(inout: dst[0 : col])
    #pragma omp teams distribute parallel for default(shared)
    for(i = 0; i < col ; i++) 
    {
        dst[i] = sqrt(dst[i]);
    }
}

void getActiveBlockVector_GPU_block(double *activeBlockVectorR, int *activeMask, double *blockVectorR, 
                                    int M, int blocksize, int currentBlockSize, int block_width, int block_id)
{
    //activeBlockVectorR -> M * currentBlockSize
    //blockVectorR -> M*blocksize
    //activeMask-> blocksize

    int i, j, k = 0, blksz;
    blksz = (block_id * block_width + block_width) > M ? M - block_id * block_width : block_width;
    // cout << "GET: " << " block_id: " << block_id << " blksz: " << blksz << endl;

    #pragma omp target map(to: blockVectorR[block_id * block_width * blocksize : blksz * blocksize], activeMask[0 : blocksize])\
    map(activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize])\
    depend(inout: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize])\
    depend(in : blockVectorR[block_id * block_width * blocksize : blksz * blocksize], activeMask[0 : blocksize], currentBlockSize)
    #pragma omp teams distribute parallel for private(j, k) 
    for(i = 0 ; i < blksz ; i++)
    {
        k = 0;
        for(j = 0 ; j < blocksize ; j++)
        {
             if(activeMask[j] == 1)
             {
                activeBlockVectorR[block_id * block_width * currentBlockSize + i * currentBlockSize + k] 
                                        = blockVectorR[block_id * block_width * blocksize + i * blocksize + j];
                k++;
             }
        }
    }
}


void updateBlockVector_GPU_block(double *activeBlockVectorR, int *activeMask, double *blockVectorR, int M, int blocksize, int currentBlockSize, int block_width, int block_id)
{
    // activeBlockVectorR -> M*currentBlockSize
    // blockVectorR -> M*blocksize
    // activeMask-> blocksize
    int i, j, k = 0, blksz;
    blksz = (block_id * block_width + block_width) > M ? M - block_id * block_width : block_width;
    // cout << "UPDATE: " << " block_id: " << block_id << " blksz: " << blksz << endl;

    #pragma omp target firstprivate(blksz, block_id, block_width)\
    map(to: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize], activeMask[0 : blocksize])\
    map(blockVectorR[block_id * block_width * blocksize : blksz * blocksize]) defaultmap(tofrom: scalar)\
    depend(in: activeBlockVectorR[block_id * block_width * currentBlockSize : blksz * currentBlockSize], activeMask[0 : blocksize], currentBlockSize)\
    depend(inout : blockVectorR[block_id * block_width * blocksize : blksz * blocksize])
    #pragma omp teams distribute parallel for private(k, j)
    for(i = 0 ; i < blksz ; i++)
    {
        k = 0 ;
        for(j = 0 ; j < blocksize ; j++)
        {
            if(activeMask[j] == 1)
            {
                blockVectorR[block_id * block_width * blocksize + i * blocksize + j] = activeBlockVectorR[block_id * block_width * currentBlockSize + i * currentBlockSize + k];
                k++;
            }
        }
    }
}

void transpose_GPU_block(double *src, double *d_dst, 
                            int row, int col, int block_width)
{
    int i, j, k, l, m , blksz, start_point;
    int nrowblk = ceil(1.0 * row/block_width);
    cudaError_t cuberror;
    

    for(i = 0 ; i < nrowblk ; i++)
	{
		blksz = block_width;
		if(i * block_width + blksz > row)
			blksz = row - i * block_width;
        
        start_point = i * block_width;

		#pragma omp target map(to: src[i * block_width * col : blksz * col])\
        defaultmap(tofrom: scalar) is_device_ptr(d_dst) firstprivate(i, start_point, row, col) private(k)\
        depend(in: src[i * block_width * col : blksz * col])
        #pragma omp teams distribute parallel for collapse(2)
        for(j = start_point; j < start_point + blksz ; j++)
        {
            for(k = 0 ; k < col ; k++)
            {
                // int p = j * k;
                // if(i == 7)
                //     printf("%d ", p);
                // cout << "Transpose: " << " block_id: " << i << " blksz: " << blksz << " start_point: " << start_point << endl;
		
                // if(i == 7)
                //     printf("(%d, %d, %lf)\n", j, k, src[j * col + k]);
                //     cout << "hellow" << endl;
                    // cout << j << " , " << k << " " << endl;
                // d_dst[k * row + j] = src[i * block_width * col + (j - start_point) * col + k];
                d_dst[1] = 0.5; //src[j * col + k];
            }
            // cout << endl;
        }

        // cuberror = cudaMemcpy(dst + (i * block_width * col), d_dst_tile, blksz * col * sizeof(double), cudaMemcpyDeviceToHost);
    	// if( cuberror != 0 ){ printf("cudaMemcpy failed d_dst_tile at blok_id: %d errocode: %d\n", i, cuberror);}
		// cudaDeviceSynchronize();
	}
}

void transpose_GPU(double *src, double *dst, int N, int M)
{
    //src - M * N
    int i, j;
    #pragma omp target map(to: src[0 : M * N]) is_device_ptr(dst) depend(out: dst[0 : N * M])
    #pragma omp teams distribute parallel for collapse(2)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

void transpose_GPU_deviceptr(double *src, double *dst, int N, int M)
{
    //src - M * N
    int i, j;
    #pragma omp target is_device_ptr(src, dst) depend(in: src[0 : N * M]) depend(out: dst[0 : N * M])
    #pragma omp teams distribute parallel for collapse(2)
    for(i = 0 ; i < M ; i++)
    {
        for(j = 0 ; j < N ; j++)
        {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

void cusparseDcsrmm_tile(int *ia, int *ja, double *acsr, 
                        double *d_activeBlockVectorR, double *activeBlockVectorAR, double *d_activeBlockVectorAR_tile, double *d_temp_actAR,
                        int *rowPtrTile, int *colIndexTile, double *coolValTile, 
                        int *nnz_per_tile, int numrows, int numcols, int nrowblk, int currentBlockSize, int block_width, int block_id, int h, int t,
                        cusparseHandle_t handle, cusparseMatDescr_t descr)
{
    /*
    ia, ja, acsr - Sparse matrix in CSR format,
    rowPtrTile, colIndexTile, coolValTile - CSR format for one signle tile
    nnz_per_tile - nnz per tile, precalculated
    d_activeBlockVectorR - RHS vector block (dense matrix)
    d_activeBlockVectorAR_tile - output from the SpMM operation on a single time
    */
    int blksz = block_width;
    int i, j, k;
    double dzero = 0.0, dtwo = 2.0, dthree = 3.0, done = 1.0;
    cudaError_t cudaStat;
    cusparseStatus_t cuSparseStatus;
    //d_temp_actAR - declare it here.
    // cudaStat = cudaMalloc((void**)&d_temp_actAR, block_width * currentBlockSize * sizeof(double));
    // if (cudaStat != cudaSuccess) {
    //     printf("Device malloc failed (d_temp_actAR) errorcode: %d\n", cudaStat);
    //     return 1;
    // }

    // for(i = 0 ; i < nrowblk; i++)
    // {

        i = block_id;
        blksz = block_width;
        if(i * block_width + blksz > numrows)
            blksz = numrows - i * block_width;
        
        // cudaStat = cudaMemcpy(rowPtrTile, ia + (i * block_width), (blksz + 1) * sizeof(int), cudaMemcpyHostToDevice);
        // if( cudaStat != cudaSuccess ){ printf("cudaMemcpy failed rowPtrTile ==> %d\n", cudaStat); }
        
        int status = omp_target_memcpy(rowPtrTile, ia + (i * block_width), (blksz + 1) * sizeof(int), 0, 0, t, h);
        if(status != 0){ printf("cudaMemcpy failed rowPtrTile ==> %d\n", status); return; }
        
        // cudaStat = cudaMemcpy(colIndexTile, ja + ia[i * block_width], nnz_per_tile[i] * sizeof(int), cudaMemcpyHostToDevice);
        // if( cudaStat != cudaSuccess ){ printf("cudaMemcpy failed colIndexTile ==> %d\n", cudaStat); }

        status = omp_target_memcpy(colIndexTile, ja + ia[i * block_width], nnz_per_tile[i] * sizeof(int), 0, 0, t, h);
        if( status != 0 ){ printf("cudaMemcpy failed colIndexTile ==> %d\n", status); return; }
        
        // cudaStat = cudaMemcpy(coolValTile, acsr + ia[i * block_width], nnz_per_tile[i] * sizeof(double), cudaMemcpyHostToDevice);
        // if( cudaStat != cudaSuccess ){ printf("cudaMemcpy failed coolValTile ==> %d\n", cudaStat); }

        status = omp_target_memcpy(coolValTile, acsr + ia[i * block_width], nnz_per_tile[i] * sizeof(double), 0, 0, t, h);
        if( status != 0 ){ printf("cudaMemcpy failed coolValTile ==> %d\n", status); return; }

        cudaDeviceSynchronize();
        //offsetting rowptr
        #pragma omp target teams distribute parallel for is_device_ptr(rowPtrTile)
        for(j = blksz; j >= 0 ; j--)
            rowPtrTile[j] = rowPtrTile[j] - rowPtrTile[0];
     
        /*cuSparseStatus = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, blksz, currentBlockSize, numrows,
                           nnz_per_tile[i], &done, descr, coolValTile, rowPtrTile, colIndexTile,
                           d_activeBlockVectorR, numrows, &dzero, d_activeBlockVectorAR_tile, blksz);
        */
        cudaDeviceSynchronize();
        if(cuSparseStatus != CUSPARSE_STATUS_SUCCESS) {
            printf("cusparseDcsrmm_tile error: %d\n", cuSparseStatus);
        }
        transpose_GPU(d_activeBlockVectorAR_tile, d_temp_actAR, blksz, currentBlockSize);
        
        // cudaStat = cudaMemcpy(activeBlockVectorAR + (i * block_width * currentBlockSize), d_temp_actAR, blksz * currentBlockSize * sizeof(double), cudaMemcpyDeviceToHost);
        // if (cudaStat != cudaSuccess){
        //     printf("cusparseDcsrmm_tile Memcpy from Device to Host failed\n");
        // }

        status = omp_target_memcpy(activeBlockVectorAR + (i * block_width * currentBlockSize), d_temp_actAR, blksz * currentBlockSize * sizeof(double), 0, 0, h, t);
        if(status != 0){ printf("cudaMemcpy failed activeBlockVectorAR ==> %d\n", status); return; }

        
        cudaDeviceSynchronize();
    // }
}

void mat_copy_GPU(double *src,  int row, int col, double *dst, int start_row, int start_col, int ld_dst)
{
    int i,j;
    #pragma omp target is_device_ptr(src, dst)
    #pragma omp teams distribute parallel for private(j) //collapse(2)
    for(i = 0 ; i < row ; i++)
    {
        for(j = 0 ; j < col ; j++)
        {
            dst[(start_row + i) * ld_dst + (start_col + j)] = src[i * col + j];
        }
    }
}

void clear_buffer(double *buf, int row, int col)
{
    int i, nbuf = 128;
    int n = row * col;

    #pragma omp target is_device_ptr(buf) firstprivate(n, nbuf)\
    depend(inout : buf[0 * n : n], buf[1 * n : n], buf[2 * n : n], buf[3 * n : n], buf[4 * n : n], buf[5 * n : n], buf[6 * n : n], buf[7 * n : n], buf[8 * n : n], buf[9 * n : n], buf[10 * n : n],\
    buf[11 * n : n], buf[12 * n : n], buf[13 * n : n], buf[14 * n : n], buf[15 * n : n], buf[16 * n : n], buf[17 * n : n], buf[18 * n : n], buf[19 * n : n], buf[20 * n : n],\
    buf[21 * n : n], buf[22 * n : n], buf[23 * n : n], buf[24 * n : n], buf[25 * n : n], buf[26 * n : n], buf[27 * n : n], buf[28 * n : n], buf[29 * n : n], buf[30 * n : n],\
    buf[31 * n : n], buf[32 * n : n], buf[33 * n : n], buf[34 * n : n], buf[35 * n : n], buf[36 * n : n], buf[37 * n : n], buf[38 * n : n], buf[39 * n : n], buf[40 * n : n],\
    buf[41 * n : n], buf[42 * n : n], buf[43 * n : n], buf[44 * n : n], buf[45 * n : n], buf[46 * n : n], buf[47 * n : n], buf[48 * n : n], buf[49 * n : n], buf[50 * n : n],\
    buf[51 * n : n], buf[52 * n : n], buf[53 * n : n], buf[54 * n : n], buf[55 * n : n], buf[56 * n : n], buf[57 * n : n], buf[58 * n : n], buf[59 * n : n], buf[60 * n : n],\
    buf[61 * n : n], buf[62 * n : n], buf[63 * n : n], buf[64 * n : n], buf[65 * n : n], buf[66 * n : n], buf[67 * n : n], buf[68 * n : n], buf[69 * n : n], buf[70 * n : n],\
    buf[71 * n : n], buf[72 * n : n], buf[73 * n : n], buf[74 * n : n], buf[75 * n : n], buf[76 * n : n], buf[77 * n : n], buf[78 * n : n], buf[79 * n : n], buf[80 * n : n],\
    buf[81 * n : n], buf[82 * n : n], buf[83 * n : n], buf[84 * n : n], buf[85 * n : n], buf[86 * n : n], buf[87 * n : n], buf[88 * n : n], buf[89 * n : n], buf[90 * n : n],\
    buf[91 * n : n], buf[92 * n : n], buf[93 * n : n], buf[94 * n : n], buf[95 * n : n], buf[96 * n : n], buf[97 * n : n], buf[98 * n : n], buf[99 * n : n], buf[100 * n : n],\
    buf[101 * n : n], buf[102 * n : n], buf[103 * n : n], buf[104 * n : n], buf[105 * n : n], buf[106 * n : n], buf[107 * n : n], buf[108 * n : n], buf[109 * n : n], buf[110 * n : n],\
    buf[111 * n : n], buf[112 * n : n], buf[113 * n : n], buf[114 * n : n], buf[115 * n : n], buf[116 * n : n], buf[117 * n : n], buf[118 * n : n], buf[119 * n : n], buf[120 * n : n],\
    buf[121 * n : n], buf[122 * n : n], buf[123 * n : n], buf[124 * n : n], buf[125 * n : n], buf[126 * n : n], buf[127 * n : n])
    #pragma omp teams distribute parallel for
    for(i = 0 ; i < nbuf * n ; i++)
    {
        // cout << "i : " << i << endl;
        buf[i] = 0.0;
    }
}

void sum_sqrt_task_RNRED(double *buf, double *dst, const int col)
{
    //code: 6
    int i, j, k, l, length, blksz, tid;
    int nbuf = 128;
    int n = col;
    double tstart, tend;
    //printf("nthrds: %d \n", nthrds);
    //adding partial sums
    // #pragma omp target is_device_ptr(buf)\
    // firstprivate(nbuf, n, col) map(tofrom: dst[0 : col])\
    // depend(in: buf[0 * n : n], buf[1 * n : n], buf[2 * n : n], buf[3 * n : n], buf[4 * n : n], buf[5 * n : n], buf[6 * n : n], buf[7 * n : n], buf[8 * n : n], buf[9 * n : n], buf[10 * n : n],\
    // buf[11 * n : n], buf[12 * n : n], buf[13 * n : n], buf[14 * n : n], buf[15 * n : n], buf[16 * n : n], buf[17 * n : n], buf[18 * n : n], buf[19 * n : n], buf[20 * n : n],\
    // buf[21 * n : n], buf[22 * n : n], buf[23 * n : n], buf[24 * n : n], buf[25 * n : n], buf[26 * n : n], buf[27 * n : n], buf[28 * n : n], buf[29 * n : n], buf[30 * n : n],\
    // buf[31 * n : n], buf[32 * n : n], buf[33 * n : n], buf[34 * n : n], buf[35 * n : n], buf[36 * n : n], buf[37 * n : n], buf[38 * n : n], buf[39 * n : n], buf[40 * n : n],\
    // buf[41 * n : n], buf[42 * n : n], buf[43 * n : n], buf[44 * n : n], buf[45 * n : n], buf[46 * n : n], buf[47 * n : n], buf[48 * n : n], buf[49 * n : n], buf[50 * n : n],\
    // buf[51 * n : n], buf[52 * n : n], buf[53 * n : n], buf[54 * n : n], buf[55 * n : n], buf[56 * n : n], buf[57 * n : n], buf[58 * n : n], buf[59 * n : n], buf[60 * n : n],\
    // buf[61 * n : n], buf[62 * n : n], buf[63 * n : n], buf[64 * n : n], buf[65 * n : n], buf[66 * n : n], buf[67 * n : n], buf[68 * n : n], buf[69 * n : n], buf[70 * n : n],\
    // buf[71 * n : n], buf[72 * n : n], buf[73 * n : n], buf[74 * n : n], buf[75 * n : n], buf[76 * n : n], buf[77 * n : n], buf[78 * n : n], buf[79 * n : n], buf[80 * n : n],\
    // buf[81 * n : n], buf[82 * n : n], buf[83 * n : n], buf[84 * n : n], buf[85 * n : n], buf[86 * n : n], buf[87 * n : n], buf[88 * n : n], buf[89 * n : n], buf[90 * n : n],\
    // buf[91 * n : n], buf[92 * n : n], buf[93 * n : n], buf[94 * n : n], buf[95 * n : n], buf[96 * n : n], buf[97 * n : n], buf[98 * n : n], buf[99 * n : n], buf[100 * n : n],\
    // buf[101 * n : n], buf[102 * n : n], buf[103 * n : n], buf[104 * n : n], buf[105 * n : n], buf[106 * n : n], buf[107 * n : n], buf[108 * n : n], buf[109 * n : n], buf[110 * n : n],\
    // buf[111 * n : n], buf[112 * n : n], buf[113 * n : n], buf[114 * n : n], buf[115 * n : n], buf[116 * n : n], buf[117 * n : n], buf[118 * n : n], buf[119 * n : n], buf[120 * n : n],\
    // buf[121 * n : n], buf[122 * n : n], buf[123 * n : n], buf[124 * n : n], buf[125 * n : n], buf[126 * n : n], buf[127 * n : n]) depend(out: dst[0 : col])
    // {
    //     //tid = omp_get_thread_num();
    //     //tstart = omp_get_wtime();
    //     for(i = 0 ; i < nbuf ; i++) //threads
    //     {
    //         for(j = 0 ; j < col ; j++) //each col
    //         {
    //             dst[j] += buf[i * col + j]; 
                
    //         }
    //     } //end for

    //     //tend = omp_get_wtime();
    //     //taskTiming[6][tid] += (tend - tstart); 
    // } //end task

    //including d_residualNorms
    #pragma omp target is_device_ptr(buf, dst) firstprivate(nbuf, n, col)\
    depend(in: buf[0 * n : n], buf[1 * n : n], buf[2 * n : n], buf[3 * n : n], buf[4 * n : n], buf[5 * n : n], buf[6 * n : n], buf[7 * n : n], buf[8 * n : n], buf[9 * n : n], buf[10 * n : n],\
    buf[11 * n : n], buf[12 * n : n], buf[13 * n : n], buf[14 * n : n], buf[15 * n : n], buf[16 * n : n], buf[17 * n : n], buf[18 * n : n], buf[19 * n : n], buf[20 * n : n],\
    buf[21 * n : n], buf[22 * n : n], buf[23 * n : n], buf[24 * n : n], buf[25 * n : n], buf[26 * n : n], buf[27 * n : n], buf[28 * n : n], buf[29 * n : n], buf[30 * n : n],\
    buf[31 * n : n], buf[32 * n : n], buf[33 * n : n], buf[34 * n : n], buf[35 * n : n], buf[36 * n : n], buf[37 * n : n], buf[38 * n : n], buf[39 * n : n], buf[40 * n : n],\
    buf[41 * n : n], buf[42 * n : n], buf[43 * n : n], buf[44 * n : n], buf[45 * n : n], buf[46 * n : n], buf[47 * n : n], buf[48 * n : n], buf[49 * n : n], buf[50 * n : n],\
    buf[51 * n : n], buf[52 * n : n], buf[53 * n : n], buf[54 * n : n], buf[55 * n : n], buf[56 * n : n], buf[57 * n : n], buf[58 * n : n], buf[59 * n : n], buf[60 * n : n],\
    buf[61 * n : n], buf[62 * n : n], buf[63 * n : n], buf[64 * n : n], buf[65 * n : n], buf[66 * n : n], buf[67 * n : n], buf[68 * n : n], buf[69 * n : n], buf[70 * n : n],\
    buf[71 * n : n], buf[72 * n : n], buf[73 * n : n], buf[74 * n : n], buf[75 * n : n], buf[76 * n : n], buf[77 * n : n], buf[78 * n : n], buf[79 * n : n], buf[80 * n : n],\
    buf[81 * n : n], buf[82 * n : n], buf[83 * n : n], buf[84 * n : n], buf[85 * n : n], buf[86 * n : n], buf[87 * n : n], buf[88 * n : n], buf[89 * n : n], buf[90 * n : n],\
    buf[91 * n : n], buf[92 * n : n], buf[93 * n : n], buf[94 * n : n], buf[95 * n : n], buf[96 * n : n], buf[97 * n : n], buf[98 * n : n], buf[99 * n : n], buf[100 * n : n],\
    buf[101 * n : n], buf[102 * n : n], buf[103 * n : n], buf[104 * n : n], buf[105 * n : n], buf[106 * n : n], buf[107 * n : n], buf[108 * n : n], buf[109 * n : n], buf[110 * n : n],\
    buf[111 * n : n], buf[112 * n : n], buf[113 * n : n], buf[114 * n : n], buf[115 * n : n], buf[116 * n : n], buf[117 * n : n], buf[118 * n : n], buf[119 * n : n], buf[120 * n : n],\
    buf[121 * n : n], buf[122 * n : n], buf[123 * n : n], buf[124 * n : n], buf[125 * n : n], buf[126 * n : n], buf[127 * n : n]) depend(out: dst[0 : col])
    {
        //tid = omp_get_thread_num();
        //tstart = omp_get_wtime();
        for(i = 0 ; i < nbuf ; i++) //threads
        {
            for(j = 0 ; j < col ; j++) //each col
            {
                dst[j] += buf[i * col + j]; 
                
            }
        } //end for

        //tend = omp_get_wtime();
        //taskTiming[6][tid] += (tend - tstart); 
    } //end task
}

void sum_sqrt_task_SQRT(double *dst, const int col)
{
    //code: 7
    int i, j, k, l, length, blksz, tid;
    int nbuf = nthrds;
    double tstart, tend;

    // #pragma omp task private(i, tid, tstart, tend)\
    // firstprivate(col)\
    // depend(inout: dst[0 : col])
    // {
    //     //tid = omp_get_thread_num();
    //     //tstart = omp_get_wtime();

    //     for(i = 0; i < col; i++) //i->col
    //     {
    //         // printf("%lf - ", dst[i]);
    //         dst[i] = sqrt(dst[i]);
    //         // printf("%lf\n", dst[i]);
    //     }
    //     //printf("\n\n");

    //     //tend = omp_get_wtime();
    //     //taskTiming[7][tid] += (tend - tstart);         
    // }

    #pragma omp target firstprivate(col) is_device_ptr(dst)\
    depend(inout: dst[0 : col])
    {
        for(i = 0; i < col; i++)
            dst[i] = sqrt(dst[i]);
    }
}


#endif

#endif
