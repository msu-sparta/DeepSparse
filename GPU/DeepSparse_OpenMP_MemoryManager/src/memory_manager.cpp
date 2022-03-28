#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

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
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <map>

using namespace std;

#include <cuda_runtime.h>
#include <omp.h>

#include "../inc/util.h"
#include "../inc/matrix_ops_cpu.h"
#include "../inc/matrix_ops_gpu_v6.h"
#include "../inc/memory_manager_v6.h"


struct pair_hash
{
	template <class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2> &pair) const
	{
		return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
	}
};

struct EQUALPAIR 
{
    inline bool operator()(const pair<double * , int> pr1, const pair<double * , int> pr2) const 
    {
        return ((pr1.first == pr2.first) && (pr1.second == pr2.second));
    }
};


double capacity = 10 * 1e+9;
double available_mem = capacity;
double allocated_mem = 0;
unsigned long int head = 0;
int device_id, host_id;
double *d_memory;
unsigned long int num_per_blk = 1048576; //524288; //1048576; //786432; //524288; //786432; //524288; // 125000;
double memGranularity = num_per_blk * sizeof(double); //# of doubles times sizeof each double
unsigned long int totalMemBlock;
unsigned long int viewIndex = 0;
unsigned long int memBlockIndex = 0; // always points to the next available device memory block
bool inEvictionMode = false;
bool errorFlag = false;
vector<double> mmTiming(10, 0);


// KEY: pair<double *, int> = <pointer to the matrix, block_id>
// VALUE: vector<double> = <offset (mem block index), size (# of elements), total memory blocks required, timestamp>
unordered_map<pair<double *, int>, vector<double>, pair_hash, EQUALPAIR> mp;
map<unsigned long int, vector<unsigned long int>> freeBlockMap;
unordered_map<double *, vector<int>> matrixDimension;
unordered_map<double *, string> matrixName;
queue<pair<double *, int>> evictionQueue;
vector<pair<double *, int>> memView;
map<long, double> h2dTime, d2hTime;
map<long, double> h2dTransfer, d2hTransfer;
map<long, double> mmLoopTime, isDeviceFunctionTime, pushPopTime, freeBlockTiming, grabRHSTiming, dsUpdateTiming, copyToHostTiming;

// how much memory is availble on device
double availbleDeviceMemory()
{
    return available_mem;
}

// capacity of the current device
double deviceCapacity()
{
    return capacity;
}

// how much device memory is used so far
size_t allocated()
{
    return capacity - available_mem;
}


// This function checks if all memory blocks on device 
// have been occupied atleast once by any memory blocks
bool isAllBlocksOccupied()
{    
    return (memBlockIndex >= totalMemBlock);
}

// check wheather the block is on device memory or not
bool isOnDevice(pair<double *, int> pr)
{
    if(mp.find(pr) != mp.end())
        return true;
    
    return false;
}

// for a given chunk size (in byte), this function determines the # of required blocks on device
unsigned long int requiredDeviceBlocks(double blkz_in_byte, double memGranularity)
{
    return ceil(blkz_in_byte/memGranularity);
}

unsigned long int availableDeviceBlocks()
{
    if(memBlockIndex >= totalMemBlock)
        return 0;
    else if(memBlockIndex < totalMemBlock)
        return totalMemBlock - memBlockIndex;
    
    return -1; // error
}

unsigned long int availableBlocksOnRight(unsigned long int memView_index)
{
    return totalMemBlock - memView_index;
}

//utility
int copyToHost(double *host_mtx, int row, int col, int block_id, int block_width, int num_element, long iterationNumber)
{
    int status = -1;
    // cudaError_t status;  
    pair<double *, int> pr = make_pair(host_mtx, block_id); 
    double tstart, tend;

    if(isOnDevice(pr))
    {
        #if defined(TIMER)
            tstart = omp_get_wtime();
        #endif

        status = omp_target_memcpy(host_mtx + (block_id * block_width * col), d_memory + (unsigned long int)mp[pr][0] * num_per_blk, num_element * sizeof(double), 0, 0, host_id, device_id);
        // status = cudaMemcpy(host_mtx + (block_id * block_width * col), d_memory + (unsigned long int)mp[pr][0] * num_per_blk, num_element * sizeof(double), cudaMemcpyDeviceToHost);

        //cudaDeviceSynchronize();

        #if defined(TIMER)
            tend = omp_get_wtime();
            d2hTime[iterationNumber - 1] += tend - tstart;
            d2hTransfer[iterationNumber - 1] += num_element * sizeof(double) * 1e-9;
            
            #if defined(DEBUG4)
            if(num_element * sizeof(double) * 1e-9/(tend - tstart) < 2.0)
                cout << "   *** copyToHost -- D2H: " << num_element * sizeof(double) * 1e-6 << " MB @ " << tend - tstart << " sec. --> Rate : " << num_element * sizeof(double) * 1e-9/(tend - tstart) << " GB/sec" << endl;
            #endif
        #endif
    }
    else
    {
        cout << "   *** copyToHost: index: " << (unsigned long int)mp[pr][0]  << " block_id: " << block_id << " row: " << row << " col: " << col  << "is not in Device" << endl;
        // cout << "   *** Mtx block is not in Device -- copyToHost" << endl;
    }

    if(status != 0)
        cout << "   *** copyToHost: Error while copying index: " << (unsigned long int)mp[pr][0]  << " block_id: " << block_id << " row: " << row << " col: " << col << endl;

    return (status == 0 ? 0 : -1);
}

template <typename T1, typename T2>
typename std::map<T1, T2>::const_iterator nearest_key(const std::map<T1, T2>& mp, T1 key)
{
    // auto exact_key = mp.find(key);
    // if(exact_key != mp.end())
    //     return exact_key;

    auto lower_bound = mp.lower_bound(key);
    if (lower_bound == mp.end()) return --lower_bound;
    // cout << lower_bound->first << endl;
    // cout << "Lower bound check" << endl;
    
    auto upper_bound = lower_bound; upper_bound++;
    // cout << "Upper bound check" << endl;
    if (upper_bound == mp.end()) return lower_bound;
    
    auto dist_to_lower = lower_bound->first - key;
    auto dist_to_upper = upper_bound->first - key;
    // cout << "Dist check" << endl;
    
    return (dist_to_upper <= dist_to_lower) ? upper_bound : lower_bound;
}


int copyToDevice(double *mtx, int row, int col, int block_id, int offset, int num_element, double isModified, long iterationNumber)
{
    // mtx + offset = takes us to the starting address of block_id that needs to be copied to device
    
    int status, temp_status;
    // int temp_status; 
    // cudaError_t status;  
    double chunk_size = num_element * sizeof(double);
    unsigned long int required_blk = requiredDeviceBlocks(chunk_size, memGranularity);
    unsigned long int ii;
    double tstart, tend, temp_start, temp_start_1;
    double timer_full;

    #if defined(DEBUG3)
        timer_full = omp_get_wtime();
    #endif

    // we have required # of unused memory blocks on device
    if(!isAllBlocksOccupied() && required_blk <= availableDeviceBlocks()) 
    {
        #if defined(TIMER)
            tstart = omp_get_wtime();
        #endif
        
        status = omp_target_memcpy(d_memory + memBlockIndex * num_per_blk, mtx + offset, chunk_size, 0, 0, device_id, host_id);
        // status = cudaMemcpy(d_memory + memBlockIndex * num_per_blk, mtx + offset, chunk_size, cudaMemcpyHostToDevice);

        //cudaDeviceSynchronize();

        #if defined(TIMER)
            tend = omp_get_wtime();
            h2dTime[iterationNumber - 1] +=  tend - tstart;
            h2dTransfer[iterationNumber - 1] += chunk_size * 1e-9;
            
            #if defined(DEBUG4)
            if(chunk_size * 1e-9/(tend - tstart) < 2.0)
                cout << "   *** copyToDevice -- (first if) H2D: " << chunk_size * 1e-6 << " MB @ " << tend - tstart << " sec. --> Rate : " << chunk_size * 1e-9/(tend - tstart) << " GB/sec" << endl;
            #endif

        #endif

        if(status == 0)
        {
            #if defined(TIMER)
                tstart = omp_get_wtime();
            #endif

            pair<double *, int> newPair = make_pair(mtx, block_id);
            mp[newPair] = {memBlockIndex * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
            
            // cout << "copyToDevice -- in first if ==> for loop" << endl;
            // update memory view before incrementing memBlockIndex
            for(ii = memBlockIndex ; ii < memBlockIndex + required_blk ; ii++)
            {
                memView[ii] = newPair;
            }
            // cout << "copyToDevice -- in first if ==> for loop end" << endl;
            // inserting newly added blocks into the eviction queue
            evictionQueue.push(newPair);
            // update head
            memBlockIndex += required_blk; 
            // available_mem -= required_blk * memGranularity;
            // cout << "copyToDevice ==> memBlockIndex : " << memBlockIndex << endl;
            
            #if defined(TIMER)
                tend = omp_get_wtime();
                mmLoopTime[iterationNumber - 1] += tend - tstart;
                dsUpdateTiming[iterationNumber - 1] += tend - tstart;
            #endif

            // #if defined(DEBUG3)
            //     cout << "   *** copyToDevice -- first if time: " << omp_get_wtime() - timer_full << " sec." << endl;
            // #endif

            return status;
        }
        else
        {
            printf("    * copyToDevice -- omp_target_memcpy failed in case of avaliable blocks on device ==> %d\n", status); 
            errorFlag = true;
            return status; 
        }
    }
    else //** enough space is not available on device memory
    {
        //** check if enough free memory blocks are available
        
        bool fille_by_free_block = false;

        if(freeBlockMap.size() > 0)
        {
            if(freeBlockMap.find(required_blk) != freeBlockMap.end() && freeBlockMap[required_blk].size() > 0) 
            {
                // so we got an empty spase of perfect size
                unsigned long int d_index = freeBlockMap[required_blk].back(); // d_memory + d_offset = starting address of the empty block
                freeBlockMap[required_blk].pop_back();

                #if defined(TIMER)
                    tstart = omp_get_wtime();
                #endif
                
                status = omp_target_memcpy(d_memory + d_index * num_per_blk, mtx + offset, num_element * sizeof(double), 0, 0, device_id, host_id);
                // status = cudaMemcpy(d_memory + d_index * num_per_blk, mtx + offset, num_element * sizeof(double), cudaMemcpyHostToDevice);
                
                //cudaDeviceSynchronize();

                #if defined(TIMER)
                    tend = omp_get_wtime();
                    h2dTime[iterationNumber - 1] += tend - tstart;
                    h2dTransfer[iterationNumber - 1] += num_element * sizeof(double) * 1e-9;

                    #if defined(DEBUG4)
                    if(num_element * sizeof(double) * 1e-9/(tend - tstart) < 2.0)
                        cout << "   *** copyToDevice -- free block (case 1) H2D: " << num_element * sizeof(double) * 1e-6 << " MB @ " << tend - tstart << " sec. --> Rate : " << num_element * sizeof(double) * 1e-9/(tend - tstart) << " GB/sec" << endl;
                    #endif
                #endif

                #if defined(DEBUG)
                    cout << "   *** copyToDevice -- free block (case 1) removing free block at index : " << d_index << " (" << matrixName[memView[d_index].first] << ", " << memView[d_index].second << ") required_blk: " << required_blk << endl;
                #endif

                if(status == 0)
                {
                    #if defined(TIMER)
                        tstart = omp_get_wtime();
                    #endif

                    fille_by_free_block = true;
                    // update major DS
                    pair<double *, int> newPair = make_pair(mtx, block_id);
                    mp[newPair] = {d_index * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                    evictionQueue.push(newPair);

                    for(ii = d_index ; ii < d_index + required_blk ; ii++)
                    {
                        memView[ii] = newPair;
                    }
                    if(freeBlockMap[required_blk].size() == 0)
                        freeBlockMap.erase(required_blk);
                    
                    #if defined(TIMER)
                        tend = omp_get_wtime();
                        mmLoopTime[iterationNumber - 1] += tend - tstart;
                        freeBlockTiming[iterationNumber - 1] += tend - tstart;
                    #endif
                    
                    #if defined(DEBUG3)
                        cout << "   *** copyToDevice -- free block time (case 1): " << omp_get_wtime() - timer_full << " sec." << endl;
                    #endif

                    return status;
                }
                else
                {
                    printf("    * copyToDevice -- free block case 1 omp_target_memcpy failed ==> %d\n", status); 
                    return status; 
                }
            }
            else
            {
                auto fesible_block = nearest_key(freeBlockMap, required_blk);
                unsigned long int free_block_key = fesible_block->first;

                #if defined(DEBUG)
                    cout << "   * copyToDevice -- fesible_block KEY: " << fesible_block->first << " VECTOR SIZE : " << fesible_block->second.size() << endl;
                #endif

                if(fesible_block->first > required_blk && !fesible_block->second.empty())
                {
                    unsigned long int memView_index = fesible_block->second.back();
                    freeBlockMap[free_block_key].pop_back();
                    unsigned long int occupied_blk = fesible_block->first;
                    unsigned long int free_block_count = occupied_blk - required_blk;

                    #if defined(TIMER)
                        tstart = omp_get_wtime();
                    #endif

                    status = omp_target_memcpy(d_memory + memView_index * num_per_blk, mtx + offset, chunk_size, 0, 0, device_id, host_id);
                    // status = cudaMemcpy(d_memory + memView_index * num_per_blk, mtx + offset, chunk_size, cudaMemcpyHostToDevice);
                    
                    //cudaDeviceSynchronize();

                    #if defined(TIMER)
                        tend = omp_get_wtime();
                        h2dTime[iterationNumber - 1] += tend - tstart;
                        h2dTransfer[iterationNumber - 1] += chunk_size * 1e-9;
                        #if defined(DEBUG4)
                        if(chunk_size * 1e-9/(tend - tstart) < 2.0)
                            cout << "   *** copyToDevice -- free block (case 2) H2D: " << chunk_size * 1e-6 << " MB @ " << tend - tstart << " sec. --> Rate : " << chunk_size * 1e-9/(tend - tstart) << " GB/sec" << endl;
                        #endif
                    #endif

                    if(status == 0)
                    {
                        #if defined(TIMER)
                            tstart = omp_get_wtime();
                        #endif

                        fille_by_free_block = true;
                        // what we need to do now??
                        pair<double *, int> newPair = make_pair(mtx, block_id);
                        mp[newPair] = {memView_index * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                        evictionQueue.push(newPair);

                        #if defined(DEBUG)
                            cout << "   *** copyToDevice -- free block (case 2) updating memView: " << memView_index << " -- " << memView_index + required_blk - 1 << " (" << matrixName[newPair.first] << " , " << newPair.second << ") rquired: " << required_blk << " occupied: " << occupied_blk << endl;
                        #endif

                        //** update memoryView
                        for(ii =  memView_index; ii < memView_index + required_blk ; ii++)
                        {
                            memView[ii] = newPair;    
                        }

                        if(free_block_count > 0)
                        {
                            #if defined(DEBUG)
                                cout << "   *** copyToDevice -- free block (case 2) free_block_count: " << free_block_count << " at index: " << memView_index + required_blk << " -- " << memView_index + occupied_blk - 1 << endl;
                            #endif

                            // ** MERGING: see if there are any mergable free blocks on the right.
                                
                            unsigned long int free_block_starting_index = memView_index + required_blk;
                            unsigned long int total_free_block = free_block_count;
                            unsigned long int look_ahead_index = free_block_starting_index + free_block_count;
                                
                            // cout << "   *** copyToDevice -- case 1 look_ahead_index: " << look_ahead_index << " total_free_block: " << total_free_block << endl;

                            while(look_ahead_index < totalMemBlock && memView[look_ahead_index].first == nullptr && memView[look_ahead_index].second > 0)
                            {
                                unsigned long int next_free_block_count = memView[look_ahead_index].second;
                        
                                #if defined(DEBUG)
                                    cout << "       *** copyToDevice -- free block (case 2) free block MERGING -- look_ahead_index: " << look_ahead_index << " --> " << matrixName[memView[look_ahead_index].first] << ", " << memView[look_ahead_index].second << ") next_free_block_count: " << next_free_block_count << " sanity: " << count(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index)  << endl;
                                #endif

                                // removing the look_ahead_index from freeBlockMap[next_free_block_count]
                                freeBlockMap[memView[look_ahead_index].second].erase(remove(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index), freeBlockMap[memView[look_ahead_index].second].end());
                                if(freeBlockMap[next_free_block_count].empty())
                                    freeBlockMap.erase(next_free_block_count);
                                // next lookup
                                look_ahead_index += next_free_block_count;
                                total_free_block += next_free_block_count;
                            } 


                            #if defined(DEBUG)
                                if(total_free_block > free_block_count)
                                {
                                    cout << "   *** copyToDevice -- free block (case 2) free block MERGING -- total_free_block: " << total_free_block << " ( " << free_block_starting_index << " -- " <<  free_block_starting_index + total_free_block - 1 << " )"<< endl;
                                    for(ii =  memView_index + occupied_blk ; ii < free_block_starting_index + total_free_block ; ii++)
                                    {
                                        if(memView[ii].first != nullptr)
                                            cout << "   *** copyToDevice -- free block (case 2) free block ALERT: " << ii << " ( " << matrixName[memView[ii].first] << ", " << memView[ii].second << " )"<< endl;
                                    }   
                                }
                            #endif

                            freeBlockMap[total_free_block].push_back(free_block_starting_index); // OR: free_block_starting_index
                                
                            unsigned long int temp_count = total_free_block; // REUSING free_block_count variable as a counter for decrementing

                            #if defined(DEBUG) 
                                cout << "   *** copyToDevice -- free block (case 2) (final) total_free_block: " << total_free_block << " at index: " << free_block_starting_index << " -- " << free_block_starting_index + total_free_block - 1 << endl;
                            #endif

                            for(ii =  free_block_starting_index ; ii < free_block_starting_index + total_free_block ; ii++)
                            {
                                memView[ii] = make_pair(nullptr, temp_count);
                                temp_count--;
                            }
                        }

                        if(fesible_block->second.empty())
                            freeBlockMap.erase(fesible_block->first);

                        #if defined(TIMER)
                            // mmLoopTime[iterationNumber - 1] += omp_get_wtime() - tstart;
                            tend = omp_get_wtime();
                            mmLoopTime[iterationNumber - 1] += tend - tstart;
                            freeBlockTiming[iterationNumber - 1] += tend - tstart;
                        #endif

                        #if defined(DEBUG3)
                            cout << "   *** copyToDevice -- free block time (case 2): " << omp_get_wtime() - timer_full << " sec." << endl;
                        #endif

                        return status;
                    } // end if(status == 0)
                    else
                    {
                        printf("    * copyToDevice -- free block case 2 omp_target_memcpy failed ==> %d\n", status); 
                        return status; 
                    }
                } // end if
            }
        }

        //evict a necessary blocks
        if(evictionQueue.size() > 0 && !fille_by_free_block) 
        {
            if(!inEvictionMode) inEvictionMode = true;

            // cout << "********* Trying to evict in copyToDevice *********" << endl;

            // *** should we check if evictableMatrixBlock is infact in mp first?? ==> I guess YES!! ==> add this checking later
            // *** we should also check if evictableMatrixBlock is in mp ==> replace this by a while loop
            #if defined(TIMER)
                tstart = omp_get_wtime();
            #endif

            int pop_count = 0;
            while(evictionQueue.size() > 0 && !isOnDevice(evictionQueue.front())) 
            {    
                evictionQueue.pop();
                pop_count++;
            }
            
            #if defined(TIMER)
                tend = omp_get_wtime();
                mmLoopTime[iterationNumber - 1] += tend - tstart;
                pushPopTime[iterationNumber - 1] += tend - tstart;
            #endif

            #if defined(DEBUG)
                cout << "   ** copyToDevice -- pop_count: " << pop_count << " evictionQueue size - " << evictionQueue.size() << " -- freeBlockMap size - " << freeBlockMap.size() << endl;
            #endif
            
            // now we need to evict one or multiple blocks from the eviction queue to put the new mtx on device
            if(evictionQueue.size() > 0)
            {
                pair<double *, int> evictableMatrixBlock = evictionQueue.front();

                // *** case - 1 & 3: # of mem blocks occupied by evictable mtx block is >= to the required mem blocks
                if((unsigned long int) mp[evictableMatrixBlock][2] >= required_blk)
                {
                    // cout << "********* Eviction case 1 -- copyToDevice" << endl;
                    // cout << "   ** copyToDevice -- Eviction case 1" << endl;
                    //first copy evictableMatrixBlock from device to host ==> then put back (mtx, block_id) at the same place

                    if((unsigned long int) mp[evictableMatrixBlock][5] == 1) // evictableMatrixBlock is modified ==> need to copy back to host
                    {
                        // cout << "   ** copyToDevice -- Eviction case 1 -- evictableMatrixBlock is modified" << endl;
                        temp_status = copyToHost(evictableMatrixBlock.first, (unsigned long int) mp[evictableMatrixBlock][3], (unsigned long int) mp[evictableMatrixBlock][4], 
                                            (unsigned long int) evictableMatrixBlock.second, block_width, mp[evictableMatrixBlock][1], iterationNumber);
                        #if defined(DEBUG)
                            cout << "   *** copyToDevice -- case 1 Evict -- (" << matrixName[evictableMatrixBlock.first]  << " , " << (unsigned long int) evictableMatrixBlock.second << " ) @ " << (unsigned long int) mp[evictableMatrixBlock][0] << " (" << (unsigned long int) mp[evictableMatrixBlock][2]<< ") --> Copy -- (" << matrixName[mtx] << " , " << block_id << ") --> " << required_blk << endl;
                            // cout << "   ** copyToDevice -- case 1 Evicting -- (" << matrixName[evictableMatrixBlock.first] << " , " << (unsigned long int) evictableMatrixBlock.second << " , " << occupied_blk << " , " << memView_index << "), Copying -- (" << matrixName[mtx] << " , " << block_id << " , " << required_blk << " , " << (unsigned long int) mp[newPair][0] << ")" << endl;
                        #endif
                    }
                    else
                    {
                        temp_status = 0; //leting the next copy to go.

                        #if defined(DEBUG)
                            cout << "   ** copyToDevice -- case 1 Purge -- (" << matrixName[evictableMatrixBlock.first]  << " , " << (unsigned long int) evictableMatrixBlock.second << " ) @ " << (unsigned long int) mp[evictableMatrixBlock][0] << " (" << (unsigned long int) mp[evictableMatrixBlock][2]<< ") --> Copy -- (" << matrixName[mtx] << " , " << block_id << ") --> " << required_blk << endl;
                        #endif

                        // cout << "   ** copyToDevice -- case 1 not Evicting -- (" << matrixName[evictableMatrixBlock.first] << " , " << (unsigned long int) evictableMatrixBlock.second << " , " << occupied_blk << " , " << memView_index << "), Copying -- (" << matrixName[mtx] << " , " << block_id << " , " << required_blk << " , " << (unsigned long int) mp[newPair][0] << ")" << endl;
                    }
                    // if we successfully copy the evictable mtx to host
                    if(temp_status == 0)
                    {
                        #if defined(TIMER)
                            tstart = omp_get_wtime();
                        #endif

                        // now copy the mtx block on the same spot 
                        status = omp_target_memcpy(d_memory + (unsigned long int) mp[evictableMatrixBlock][0] * num_per_blk, mtx + offset, chunk_size, 0, 0, device_id, host_id);
                        // status = cudaMemcpy(d_memory + (unsigned long int) mp[evictableMatrixBlock][0] * num_per_blk, mtx + offset, chunk_size, cudaMemcpyHostToDevice);
                        
                        //cudaDeviceSynchronize();

                        #if defined(TIMER)
                            tend = omp_get_wtime();
                            h2dTime[iterationNumber - 1] += tend - tstart;
                            h2dTransfer[iterationNumber - 1] += chunk_size * 1e-9;
                            
                            #if defined(DEBUG4)
                            if(chunk_size * 1e-9/(tend - tstart) < 2.0)
                                cout << "   *** copyToDevice -- eviction (case 1) H2D: " << chunk_size * 1e-6 << " MB @ " << tend - tstart << " sec. --> Rate : " << chunk_size * 1e-9/(tend - tstart) << " GB/sec" << endl;
                            #endif
                        #endif

                        if(status == 0)
                        {
                            #if defined(TIMER)
                                tstart = omp_get_wtime();
                            #endif
                            
                            // what we need to do now??
                            pair<double *, int> newPair = make_pair(mtx, block_id);
                            mp[newPair] = {mp[evictableMatrixBlock][0], num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};

                            // cout << "   ** copyToDevice -- case 1 Evicttion reconfirming index - " << mp[evictableMatrixBlock][0] << endl;
                            
                            pair<double *, int> nullPr = make_pair(nullptr, -1);
                            unsigned long int memView_index = (unsigned long int) mp[evictableMatrixBlock][0];
                            unsigned long int occupied_blk = (unsigned long int) mp[evictableMatrixBlock][2];
                            unsigned long int free_block_count = occupied_blk - required_blk;
                            
                            // if(free_block_count > 0)
                            //     cout << "   *** copyToDevice -- case 1 free_block_count: " << free_block_count << endl;
                            
                            #if defined(DEBUG)
                                cout << "   *** copyToDevice -- case 1 updating memView: " << memView_index << " -- " << memView_index + required_blk - 1 << " (" << matrixName[newPair.first] << " , " << newPair.second << ") rquired: " << required_blk << " occupied: " << occupied_blk << endl;
                            #endif

                            //*** update memoryView
                            for(ii =  memView_index; ii < memView_index + required_blk ; ii++)
                            {
                                memView[ii] = newPair;    
                            }
                            
                            #if defined(TIMER)
                                dsUpdateTiming[iterationNumber - 1] += omp_get_wtime() - tstart;
                                temp_start = omp_get_wtime();
                            #endif

                            // *** Merging & Updating free block info
                            if(free_block_count > 0)
                            {
                                #if defined(DEBUG)
                                    cout << "   *** copyToDevice -- case 1 free_block_count: " << free_block_count << " at index: " << memView_index + required_blk << " -- " << memView_index + occupied_blk - 1 << endl;
                                #endif

                                // //** MERGING: see if there are any mergable free blocks on the right.
                                
                                unsigned long int free_block_starting_index = memView_index + required_blk;
                                unsigned long int total_free_block = free_block_count;
                                unsigned long int look_ahead_index = free_block_starting_index + free_block_count;
                                
                                // cout << "   *** copyToDevice -- case 1 look_ahead_index: " << look_ahead_index << " total_free_block: " << total_free_block << endl;

                                while(look_ahead_index < totalMemBlock && memView[look_ahead_index].first == nullptr && memView[look_ahead_index].second > 0)
                                {
                                    unsigned long int next_free_block_count = memView[look_ahead_index].second;
                                    
                                    #if defined(DEBUG)
                                        cout << "       *** copyToDevice -- case 1 free block MERGING -- look_ahead_index: " << look_ahead_index << " --> " << matrixName[memView[look_ahead_index].first] << ", " << memView[look_ahead_index].second << ") next_free_block_count: " << next_free_block_count << " sanity: " << count(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index)  << endl;
                                    #endif
                                    
                                    // removing the look_ahead_index from freeBlockMap[next_free_block_count]
                                    freeBlockMap[memView[look_ahead_index].second].erase(remove(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index), freeBlockMap[memView[look_ahead_index].second].end());
                                    if(freeBlockMap[next_free_block_count].empty())
                                        freeBlockMap.erase(next_free_block_count);
                                    // next lookup
                                    look_ahead_index += next_free_block_count;
                                    total_free_block += next_free_block_count;
                                } 

                                #if defined(DEBUG)
                                    if(total_free_block > free_block_count)
                                    {
                                        cout << "   *** copyToDevice -- case 1 free block MERGING -- total_free_block: " << total_free_block << " ( " << free_block_starting_index << " -- " <<  free_block_starting_index + total_free_block - 1 << " )"<< endl;
                                        for(ii =  memView_index + occupied_blk ; ii < free_block_starting_index + total_free_block ; ii++)
                                        {
                                            if(memView[ii].first != nullptr)
                                                cout << "       *** copyToDevice -- case 1 free block ALERT: " << ii << " ( " << matrixName[memView[ii].first] << ", " << memView[ii].second << " )"<< endl;
                                        }   
                                    }
                                #endif
                                
                                freeBlockMap[total_free_block].push_back(free_block_starting_index); // OR: free_block_starting_index
                                
                                unsigned long int temp_count = total_free_block; // REUSING free_block_count variable as a counter for decrementing
                                
                                #if defined(DEBUG)
                                    cout << "   *** copyToDevice -- case 1 (final) total_free_block: " << total_free_block << " at index: " << free_block_starting_index << " -- " << free_block_starting_index + total_free_block - 1 << endl;
                                #endif

                                for(ii =  free_block_starting_index ; ii < free_block_starting_index + total_free_block ; ii++)
                                {
                                    memView[ii] = make_pair(nullptr, temp_count);
                                    temp_count--;
                                }


                                //WORKING:

                                // freeBlockMap[free_block_count].push_back(memView_index + required_blk);

                                // for(ii =  memView_index + required_blk; ii < memView_index + occupied_blk ; ii++)
                                // {
                                //     memView[ii] = make_pair(nullptr, free_block_count);
                                //     free_block_count--;
                                // }
                            }

                            evictionQueue.pop(); // removing the first element fromt the queue
                            evictionQueue.push(newPair);
                            mp.erase(evictableMatrixBlock);

                            #if defined(TIMER)
                                tend = omp_get_wtime();
                                mmLoopTime[iterationNumber - 1] += tend - tstart;
                                freeBlockTiming[iterationNumber - 1] += tend - temp_start;
                            #endif

                            #if defined(DEBUG3)
                                cout << "   *** copyToDevice -- eviction (case 1): " << omp_get_wtime() - timer_full << " sec." << endl;
                            #endif

                            return status;
                        }
                        else
                        {
                            printf("    * copyToDevice -- omp_target_memcpy failed in case of avaliable blocks on device ==> %d\n", status); 
                            errorFlag = true;
                            return status; 
                        }
                    }
                    else
                    {
                        printf("    ** copyToDevice -- evictable mtx is not copied back to host (case 1)==> %d\n", status); 
                        return status;
                    }
                    // }
                    // else // evictableMatrixBlock is not modified ==> no need to copy back to host
                    // {
                        
                    //     pair<double *, int> newPair = make_pair(mtx, block_id);
                    //     mp[newPair] = {mp[evictableMatrixBlock][0], num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                            
                    //     pair<double *, int> nullPr = make_pair(nullptr, -1);
                    //     unsigned long int memView_index = (unsigned long int) mp[evictableMatrixBlock][0];
                    //     unsigned long int occupied_blk = (unsigned long int) mp[evictableMatrixBlock][2];
                    //     cout << "   ** copyToDevice -- case 1 not Evicting -- (" << matrixName[evictableMatrixBlock.first] << " , " << (unsigned long int) evictableMatrixBlock.second << " , " << occupied_blk << " , " << memView_index << "), Copying -- (" << matrixName[mtx] << " , " << block_id << " , " << required_blk << " , " << (unsigned long int) mp[newPair][0] << ")" << endl;
                    //     // update memory view before incrementing memBlockIndex
                    //     for(ii =  memView_index; ii < memView_index + occupied_blk ; ii++)
                    //     {
                    //         if(ii < memView_index + required_blk)
                    //             memView[ii] = newPair;
                    //         else
                    //             memView[ii] = nullPr;
                    //         // *** update empty space later as we are dropping it off now
                    //     }
                    //     mp.erase(evictableMatrixBlock);
                    //     evictionQueue.pop(); // removing the first element fromt the queue
                    //     evictionQueue.push(newPair);
                    //     return 0; 
                    // }
                }
                //*** case - 2: # of mem blocks occupied by evictable mtx block is less than required mem blocks
                else if((unsigned long int) mp[evictableMatrixBlock][2] < required_blk)
                {
                    #if defined(TIMER)
                        tstart = omp_get_wtime();
                    #endif

                    #if defined(DEBUG)
                        cout << "   *** copyToDevice -- case 2 Evicting -- (" << matrixName[evictableMatrixBlock.first]  << " , " << (unsigned long int) evictableMatrixBlock.second << " , " << (unsigned long int) mp[evictableMatrixBlock][0] << " , " << (unsigned long int) mp[evictableMatrixBlock][2]<< "), Copying -- (" << matrixName[mtx] << " , " << block_id << " , " << required_blk <<")"  << " isModified: " << mp[evictableMatrixBlock][5] << endl;
                    #endif
                    int push_back_count = 0;
                    // we can only go to the right and we need to make sure there are enough spaces to my right to hold the matrix block needs to be copied
                    unsigned long int blocksOnRight = availableBlocksOnRight((unsigned long int) mp[evictableMatrixBlock][0]);
                    while(evictionQueue.size() > 0 && blocksOnRight < required_blk)
                    {
                        if(isOnDevice(evictionQueue.front()))
                        {
                            evictableMatrixBlock = evictionQueue.front();
                            blocksOnRight = availableBlocksOnRight((unsigned long int) mp[evictableMatrixBlock][0]);
                            
                            #if defined(DEBUG)
                                cout << "   ** copyToDevice -- Eviction case 2 - blocksOnRight: " << blocksOnRight <<  " required_blk: " << required_blk << " front: (" << matrixName[evictableMatrixBlock.first] << " , " << evictableMatrixBlock.second << ")"<< endl;
                            #endif

                            if(blocksOnRight < required_blk)
                            {
                                #if defined(DEBUG)
                                    cout << "   ** copyToDevice -- pushed back" << endl;
                                #endif

                                evictionQueue.pop();
                                evictionQueue.push(evictableMatrixBlock);
                                push_back_count++;
                            }
                        }
                        else
                        {
                            evictionQueue.pop();
                        }
                    }

                    #if defined(TIMER)
                        pushPopTime[iterationNumber - 1] += omp_get_wtime() - tstart;
                    #endif

                    #if defined(DEBUG)
                        cout << "   *** copyToDevice -- case 2 Evicting -- push_back_count: " << push_back_count << endl;
                    #endif

                    // need to grab some more space from the adjacent mem blocks ==> we may need to move multiple mtx to host
                    unsigned long int eviction_offset = (unsigned long int) mp[evictableMatrixBlock][0] * num_per_blk;
                    unsigned long int memView_index = (unsigned long int) mp[evictableMatrixBlock][0];
                    unsigned long int occupied_blk = 0; //(unsigned long int) mp[evictableMatrixBlock][2];
                    unsigned long int temp_index, temp_blk;
                    temp_index = memView_index; //+ occupied_blk;

                    pair<double *, int> nullPr = make_pair(nullptr, -1);
                    
                    #if defined(DEBUG)
                        cout << "   ** copyToDevice -- Eviction case 2 - memView_index: " << memView_index << " occupied_blk: " << (unsigned long int) mp[evictableMatrixBlock][2] <<  " required_blk: " << required_blk << " evicting: (" << matrixName[evictableMatrixBlock.first] << " , " << evictableMatrixBlock.second << ")" << " isModified: " << mp[evictableMatrixBlock][5]<< endl;
                        cout << "   ** copyToDevice -- Eviction case 2 - acquiring more blocks from current and right" << endl;
                    #endif

                    bool foundFreeBlock = false;
                    unsigned long int free_block_index = -1;
                    pair<double *, unsigned long int> firstFreeBlock;
                    // look at right side first

                    #if defined(TIMER)
                        temp_start_1 = omp_get_wtime();
                        double copyToHostTime = 0;
                    #endif

                    while(occupied_blk < required_blk && temp_index < totalMemBlock)
                    {
                        pair<double *, int> rightPair = memView[temp_index];
                        
                        // if(isOnDevice(rightPair)) { cout << "   ** copyToDevice -- rightPair is on device. index: " << (unsigned long int) mp[rightPair][0] << endl;}
                        //*** right side's pair could be a null space
                        if(rightPair.first != nullptr && isOnDevice(rightPair))
                        {
                            #if defined(DEBUG)
                                cout << "   ** copyToDevice -- rightPair: "<< temp_index << " (" << matrixName[rightPair.first] << " , "<< rightPair.second  << ") -- on device -- " << " isModified: " << mp[rightPair][5]<< endl;
                            #endif

                            // now evict rightPair pair
                            // first copy rightPair from device to host ==> matrixName[rightPair.first].compare("acsr") != 0 || 
                            if((unsigned long int) mp[rightPair][5] == 1) //we don't need to copy back the sparse matrix
                            {
                                #if defined(TIMER)
                                    temp_start = omp_get_wtime();
                                #endif

                                temp_status = copyToHost(rightPair.first, (unsigned long int) mp[rightPair][3], (unsigned long int) mp[rightPair][4], 
                                                (unsigned long int) rightPair.second, block_width, mp[rightPair][1], iterationNumber);
                                
                                #if defined(TIMER)
                                    copyToHostTiming[iterationNumber - 1] += omp_get_wtime() - temp_start;
                                    copyToHostTime += omp_get_wtime() - temp_start;
                                #endif
                            }
                            else
                            {
                                temp_status = 0;
                            }

                            //*** if we successfully copy the rightPair mtx to host
                            if(temp_status == 0)
                            {
                                //*** I have passed a consecutive free blocks cluster and found a real matrix block
                                if(foundFreeBlock && temp_index > free_block_index) 
                                {
                                    #if defined(DEBUG)
                                        cout << "       ** copyToDevice -- removing free block at: " << free_block_index << " # of block: " <<  firstFreeBlock.second << endl;
                                    #endif

                                    freeBlockMap[firstFreeBlock.second].erase(remove(freeBlockMap[firstFreeBlock.second].begin(), freeBlockMap[firstFreeBlock.second].end(), free_block_index), freeBlockMap[firstFreeBlock.second].end());

                                    if(freeBlockMap[firstFreeBlock.second].empty())
                                        freeBlockMap.erase(firstFreeBlock.second);

                                    foundFreeBlock = false;
                                    free_block_index = -1;
                                }

                                temp_blk = (unsigned long int) mp[rightPair][2];
                                occupied_blk += temp_blk;
                                temp_index += temp_blk;
                                mp.erase(rightPair); 
                            }
                            else
                            {
                                printf("    ** copyToDevice -- right sides's mtx is not copied back to host (case 2)==> %d\n", status); 
                                return status;
                            }
                        }
                        //*** free block, grab it and increment temp_index, occupied_blk by 1
                        else
                        {
                            #if defined(DEBUG)
                                cout << "       ** copyToDevice -- rightPair: "<< temp_index << " (" << matrixName[rightPair.first] << " , "<< rightPair.second  << ") -- not on device"<< endl;
                            #endif
                            // *** we have a serious issue here, if we grab this empty space then we need to update freeBlock list as well??

                            if(!foundFreeBlock) 
                            {
                                foundFreeBlock = true;
                                firstFreeBlock = rightPair;
                                free_block_index = temp_index;
                                
                                #if defined(DEBUG)
                                    cout << "   ** copyToDevice -- found free block at: " << free_block_index << " # of block: " <<  firstFreeBlock.second << endl;
                                #endif
                            }

                            temp_index++;
                            occupied_blk++;
                        }
                    }
                    
                    #if defined(TIMER)
                        tend = omp_get_wtime();
                        mmLoopTime[iterationNumber - 1] += tend - tstart - copyToHostTime;
                        grabRHSTiming[iterationNumber - 1] += tend - temp_start_1 - copyToHostTime;
                    #endif

                    if(occupied_blk < required_blk)
                        cout << "   ** copyToDevice -- Eviction case 2 - acquiring more blocks from left if needed - occupied_blk: " << occupied_blk  << " temp_index: " << temp_index << " memView_index: " << memView_index << endl;
                    // if(foundFreeBlock)
                    //     cout << "   ** copyToDevice -- Eviction case 2 -- start of freeBlock : " << free_block_index << " # of consecutive free blocks: " << firstFreeBlock.second << " sanity : " << count(freeBlockMap[firstFreeBlock.second].begin() , freeBlockMap[firstFreeBlock.second].end() , free_block_index) << endl;

                    // EVERYTHING is OK now, so copy the desired mtx in eviction_offset
                    if(occupied_blk >= required_blk)
                    {
                        #if defined(TIMER)
                            tstart = omp_get_wtime();
                        #endif

                        status = omp_target_memcpy(d_memory + eviction_offset, mtx + offset, chunk_size, 0, 0, device_id, host_id);
                        // status = cudaMemcpy(d_memory + eviction_offset, mtx + offset, chunk_size, cudaMemcpyHostToDevice);
                        
                        //cudaDeviceSynchronize();

                        #if defined(TIMER)
                            tend = omp_get_wtime();
                            h2dTime[iterationNumber - 1] += tend - tstart;
                            h2dTransfer[iterationNumber - 1] += chunk_size * 1e-9;
                            
                            #if defined(DEBUG4)
                            if(chunk_size * 1e-9/(tend - tstart) < 2.0)
                                cout << "   *** copyToDevice -- eviction (case 2) H2D: " << chunk_size * 1e-6 << " MB @ " << tend - tstart << " sec. --> Rate : " << chunk_size * 1e-9/(tend - tstart) << " GB/sec" << endl;
                            #endif
                        #endif

                        if(status == 0)
                        {
                            #if defined(TIMER)
                                tstart = omp_get_wtime();
                            #endif

                            // what we need to do now??
                            pair<double *, int> newPair = make_pair(mtx, block_id);
                            mp[newPair] = {memView_index * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                            mp.erase(evictableMatrixBlock);
                            evictionQueue.pop(); // removing the first element fromt the queue
                            evictionQueue.push(newPair);
                            
                            unsigned long int free_block_count = occupied_blk - required_blk;
                            
                            #if defined(DEBUG)
                                cout << "   *** copyToDevice -- case 2 updating memView: " << memView_index << " -- " << memView_index + required_blk - 1 << " (" << matrixName[newPair.first] << " , " << newPair.second << ")"<< endl;
                            #endif
    
                            //*** update memoryView
                            for(ii =  memView_index; ii < memView_index + required_blk ; ii++)
                            {
                                memView[ii] = newPair;    
                            }
                            
                            #if defined(TIMER)
                                dsUpdateTiming[iterationNumber - 1] += omp_get_wtime() - tstart;
                                temp_start = omp_get_wtime();
                            #endif

                            // *** updating free block info
                            if(free_block_count > 0)
                            {
                                #if defined(DEBUG)
                                    cout << "   *** copyToDevice -- case 2 free_block_count: " << free_block_count << " at index: " << memView_index + required_blk << endl;
                                #endif
    
                                //** MERGING: see if there are any mergable free blocks on the right.
                                
                                unsigned long int free_block_starting_index = memView_index + required_blk;
                                unsigned long int total_free_block = free_block_count;
                                unsigned long int look_ahead_index = free_block_starting_index + free_block_count;
                                
                                // cout << "   *** copyToDevice -- case 2 look_ahead_index: " << look_ahead_index << " total_free_block: " << total_free_block << endl;

                                while(look_ahead_index < totalMemBlock && memView[look_ahead_index].first == nullptr && memView[look_ahead_index].second > 0)
                                {
                                    unsigned long int next_free_block_count = memView[look_ahead_index].second;
                                    
                                    #if defined(DEBUG)
                                        cout << "       *** copyToDevice -- case 2 free block MERGING -- look_ahead_index: " << look_ahead_index << " next_free_block_count: " << next_free_block_count << " sanity: " << count(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index) << endl;
                                    #endif
                                    // removing the look_ahead_index from freeBlockMap[next_free_block_count]
                                    freeBlockMap[memView[look_ahead_index].second].erase(remove(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index), freeBlockMap[memView[look_ahead_index].second].end());
                                    if(freeBlockMap[next_free_block_count].empty())
                                        freeBlockMap.erase(next_free_block_count);
                                    // next lookup
                                    look_ahead_index += next_free_block_count;
                                    total_free_block += next_free_block_count;
                                } 

                                #if defined(DEBUG)
                                    if(total_free_block > free_block_count)
                                    {
                                        // cout << "   *** copyToDevice -- case 2 free block MERGING -- total_free_block: " << total_free_block << endl;
                                        cout << "   *** copyToDevice -- case 2 free block MERGING -- total_free_block: " << total_free_block << " ( " << free_block_starting_index << " -- " <<  free_block_starting_index + total_free_block - 1 << " )"<< endl;
                                        for(ii =  memView_index + occupied_blk ; ii < free_block_starting_index + total_free_block ; ii++)
                                        {
                                            if(memView[ii].first != nullptr)
                                                cout << "       *** copyToDevice -- case 2 free block ALERT: " << ii << " ( " << matrixName[memView[ii].first] << ", " << memView[ii].second << " )"<< endl;
                                        }
                                    }
                                #endif    
                                
                                freeBlockMap[total_free_block].push_back(free_block_starting_index); // OR: free_block_starting_index
                                
                                #if defined(DEBUG)
                                    cout << "   *** copyToDevice -- case 2 (final) total_free_block: " << total_free_block << " at index: " << free_block_starting_index << " -- " << free_block_starting_index + total_free_block - 1 << endl;
                                #endif

                                unsigned long int temp_count = total_free_block; // REUSING free_block_count variable as a counter for decrementing
                                for(ii =  free_block_starting_index ; ii < free_block_starting_index + total_free_block ; ii++)
                                {
                                    memView[ii] = make_pair(nullptr, temp_count);
                                    temp_count--;
                                }

                                // WORKING -- giving correct result with flaws in free block counting
                                // freeBlockMap[free_block_count].push_back(memView_index + required_blk);

                                // for(ii =  memView_index + required_blk; ii < memView_index + occupied_blk ; ii++)
                                // {
                                //     memView[ii] = make_pair(nullptr, free_block_count);
                                //     free_block_count--;
                                //     // *** update free block later as we are dropping it off now
                                // }
                            }

                            if(foundFreeBlock)
                            {
                                #if defined(DEBUG)
                                    cout << "   *** copyToDevice -- case 2 (foundFreeBlock) removing free block at: " << free_block_index << " # of block: " <<  firstFreeBlock.second << " temp_index: " << temp_index << endl;
                                #endif
                                freeBlockMap[firstFreeBlock.second].erase(remove(freeBlockMap[firstFreeBlock.second].begin(), freeBlockMap[firstFreeBlock.second].end(), free_block_index), freeBlockMap[firstFreeBlock.second].end());


                                if(memView[temp_index].first == nullptr && memView[temp_index].second > 0 && free_block_index + firstFreeBlock.second > temp_index)
                                {
                                    #if defined(DEBUG)
                                        cout << "   *** copyToDevice -- case 2 adding free blocks at temp_index: " << temp_index << "(" << matrixName[memView[temp_index].first] << ", " << memView[temp_index].second << ") occupied_blk : " << occupied_blk << " required_blk : " << required_blk << endl;
                                    #endif
                                    // freeBlockMap[memView[temp_index].second].push_back(temp_index);

                                    //ADDING FREE BLOCK TRACKING PART
                                    unsigned long int free_block_starting_index = temp_index;
                                    unsigned long int total_free_block = memView[temp_index].second;
                                    unsigned long int look_ahead_index = free_block_starting_index + total_free_block;
                                    
                                    // cout << "       ** copyToDevice -- case 2 (foundFreeBlock) look_ahead_index: " << look_ahead_index << " total_free_block: " << total_free_block << endl;

                                    while(look_ahead_index < totalMemBlock && memView[look_ahead_index].first == nullptr && memView[look_ahead_index].second > 0)
                                    {
                                        unsigned long int next_free_block_count = memView[look_ahead_index].second;

                                        #if defined(DEBUG)
                                            cout << "       *** copyToDevice -- case 2 (foundFreeBlock) free block MERGING -- look_ahead_index: " << look_ahead_index << " --> " << matrixName[memView[look_ahead_index].first] << ", " << memView[look_ahead_index].second << ") next_free_block_count: " << next_free_block_count << " sanity: " << count(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index)  << endl;
                                        #endif
    
                                        // removing the look_ahead_index from freeBlockMap[next_free_block_count]
                                        freeBlockMap[memView[look_ahead_index].second].erase(remove(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index), freeBlockMap[memView[look_ahead_index].second].end());
                                        if(freeBlockMap[next_free_block_count].empty())
                                            freeBlockMap.erase(next_free_block_count);
                                        // next lookup
                                        look_ahead_index += next_free_block_count;
                                        total_free_block += next_free_block_count;
                                    } 
                                    
                                    #if defined(DEBUG)
                                        if(total_free_block > memView[temp_index].second)
                                        {
                                            // cout << "       *** copyToDevice -- case 2 (foundFreeBlock) free block MERGING -- total_free_block: " << total_free_block << endl;
                                            cout << "   *** copyToDevice -- case 2 (foundFreeBlock) free block MERGING -- total_free_block: " << total_free_block << " ( " << free_block_starting_index << " -- " <<  free_block_starting_index + total_free_block - 1 << " )"<< endl;
                                            for(ii =  temp_index + memView[temp_index].second ; ii < free_block_starting_index + total_free_block ; ii++)
                                            {
                                                if(memView[ii].first != nullptr)
                                                    cout << "       *** copyToDevice -- case 2 (foundFreeBlock) free block ALERT: " << ii << " ( " << matrixName[memView[ii].first] << ", " << memView[ii].second << " )"<< endl;
                                            }
                                        }
                                    #endif

                                    freeBlockMap[total_free_block].push_back(free_block_starting_index); // OR: free_block_starting_index

                                    #if defined(DEBUG)
                                        cout << "   *** copyToDevice -- case 2 (final) (foundFreeBlock) total_free_block: " << total_free_block << " at index: " << free_block_starting_index << " -- " << free_block_starting_index + total_free_block - 1 << endl;
                                    #endif

                                    unsigned long int temp_count = total_free_block; // REUSING free_block_count variable as a counter for decrementing
                                    for(ii =  free_block_starting_index ; ii < free_block_starting_index + total_free_block ; ii++)
                                    {
                                        memView[ii] = make_pair(nullptr, temp_count);
                                        temp_count--;
                                    }
                                }
                            }

                            #if defined(DEBUG)
                                if(foundFreeBlock && free_block_count > 0)
                                    cout <<"    ** copyToDevice -- Eviction case 2 -- should it be the case??" << endl;

                                cout << "   ** copyToDevice -- Eviction case 2 -- returning successfully" << endl;
                            #endif

                            #if defined(TIMER)
                                tend = omp_get_wtime();
                                mmLoopTime[iterationNumber - 1] += tend - tstart;
                                freeBlockTiming[iterationNumber - 1] += tend - temp_start;
                            #endif

                            #if defined(DEBUG3)
                                cout << "   *** copyToDevice -- eviction (case 2): " << omp_get_wtime() - timer_full << " sec." << endl;
                            #endif

                            return status;
                        }
                        else
                        {
                            printf("    ** copyToDevice -- right sides's mtx is not copied back to host (case 2)==> %d\n", status);
                            errorFlag = true;
                            return status; 
                        }
                    }
                    else
                    {
                        printf("    ** copyToDevice -- eviction case-2 failed, couldn't occupy left or right side cells properly\n");
                        errorFlag = true;
                        return -1;
                    }
                } // end case-2
            }
            else
            {
                cout << "   ** copyToDevice -- evictionQueue is empty. There is nothing to evict." << endl;
                errorFlag = true;
            } // end eviction policy
            
        }
    } // end of all case

    return status;
}

int reserveOnDevice(double *mtx, int row, int col, int block_id, int offset, int num_element, double isModified, long iterationNumber)
{
    // cout << "In reserveOnDevice: " << num_element << endl;
    // mtx + offset = takes us to the starting address of block_id that needs to be copied to device
    
    int temp_status, status = -1;  
    double chunk_size = num_element * sizeof(double);
    unsigned long int required_blk = requiredDeviceBlocks(chunk_size, memGranularity);
    // cout << "   * reserveOnDevice memBlockIndex: " << memBlockIndex << " availableDeviceBlocks: " << availableDeviceBlocks() << " required_blk: " << required_blk  << endl;
    unsigned long int ii;
    double tstart, tend, timer_full, temp_start, temp_start_1;

    #if defined(DEBUG3)
        timer_full = omp_get_wtime();
    #endif

    // we have required # of blks on device
    // cout << "first if - checking: " << availableDeviceBlocks << endl;
    if(!isAllBlocksOccupied() && required_blk <= availableDeviceBlocks()) 
    {
        // cout << "first if" << endl;
        #if defined(TIMER)
            tstart = omp_get_wtime();
        #endif

        pair<double *, int> newPair = make_pair(mtx, block_id);
        mp[newPair] = {memBlockIndex * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};

        // cout << "first if - before for loop" << endl;

        // update memory view before incrementing memBlockIndex
        for(ii = memBlockIndex ; ii < memBlockIndex + required_blk ; ii++)
        {
            memView[ii] = newPair;
            // cout << "ii: " << ii << endl;
        }
        // cout << "first if - for loop" << endl;

        evictionQueue.push(newPair);
        // cout << "first if - push" << endl;
        memBlockIndex += required_blk; 
        available_mem -= required_blk * memGranularity;

        #if defined(TIMER)
            tend = omp_get_wtime();
            mmLoopTime[iterationNumber - 1] += tend - tstart;
            dsUpdateTiming[iterationNumber - 1] += tend - tstart;
        #endif

        #if defined(DEBUG3)
            cout << "   *** reserveOnDevice -- first if time: " << omp_get_wtime() - timer_full << " sec." << endl;
        #endif

        // cout << "   * reserveOnDevice ==> memBlockIndex: " << memBlockIndex << endl;
        return 0;
    }
    else // enough space is not available on device memory
    {
        #if defined(TIMER)
            tstart = omp_get_wtime();
        #endif

        if(!inEvictionMode) inEvictionMode = true;


        //** check if enough free memory blocks are available
        
        bool fille_by_free_block = false;

        if(freeBlockMap.size() > 0)
        {
            if(freeBlockMap.find(required_blk) != freeBlockMap.end() && freeBlockMap[required_blk].size() > 0) 
            {
                // so we got an empty spase of perfect size
                unsigned long int d_index = freeBlockMap[required_blk].back(); // d_memory + d_offset = starting address of the empty block
                freeBlockMap[required_blk].pop_back();

                // status = omp_target_memcpy(d_memory + d_index * num_per_blk, mtx + offset, num_element * sizeof(double), 0, 0, device_id, host_id);
                #if defined(DEBUG)
                    cout << "       *** reserveOnDevice -- case 1 removing free block at index : " << d_index << " (" << matrixName[memView[d_index].first] << ", " << memView[d_index].second << ") required_blk: " << required_blk << endl;
                #endif
                
                fille_by_free_block = true;
                
                pair<double *, int> newPair = make_pair(mtx, block_id);
                mp[newPair] = {d_index * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                evictionQueue.push(newPair);

                for(ii = d_index ; ii < d_index + required_blk ; ii++)
                {
                    memView[ii] = newPair;
                }
                if(freeBlockMap[required_blk].size() == 0)
                    freeBlockMap.erase(required_blk);
                
                #if defined(TIMER)
                    tend = omp_get_wtime();
                    mmLoopTime[iterationNumber - 1] += tend - tstart;
                    freeBlockTiming[iterationNumber - 1] += tend - tstart;
                #endif

                #if defined(DEBUG3)
                    cout << "   *** reserveOnDevice -- free block time (case 1): " << omp_get_wtime() - timer_full << " sec." << endl;
                #endif

                return 0; 
            }
            else
            {
                auto fesible_block = nearest_key(freeBlockMap, required_blk);
                unsigned long int free_block_key = fesible_block->first;

                #if defined(DEBUG)
                    cout << "   * reserveOnDevice -- fesible_block KEY: " << fesible_block->first << " VECTOR SIZE : " << fesible_block->second.size() << endl;
                #endif

                if(fesible_block->first > required_blk && !fesible_block->second.empty())
                {
                    unsigned long int memView_index = fesible_block->second.back();
                    freeBlockMap[free_block_key].pop_back();
                    unsigned long int occupied_blk = fesible_block->first;
                    unsigned long int free_block_count = occupied_blk - required_blk;

                    fille_by_free_block = true;
                    // what we need to do now??
                    pair<double *, int> newPair = make_pair(mtx, block_id);
                    mp[newPair] = {memView_index * 1.0, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                    evictionQueue.push(newPair);

                    #if defined(DEBUG)
                        cout << "   *** reserveOnDevice -- free block case 2 updating memView: " << memView_index << " -- " << memView_index + required_blk - 1 << " (" << matrixName[newPair.first] << " , " << newPair.second << ") rquired: " << required_blk << " occupied: " << occupied_blk << endl;
                    #endif

                    //** update memoryView
                    for(ii =  memView_index; ii < memView_index + required_blk ; ii++)
                    {
                        memView[ii] = newPair;    
                    }

                    if(free_block_count > 0)
                    {
                        #if defined(DEBUG)        
                            cout << "   *** reserveOnDevice -- free block case 2 free_block_count: " << free_block_count << " at index: " << memView_index + required_blk << " -- " << memView_index + occupied_blk - 1 << endl;
                        #endif
                            // ** MERGING: see if there are any mergable free blocks on the right.
                                
                            unsigned long int free_block_starting_index = memView_index + required_blk;
                            unsigned long int total_free_block = free_block_count;
                            unsigned long int look_ahead_index = free_block_starting_index + free_block_count;
                                
                            // cout << "   *** copyToDevice -- case 1 look_ahead_index: " << look_ahead_index << " total_free_block: " << total_free_block << endl;

                            while(look_ahead_index < totalMemBlock && memView[look_ahead_index].first == nullptr && memView[look_ahead_index].second > 0)
                            {
                                unsigned long int next_free_block_count = memView[look_ahead_index].second;
                            
                                #if defined(DEBUG)
                                    cout << "       *** reserveOnDevice -- free block case 2 free block MERGING -- look_ahead_index: " << look_ahead_index << " --> " << matrixName[memView[look_ahead_index].first] << ", " << memView[look_ahead_index].second << ") next_free_block_count: " << next_free_block_count << " sanity: " << count(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index)  << endl;
                                #endif
                                // removing the look_ahead_index from freeBlockMap[next_free_block_count]
                                freeBlockMap[memView[look_ahead_index].second].erase(remove(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index), freeBlockMap[memView[look_ahead_index].second].end());
                                if(freeBlockMap[next_free_block_count].empty())
                                    freeBlockMap.erase(next_free_block_count);
                                // next lookup
                                look_ahead_index += next_free_block_count;
                                total_free_block += next_free_block_count;
                            } 
                            #if defined(DEBUG)
                                if(total_free_block > free_block_count)
                                {
                                    cout << "   *** reserveOnDevice -- free block case 2 free block MERGING -- total_free_block: " << total_free_block << " ( " << free_block_starting_index << " -- " <<  free_block_starting_index + total_free_block - 1 << " )"<< endl;
                                    for(ii =  memView_index + occupied_blk ; ii < free_block_starting_index + total_free_block ; ii++)
                                    {
                                        if(memView[ii].first != nullptr)
                                            cout << "       *** reserveOnDevice -- free block case 2 free block ALERT: " << ii << " ( " << matrixName[memView[ii].first] << ", " << memView[ii].second << " )"<< endl;
                                    }   
                                }
                            #endif

                            freeBlockMap[total_free_block].push_back(free_block_starting_index); // OR: free_block_starting_index
                                
                            unsigned long int temp_count = total_free_block; // REUSING free_block_count variable as a counter for decrementing
                        
                            #if defined(DEBUG) 
                                cout << "   *** reserveOnDevice -- free block case 2 (final) total_free_block: " << total_free_block << " at index: " << free_block_starting_index << " -- " << free_block_starting_index + total_free_block - 1 << endl;
                            #endif

                            for(ii =  free_block_starting_index ; ii < free_block_starting_index + total_free_block ; ii++)
                            {
                                memView[ii] = make_pair(nullptr, temp_count);
                                temp_count--;
                            }
                    } // end if(free_block_count > 0)

                    if(fesible_block->second.empty())
                        freeBlockMap.erase(fesible_block->first);

                    #if defined(TIMER)
                        tend = omp_get_wtime();
                        mmLoopTime[iterationNumber - 1] += tend - tstart;
                        freeBlockTiming[iterationNumber - 1] += tend - tstart;
                    #endif

                    #if defined(DEBUG3)
                        cout << "   *** reserveOnDevice -- free block time (case 2): " << omp_get_wtime() - timer_full << " sec." << endl;
                    #endif

                    return 0;
                }
                else
                {
                    cout << "   *** reserveOnDevice -- iterationNumber: " << iterationNumber  << " required_blk: " << required_blk << " fesible_block->first: " << fesible_block->first <<  " fesible_block->second.size(): "<< fesible_block->second.size() << endl;
                }
            } // end else
        } // end freeBlocMap.size() > 0
        
        // cout << "********* Trying to evict in reserveOnDevice ********* block_id: " << block_id << endl;

        //evict blocks from device
        if(evictionQueue.size() > 0 && !fille_by_free_block) 
        {
            #if defined(TIMER)
                tstart = omp_get_wtime();
            #endif

            int pop_count = 0;
            
            while(evictionQueue.size() > 0 && !isOnDevice(evictionQueue.front())) 
            {
                evictionQueue.pop();
                pop_count++;
            }

            #if defined(TIMER)
                tend = omp_get_wtime();
                mmLoopTime[iterationNumber - 1] += tend - tstart;
                pushPopTime[iterationNumber - 1] += tend - tstart;
            #endif

            #if defined(DEBUG)
                cout << "   ** reserveOnDevice -- pop_count: " << pop_count << " evictionQueue size - " << evictionQueue.size() << " -- freeBlockMap size - " << freeBlockMap.size() << endl;
            #endif

            // now we need to evict one or multiple blocks from the eviction queue to put the new mtx on device
            if(evictionQueue.size() > 0)
            {
                // cout << "********* Found one mtx block to evict -- reserveOnDevice" << endl;

                pair<double *, int> evictableMatrixBlock = evictionQueue.front();

                // *** case - 1 & 3: # of mem blocks occupied by evictable mtx block is >= to the required mem blocks
                if((unsigned long int) mp[evictableMatrixBlock][2] >= required_blk)
                {
                    
                    
                    // cout << "   * reserveOnDevice -- Eviction case 1" << endl; matrixName[evictableMatrixBlock.first].compare("acsr") != 0 || 
                    //first copy evictableMatrixBlock from device to host ==> then put back (mtx, block_id) at the same place
                    if((unsigned long int) mp[evictableMatrixBlock][5] == 1)//(mp[evictableMatrixBlock][5] > 0.0) // evictableMatrixBlock is modified ==> need to copy back to host
                    {
                        // cout << "   * reserveOnDevice -- Eviction case 1 -- evictableMatrixBlock is modified" << endl;
                        status = copyToHost(evictableMatrixBlock.first, (unsigned long int) mp[evictableMatrixBlock][3], (unsigned long int) mp[evictableMatrixBlock][4], 
                                            (unsigned long int) evictableMatrixBlock.second, block_width, mp[evictableMatrixBlock][1], iterationNumber);
                            // cout << "   * reserveOnDevice -- case 1 Evicting matrix -- " << matrixName[evictableMatrixBlock.first] << " block_id: " << (unsigned long int) evictableMatrixBlock.second << endl;
                        // cout << "   * reserveOnDevice -- case 1 Evicting -- (" << matrixName[evictableMatrixBlock.first] << " , block_id: " << (unsigned long int) evictableMatrixBlock.second << "), Copying -- (" << matrixName[mtx] << " , " << block_id << ")" << endl;
                        #if defined(DEBUG)
                            cout << "   * reserveOnDevice -- case 1 Evict -- (" << matrixName[evictableMatrixBlock.first]  << " , " << (unsigned long int) evictableMatrixBlock.second << ") @ " << (unsigned long int) mp[evictableMatrixBlock][0] << " (" << (unsigned long int) mp[evictableMatrixBlock][2]<< ") --> Copy -- (" << matrixName[mtx] << " , " << block_id << " , " << required_blk <<")" << endl;
                        #endif
                    }
                    else // evictableMatrixBlock is not modified ==> no need to copy back to host
                    {
                        status = 0;
                        
                        #if defined(DEBUG)
                            cout << "   * reserveOnDevice -- case 1 Purge -- (" << matrixName[evictableMatrixBlock.first] << " , block_id: " << (unsigned long int) evictableMatrixBlock.second << ") --> Copying -- (" << matrixName[mtx] << " , " << block_id << ")" << endl;
                        #endif
                    }
                    // if we successfully copy the evictable mtx to host
                    if(status == 0)
                    {
                        #if defined(TIMER)
                            tstart = omp_get_wtime();
                        #endif

                        pair<double *, int> newPair = make_pair(mtx, block_id);
                        mp[newPair] = {mp[evictableMatrixBlock][0], num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                        // cout << "newPair is saved in mp" << endl;
                        // evictionQueue.pop(); // removing the first element fromt the queue
                        // evictionQueue.push(newPair);
                        pair<double *, int> nullPr = make_pair(nullptr, -1);
                        unsigned long int memView_index = (unsigned long int) mp[evictableMatrixBlock][0];
                        unsigned long int occupied_blk = (unsigned long int) mp[evictableMatrixBlock][2];

                        unsigned long int free_block_count = occupied_blk - required_blk;
                            
                        //*** update memoryView
                        for(ii =  memView_index ; ii < memView_index + required_blk ; ii++)
                        {
                            memView[ii] = newPair;        
                        }

                        #if defined(TIMER)
                            dsUpdateTiming[iterationNumber - 1] += omp_get_wtime() - tstart;
                            temp_start = omp_get_wtime();
                        #endif
                        
                        // *** updating free block info
                        if(free_block_count > 0)
                        {
                            #if defined(DEBUG)
                                cout << "   ** reserveOnDevice -- case 1 free_block_count: " << free_block_count << " at index : " << memView_index + required_blk << endl;
                            #endif
                            //** MERGING: see if there are any mergable free blocks on the right.
                        
                            unsigned long int free_block_starting_index = memView_index + required_blk;
                            unsigned long int total_free_block = free_block_count;
                            unsigned long int look_ahead_index = free_block_starting_index + free_block_count;
                                
                            // cout << "   *** reserveOnDevice -- case 1 look_ahead_index: " << look_ahead_index << " total_free_block: " << total_free_block << endl;

                            while(look_ahead_index < totalMemBlock && memView[look_ahead_index].first == nullptr && memView[look_ahead_index].second > 0)
                            {
                                unsigned long int next_free_block_count = memView[look_ahead_index].second;

                                #if defined(DEBUG)
                                    cout << "       *** reserveOnDevice -- case 1 free block MERGING -- look_ahead_index: " << look_ahead_index << " --> " << matrixName[memView[look_ahead_index].first] << ", " << memView[look_ahead_index].second << ") next_free_block_count: " << next_free_block_count << " sanity: " << count(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index)  << endl;
                                #endif
                                // removing the look_ahead_index from freeBlockMap[next_free_block_count]
                                freeBlockMap[memView[look_ahead_index].second].erase(remove(freeBlockMap[memView[look_ahead_index].second].begin(), freeBlockMap[memView[look_ahead_index].second].end(), look_ahead_index), freeBlockMap[memView[look_ahead_index].second].end());
                                if(freeBlockMap[next_free_block_count].empty())
                                        freeBlockMap.erase(next_free_block_count);
                                // next lookup
                                look_ahead_index += next_free_block_count;
                                total_free_block += next_free_block_count;
                            } 

                            #if defined(DEBUG)
                                if(total_free_block > free_block_count)
                                {
                                    // cout << "       *** reserveOnDevice -- case 1 free block MERGING -- total_free_block: " << total_free_block << endl;
                                    cout << "   *** reserveOnDevice -- case 1 free block MERGING -- total_free_block: " << total_free_block << " ( " << free_block_starting_index << " -- " <<  free_block_starting_index + total_free_block - 1 << " )"<< endl;
                                    for(ii =  memView_index + occupied_blk ; ii < free_block_starting_index + total_free_block ; ii++)
                                    {
                                        if(memView[ii].first != nullptr)
                                            cout << "       *** reserveOnDevice -- case 1 free block ALERT: " << ii << " ( " << matrixName[memView[ii].first] << ", " << memView[ii].second << " )"<< endl;
                                    }
                                }

                                cout << "   *** reserveOnDevice -- case 1 total_free_block: " << total_free_block << " at index: " << free_block_starting_index << " -- " << free_block_starting_index + total_free_block - 1 << endl;
                            #endif

                            freeBlockMap[total_free_block].push_back(free_block_starting_index); // OR: free_block_starting_index
                            
                            unsigned long int temp_count = total_free_block; // REUSING free_block_count variable as a counter for decrementing
                            for(ii =  free_block_starting_index ; ii < free_block_starting_index + total_free_block ; ii++)
                            {
                                memView[ii] = make_pair(nullptr, temp_count);
                                temp_count--;
                            }

                            //WORKING:

                            // freeBlockMap[free_block_count].push_back(memView_index + required_blk);

                            // for(ii =  memView_index + required_blk; ii < memView_index + occupied_blk ; ii++)
                            // {
                            //     memView[ii] = make_pair(nullptr, free_block_count);
                            //     free_block_count--;
                            // }
                        }

                        // cout << "memView updating finished...." << endl;
                        mp.erase(evictableMatrixBlock);
                        // cout << "evictableMatrixBlock is erased from mp" << endl;
                        evictionQueue.pop();
                        // cout << "evictionQueue poped" << endl;
                        evictionQueue.push(newPair);
                        // cout << "evictionQueue size: " << evictionQueue.size() << endl;
                        // cout << "newPair is pushed in evictionQueue. Returing -- reserveOnDevice..." << endl;

                        #if defined(TIMER)
                            tend = omp_get_wtime();
                            mmLoopTime[iterationNumber - 1] += tend - tstart;
                            freeBlockTiming[iterationNumber - 1] += tend - temp_start;
                        #endif

                        #if defined(DEBUG3)
                            cout << "   *** reserveOnDevice -- eviction (case 1): " << omp_get_wtime() - timer_full << " sec." << endl;
                        #endif

                        return 0;
                    }
                    else
                    {
                        printf("    * reserveOnDevice -- evictable mtx is not copied back to host (case 1)==> %d\n", status); 
                        return -1;
                    }
                    // }
                    // else // evictableMatrixBlock is not modified ==> no need to copy back to host
                    // {
                    //     cout << "   * reserveOnDevice -- case 1 not Evicting -- (" << matrixName[evictableMatrixBlock.first] << " , block_id: " << (unsigned long int) evictableMatrixBlock.second << "), Copying -- (" << matrixName[mtx] << " , " << block_id << ")"  << " isModified: " << mp[evictableMatrixBlock][5] << endl;

                    //     pair<double *, int> newPair = make_pair(mtx, block_id);
                    //     mp[newPair] = {mp[evictableMatrixBlock][0], num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                    //     // cout << "newPair is saved in mp" << endl;
                    //     // evictionQueue.pop(); // removing the first element fromt the queue
                    //     // evictionQueue.push(newPair);
                    //     pair<double *, int> nullPr = make_pair(nullptr, -1);
                    //     unsigned long int memView_index = (unsigned long int) mp[evictableMatrixBlock][0];
                    //     unsigned long int occupied_blk = (unsigned long int) mp[evictableMatrixBlock][2];
                    //     // update memory view before incrementing memBlockIndex
                    //     // cout << "Updating memView.." << endl;
                    //     for(ii =  memView_index; ii < memView_index + occupied_blk ; ii++)
                    //     {
                    //         if(ii < memView_index + required_blk)
                    //             memView[ii] = newPair;
                    //         else
                    //             memView[ii] = nullPr;
                    //             // *** update empty space later as we are dropping it off now
                    //     }
                    //     // cout << "memView updating finished...." << endl;
                    //     mp.erase(evictableMatrixBlock);
                    //     // cout << "evictableMatrixBlock is erased from mp" << endl;
                    //     evictionQueue.pop();
                    //     // cout << "evictionQueue poped" << endl;
                    //     evictionQueue.push(newPair);
                    //     // cout << "evictionQueue size: " << evictionQueue.size() << endl;
                    //     // cout << "newPair is pushed in evictionQueue. Returing -- reserveOnDevice..." << endl;
                    //     return 0;
                    // }
                }
                //*** case - 2: # of mem blocks occupied by evictable mtx block is less than required mem blocks
                else if((unsigned long int) mp[evictableMatrixBlock][2] < required_blk)
                {
                    cout << "   * reserveOnDevice -- Eviction case 2" << endl;
                    // need to grab some more space from the adjacent mem blocks ==> we may need to move multiple mtx to host
                    unsigned long int eviction_offset = (unsigned long int) mp[evictableMatrixBlock][0] * num_per_blk;
                    unsigned long int memView_index = (unsigned long int) mp[evictableMatrixBlock][0];
                    unsigned long int occupied_blk = (unsigned long int) mp[evictableMatrixBlock][2];
                    unsigned long int temp_index, temp_blk;
                    temp_index = memView_index + occupied_blk;
                    pair<double *, int> nullPr = make_pair(nullptr, -1);
                    
                    // look at right side first
                    while(occupied_blk < required_blk && temp_index < totalMemBlock)
                    {
                        pair<double *, int> rightPair = memView[temp_index];
                        // right side's pair could be a null space
                        if(rightPair.first != nullptr)
                        {
                            // now evict rightPair pair
                            // first copy rightPair from device to host
                            status = copyToHost(rightPair.first, (unsigned long int) mp[evictableMatrixBlock][3], (unsigned long int) mp[evictableMatrixBlock][4], 
                                                (unsigned long int) rightPair.second, block_width, mp[rightPair][1], iterationNumber);
                            // if we successfully copy the rightPair mtx to host
                            if(status == 0)
                            {
                                temp_blk = (unsigned long int) mp[rightPair][2];
                                if(occupied_blk + temp_blk >= required_blk)
                                {
                                    // update memory view accordingly
                                    for(ii =  temp_index; ii < temp_index + temp_blk ; ii++)
                                    {
                                        memView[ii] = nullPr;
                                    }
                                    // if there would be some empty space, add those to freeBlockMap list 
                                    // if(occupied_blk + temp_blk > required_blk)
                                    // {
                                    //     unsigned long int empty_blk = occupied_blk + temp_blk - required_blk;
                                    //     unsigned long int empty_offset = (unsigned long int) mp[evictableMatrixBlock][0] + required_blk * num_per_blk;
                                    //     freeBlockMap[empty_blk].push_back(empty_offset);
                                    //     occupied_blk = required_blk; // breaking this while loop
                                    // }

                                    occupied_blk = required_blk; // breaking this while loop
                                    // erase rightPair from mp
                                    mp.erase(rightPair); // should we check if rightPair is infact available on mp?? ==> think about it later
                                }
                                else
                                {
                                    occupied_blk += temp_blk;
                                    temp_index += temp_blk;
                                }
                            }
                            else
                            {
                                printf("    * reserveOnDevice -- right sides's mtx is not copied back to host (case 2)==> %d\n", status); 
                                return status;
                            }
                        }
                        // null space, grab it and increment temp_index, occupied_blk by 1
                        else
                        {
                            // *** we have a serious issue here, if we grab this empty space then we need to update freeBlockMap list as well??
                            // *** ==> DROP the empty blocks idea for now
                            temp_index++;
                            occupied_blk++;
                        }
                    }

                    temp_index = memView_index - 1; // next available left space
                    //*** still need few blks but no righ side blks available ==> SO LOOP AT YOUR LEFT!!
                    while(occupied_blk < required_blk && temp_index >=0)
                    {
                        pair<double *, int> leftPair = memView[temp_index];
                        // right side's pair could be a null space
                        if(leftPair.first != nullptr)
                        {
                            // now evict rightPair pair
                            // first copy rightPair from device to host
                            status = copyToHost(leftPair.first, (unsigned long int) mp[evictableMatrixBlock][3], (unsigned long int) mp[evictableMatrixBlock][4], 
                                                (unsigned long int) leftPair.second, block_width, mp[leftPair][1], iterationNumber);
                            // if we successfully copy the rightPair mtx to host
                            if(status == 0)
                            {
                                temp_blk = (unsigned long int) mp[leftPair][2];
                                if(occupied_blk + temp_blk >= required_blk)
                                {
                                    // if there would be some empty space, add those to freeBlockMap list 
                                    if(occupied_blk + temp_blk > required_blk)
                                    {
                                        unsigned long int empty_blk = occupied_blk + temp_blk - required_blk;
                                        unsigned long int empty_offset = (unsigned long int) mp[leftPair][0];
                                        // freeBlockMap[empty_blk].push_back(empty_offset); // *** hold this off for a while
                                        occupied_blk = required_blk; // breaking this while loop

                                        // update memory view accordingly
                                        for(ii =  temp_index - temp_blk + 1 ; ii <= empty_blk  ; ii++)
                                        {
                                            memView[ii] = nullPr;
                                        }
                                        eviction_offset = empty_offset + empty_blk * num_per_blk;
                                    }
                                    else
                                    {
                                        eviction_offset = (unsigned long int) mp[leftPair][0];
                                    }
                                    // erase rightPair from mp
                                    mp.erase(leftPair); // should we check if rightPair is infact available on mp?? ==> think about it later
                                }
                                else
                                {
                                    occupied_blk += temp_blk;
                                    temp_index -= temp_blk;
                                }
                            }
                            else
                            {
                                printf("    * reserveOnDevice -- right sides's mtx is not copied back to host (case 2)==> %d\n", status); 
                                return status;
                            }
                        }
                        // null space, grab it and increment temp_index, occupied_blk by 1
                        else
                        {
                            // *** we have a serious issue here, if we grab this empty space then we need to update freeBlockMap as well??
                            // *** ==> DROP the empty blocks idea for now
                            temp_index--;
                            occupied_blk++;
                        }
                    }

                    // EVERYTHING is ok now, so copy the desired mtx in eviction_offset
                    if(occupied_blk == required_blk)
                    {
                        pair<double *, int> newPair = make_pair(mtx, block_id);
                        mp[newPair] = {(eviction_offset * 1.0)/num_per_blk, num_element * 1.0, required_blk * 1.0, row * 1.0, col * 1.0, isModified, omp_get_wtime()};
                        mp.erase(evictableMatrixBlock);
                        evictionQueue.pop(); // removing the first element fromt the queue
                        evictionQueue.push(newPair);

                        unsigned long int memView_index = (unsigned long int) eviction_offset/num_per_blk;
                            // update memory view before incrementing memBlockIndex
                        for(ii =  memView_index; ii < memView_index + required_blk ; ii++)
                        {
                            memView[ii] = newPair;
                        }
                        cout << "   * reserveOnDevice -- Eviction case 2 -- returning successfully" << endl;
                        return 0;
                    }
                    else
                    {
                        printf("    * reserveOnDevice -- eviction case-2 failed, couldn't occupy left or right side cells properly\n");
                        return -1;
                    }
                } // end case-2
            }
            else
            {
                cout << "   * reserveOnDevice -- evictionQueue is empty. There is nothing to evict." << endl;
                return -1;
            } // end eviction policy
        }
    } // end of all case

    return status;
}

//*** OLD reserveOnDevice
/*int reserveOnDevice(double *mtx, int row, int col, int block_id, int offset, int num_element)
{
    if(availbleDeviceMemory() >= num_element * sizeof(double))
    {
        // status = omp_target_memcpy(d_memory + head, mtx + offset, blksz * sizeof(double), 0, 0, device_id, host_id);
        mp[make_pair(mtx, block_id)] = {head  * 1.0, num_element * sizeof(double) * 1.0, omp_get_wtime()};
        head = head + num_element; //* sizeof(double);
        available_mem -= num_element * sizeof(double);
        return 0;
    }
    else
    {
        printf("Enought memory is not available on GPU\n");
        return -1;
    }
    return -1;
}*/

double setMemoryGranularity(double size_in_byte)
{
    return ceil(size_in_byte/sizeof(double)) * sizeof(double);
}

void printMapInfo()
{
    cout << "\n------- Printing Memory Info -------" << endl;
    cout << "         Device Capacity: " << deviceCapacity()/1e+9 << "GBs" << endl;
    // cout << "    availbleDeviceMemory: " << availbleDeviceMemory()/1e+9 << "GBs" << endl;
    // cout << "        Allocated Memory: " << allocated()/1e+9 << "GBs" << endl;
    cout << "           totalMemBlock: " << totalMemBlock << endl;
    cout << "           memBlockIndex: " << memBlockIndex << endl;
    cout << "   availableDeviceBlocks: " << availableDeviceBlocks() << endl;
    cout << "          MemoryMap size: " << mp.size() << endl;
    cout << "     Eviction Queue size: " << evictionQueue.size() << endl;
    cout << "matrixDimension map size: " << matrixDimension.size() << endl;
    cout << "       freeBlockMap Size: " << freeBlockMap.size() << endl;
    cout << "--------------------------------------" << endl;
}

void printDuplicates(vector<unsigned long int> memview_index) 
{ 
    int i;
    map<unsigned long int, int> CountMap;

    for (i = 0 ; i < memview_index.size() ; i++) 
    {
        CountMap[memview_index[i]]++;
    }
    if(CountMap.size() > 0)
        cout << "DUPLICATE: ";

    bool first = true;
    for (auto it = CountMap.begin(); it != CountMap.end(); it++, i++)
    {
        if (it->second > 1)
        {
            if(first)
            {
                cout << it->first << " (" << it->second << ")";
                first = false;
            }
            else
                cout << " , " << it->first << " (" << it->second << ")";
        }
    }
    cout << endl; 
}

void printMemoryUsageStat(long iterationNumber)
{
    int i, j, nullBlocks = 0, mtxBlocks = 0;

    cout << "\n----- Printing Memory Usage Stat -----" << endl;
   
    // cout << "Capacity: " << deviceCapacity()/1e+9 << "GBs" << endl;
    // cout << "availbleDeviceMemory: " << availbleDeviceMemory()/1e+9 << "GBs" << endl;
    // cout << "allocated memory: " << allocated()/1e+9 << "GBs" << endl;
    // cout << "totalMemBlock: " << totalMemBlock << endl;
    // cout << "memBlockIndex: " << memBlockIndex << endl;
    // cout << "availableDeviceBlocks: " << availableDeviceBlocks << endl;
    // cout << "Map size: " << mp.size() << endl;
    // cout << "Eviction Queue size: " << evictionQueue.size() << endl;
    // cout << "matrixDimension map size: " << matrixDimension.size() << endl;

    pair<double *, int> currentPair;
    vector<int> sanity_vector(totalMemBlock, 0);

    for(i = 0 ; i < totalMemBlock ; i++)
    {
        currentPair = memView[i];
        if(currentPair.first == nullptr)
        {
            nullBlocks++;
            sanity_vector[i] = 1;
        }
        else
            mtxBlocks++;
    }


    for(auto block : freeBlockMap)
    {
        if(block.second.size() > 0)
        {
            for(auto index : block.second)
            {
                for(i = index ; i < index + block.first ; i++)
                {
                    if(memView[index].first == nullptr)
                    {
                        if(sanity_vector[i] == 1)
                            sanity_vector[i] = -1;
                        else
                            sanity_vector[i] = 2;

                    }
                    else
                        sanity_vector[i] = 3;
                }
            }
        }
    }

    int mtx_in_map = 0, common = 0, map_only = 0, sanity_only = 0;
    
    for(i = 0 ; i < totalMemBlock ; i++)
    {
        if(sanity_vector[i] == -1)
            common++;
        else if(sanity_vector[i] == 1)
            map_only++;
        else if(sanity_vector[i] == 2)
            sanity_only++;
        else if(sanity_vector[i] == 3)
        {
            sanity_only++;
            mtx_in_map++;
        }    
    }

    cout << "          Free Blocks: " << nullBlocks << endl;
    cout << "           MTX Blocks: " << mtxBlocks << endl;
    
     

    unsigned long int free_block_count = 0;
    for(auto block : freeBlockMap)
    {
        if(block.second.size() > 0)
            free_block_count += block.first * block.second.size();
    }
    cout << " freeBlockMap content: " << free_block_count << endl;
    cout << "         Total Blocks: " << totalMemBlock << " Memory Block Utilization: " << (mtxBlocks * 1.00 / totalMemBlock) * 100.0 << " %" << endl;
    cout << "    freeBlockMap Size: " << freeBlockMap.size() << endl; 
    cout << "        Sanity vector: " <<  "common - " << common << " map_only - " << map_only << " sanity_only - " << sanity_only <<  " mtx_in_map - " << mtx_in_map << endl;
    cout << endl; 
    cout << "\n--------------------------------------" << endl;
    
    

    // cout << "h2dTime:" << endl;
    // for(i = 0 ; i < 5 ; i++)
    //     cout << h2dTime[i] << ",";
    // cout << endl;
    // cout << endl;

    // cout << "d2hTime:" << endl;
    // for(i = 0 ; i < 5 ; i++)
    //     cout << d2hTime[i] << ",";
    // cout << endl;
    // cout << endl;

    // cout << "mmLoopTime:" << endl;
    // for(i = 0 ; i < 5 ; i++)
    //     cout << mmLoopTime[i] << ",";
    // cout << endl;
    // cout << endl;

    // cout << "Total data transfer + mmLoopTime time:" << endl;
    // for(i = 0 ; i < 5 ; i++)
    //     cout << d2hTime[i] + h2dTime[i] + mmLoopTime[i] << ",";
    // cout << endl;
    // cout << endl;

    for(auto block : freeBlockMap)
    {
        
        if(block.second.size() > 0)
        {
            cout << "freeBlockMap[ " << block.first << " ] : " ;
            for(auto index : block.second)
                cout << index << " , ";
            cout << endl;

            printDuplicates(block.second); 
        }

        if(block.second.size() > 0)
        {
            cout << "MTX BLOCKS: " ;
            for(auto index : block.second)
            {
                for(i = index ; i < index + block.first ; i++)
                {
                    if(memView[index].first != nullptr)
                        cout << index << ": (" << matrixName[memView[index].first] << ", " << memView[index].second << "), ";
                }
            }   
            cout << endl;
        }
            // free_block_count += block.first * block.second.size();
    }

    cout << "mmTiming,mmLoopTime,h2dTime,d2hTime,,h2dTransfer,d2hTransfer,h2dRate,d2hRate" << endl;
    for(i = 0 ; i < iterationNumber ; i++)
        cout << mmTiming[i] << "," << mmLoopTime[i] << "," << h2dTime[i] << "," << d2hTime[i] << "," << h2dTransfer[i] << "," << d2hTransfer[i] << "," << h2dTransfer[i]/ h2dTime[i] << "," << d2hTransfer[i]/d2hTime[i] << endl;
    cout << endl << endl;

    cout << "mmLoopTime,freeBlockTiming,grabRHSTiming,dsUpdateTiming,copyToHostTiming,pushPopTime" << endl;
    for(i = 0 ; i < iterationNumber ; i++)
        cout << mmLoopTime[i] << "," << freeBlockTiming[i] << "," << grabRHSTiming[i] << "," << dsUpdateTiming[i] << "," << copyToHostTiming[i] << "," << pushPopTime[i] << endl;

    
    cout << "----------------------------------------" << endl;
}

void setMatrixInfo(double *mtx, int row, int col, string mtxName)
{
  matrixDimension[mtx] = {row, col};
  matrixName[mtx] = mtxName;
}

#endif
