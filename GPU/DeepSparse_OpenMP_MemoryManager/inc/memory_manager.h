#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

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
#include <unordered_map>
#include <exception>
#include <vector>
#include <queue>
#include <string>
#include <map>

using namespace std;

#include <omp.h>
#include "../inc/util.h"
#include "../inc/matrix_ops_cpu.h"
#include "../inc/matrix_ops_gpu_v11.h"



// Base class for exceptions.
class Exception : public std::exception 
{
public:
    Exception()
        : std::exception()
    {}

    // Sets the what() message to msg.
    Exception(std::string const& msg)
        : std::exception(),
          msg_(msg)
    {}

    // Sets the what() message to msg with func, file, line appended.
    Exception(std::string const& msg,
              const char* func, const char* file, int line)
        : std::exception(),
          msg_(msg + " in " + func + " at " + file + ":" + std::to_string(line))
    {}

    // @return message describing the execption.
    virtual char const* what() const noexcept override
    {
        return msg_.c_str();
    }

protected:
    // Sets the what() message to msg with func, file, line appended.
    void what(std::string const& msg,
              const char* func, const char* file, int line)
    {
        msg_ = msg + " in " + func + " at " + file + ":" + std::to_string(line);
    }

    std::string msg_;
};


class CudaException : public Exception 
{
public:
    CudaException(const char* call,
                  cudaError_t code,
                  const char* func,
                  const char* file,
                  int line)
        : Exception()
    {
        const char* name = cudaGetErrorName(code);
        const char* string = cudaGetErrorString(code);

        what(std::string("CUDA ERROR: ")
             + call + " failed: " + string
             + " (" + name + "=" + std::to_string(code) + ")",
             func, file, line);
    }
};

#define cuda_call(call) \
    do { \
        cudaError_t cuda_call_ = call; \
        if (cuda_call_ != cudaSuccess) \
            throw CudaException( \
                #call, cuda_call, __func__, __FILE__, __LINE__); \
    } while(0)



// Map
// https://stackoverflow.com/questions/3277172/using-pair-as-key-in-a-map-c-stl
// https://codeforces.com/blog/entry/21853

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

// Device memory size in bytes
extern double capacity;
extern double available_mem;
extern double allocated_mem;
extern int head; // do we need to keep a array of available head? ==> sure??
extern int device_id, host_id;
extern double *d_memory;
extern unsigned long int num_per_blk;
extern double memGranularity;
extern unsigned long int totalMemBlock;
extern unsigned long int viewIndex;
extern unsigned long int memBlockIndex;

extern unordered_map< pair<double *, int>, vector<double>, pair_hash, EQUALPAIR> mp;
extern map<int, vector<unsigned long int>> freeBlock;
extern unordered_map<double *, vector<int>> matrixDimension;
extern unordered_map<double *, string> matrixName;
extern vector<pair<double *, int>> memView;
extern queue<pair<double *, int>> evictionQueue; //cann't be more then once ==> if on device then don't insert otherwise insert
extern bool inEvictionMode;
extern bool errorFlag;
extern vector<double> mmTiming;
extern map<long, double> h2dTime, d2hTime;
extern map<long, double> h2dTransfer, d2hTransfer;
extern map<long, double> mmLoopTime, isDeviceFunctionTime, pushPopTime, freeBlockTiming, grabRHSTiming, dsUpdateTiming, copyToHostTiming;

// Memory util funcitons
double availbleDeviceMemory();
double deviceCapacity();
size_t allocated();
bool isAllBlocksOccupied();

bool isOnDevice(pair<double *, int> pr);
unsigned long int requiredDeviceBlocks(double blkz_in_byte, double memGranularity);
int copyToDevice(double *mtx, int row, int col, int block_id, int offset, int blksz, double isModified, long iterationNumber);
int reserveOnDevice(double *mtx, int row, int col, int block_id, int offset, int blksz, double isModified, long iterationNumber);
int copyToHost(double *host_mtx, int row, int col, int block_id, int block_width, int blksz, long iterationNumber);
double setMemoryGranularity(double size_in_byte);
unsigned long int availableBlocksOnRight(unsigned long int memView_index);

void printMapInfo();
void printMemoryUsageStat(long iterationNumber);
void setMatrixInfo(double *mtx, int row, int col, string mtxName);

template <typename T1, typename T2>
typename std::map<T1, T2>::const_iterator nearest_key(const std::map<T1, T2>& mp, T1 key);


#endif
