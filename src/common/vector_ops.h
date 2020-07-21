#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

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
using namespace std;

#include <mkl.h>
#include <omp.h>
#include "exec_util.h"

void norm_task(double *Y, double *save_norm, int iterationNumber);
void normalize_task(double *Y, int n, int block_id, int block_width, double *save_norm, int iterationNumber);
void custom_dlacpy_vector(double *src, double *dst, int n, int block_width, int block_id);
void dotV(double *Y, int n, double *squared_norm, int block_id, int buf_id, int block_width);
void _XTY(double *X, double *Y, double *result ,int M, int N, int P, int blocksize);

#endif
