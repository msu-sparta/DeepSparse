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

void _XTY(double *X, double *Y, double *result ,int M, int N, int P, int blocksize);

#endif