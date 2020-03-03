CC=g++
#CC=icpc
#CFLAGS=-std=c++11 -Wall -Wextra -Wconversion -O3  -qopenmp
#CFLAGS= -fopenmp -O3 -std=c++11 -m64 -I${MKL_ROOT}/include
CFLAGS= -march=knl -fopenmp -O3 -std=c++11 -m64 -I${MKL_ROOT}/include
#CFLAGS= -xMIC-AVX512 -qopenmp -O3 -std=c++11 -m64 -I${MKL_ROOT}/include
#CFLAGS= -qopenmp -O3 -std=c++11 -m64 -I${MKL_ROOT}/include
LINKER_INT32=-L${MKL_ROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl


EXEDIR=../exe

lobpcg_libcsr: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/vector_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c lobpcg_libcsr_executor.cpp   exec_util.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -o $(EXEDIR)/lobpcg_libcsr_executor.x exec_util.o lobpcg_libcsr_executor.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)


lobpcg_libcsb: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/vector_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c lobpcg_libcsb_executor.cpp   exec_util.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -o $(EXEDIR)/lobpcg_libcsb_executor.x exec_util.o lobpcg_libcsb_executor.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)

lobpcg_global: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/vector_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c lobpcg_global_executor.cpp   exec_util.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -o $(EXEDIR)/lobpcg_global_executor.x exec_util.o lobpcg_global_executor.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)


lobpcg_part: 
	$(CC) $(CFLAGS) -c ../../common/exec_util.cpp    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/matrix_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c ../../common/vector_ops.cpp exec_util.o    -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -c lobpcg_part_executor.cpp   exec_util.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)
	$(CC) $(CFLAGS) -o $(EXEDIR)/lobpcg_part_executor.x exec_util.o lobpcg_part_executor.o matrix_ops.o vector_ops.o  -g $(LINKER_INT32)



clean: 
	rm -rf *.o 
